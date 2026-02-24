"""Standalone OR Reversal strategy — fade failed OR breakouts.

After the opening range completes, fades price rejection at OR H/L
extremes using hammer / engulfing patterns.  Supports optional VWAP
disagreement filter, gap-fade context, low-ADX regime filter,
breakout-then-reclaim detection, and OR-opposite-side targets.

R:R filtered.  Fire-once per session.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategy.intraday.base_strategy import InfraExitConfig
from stockdownloader.strategy.intraday.trade_management import (
    IntradayExitManager,
    clamp_sl_dist,
    detect_reversal_patterns,
    make_entry_signal,
    reversal_pattern_label,
)
from stockdownloader.strategy.intraday.session_state import BarContext
from stockdownloader.strategy.intraday.infra import IntradayInfra
from stockdownloader.strategy.intraday.trail_strategy import VwapRatchetTrail
from stockdownloader.strategy.intraday.base_strategy import BaseIntradayStrategy
from stockdownloader.core.math import TWO, ZERO
from stockdownloader.util.pinescript.models import ModeDefinition
from stockdownloader.util.pinescript.modes import orr_mode

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategy.intraday.session_state import SessionState


@dataclass(frozen=True, slots=True)
class ORReversalStrategyConfig(InfraExitConfig):
    """All configurable parameters for the OR Reversal strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── ORR entry ───────────────────────────────────────────────────────
    orr_enable: bool = True
    orr_window: int = 78
    orr_prox: Decimal = Decimal("0.6")
    orr_sl_atr: Decimal = Decimal("0.5")
    orr_sl_cap: Decimal = Decimal("2.00")
    orr_tp_mode: str = "OR Mid"
    orr_min_rr: Decimal = Decimal("0.5")
    orr_max_rr: Decimal = Decimal("3.0")
    orr_rvol: Decimal = Decimal("1.0")
    orr_vwap_disagree: bool = False
    orr_gap_filter: bool = True
    orr_adx_filter: bool = True
    orr_adx_max: Decimal = Decimal("30")   # Block ORR when ADX >= this (strongly trending)
    orr_require_break: bool = True

    # ── Overrides (base provides allow_longs, w_sr, be_trigger) ──────────
    ps_engulf: Decimal = Decimal("0.30")       # 30% body-to-range for quality patterns
    allow_shorts: bool = True                  # ORR trades both sides (base: False)
    adx_thresh: Decimal = Decimal("21")        # Lower for mean-reversion (base: 22)


class ORReversalStrategy(BaseIntradayStrategy):
    """Standalone OR Reversal strategy.

    Fades OR extreme retests with hammer/engulfing pattern
    confirmation. Supports VWAP disagreement filter, gap-fade context,
    low-ADX regime filter, breakout-then-reclaim detection, and
    OR-opposite-side targets.  R:R filtered, fire-once per session.
    """

    def __init__(self, config: ORReversalStrategyConfig | None = None) -> None:
        c = config or ORReversalStrategyConfig()
        self._c = c
        self._infra = IntradayInfra(c, IntradayExitManager(VwapRatchetTrail()))

    _ENTRY_FLAGS = {"fire_once": True}

    @staticmethod
    def pinescript_mode() -> ModeDefinition:
        """Return the PineScript mode definition for OR Reversal."""
        return orr_mode()

    @property
    def name(self) -> str:
        return "OR Reversal"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        state = ctx.state

        if not c.orr_enable:
            return None
        if state.fired_today:
            return None
        if not state.or_done:
            return None

        # -- Track breakout flags (for require_break mode) --
        bar = ctx.bar
        if bar.close > state.or_high:
            state.orr_break_above = True
        if bar.close < state.or_low:
            state.orr_break_below = True

        # -- Window check --
        if ctx.bar_of_day <= c.or_bars or ctx.bar_of_day > c.orr_window:
            return None

        # -- Proximity to OR extremes --
        prox_dist = ctx.atr_val * c.orr_prox
        near_or_high = ctx.bar.high >= state.or_high - prox_dist
        near_or_low = ctx.bar.low <= state.or_low + prox_dist

        # -- Pattern detection (hammer / engulfing only) --
        bull_hammer, bear_hammer, bull_engulf, bear_engulf = detect_reversal_patterns(
            ctx.bar, ctx.prev_bar, ctx.atr_val, c.ps_engulf,
        )
        bull_signal = bull_hammer or bull_engulf
        bear_signal = bear_hammer or bear_engulf

        # -- Direction --
        go_long = near_or_low and bull_signal and c.allow_longs
        go_short = near_or_high and bear_signal and c.allow_shorts

        if not go_long and not go_short:
            return None

        # -- Require breakout-then-reclaim --
        if c.orr_require_break:
            # Long ORR: require a prior close below OR low (failed breakdown), now reclaiming
            if go_long and not state.orr_break_below:
                return None
            # Short ORR: require a prior close above OR high (failed breakout), now reclaiming
            if go_short and not state.orr_break_above:
                return None

        # -- RVOL filter --
        if ctx.rel_vol < c.orr_rvol:
            return None

        # -- VWAP disagreement filter --
        vwap = ctx.vwap_bands.vwap
        if c.orr_vwap_disagree:
            # Short ORR: want close < VWAP (breakout above OR high was against VWAP = good fade)
            if go_short and bar.close > vwap:
                return None
            # Long ORR: want close > VWAP (breakdown below OR low was against VWAP = good fade)
            if go_long and bar.close < vwap:
                return None

        # -- Gap fade filter --
        if c.orr_gap_filter:
            # Gap-fade: ORR should be AGAINST the gap direction
            if go_long and state.gap_dir > 0:
                return None  # Gap up, long ORR doesn't help fill the gap
            if go_short and state.gap_dir < 0:
                return None  # Gap down, short ORR doesn't help fill the gap

        # -- ADX filter (inverted — require LOW ADX for mean-reversion) --
        if c.orr_adx_filter and ctx.adx_val >= c.orr_adx_max:
            return None  # Too trendy for mean-reversion

        # -- Time-of-day gate: skip lunch chop --
        if not ctx.is_good_time:
            return None

        # -- SL / TP pre-calc and R:R filter --
        or_mid = (state.or_high + state.or_low) / TWO

        if go_long:
            sl_raw = ctx.bar.close - (state.or_low - ctx.atr_val * c.orr_sl_atr)
            sl_dist = clamp_sl_dist(sl_raw, c.orr_sl_cap)
            if sl_dist is None:
                return None
            sl_price = ctx.bar.close - sl_dist
            tp_price = self._compute_tp(True, vwap, or_mid, state, c)
            reward = tp_price - ctx.bar.close
        else:
            sl_raw = (state.or_high + ctx.atr_val * c.orr_sl_atr) - ctx.bar.close
            sl_dist = clamp_sl_dist(sl_raw, c.orr_sl_cap)
            if sl_dist is None:
                return None
            sl_price = ctx.bar.close + sl_dist
            tp_price = self._compute_tp(False, vwap, or_mid, state, c)
            reward = ctx.bar.close - tp_price

        rr = reward / sl_dist if sl_dist > ZERO else ZERO
        if rr < c.orr_min_rr or rr > c.orr_max_rr:
            return None

        # -- Scoring --
        score, max_score = self._compute_score(ctx, go_long, go_short, state, c)

        # -- Pattern label --
        pat = reversal_pattern_label(bull_hammer, bear_hammer, bull_engulf, bear_engulf)
        level = "ORL" if go_long else "ORH"

        state.fired_today = True
        return make_entry_signal(
            go_long=go_long,
            mode="ORR",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=f"ORR {pat} at {level}",
        )

    # -- Helpers --

    @staticmethod
    def _compute_tp(
        go_long: bool,
        vwap: Decimal,
        or_mid: Decimal,
        state: SessionState,
        c: ORReversalStrategyConfig,
    ) -> Decimal:
        """Compute take-profit price based on configured TP mode."""
        if c.orr_tp_mode == "VWAP":
            return vwap
        if c.orr_tp_mode == "OR Opposite":
            # Long ORR (entered near OR low) → target OR high (opposite)
            # Short ORR (entered near OR high) → target OR low (opposite)
            return state.or_high if go_long else state.or_low
        # "OR Mid"
        return or_mid

    @staticmethod
    def _compute_score(
        ctx: BarContext,
        go_long: bool,
        go_short: bool,
        state: SessionState,
        c: ORReversalStrategyConfig,
    ) -> tuple[int, int]:
        """Compute confluence score and max possible score."""
        pts_vol = 3 if ctx.rel_vol >= Decimal("2") else (1 if ctx.rel_vol >= Decimal("1") else 0)
        pts_vwap = 1 if (
            (go_short and ctx.bar.close < ctx.vwap_bands.vwap)
            or (go_long and ctx.bar.close > ctx.vwap_bands.vwap)
        ) else 0
        pts_gap = 1 if (
            (go_long and getattr(state, "gap_dir", 0) < 0)  # gap-down + long fade
            or (go_short and getattr(state, "gap_dir", 0) > 0)  # gap-up + short fade
        ) else 0
        pts_adx = 1 if ctx.adx_val < c.orr_adx_max else 0  # low ADX = good for ORR
        pts_sr = 1 if ctx.sr_any else 0
        score = pts_vol + pts_vwap + pts_gap + pts_adx + pts_sr
        max_score = 3 + 1 + 1 + 1 + c.w_sr
        return score, max_score
