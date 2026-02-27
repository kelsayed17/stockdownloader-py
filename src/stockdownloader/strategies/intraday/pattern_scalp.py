"""Standalone Pattern Scalp strategy — opening-range manipulation fade.

Fires when the opening range is >= 30% of daily ATR (manipulation),
a reversal pattern (hammer / engulfing) appears in the opposite
direction, and RVOL confirms.  TP = fraction of OR range.

This strategy does **not** use VWAP for entry decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.base import InfraExitConfig
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    clamp_sl_dist,
    detect_reversal_patterns,
    directional_sl_tp,
    make_entry_signal,
    reversal_pattern_label,
)
from stockdownloader.strategies.intraday.session import BarContext
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trail import BreakevenTrail
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy
from stockdownloader.core.math import HUNDRED, ZERO
from stockdownloader.pinescript.models import ModeDefinition
from stockdownloader.pinescript.modes import ps_mode

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData


@dataclass(frozen=True, slots=True)
class PatternScalpStrategyConfig(InfraExitConfig):
    """All configurable parameters for the Pattern Scalp strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── PS entry ────────────────────────────────────────────────────────
    ps_enable: bool = True
    ps_atr_pct: Decimal = Decimal("20.0")      # 20% of daily ATR → more manipulation days
    ps_window: int = 35                        # Wider window for more setups
    ps_engulf: Decimal = Decimal("0.25")       # 25% body-to-range (research minimum)
    ps_rvol: Decimal = Decimal("1.5")          # Require above-average volume
    ps_sma_filter: bool = True                 # Require SMA alignment
    ps_sl_mode: str = "ATR-Based"
    ps_sl_atr: Decimal = Decimal("1.3")        # Tighter SL
    ps_sl_cap: Decimal = Decimal("1.50")       # Tighter cap
    ps_tp_pct: Decimal = Decimal("75.0")
    ps_min_rr: Decimal = Decimal("1.5")        # Higher R:R to protect against losses
    ps_htf_align: bool = True
    ps_time_gate: bool = True                   # Skip lunch chop (not in PineScript)

    # allow_longs, allow_shorts, w_sr inherited from base


class PatternScalpStrategy(BaseIntradayStrategy):
    """Standalone Pattern Scalp strategy.

    Fades opening-range manipulation: fires when OR range >= daily ATR
    percentage, a reversal pattern appears in the opposite direction,
    and RVOL confirms.
    """

    def __init__(self, config: PatternScalpStrategyConfig | None = None, **overrides: object) -> None:
        if config is not None and overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if overrides:
            c = PatternScalpStrategyConfig(**overrides)
        else:
            c = config or PatternScalpStrategyConfig()
        self._c = c
        self._infra = IntradayInfra(c, IntradayExitManager(BreakevenTrail()))

    _ENTRY_FLAGS = {"fire_once": True}

    @staticmethod
    def pinescript_mode() -> ModeDefinition:
        """Return the PineScript mode definition for Pattern Scalp."""
        return ps_mode()

    @property
    def name(self) -> str:
        return "Pattern Scalp"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        state = ctx.state

        if not c.ps_enable:
            return None
        if state.fired_today:
            return None
        if not state.is_manip:
            return None

        # -- Window check --
        if ctx.bar_of_day < 4 or ctx.bar_of_day > c.ps_window:
            return None

        # -- Time-of-day gate: skip lunch chop (configurable) --
        if c.ps_time_gate and not ctx.is_good_time:
            return None

        # -- Pattern detection --
        bull_hammer, bear_hammer, bull_engulf, bear_engulf = detect_reversal_patterns(
            ctx.bar, ctx.prev_bar, ctx.atr_val, c.ps_engulf,
        )

        bull_reversal = bull_hammer or bull_engulf
        bear_reversal = bear_hammer or bear_engulf

        # -- Direction (fade the OR direction) --
        go_long = state.or_dir == -1 and bull_reversal and c.allow_longs
        go_short = state.or_dir == 1 and bear_reversal and c.allow_shorts

        if not go_long and not go_short:
            return None

        # -- RVOL filter --
        if ctx.rel_vol < c.ps_rvol:
            return None

        # -- SMA bias filter (optional) --
        if c.ps_sma_filter and state.daily_sma > ZERO:
            sma_dist = (ctx.bar.close - state.daily_sma) / state.daily_sma * HUNDRED
            daily_bias = 1 if sma_dist > Decimal("0.5") else (-1 if sma_dist < Decimal("-0.5") else 0)
            sma_ok = (
                (state.or_dir == -1 and daily_bias >= 1)
                or (state.or_dir == 1 and daily_bias <= -1)
            )
            if not sma_ok:
                return None

        # -- HTF trend alignment --
        if c.ps_htf_align:
            if go_long and ctx.htf_trend < 0:
                go_long = False
            if go_short and ctx.htf_trend > 0:
                go_short = False
            if not go_long and not go_short:
                return None

        # -- SL calculation --
        if c.ps_sl_mode == "Day Extreme":
            if go_long:
                sl_raw = ctx.bar.close - state.day_lod + ctx.atr_val * Decimal("0.1")
            else:
                sl_raw = state.day_hod - ctx.bar.close + ctx.atr_val * Decimal("0.1")
        else:
            sl_raw = ctx.atr_val * c.ps_sl_atr

        sl_dist = clamp_sl_dist(sl_raw, c.ps_sl_cap)
        if sl_dist is None:
            return None

        # -- TP calculation --
        tp_dist = state.or_range * (c.ps_tp_pct / HUNDRED)

        # -- Minimum R:R check --
        if sl_dist > ZERO and tp_dist / sl_dist < c.ps_min_rr:
            return None

        sl_price, tp_price = directional_sl_tp(go_long, ctx.bar.close, sl_dist, tp_dist)

        # -- Pattern label --
        pat = reversal_pattern_label(bull_hammer, bear_hammer, bull_engulf, bear_engulf)

        # -- Scoring --
        pts_vol = 3 if ctx.rel_vol >= Decimal("2") else (1 if ctx.rel_vol >= Decimal("1") else 0)
        pts_sr = c.w_sr if state.or_done else 0
        score = pts_vol + pts_sr
        max_score = 3 + c.w_sr + 1

        state.fired_today = True
        return make_entry_signal(
            go_long=go_long,
            mode="PS",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=f"PS {pat}",
        )
