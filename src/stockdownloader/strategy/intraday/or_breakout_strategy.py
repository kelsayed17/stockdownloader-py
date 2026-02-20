"""Standalone OR Breakout strategy — momentum continuation.

Enters when a 5-minute candle closes beyond the OR extreme with strong
volume confirmation (RVOL >= 2.0) and VWAP alignment.

Supports two entry modes:
  - **aggressive** (default): enter immediately on the breakout candle close
  - **retest**: wait for price to pull back and touch the broken OR level,
    then enter on the bounce

Stop-loss variants: ``"OR Opposite"``, ``"OR Midpoint"``, ``"ATR-Based"``.
Take-profit modes: ``"trail_only"`` (ATR chandelier), ``"or_range"`` (1× OR
height), ``"2x_or_range"`` (2× OR height).

Optional filters: gap context, ADX regime, VWAP alignment.
Optional exits: OR re-entry invalidation, time-based force-close.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import IntradaySignal
from stockdownloader.strategy.intraday.base_config import InfraExitConfig
from stockdownloader.strategy.intraday.entry_helpers import make_entry_signal
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.session_state import BarContext
from stockdownloader.strategy.intraday.infra import IntradayInfra
from stockdownloader.strategy.intraday.trail_strategy import AtrChandelierTrail
from stockdownloader.strategy.intraday.base_strategy import BaseIntradayStrategy
from stockdownloader.util.intraday_indicators import candle_strength
from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.pinescript_models import ModeDefinition
from stockdownloader.util.pinescript_modes import orb_mode

if TYPE_CHECKING:
    from stockdownloader.model.price_data import IntradayPriceData
    from stockdownloader.strategy.intraday.session_state import SessionState


@dataclass(frozen=True, slots=True)
class ORBreakoutStrategyConfig(InfraExitConfig):
    """All configurable parameters for the OR Breakout strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── ORB entry ───────────────────────────────────────────────────────
    orb_enable: bool = True
    orb_window: int = 30                            # 30-min OR (research-backed)
    orb_rvol: Decimal = Decimal("2.0")             # Higher volume requirement
    orb_sl_mode: str = "OR Midpoint"               # Tighter SL using OR midpoint
    orb_sl_atr: Decimal = Decimal("1.5")
    orb_sl_cap: Decimal = Decimal("2.00")          # Tighter cap
    orb_vwap_align: bool = True
    orb_body_min: Decimal = Decimal("0.25")        # Stronger breakout candle
    orb_entry_mode: str = "aggressive"
    orb_retest_bars: int = 5
    orb_tp_mode: str = "2x_or_range"               # Larger TP target
    orb_gap_filter: bool = True
    orb_adx_filter: bool = True
    orb_nr7_filter: bool = False
    orb_htf_align: bool = True                     # Require HTF alignment

    # ── Overrides (base provides allow_longs, allow_shorts, w_sr) ────────
    adx_thresh: Decimal = Decimal("25")            # Higher for trend-following (base: 22)

_TWO = Decimal("2")
_ONE_HALF = Decimal("1.5")


class ORBreakoutStrategy(BaseIntradayStrategy):
    """Standalone OR Breakout strategy.

    Momentum continuation: enters when a candle closes beyond the
    opening range extreme with strong volume (RVOL >= 2.0) and VWAP
    alignment.  Supports aggressive and conservative (retest) entry
    modes, multiple SL/TP variants, and optional gap/ADX filters.
    """

    def __init__(self, config: ORBreakoutStrategyConfig | None = None) -> None:
        c = config or ORBreakoutStrategyConfig()
        self._c = c
        self._infra = IntradayInfra(c, IntradayExitManager(AtrChandelierTrail()))

    _ENTRY_FLAGS = {"fire_once": True}

    @staticmethod
    def pinescript_mode() -> ModeDefinition:
        """Return the PineScript mode definition for OR Breakout."""
        return orb_mode()

    @property
    def name(self) -> str:
        return "OR Breakout"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        state = ctx.state

        if not c.orb_enable:
            return None
        if state.fired_today:
            return None
        if not state.or_done:
            return None

        # -- NR7 compression filter --
        if c.orb_nr7_filter and not state.is_nr7:
            return None

        # -- Retest mode: check for pending retest first --
        if c.orb_entry_mode == "retest" and state.orb_breakout_pending:
            return self._check_retest(ctx)

        # -- Window check --
        if ctx.bar_of_day <= c.or_bars or ctx.bar_of_day > c.orb_window:
            return None

        # -- Time-of-day gate: skip lunch chop --
        if not ctx.is_good_time:
            return None

        # -- Close beyond OR extreme --
        bar = ctx.bar
        close_long = bar.close > state.or_high and bar.close > bar.open
        close_short = bar.close < state.or_low and bar.close < bar.open

        if not close_long and not close_short:
            return None

        # -- Body filter --
        cs = candle_strength(bar, ctx.atr_val)
        if cs.body_atr < c.orb_body_min:
            return None

        # -- RVOL filter --
        if ctx.rel_vol < c.orb_rvol:
            return None

        # -- VWAP alignment --
        vwap = ctx.vwap_bands.vwap
        if c.orb_vwap_align:
            if close_long and bar.close <= vwap:
                return None
            if close_short and bar.close >= vwap:
                return None

        # -- Direction --
        go_long = close_long and c.allow_longs
        go_short = close_short and c.allow_shorts

        if not go_long and not go_short:
            return None

        # -- Gap filter --
        if c.orb_gap_filter:
            if go_long and state.gap_dir < 0:
                return None
            if go_short and state.gap_dir > 0:
                return None

        # -- ADX filter --
        if c.orb_adx_filter and ctx.adx_val < c.adx_thresh:
            return None

        # -- HTF trend alignment --
        if c.orb_htf_align:
            if go_long and ctx.htf_trend < 0:
                go_long = False
            if go_short and ctx.htf_trend > 0:
                go_short = False
            if not go_long and not go_short:
                return None

        # -- SL calculation --
        sl_dist = self._compute_sl_dist(go_long, bar.close, state, ctx.atr_val, c)
        if sl_dist is None or sl_dist <= ZERO:
            return None

        # -- Retest mode: store pending breakout instead of entering --
        if c.orb_entry_mode == "retest":
            state.orb_breakout_pending = True
            state.orb_breakout_long = go_long
            state.orb_breakout_level = state.or_high if go_long else state.or_low
            state.orb_breakout_bar = ctx.bar_of_day
            state.orb_breakout_sl = sl_dist
            return None

        # -- TP calculation --
        tp_price = self._compute_tp(go_long, bar.close, state.or_range, c)

        # -- Scoring --
        score, max_score = self._compute_score(ctx, go_long, state, c)

        level = "ORH" if go_long else "ORL"
        sl_price = bar.close - sl_dist if go_long else bar.close + sl_dist

        state.fired_today = True
        return make_entry_signal(
            go_long=go_long,
            mode="ORB",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=f"ORB Break {level}",
        )

    # -- Retest logic --

    def _check_retest(self, ctx: BarContext) -> IntradaySignal | None:
        """Check if a pending breakout retest has been confirmed."""
        c = self._c
        state = ctx.state
        bar = ctx.bar

        # Timeout: cancel if too many bars have passed
        if ctx.bar_of_day - state.orb_breakout_bar > c.orb_retest_bars:
            state.orb_breakout_pending = False
            return None

        go_long = state.orb_breakout_long
        level = state.orb_breakout_level

        # Retest detection: price touches near the broken level and closes
        # on the correct side (bounce confirmation).
        if go_long:
            # Bar low touches or dips near OR high, closes above it
            if bar.low <= level and bar.close > level:
                return self._enter_retest(ctx, go_long=True)
        else:
            # Bar high touches or spikes near OR low, closes below it
            if bar.high >= level and bar.close < level:
                return self._enter_retest(ctx, go_long=False)

        return None

    def _enter_retest(self, ctx: BarContext, go_long: bool) -> IntradaySignal:
        """Fire the retest entry using stored breakout data."""
        c = self._c
        state = ctx.state
        bar = ctx.bar

        sl_dist = state.orb_breakout_sl
        sl_price = bar.close - sl_dist if go_long else bar.close + sl_dist
        tp_price = self._compute_tp(go_long, bar.close, state.or_range, c)
        score, max_score = self._compute_score(ctx, go_long, state, c)

        level = "ORH" if go_long else "ORL"

        state.orb_breakout_pending = False
        state.fired_today = True
        return make_entry_signal(
            go_long=go_long,
            mode="ORB",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=f"ORB Retest {level}",
        )

    # -- Helpers --

    @staticmethod
    def _compute_sl_dist(
        go_long: bool,
        close: Decimal,
        state: SessionState,
        atr_val: Decimal,
        c: ORBreakoutStrategyConfig,
    ) -> Decimal | None:
        """Compute stop-loss distance based on configured SL mode."""
        if go_long:
            if c.orb_sl_mode == "OR Opposite":
                sl_dist = min(close - state.or_low, atr_val * c.orb_sl_atr)
            elif c.orb_sl_mode == "OR Midpoint":
                or_mid = (state.or_high + state.or_low) / _TWO
                sl_dist = min(close - or_mid, atr_val * c.orb_sl_atr)
            else:  # ATR-Based
                sl_dist = atr_val * c.orb_sl_atr
        else:
            if c.orb_sl_mode == "OR Opposite":
                sl_dist = min(state.or_high - close, atr_val * c.orb_sl_atr)
            elif c.orb_sl_mode == "OR Midpoint":
                or_mid = (state.or_high + state.or_low) / _TWO
                sl_dist = min(or_mid - close, atr_val * c.orb_sl_atr)
            else:  # ATR-Based
                sl_dist = atr_val * c.orb_sl_atr
        sl_dist = min(sl_dist, c.orb_sl_cap)
        if sl_dist <= ZERO:
            return None
        return sl_dist

    @staticmethod
    def _compute_tp(
        go_long: bool,
        close: Decimal,
        or_range: Decimal,
        c: ORBreakoutStrategyConfig,
    ) -> Decimal:
        """Compute take-profit price based on configured TP mode."""
        if c.orb_tp_mode == "or_range":
            return close + or_range if go_long else close - or_range
        if c.orb_tp_mode == "2x_or_range":
            return close + or_range * _TWO if go_long else close - or_range * _TWO
        # "trail_only" — no fixed TP
        return ZERO

    @staticmethod
    def _compute_score(
        ctx: BarContext,
        go_long: bool,
        state: SessionState,
        c: ORBreakoutStrategyConfig,
    ) -> tuple[int, int]:
        """Compute confluence score and max possible score."""
        pts_vol = 3 if ctx.rel_vol >= _TWO else (2 if ctx.rel_vol >= _ONE_HALF else 0)
        pts_adx = 1 if ctx.adx_val >= c.adx_thresh else 0
        pts_gap = 1 if (
            (go_long and getattr(state, "gap_dir", 0) > 0)
            or (not go_long and getattr(state, "gap_dir", 0) < 0)
        ) else 0
        pts_sr = 1 if ctx.sr_any else 0
        score = pts_vol + pts_adx + pts_gap + pts_sr
        max_score = 3 + 1 + 1 + 1 + c.w_sr
        return score, max_score
