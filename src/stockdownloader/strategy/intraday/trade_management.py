"""Trade management layer for intraday strategies.

Combines entry helpers (SL/TP pricing, signal construction, pattern detection)
with exit management (hard SL, fixed TP, breakeven, trail, EOD flatten).
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import IntradayAction, IntradaySignal, HOLD
from stockdownloader.core.models.trade import Direction
from stockdownloader.strategy.intraday.session_state import SessionState
from stockdownloader.strategy.intraday.trail_strategy import (
    BreakevenTrail,
    TrailStrategy,
)
from stockdownloader.core.math import ZERO
from stockdownloader.indicators.volume import ExtendedSessionVWAP
from stockdownloader.indicators.intraday import (
    is_bear_engulfing,
    is_bull_engulfing,
    is_hammer,
    is_inv_hammer,
)

if TYPE_CHECKING:
    from stockdownloader.strategy.intraday.base_strategy import InfraExitConfig

# ===================================================================
# SL / TP helpers
# ===================================================================


def directional_sl_tp(
    go_long: bool,
    close: Decimal,
    sl_dist: Decimal,
    tp_dist: Decimal,
) -> tuple[Decimal, Decimal]:
    """Compute SL and TP prices from distance values.

    Parameters
    ----------
    go_long:
        ``True`` for long entries, ``False`` for short.
    close:
        Current bar close price.
    sl_dist:
        Stop-loss distance (positive).
    tp_dist:
        Take-profit distance (positive).  Pass ``Decimal("0")`` for
        no fixed TP (e.g. trail-only modes).

    Returns
    -------
    (sl_price, tp_price)
    """
    if go_long:
        return close - sl_dist, close + tp_dist
    return close + sl_dist, close - tp_dist


def clamp_sl_dist(
    raw: Decimal,
    cap: Decimal,
) -> Decimal | None:
    """Clamp *raw* SL distance to *cap* and reject if non-positive.

    Returns ``None`` when the clamped value is <= 0 (caller should
    return ``None`` / skip entry).
    """
    dist = min(raw, cap)
    if dist <= ZERO:
        return None
    return dist


# ===================================================================
# Entry signal construction
# ===================================================================


def make_entry_signal(
    *,
    go_long: bool,
    mode: str,
    sl_price: Decimal,
    tp_price: Decimal,
    score: int,
    max_score: int,
    risk_per_share: Decimal,
    reason: str,
) -> IntradaySignal:
    """Build an ``IntradaySignal`` for a long or short entry.

    Consolidates the identical ``IntradaySignal(...)`` construction
    that appears at the end of every strategy's entry method.
    """
    return IntradaySignal(
        action=IntradayAction.ENTER_LONG if go_long else IntradayAction.ENTER_SHORT,
        mode=mode,
        stop_loss=sl_price,
        take_profit=tp_price,
        confluence_score=score,
        max_score=max_score,
        risk_per_share=risk_per_share,
        reason=reason,
    )


# ===================================================================
# Pattern detection helpers (shared by ORR + PS)
# ===================================================================


def detect_reversal_patterns(
    bar: IntradayPriceData,
    prev_bar: IntradayPriceData | None,
    atr_val: Decimal,
    engulf_ratio: Decimal,
) -> tuple[bool, bool, bool, bool]:
    """Detect hammer and engulfing reversal patterns.

    Returns
    -------
    (bull_hammer, bear_hammer, bull_engulf, bear_engulf)
    """
    bull_hammer = is_hammer(bar, atr_val)
    bear_hammer = is_inv_hammer(bar, atr_val)
    bull_engulf = prev_bar is not None and is_bull_engulfing(bar, prev_bar, engulf_ratio)
    bear_engulf = prev_bar is not None and is_bear_engulfing(bar, prev_bar, engulf_ratio)
    return bull_hammer, bear_hammer, bull_engulf, bear_engulf


def reversal_pattern_label(
    bull_hammer: bool,
    bear_hammer: bool,
    bull_engulf: bool,
    bear_engulf: bool,
) -> str:
    """Return a human-readable label for the matched reversal pattern.

    Must be called only when at least one of the flags is ``True``.
    """
    if bull_hammer:
        return "Hammer"
    if bear_hammer:
        return "Inv Hammer"
    if bull_engulf:
        return "Engulfing"
    return "Bear Engulf"


# ===================================================================
# Exit management
# ===================================================================

# Default trail when none is injected
_BE_TRAIL = BreakevenTrail()


class IntradayExitManager:
    """Evaluates exit conditions each bar while in a position.

    Accepts an optional ``trail_strategy`` for standalone strategies.
    The orchestrator calls :meth:`set_active_trail` before each entry
    to inject the appropriate trail for the entering strategy.
    """

    def __init__(self, trail_strategy: TrailStrategy | None = None) -> None:
        self._trail = trail_strategy

    def set_active_trail(self, trail: TrailStrategy) -> None:
        """Set the active trail strategy (used by orchestrator)."""
        self._trail = trail

    def evaluate(
        self,
        bar: IntradayPriceData,
        state: SessionState,
        config: InfraExitConfig,
        vwap_bands: ExtendedSessionVWAP,
        current_atr: Decimal,
        bar_of_day: int,
    ) -> IntradaySignal:
        """Return EXIT signal if exit conditions met, else HOLD."""
        if not state.in_position or state.position_direction is None:
            return HOLD

        is_long = state.position_direction == Direction.LONG
        entry = state.entry_price
        sl = state.stop_loss
        tp = state.take_profit

        # -- 1. Hard SL --
        if is_long and bar.low <= sl:
            return self._exit(state, sl, "hard_stop")
        if not is_long and bar.high >= sl:
            return self._exit(state, sl, "hard_stop")

        # -- 1b. OR re-entry invalidation (ORB only) --
        if config.orb_reentry_exit and state.entry_mode == "ORB":
            if is_long and bar.close < state.or_high:
                return self._exit(state, bar.close, "or_reentry")
            if not is_long and bar.close > state.or_low:
                return self._exit(state, bar.close, "or_reentry")

        # -- 1c. ORR re-breakout invalidation --
        if config.orr_rebreak_exit and state.entry_mode == "ORR":
            # ORR long (faded OR low): if price breaks below OR low again -> exit
            if is_long and bar.close < state.or_low:
                return self._exit(state, bar.close, "orr_rebreak")
            # ORR short (faded OR high): if price breaks above OR high again -> exit
            if not is_long and bar.close > state.or_high:
                return self._exit(state, bar.close, "orr_rebreak")

        # -- 2. Fixed TP --
        if tp > ZERO:
            if is_long and bar.high >= tp:
                return self._exit(state, tp, "take_profit")
            if not is_long and bar.low <= tp:
                return self._exit(state, tp, "take_profit")

        # -- 3. Update peak favorable excursion --
        if is_long:
            if bar.high > state.orb_extreme:
                state.orb_extreme = bar.high
        else:
            if state.orb_extreme == ZERO or bar.low < state.orb_extreme:
                state.orb_extreme = bar.low

        # -- 4. BE trigger + trail activation --
        risk = state.risk_amount
        if not state.be_triggered and risk > ZERO:
            unrealized = (bar.close - entry) if is_long else (entry - bar.close)
            if unrealized >= risk * config.be_trigger:
                state.be_triggered = True
                trail = self._resolve_trail()
                trail.activate(state, is_long, entry, current_atr, vwap_bands, config)

        # -- 5. Trail ratchet (per-bar) --
        trail = self._resolve_trail()
        if trail.ratchet(state, bar, is_long, entry, current_atr, vwap_bands, config):
            return self._exit(state, state.trail_level, trail.exit_reason)

        # -- 6. ORB time-based exit --
        if config.orb_time_exit > 0 and state.entry_mode == "ORB":
            if bar_of_day >= config.orb_time_exit:
                return self._exit(state, bar.close, "orb_time_exit")

        # -- 7. EOD close --
        if config.close_eod and bar_of_day >= config.bars_per_day:
            return self._exit(state, bar.close, "eod")

        return HOLD

    def _resolve_trail(self) -> TrailStrategy:
        """Return the active trail strategy.

        Falls back to the breakeven trail when no trail was injected.
        """
        if self._trail is not None:
            return self._trail
        return _BE_TRAIL

    # -- helpers --

    @staticmethod
    def _exit(state: SessionState, price: Decimal, reason: str) -> IntradaySignal:
        """Record exit in state and return EXIT signal."""
        pnl_per_share: Decimal
        if state.position_direction == Direction.LONG:
            pnl_per_share = price - state.entry_price
        else:
            pnl_per_share = state.entry_price - price

        if state.risk_amount > ZERO and state.session_start_equity > ZERO:
            shares_est = max(
                1,
                int(state.session_start_equity * Decimal("0.01") / state.risk_amount),
            )
        else:
            shares_est = 1

        state.session_pnl += pnl_per_share * Decimal(str(shares_est))

        if pnl_per_share <= ZERO:
            state.consec_losses += 1
        else:
            state.consec_losses = 0

        state.in_position = False
        state.position_direction = None
        state.be_triggered = False
        state.trailing_vwap = False
        state.trailing_atr = False
        state.trail_level = ZERO
        state.orb_extreme = ZERO

        return IntradaySignal(
            action=IntradayAction.EXIT,
            mode=state.entry_mode,
            stop_loss=price,
            reason=reason,
        )
