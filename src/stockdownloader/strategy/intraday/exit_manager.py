"""Exit management for intraday strategies.

Handles per-bar exit evaluation: hard SL, fixed TP, breakeven trigger,
trail activation/ratcheting via pluggable TrailStrategy, and EOD flatten.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.model.trade import IntradayAction, IntradaySignal, HOLD
from stockdownloader.model.trade import Direction
from stockdownloader.strategy.intraday.session_state import SessionState
from stockdownloader.strategy.intraday.trail_strategy import (
    BreakevenTrail,
    TrailStrategy,
)
from stockdownloader.util.intraday_indicators import ExtendedSessionVWAP
from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from stockdownloader.strategy.intraday.base_config import InfraExitConfig

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
            # ORR long (faded OR low): if price breaks below OR low again → exit
            if is_long and bar.close < state.or_low:
                return self._exit(state, bar.close, "orr_rebreak")
            # ORR short (faded OR high): if price breaks above OR high again → exit
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
