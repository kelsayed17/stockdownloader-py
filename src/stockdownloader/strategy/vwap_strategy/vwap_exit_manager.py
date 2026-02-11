"""Exit management for the VWAP v11.2 strategy.

Handles per-bar exit evaluation: hard SL, fixed TP, breakeven trigger,
VWAP ratcheting trail (PB / ORR), ATR chandelier trail (ORB), and EOD
flatten.
"""

from __future__ import annotations

from decimal import Decimal

from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import IntradayAction, IntradaySignal, HOLD
from stockdownloader.model.trade import Direction
from stockdownloader.strategy.vwap_strategy.session_state import SessionState
from stockdownloader.strategy.vwap_strategy.vwap_config import VwapStrategyConfig
from stockdownloader.util.intraday_indicators import ExtendedSessionVWAP

_ZERO = Decimal("0")
_BE_BUF = Decimal("0.05")


class VwapExitManager:
    """Evaluates exit conditions each bar while in a position."""

    def evaluate(
        self,
        bar: IntradayPriceData,
        state: SessionState,
        config: VwapStrategyConfig,
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

        # ── 1. Hard SL ──────────────────────────────────────────────────
        if is_long and bar.low <= sl:
            return self._exit(state, sl, "hard_stop")
        if not is_long and bar.high >= sl:
            return self._exit(state, sl, "hard_stop")

        # ── 2. Fixed TP (not ORB — ORB uses trail only) ─────────────────
        if tp > _ZERO and state.entry_mode != "ORB":
            if is_long and bar.high >= tp:
                return self._exit(state, tp, "take_profit")
            if not is_long and bar.low <= tp:
                return self._exit(state, tp, "take_profit")

        # ── 3. Update peak favorable excursion ──────────────────────────
        if is_long:
            if bar.high > state.orb_extreme:
                state.orb_extreme = bar.high
        else:
            if state.orb_extreme == _ZERO or bar.low < state.orb_extreme:
                state.orb_extreme = bar.low

        # ── 4. BE trigger + trail activation ────────────────────────────
        risk = state.risk_amount
        if not state.be_triggered and risk > _ZERO:
            unrealized = (bar.close - entry) if is_long else (entry - bar.close)
            if unrealized >= risk * config.be_trigger:
                state.be_triggered = True

                if state.is_orb_trade:
                    # ORB: chandelier ATR trail
                    state.trailing_atr = True
                    if is_long:
                        state.trail_level = max(
                            state.orb_extreme - current_atr * config.orb_trail_atr,
                            entry + _BE_BUF,
                        )
                    else:
                        state.trail_level = min(
                            state.orb_extreme + current_atr * config.orb_trail_atr,
                            entry - _BE_BUF,
                        )
                elif state.is_pb_trade and config.trail_vwap:
                    # PB / ORR: VWAP ratcheting trail
                    state.trailing_vwap = True
                    vwap = vwap_bands.vwap
                    buf = current_atr * config.trail_buf
                    if is_long:
                        state.trail_level = max(vwap - buf, entry + _BE_BUF)
                    else:
                        state.trail_level = min(vwap + buf, entry - _BE_BUF)
                else:
                    # Simple BE
                    if is_long:
                        state.stop_loss = entry + _BE_BUF
                    else:
                        state.stop_loss = entry - _BE_BUF

        # ── 5. VWAP ratchet (per-bar, PB / ORR) ────────────────────────
        if state.trailing_vwap:
            vwap = vwap_bands.vwap
            buf = current_atr * config.trail_buf
            if is_long:
                new_trail = max(vwap - buf, entry + _BE_BUF)
                new_trail = max(new_trail, state.trail_level)
                if new_trail > state.trail_level:
                    state.trail_level = new_trail
                    state.stop_loss = new_trail
                    if not config.trail_keep_tp:
                        state.take_profit = _ZERO
                # Check trail stop
                if bar.low <= state.trail_level:
                    return self._exit(state, state.trail_level, "vwap_trail")
            else:
                new_trail = min(vwap + buf, entry - _BE_BUF)
                new_trail = min(new_trail, state.trail_level)
                if new_trail < state.trail_level:
                    state.trail_level = new_trail
                    state.stop_loss = new_trail
                    if not config.trail_keep_tp:
                        state.take_profit = _ZERO
                if bar.high >= state.trail_level:
                    return self._exit(state, state.trail_level, "vwap_trail")

        # ── 6. ATR chandelier ratchet (per-bar, ORB) ───────────────────
        if state.trailing_atr:
            if is_long:
                new_trail = max(
                    state.orb_extreme - current_atr * config.orb_trail_atr,
                    entry + _BE_BUF,
                )
                new_trail = max(new_trail, state.trail_level)
                if new_trail > state.trail_level:
                    state.trail_level = new_trail
                    state.stop_loss = new_trail
                if bar.low <= state.trail_level:
                    return self._exit(state, state.trail_level, "atr_trail")
            else:
                new_trail = min(
                    state.orb_extreme + current_atr * config.orb_trail_atr,
                    entry - _BE_BUF,
                )
                new_trail = min(new_trail, state.trail_level)
                if new_trail < state.trail_level:
                    state.trail_level = new_trail
                    state.stop_loss = new_trail
                if bar.high >= state.trail_level:
                    return self._exit(state, state.trail_level, "atr_trail")

        # ── 7. EOD close ────────────────────────────────────────────────
        if config.close_eod and bar_of_day >= config.bars_per_day:
            return self._exit(state, bar.close, "eod")

        return HOLD

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _exit(state: SessionState, price: Decimal, reason: str) -> IntradaySignal:
        """Record exit in state and return EXIT signal."""
        pnl: Decimal
        if state.position_direction == Direction.LONG:
            pnl = price - state.entry_price
        else:
            pnl = state.entry_price - price

        state.session_pnl += pnl

        if pnl <= _ZERO:
            state.consec_losses += 1
        else:
            state.consec_losses = 0

        state.in_position = False
        state.position_direction = None
        state.be_triggered = False
        state.trailing_vwap = False
        state.trailing_atr = False
        state.trail_level = _ZERO
        state.orb_extreme = _ZERO
        state.is_pb_trade = False
        state.is_orb_trade = False

        return IntradaySignal(
            action=IntradayAction.EXIT,
            mode=state.entry_mode,
            stop_loss=price,
            reason=reason,
        )
