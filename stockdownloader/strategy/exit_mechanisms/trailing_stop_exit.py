"""CURRENT_TRAIL exit mechanism: fixed R-based trailing stop.

Once the trade reaches *activation_r* R profit, a breakeven stop is set.
Thereafter a trail of *trail_r* R follows below (long) or above (short)
the peak favorable excursion.
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import Direction
from stockdownloader.strategy.exit_mechanism import ExitMechanism

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade

_ZERO = Decimal("0")


class TrailingStopExit(ExitMechanism):
    """0.5R trailing stop from peak after 1R breakeven trigger.

    Args:
        activation_r: R-multiple at which the breakeven stop activates.
        trail_r: R-multiple width of the trailing stop from peak.
        be_buffer_pct: Percentage of 1R to add as a breakeven buffer.
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("1"),
        trail_r: Decimal = Decimal("0.5"),
        be_buffer_pct: Decimal = Decimal("0.05"),
    ) -> None:
        self._activation_r = activation_r
        self._trail_r = trail_r
        self._be_buffer_pct = be_buffer_pct
        self.reset()

    def get_name(self) -> str:
        return "CURRENT_TRAIL"

    def reset(self) -> None:
        self._stop: Decimal = _ZERO
        self._peak: Decimal = _ZERO
        self._be_triggered: bool = False
        self._exit_price: Decimal = _ZERO
        self._exit_reason: str = ""
        self._initialized: bool = False

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        direction = trade.direction
        entry = trade.entry_price
        stop_dist = trade.stop_distance
        trail_buffer = stop_dist * self._trail_r
        be_buffer = stop_dist * self._be_buffer_pct

        if not self._initialized:
            self._stop = (
                entry - stop_dist if direction == Direction.LONG
                else entry + stop_dist
            )
            self._peak = entry
            self._initialized = True

        if direction == Direction.LONG:
            if bar.low <= self._stop:
                self._exit_price = self._stop
                self._exit_reason = "trail_stop" if self._be_triggered else "hard_stop"
                return True

            if bar.high > self._peak:
                self._peak = bar.high

            if not self._be_triggered and self._peak >= entry + stop_dist * self._activation_r:
                self._be_triggered = True
                self._stop = entry + be_buffer

            if self._be_triggered:
                new_stop = self._peak - trail_buffer
                if new_stop > self._stop:
                    self._stop = new_stop
        else:
            if bar.high >= self._stop:
                self._exit_price = self._stop
                self._exit_reason = "trail_stop" if self._be_triggered else "hard_stop"
                return True

            if bar.low < self._peak:
                self._peak = bar.low

            if not self._be_triggered and self._peak <= entry - stop_dist * self._activation_r:
                self._be_triggered = True
                self._stop = entry - be_buffer

            if self._be_triggered:
                new_stop = self._peak + trail_buffer
                if new_stop < self._stop:
                    self._stop = new_stop

        return False

    def get_exit_price(self) -> Decimal:
        return self._exit_price

    def get_exit_reason(self) -> str:
        return self._exit_reason

    def get_peak_favorable(self) -> Decimal:
        return self._peak
