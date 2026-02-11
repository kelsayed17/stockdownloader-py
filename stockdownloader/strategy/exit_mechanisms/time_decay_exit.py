"""TIME_DECAY exit mechanism: trail tightens as the session progresses.

Before 11:00: 0.6R trail width.
11:00 -- 13:00: 0.4R trail width.
After 13:00: 0.2R trail width.

Activation and breakeven trigger at *activation_r* R.
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


class TimeDecayExit(ExitMechanism):
    """Trailing stop that tightens by time-of-day.

    Args:
        activation_r: R-multiple threshold for breakeven trigger.
        early_trail_r: Trail width before 11:00 (default 0.6R).
        mid_trail_r: Trail width 11:00-13:00 (default 0.4R).
        late_trail_r: Trail width after 13:00 (default 0.2R).
        be_buffer_pct: Percentage of 1R added as breakeven buffer.
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("1"),
        early_trail_r: Decimal = Decimal("0.6"),
        mid_trail_r: Decimal = Decimal("0.4"),
        late_trail_r: Decimal = Decimal("0.2"),
        be_buffer_pct: Decimal = Decimal("0.05"),
    ) -> None:
        self._activation_r = activation_r
        self._early_trail_r = early_trail_r
        self._mid_trail_r = mid_trail_r
        self._late_trail_r = late_trail_r
        self._be_buffer_pct = be_buffer_pct
        self.reset()

    def get_name(self) -> str:
        return "TIME_DECAY"

    def reset(self) -> None:
        self._stop: Decimal = _ZERO
        self._peak: Decimal = _ZERO
        self._be_triggered: bool = False
        self._exit_price: Decimal = _ZERO
        self._exit_reason: str = ""
        self._initialized: bool = False

    def _trail_multiplier(self, time_str: str) -> Decimal:
        """Choose trail width based on time of day.

        *time_str* is expected in ``HH:MM:SS`` format.
        """
        hour_str = time_str[:2]
        try:
            hour = int(hour_str)
        except ValueError:
            return self._mid_trail_r

        if hour < 11:
            return self._early_trail_r
        if hour < 13:
            return self._mid_trail_r
        return self._late_trail_r

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        direction = trade.direction
        entry = trade.entry_price
        stop_dist = trade.stop_distance
        be_buffer = stop_dist * self._be_buffer_pct

        if not self._initialized:
            self._stop = (
                entry - stop_dist if direction == Direction.LONG
                else entry + stop_dist
            )
            self._peak = entry
            self._initialized = True

        # Determine trail width from time of day
        time_str = bar.time_str
        trail_mult = self._trail_multiplier(time_str)
        trail_buffer = stop_dist * trail_mult

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
