"""ATR_TRAIL exit mechanism: adaptive ATR-based trailing stop.

After the trade reaches *activation_r* R profit, a trailing stop is set at
``atr_multiplier * ATR(14)`` from the peak.  Unlike the fixed R-trail,
the distance adapts to current volatility.
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import Direction
from stockdownloader.strategy.exit_mechanism import ExitMechanism
from stockdownloader.util.technical_indicators import atr

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade

_ZERO = Decimal("0")


class AtrTrailExit(ExitMechanism):
    """0.3x ATR(14) trailing stop from peak after activation.

    Args:
        atr_multiplier: Fraction of ATR to use as trail distance.
        activation_r: R-multiple threshold before trailing begins.
        atr_period: ATR lookback period.
    """

    def __init__(
        self,
        atr_multiplier: Decimal = Decimal("0.3"),
        activation_r: Decimal = Decimal("1"),
        atr_period: int = 14,
    ) -> None:
        self._atr_multiplier = atr_multiplier
        self._activation_r = activation_r
        self._atr_period = atr_period
        self._data: list[IntradayPriceData] = []
        self._data_offset: int = 0
        self.reset()

    def get_name(self) -> str:
        return "ATR_TRAIL"

    def set_data_context(
        self, data: list[IntradayPriceData], entry_data_index: int
    ) -> None:
        """Set the full data list and entry index for ATR computation."""
        self._data = data
        self._data_offset = entry_data_index

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

        if not self._initialized:
            self._stop = (
                entry - stop_dist if direction == Direction.LONG
                else entry + stop_dist
            )
            self._peak = entry
            self._initialized = True

        data_index = self._data_offset + bar_index

        # Compute current ATR
        current_atr = _ZERO
        if self._data and data_index >= self._atr_period:
            current_atr = atr(self._data, data_index, self._atr_period)
        if current_atr <= _ZERO:
            current_atr = stop_dist  # fallback

        trail_dist = current_atr * self._atr_multiplier

        if direction == Direction.LONG:
            if bar.low <= self._stop:
                self._exit_price = self._stop
                self._exit_reason = "trail_stop" if self._be_triggered else "hard_stop"
                return True

            if bar.high > self._peak:
                self._peak = bar.high

            if not self._be_triggered and self._peak >= entry + stop_dist * self._activation_r:
                self._be_triggered = True

            if self._be_triggered:
                atr_stop = self._peak - trail_dist
                if atr_stop > self._stop:
                    self._stop = atr_stop
        else:
            if bar.high >= self._stop:
                self._exit_price = self._stop
                self._exit_reason = "trail_stop" if self._be_triggered else "hard_stop"
                return True

            if bar.low < self._peak:
                self._peak = bar.low

            if not self._be_triggered and self._peak <= entry - stop_dist * self._activation_r:
                self._be_triggered = True

            if self._be_triggered:
                atr_stop = self._peak + trail_dist
                if atr_stop < self._stop:
                    self._stop = atr_stop

        return False

    def get_exit_price(self) -> Decimal:
        return self._exit_price

    def get_exit_reason(self) -> str:
        return self._exit_reason

    def get_peak_favorable(self) -> Decimal:
        return self._peak
