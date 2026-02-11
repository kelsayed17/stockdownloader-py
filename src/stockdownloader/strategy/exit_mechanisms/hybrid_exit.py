"""HYBRID exit mechanism: ATR trail + VWAP cross combination.

After activation, both ATR trailing stop and VWAP cross are active.
The trade exits at whichever triggers first.

Use ``activation_r=0.5`` for ``HYBRID`` and ``activation_r=1.0`` for
``HYBRID_LATE``.
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import Direction
from stockdownloader.strategy.exit_mechanism import ExitMechanism
from stockdownloader.util.technical_indicators import atr, session_vwap

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade

_ZERO = Decimal("0")


class HybridExit(ExitMechanism):
    """ATR trail + VWAP cross, whichever triggers first.

    Args:
        activation_r: R-multiple threshold before both trails activate.
        atr_multiplier: Fraction of ATR to use as trail distance.
        atr_period: ATR lookback period.
        name: Display name override.
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("0.5"),
        atr_multiplier: Decimal = Decimal("0.3"),
        atr_period: int = 14,
        name: str | None = None,
    ) -> None:
        self._activation_r = activation_r
        self._atr_multiplier = atr_multiplier
        self._atr_period = atr_period
        self._name = name or "HYBRID"
        self._data: list[IntradayPriceData] = []
        self._data_offset: int = 0
        self.reset()

    def get_name(self) -> str:
        return self._name

    def set_data_context(
        self, data: list[IntradayPriceData], entry_data_index: int
    ) -> None:
        """Set the full data list and entry index for indicator computation."""
        self._data = data
        self._data_offset = entry_data_index

    def reset(self) -> None:
        self._stop: Decimal = _ZERO
        self._peak: Decimal = _ZERO
        self._trailing_active: bool = False
        self._exit_price: Decimal = _ZERO
        self._exit_reason: str = ""
        self._initialized: bool = False
        self._be_buffer: Decimal = _ZERO

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        direction = trade.direction
        entry = trade.entry_price
        stop_dist = trade.stop_distance
        activation = stop_dist * self._activation_r

        if not self._initialized:
            self._stop = (
                entry - stop_dist if direction == Direction.LONG
                else entry + stop_dist
            )
            self._peak = entry
            self._be_buffer = stop_dist * Decimal("0.05")
            self._initialized = True

        data_index = self._data_offset + bar_index

        # Compute current ATR
        current_atr = _ZERO
        if self._data and data_index >= self._atr_period:
            current_atr = atr(self._data, data_index, self._atr_period)
        if current_atr <= _ZERO:
            current_atr = stop_dist

        trail_dist = current_atr * self._atr_multiplier

        if direction == Direction.LONG:
            if bar.low <= self._stop:
                self._exit_price = self._stop
                self._exit_reason = "hard_stop" if not self._trailing_active else "trail_stop"
                return True

            if bar.high > self._peak:
                self._peak = bar.high

            if not self._trailing_active and self._peak >= entry + activation:
                self._trailing_active = True
                self._stop = entry + self._be_buffer

            if self._trailing_active:
                atr_stop = self._peak - trail_dist
                if atr_stop > self._stop:
                    self._stop = atr_stop

                if self._data:
                    vwap_val = session_vwap(self._data, data_index)
                    if bar.close < vwap_val:
                        self._exit_price = bar.close
                        self._exit_reason = "vwap_cross"
                        return True
        else:
            if bar.high >= self._stop:
                self._exit_price = self._stop
                self._exit_reason = "hard_stop" if not self._trailing_active else "trail_stop"
                return True

            if bar.low < self._peak:
                self._peak = bar.low

            if not self._trailing_active and self._peak <= entry - activation:
                self._trailing_active = True
                self._stop = entry - self._be_buffer

            if self._trailing_active:
                atr_stop = self._peak + trail_dist
                if atr_stop < self._stop:
                    self._stop = atr_stop

                if self._data:
                    vwap_val = session_vwap(self._data, data_index)
                    if bar.close > vwap_val:
                        self._exit_price = bar.close
                        self._exit_reason = "vwap_cross"
                        return True

        return False

    def get_exit_price(self) -> Decimal:
        return self._exit_price

    def get_exit_reason(self) -> str:
        return self._exit_reason

    def get_peak_favorable(self) -> Decimal:
        return self._peak
