"""VWAP_CROSS exit mechanism: exit on session VWAP cross.

After the trade reaches *activation_r* R profit, the session VWAP becomes
the trailing reference.  A long exits if close < VWAP; a short exits if
close > VWAP.

Use ``activation_r=0.5`` for ``VWAP_CROSS`` and ``activation_r=1.0`` for
``VWAP_CROSS_LATE``.
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import Direction
from stockdownloader.strategy.exit_mechanism import ExitMechanism
from stockdownloader.util.technical_indicators import session_vwap

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade

_ZERO = Decimal("0")


class VwapCrossExit(ExitMechanism):
    """Exit on VWAP cross after activation threshold.

    Args:
        activation_r: R-multiple threshold before VWAP trailing begins.
        name: Display name override (defaults to ``'VWAP_CROSS'``).
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("0.5"),
        name: str | None = None,
    ) -> None:
        self._activation_r = activation_r
        self._name = name or "VWAP_CROSS"
        # Data list set by engine before replay
        self._data: list[IntradayPriceData] = []
        self._data_offset: int = 0
        self.reset()

    def get_name(self) -> str:
        return self._name

    def set_data_context(
        self, data: list[IntradayPriceData], entry_data_index: int
    ) -> None:
        """Set the full data list and entry index for VWAP computation."""
        self._data = data
        self._data_offset = entry_data_index

    def reset(self) -> None:
        self._stop: Decimal = _ZERO
        self._peak: Decimal = _ZERO
        self._vwap_active: bool = False
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
        activation = stop_dist * self._activation_r

        if not self._initialized:
            self._stop = (
                entry - stop_dist if direction == Direction.LONG
                else entry + stop_dist
            )
            self._peak = entry
            self._initialized = True

        data_index = self._data_offset + bar_index

        if direction == Direction.LONG:
            if bar.low <= self._stop:
                self._exit_price = self._stop
                self._exit_reason = "hard_stop"
                return True

            if bar.high > self._peak:
                self._peak = bar.high

            if not self._vwap_active and self._peak >= entry + activation:
                self._vwap_active = True

            if self._vwap_active and self._data:
                vwap_val = session_vwap(self._data, data_index)
                if bar.close < vwap_val:
                    self._exit_price = bar.close
                    self._exit_reason = "vwap_cross"
                    return True
        else:
            if bar.high >= self._stop:
                self._exit_price = self._stop
                self._exit_reason = "hard_stop"
                return True

            if bar.low < self._peak:
                self._peak = bar.low

            if not self._vwap_active and self._peak <= entry - activation:
                self._vwap_active = True

            if self._vwap_active and self._data:
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
