"""Abstract base class for intraday trading strategies.

Unlike :class:`TradingStrategy` which operates on daily bars and returns a
simple BUY/SELL/HOLD signal, this ABC targets 5-minute intraday bars and
returns a rich :class:`IntradaySignal` carrying direction, stop/target levels,
confluence score, and entry mode metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import IntradaySignal


class IntradayTradingStrategy(ABC):
    """Base class for strategies that operate on intraday bars.

    Implementations maintain per-session state that resets at each new
    trading day.  The :meth:`evaluate` method is called once per bar in
    chronological order and must handle session-boundary detection
    internally (via ``IntradayPriceData.trading_date``).
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of this strategy."""

    @abstractmethod
    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        """Evaluate the strategy at *current_index* and return a signal.

        The full *data* list is provided so that the implementation can
        compute look-back indicators.  Implementations **must** detect
        session boundaries (new trading day) and reset state accordingly.
        """

    @abstractmethod
    def on_session_start(self, trading_date: str) -> None:
        """Reset session state for a new trading day.

        Called automatically by :meth:`evaluate` when it detects a day
        change, but may also be invoked externally for explicit control.
        """

    @abstractmethod
    def get_warmup_period(self) -> int:
        """Return the number of bars needed before signals are valid."""
