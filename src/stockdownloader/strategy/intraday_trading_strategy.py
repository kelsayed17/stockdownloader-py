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

    @property
    @abstractmethod
    def name(self) -> str:
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

    @property
    @abstractmethod
    def warmup_period(self) -> int:
        """Return the number of bars needed before signals are valid."""

    # ------------------------------------------------------------------
    # Engine callbacks — non-abstract with default no-ops.
    #
    # The :class:`IntradayBacktestEngine` calls these after it actually
    # opens or closes a position, allowing strategies that track position
    # state internally to stay in sync with the engine's ground truth.
    # ------------------------------------------------------------------

    def on_position_opened(self, is_long: bool) -> None:
        """Called by the engine after a position is successfully opened.

        Strategies that track ``_in_position`` / ``_position_is_long``
        should override this to set those flags **only** here, rather
        than optimistically in :meth:`evaluate`.
        """

    def on_position_closed(self) -> None:
        """Called by the engine after a position is closed.

        Covers strategy-initiated EXIT signals, engine-level
        force-closes (end-of-data), and any future SL/TP handling.
        """
