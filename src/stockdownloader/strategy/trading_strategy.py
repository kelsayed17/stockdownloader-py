"""Interface defining the contract for trading strategies.

Provides the base classes for both daily and intraday trading strategies:

- :class:`Signal` — BUY/SELL/HOLD enum for daily strategies
- :class:`TradingStrategy` — ABC for daily strategies operating on
  :class:`~stockdownloader.model.price_data.PriceData` bars
- :class:`IntradayTradingStrategy` — ABC for intraday strategies operating on
  :class:`~stockdownloader.model.price_data.IntradayPriceData` bars and
  returning rich :class:`~stockdownloader.model.trade.IntradaySignal` objects
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING

from stockdownloader.util.pinescript.models import StrategyDefinition

if TYPE_CHECKING:
    from stockdownloader.model.price_data import IntradayPriceData, PriceData
    from stockdownloader.model.trade import IntradaySignal


class Signal(Enum):
    """Trading signal produced by a strategy evaluation."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of the strategy."""

    @abstractmethod
    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        """Evaluate the strategy at the given index and return a signal.

        Args:
            data: List of price data bars.
            current_index: The current bar index to evaluate.

        Returns:
            A Signal indicating BUY, SELL, or HOLD.
        """

    @property
    @abstractmethod
    def warmup_period(self) -> int:
        """Return the number of bars needed before the strategy can generate signals."""

    def to_pinescript(self) -> StrategyDefinition:
        """Convert this strategy to a PineScript v6 strategy definition.

        Returns a :class:`~stockdownloader.util.pinescript_models.StrategyDefinition`
        that :class:`~stockdownloader.util.pinescript_generator.PineScriptGenerator`
        can render into Pine Script v6 code.

        Override in subclasses that support PineScript generation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support PineScript generation"
        )


class IntradayTradingStrategy(ABC):
    """Base class for strategies that operate on intraday bars.

    Unlike :class:`TradingStrategy` which operates on daily bars and returns a
    simple BUY/SELL/HOLD signal, this ABC targets 5-minute intraday bars and
    returns a rich :class:`IntradaySignal` carrying direction, stop/target levels,
    confluence score, and entry mode metadata.

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
    # Engine callbacks -- non-abstract with default no-ops.
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
