"""Interface defining the contract for trading strategies.

Each implementation evaluates price data and produces BUY/SELL/HOLD signals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


class Signal(Enum):
    """Trading signal produced by a strategy evaluation."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def get_name(self) -> str:
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

    @abstractmethod
    def get_warmup_period(self) -> int:
        """Return the number of bars needed before the strategy can generate signals."""
