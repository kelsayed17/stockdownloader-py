"""Interface for options trading strategies.

Operates on underlying price data and generates signals for opening/closing
options positions at specific strikes and expirations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.model.option_type import OptionType
    from stockdownloader.model.price_data import PriceData


class OptionsSignal(Enum):
    """Signal produced by an options strategy evaluation."""
    OPEN = auto()
    CLOSE = auto()
    HOLD = auto()


class OptionsStrategy(ABC):
    """Abstract base class for options trading strategies."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of the strategy."""

    @abstractmethod
    def evaluate(self, data: list[PriceData], current_index: int) -> OptionsSignal:
        """Evaluate the strategy and return a signal.

        Args:
            data: Underlying price history.
            current_index: Current bar index.

        Returns:
            OPEN to enter a position, CLOSE to exit, HOLD to do nothing.
        """

    @abstractmethod
    def get_option_type(self) -> OptionType:
        """Get the option type this strategy trades."""

    @abstractmethod
    def is_short(self) -> bool:
        """Whether this strategy sells options (writes) or buys them."""

    @abstractmethod
    def get_target_strike(self, current_price: Decimal) -> Decimal:
        """Calculate the target strike price based on current market conditions.

        Args:
            current_price: Current underlying price.

        Returns:
            Target strike price.
        """

    @abstractmethod
    def get_target_days_to_expiry(self) -> int:
        """Get the target days to expiration for new positions."""

    @abstractmethod
    def get_warmup_period(self) -> int:
        """Number of warmup bars needed before the strategy can generate signals."""
