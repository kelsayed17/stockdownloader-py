"""Abstract base class for exit mechanisms used in the exit tournament.

An :class:`ExitMechanism` evaluates whether an existing position should be
exited on each bar, tracking its own internal state (peak price, trailing
stop level, activation flag, etc.).

Unlike :class:`TradingStrategy` which produces entry *and* exit signals,
an ``ExitMechanism`` is only concerned with the exit side.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade


class ExitMechanism(ABC):
    """Abstract base class for exit mechanisms."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of this exit mechanism."""

    @abstractmethod
    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        """Evaluate whether the trade should be exited at this bar.

        Called once per bar, in chronological order, from entry forward.
        The mechanism must maintain its own internal state (peak, stop,
        activation) across calls.

        Args:
            bar: The current 5-minute bar.
            trade: The trade being evaluated.
            bar_index: Ordinal bar index from trade entry (0 = entry bar).

        Returns:
            ``True`` if the trade should be exited now.
        """

    @abstractmethod
    def get_exit_price(self) -> Decimal:
        """Return the exact exit price after :meth:`evaluate_bar` returns True.

        The exit price may differ from the bar's close (e.g. the stop level
        rather than the close).
        """

    @abstractmethod
    def get_exit_reason(self) -> str:
        """Return a short string describing why the exit was triggered.

        Examples: ``'trail_stop'``, ``'vwap_cross'``, ``'session_end'``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new trade evaluation."""

    def get_peak_favorable(self) -> Decimal:
        """Return the peak favorable excursion (from entry) seen so far.

        Subclasses should track this internally.  Default returns zero.
        """
        return Decimal("0")
