"""Abstract base class for exit mechanisms used in the exit tournament.

An :class:`ExitMechanism` evaluates whether an existing position should be
exited on each bar, tracking its own internal state (peak price, trailing
stop level, activation flag, etc.).

Unlike :class:`TradingStrategy` which produces entry *and* exit signals,
an ``ExitMechanism`` is only concerned with the exit side.

:class:`TrailingExitBase` is a convenience subclass that manages the shared
state (stop, peak, exit price/reason, data context) and initial stop
placement logic common to all trailing-stop variants.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import Direction
from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade

class ExitMechanism(ABC):
    """Abstract base class for exit mechanisms."""

    @property
    @abstractmethod
    def name(self) -> str:
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

    @property
    @abstractmethod
    def exit_price(self) -> Decimal:
        """Return the exact exit price after :meth:`evaluate_bar` returns True.

        The exit price may differ from the bar's close (e.g. the stop level
        rather than the close).
        """

    @property
    @abstractmethod
    def exit_reason(self) -> str:
        """Return a short string describing why the exit was triggered.

        Examples: ``'trail_stop'``, ``'vwap_cross'``, ``'session_end'``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new trade evaluation."""

    @property
    def peak_favorable(self) -> Decimal:
        """Return the peak favorable excursion (from entry) seen so far.

        Subclasses should track this internally.  Default returns zero.
        """
        return ZERO

class TrailingExitBase(ExitMechanism, ABC):
    """Shared boilerplate for trailing-stop exit mechanisms.

    Manages: ``_stop``, ``_peak``, ``_exit_price``, ``_exit_reason``,
    ``_initialized``, ``_data``, ``_data_offset``.

    Provides:
    - :meth:`set_data_context` for engine integration
    - :meth:`reset` to clear state between trade evaluations
    - :meth:`_init_stop` to set the initial hard stop and peak on first bar
    - :meth:`_update_peak` to track peak favorable excursion
    - :meth:`_check_hard_stop` to test if the bar hit the stop level
    - :meth:`_trigger_exit` to record exit price and reason
    - :attr:`exit_price`, :attr:`exit_reason`,
      :attr:`peak_favorable` property implementations

    Subclasses implement :meth:`evaluate_bar` and :attr:`name`.
    """

    def __init__(self) -> None:
        self._data: list[IntradayPriceData] = []
        self._data_offset: int = 0
        self._stop: Decimal = ZERO
        self._peak: Decimal = ZERO
        self._exit_price: Decimal = ZERO
        self._exit_reason: str = ""
        self._initialized: bool = False
        self._be_triggered: bool = False

    def set_data_context(
        self, data: list[IntradayPriceData], entry_data_index: int
    ) -> None:
        """Set the full data list and entry index for indicator computation."""
        self._data = data
        self._data_offset = entry_data_index

    def reset(self) -> None:
        self._stop = ZERO
        self._peak = ZERO
        self._exit_price = ZERO
        self._exit_reason = ""
        self._initialized = False
        self._be_triggered = False

    # -- Helpers for subclasses ------------------------------------------

    def _init_stop(self, entry: Decimal, stop_dist: Decimal, direction: Direction) -> None:
        """Set initial hard stop and peak on first bar (call once)."""
        if not self._initialized:
            self._stop = (
                entry - stop_dist if direction == Direction.LONG
                else entry + stop_dist
            )
            self._peak = entry
            self._initialized = True

    def _update_peak(self, bar: IntradayPriceData, direction: Direction) -> None:
        """Track peak favorable excursion."""
        if direction == Direction.LONG:
            if bar.high > self._peak:
                self._peak = bar.high
        else:
            if bar.low < self._peak:
                self._peak = bar.low

    def _check_hard_stop(self, bar: IntradayPriceData, direction: Direction) -> bool:
        """Return ``True`` if the bar breached the current stop level."""
        if direction == Direction.LONG:
            return bar.low <= self._stop
        return bar.high >= self._stop

    def _trigger_exit(self, price: Decimal, reason: str) -> bool:
        """Record exit price and reason, return ``True`` (for chaining)."""
        self._exit_price = price
        self._exit_reason = reason
        return True

    def _evaluate_trail(
        self,
        bar: IntradayPriceData,
        direction: Direction,
        entry: Decimal,
        stop_dist: Decimal,
        trail_buffer: Decimal,
        activation_r: Decimal,
        be_buffer: Decimal | None = None,
    ) -> bool:
        """Shared trailing-stop evaluation logic.

        Handles initialization, hard-stop check, peak tracking,
        breakeven activation, and trail update for both LONG and SHORT.

        Args:
            bar: Current price bar.
            direction: Trade direction.
            entry: Trade entry price.
            stop_dist: Initial stop distance (1R).
            trail_buffer: Current trail width from peak.
            activation_r: R-multiple threshold for breakeven activation.
            be_buffer: Buffer added above/below entry for breakeven stop.
                If ``None``, the stop is not moved to breakeven on
                activation (trail starts immediately from current stop).

        Returns:
            ``True`` if exit triggered.
        """
        self._init_stop(entry, stop_dist, direction)

        # Hard stop check
        if self._check_hard_stop(bar, direction):
            reason = "trail_stop" if self._be_triggered else "hard_stop"
            return self._trigger_exit(self._stop, reason)

        # Track peak
        self._update_peak(bar, direction)

        # Activation check
        if not self._be_triggered:
            if direction == Direction.LONG:
                if self._peak >= entry + stop_dist * activation_r:
                    self._be_triggered = True
                    if be_buffer is not None:
                        self._stop = entry + be_buffer
            else:
                if self._peak <= entry - stop_dist * activation_r:
                    self._be_triggered = True
                    if be_buffer is not None:
                        self._stop = entry - be_buffer

        # Trail stop update
        if self._be_triggered:
            if direction == Direction.LONG:
                new_stop = self._peak - trail_buffer
                if new_stop > self._stop:
                    self._stop = new_stop
            else:
                new_stop = self._peak + trail_buffer
                if new_stop < self._stop:
                    self._stop = new_stop

        return False

    # -- ExitMechanism interface -----------------------------------------

    @property
    def exit_price(self) -> Decimal:
        return self._exit_price

    @property
    def exit_reason(self) -> str:
        return self._exit_reason

    @property
    def peak_favorable(self) -> Decimal:
        return self._peak
