"""Shared boilerplate for trailing-stop exit mechanisms.

:class:`TrailingExitBase` manages: ``_stop``, ``_peak``, ``_exit_price``,
``_exit_reason``, ``_initialized``, ``_data``, ``_data_offset``.

Provides helpers for stop initialization, peak tracking, hard-stop
checking, and a template-method hook for custom post-activation logic.
Subclasses implement :meth:`evaluate_bar` and :attr:`name`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import Direction
from stockdownloader.strategy.exit_mechanism import ExitMechanism
from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from stockdownloader.model.price_data import IntradayPriceData


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
        data_index: int = 0,
    ) -> bool:
        """Shared trailing-stop evaluation logic.

        Handles initialization, hard-stop check, peak tracking,
        breakeven activation, optional post-activation custom exit
        (via :meth:`_post_activation_exit_check`), and trail update
        for both LONG and SHORT.

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
            data_index: Index into ``self._data`` for the current bar.
                Passed to :meth:`_post_activation_exit_check` for
                indicator computation.  Defaults to ``0``.

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

        # Post-activation custom exit (template method hook)
        if self._be_triggered:
            custom = self._post_activation_exit_check(bar, direction, data_index)
            if custom is not None:
                return self._trigger_exit(custom[0], custom[1])

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

    def _post_activation_exit_check(
        self,
        bar: IntradayPriceData,
        direction: Direction,
        data_index: int,
    ) -> tuple[Decimal, str] | None:
        """Hook for subclasses to add custom exit logic after activation.

        Called on every bar once the activation threshold has been reached,
        *before* the standard trailing stop update.  Override to add
        VWAP-cross, band-touch, or other custom exit conditions.

        Args:
            bar: Current price bar.
            direction: Trade direction.
            data_index: Index into ``self._data`` for indicator lookups.

        Returns:
            ``(exit_price, reason)`` to trigger an immediate exit, or
            ``None`` to continue with the standard trailing logic.
        """
        return None

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
