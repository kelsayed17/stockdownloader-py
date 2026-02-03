"""Tracks an individual trade position from entry to exit."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum


class Direction(Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    """Trade status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class Trade:
    """Tracks an individual trade position from entry to exit."""

    def __init__(
        self,
        direction: Direction,
        entry_date: str,
        entry_price: Decimal,
        shares: int,
    ) -> None:
        if direction is None:
            raise ValueError("direction must not be null")
        if entry_date is None:
            raise ValueError("entry_date must not be null")
        if entry_price is None:
            raise ValueError("entry_price must not be null")
        if shares <= 0:
            raise ValueError("shares must be positive")

        self._direction: Direction = direction
        self._entry_date: str = entry_date
        self._entry_price: Decimal = entry_price
        self._shares: int = shares

        self._status: TradeStatus = TradeStatus.OPEN
        self._exit_date: str | None = None
        self._exit_price: Decimal | None = None
        self._profit_loss: Decimal = Decimal("0")
        self._return_pct: Decimal = Decimal("0")

    def close(self, exit_date: str, exit_price: Decimal) -> None:
        """Close the trade at the given price."""
        if exit_date is None:
            raise ValueError("exit_date must not be null")
        if exit_price is None:
            raise ValueError("exit_price must not be null")
        if self._status == TradeStatus.CLOSED:
            raise RuntimeError("Trade is already closed")

        self._exit_date = exit_date
        self._exit_price = exit_price
        self._status = TradeStatus.CLOSED

        if self._direction == Direction.LONG:
            price_diff = exit_price - self._entry_price
        else:
            price_diff = self._entry_price - exit_price

        self._profit_loss = price_diff * Decimal(self._shares)
        self._return_pct = (
            price_diff / self._entry_price
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * Decimal("100")

    def is_win(self) -> bool:
        """Return True if the trade was profitable."""
        return self._profit_loss > Decimal("0")

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def status(self) -> TradeStatus:
        return self._status

    @property
    def entry_date(self) -> str:
        return self._entry_date

    @property
    def exit_date(self) -> str | None:
        return self._exit_date

    @property
    def entry_price(self) -> Decimal:
        return self._entry_price

    @property
    def exit_price(self) -> Decimal | None:
        return self._exit_price

    @property
    def shares(self) -> int:
        return self._shares

    @property
    def profit_loss(self) -> Decimal:
        return self._profit_loss

    @property
    def return_pct(self) -> Decimal:
        return self._return_pct

    def __str__(self) -> str:
        exit_date_str = self._exit_date if self._exit_date is not None else "N/A"
        exit_price_str = (
            f"${self._exit_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
            if self._exit_price is not None
            else "N/A"
        )
        return (
            f"{self._direction.value} {self._status.value}: "
            f"Entry {self._entry_date} @ "
            f"${self._entry_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} -> "
            f"Exit {exit_date_str} @ {exit_price_str} | "
            f"P/L: ${self._profit_loss.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} "
            f"({float(self._return_pct):.2f}%)"
        )
