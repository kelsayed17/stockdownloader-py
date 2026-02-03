"""Tracks an individual options trade from entry to exit."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

from stockdownloader.model.option_type import OptionType


class OptionsDirection(Enum):
    """Options trade direction."""

    BUY = "BUY"
    SELL = "SELL"


class OptionsTradeStatus(Enum):
    """Options trade status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


CONTRACT_MULTIPLIER: int = 100


class OptionsTrade:
    """Tracks an individual options trade from entry to exit.

    Each contract represents 100 shares of the underlying.
    """

    def __init__(
        self,
        option_type: OptionType,
        direction: OptionsDirection,
        strike: Decimal,
        expiration_date: str,
        entry_date: str,
        entry_premium: Decimal,
        contracts: int,
        entry_volume: int,
    ) -> None:
        if option_type is None:
            raise ValueError("option_type must not be null")
        if direction is None:
            raise ValueError("direction must not be null")
        if strike is None:
            raise ValueError("strike must not be null")
        if expiration_date is None:
            raise ValueError("expiration_date must not be null")
        if entry_date is None:
            raise ValueError("entry_date must not be null")
        if entry_premium is None:
            raise ValueError("entry_premium must not be null")
        if contracts <= 0:
            raise ValueError("contracts must be positive")

        self._option_type: OptionType = option_type
        self._direction: OptionsDirection = direction
        self._strike: Decimal = strike
        self._expiration_date: str = expiration_date
        self._entry_date: str = entry_date
        self._entry_premium: Decimal = entry_premium
        self._contracts: int = contracts
        self._entry_volume: int = entry_volume

        self._status: OptionsTradeStatus = OptionsTradeStatus.OPEN
        self._exit_date: str | None = None
        self._exit_premium: Decimal | None = None
        self._profit_loss: Decimal = Decimal("0")
        self._return_pct: Decimal = Decimal("0")

    def _calculate_profit_loss(self) -> None:
        """Calculate P/L and return percentage after close or expiry."""
        assert self._exit_premium is not None

        if self._direction == OptionsDirection.BUY:
            premium_diff = self._exit_premium - self._entry_premium
        else:
            premium_diff = self._entry_premium - self._exit_premium

        self._profit_loss = (
            premium_diff * Decimal(self._contracts) * Decimal(CONTRACT_MULTIPLIER)
        )

        total_cost = (
            self._entry_premium
            * Decimal(self._contracts)
            * Decimal(CONTRACT_MULTIPLIER)
        )
        if total_cost != Decimal("0"):
            self._return_pct = (
                self._profit_loss / total_cost
            ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * Decimal("100")

    def close(self, exit_date: str, exit_premium: Decimal) -> None:
        """Close the trade at the given premium."""
        if exit_date is None:
            raise ValueError("exit_date must not be null")
        if exit_premium is None:
            raise ValueError("exit_premium must not be null")
        if self._status != OptionsTradeStatus.OPEN:
            raise RuntimeError(f"Trade is not open, current status: {self._status.value}")

        self._exit_date = exit_date
        self._exit_premium = exit_premium
        self._status = OptionsTradeStatus.CLOSED
        self._calculate_profit_loss()

    def expire(self, expiry_date: str, settlement_premium: Decimal) -> None:
        """Mark the trade as expired (option expired worthless or exercised)."""
        if expiry_date is None:
            raise ValueError("expiry_date must not be null")
        if settlement_premium is None:
            raise ValueError("settlement_premium must not be null")
        if self._status != OptionsTradeStatus.OPEN:
            raise RuntimeError(f"Trade is not open, current status: {self._status.value}")

        self._exit_date = expiry_date
        self._exit_premium = settlement_premium
        self._status = OptionsTradeStatus.EXPIRED
        self._calculate_profit_loss()

    def is_win(self) -> bool:
        """Return True if the trade was profitable."""
        return self._profit_loss > Decimal("0")

    def total_entry_cost(self) -> Decimal:
        """Total premium paid/received at entry (contracts * 100 * premium)."""
        return self._entry_premium * Decimal(self._contracts * CONTRACT_MULTIPLIER)

    @property
    def option_type(self) -> OptionType:
        return self._option_type

    @property
    def direction(self) -> OptionsDirection:
        return self._direction

    @property
    def strike(self) -> Decimal:
        return self._strike

    @property
    def expiration_date(self) -> str:
        return self._expiration_date

    @property
    def entry_date(self) -> str:
        return self._entry_date

    @property
    def entry_premium(self) -> Decimal:
        return self._entry_premium

    @property
    def contracts(self) -> int:
        return self._contracts

    @property
    def entry_volume(self) -> int:
        return self._entry_volume

    @property
    def status(self) -> OptionsTradeStatus:
        return self._status

    @property
    def exit_date(self) -> str | None:
        return self._exit_date

    @property
    def exit_premium(self) -> Decimal | None:
        return self._exit_premium

    @property
    def profit_loss(self) -> Decimal:
        return self._profit_loss

    @property
    def return_pct(self) -> Decimal:
        return self._return_pct

    def __str__(self) -> str:
        exit_date_str = self._exit_date if self._exit_date is not None else "N/A"
        exit_premium_str = (
            f"${self._exit_premium.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
            if self._exit_premium is not None
            else "N/A"
        )
        return (
            f"{self._direction.value} {self._option_type.value} "
            f"{self._status.value} "
            f"${self._strike.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} "
            f"exp:{self._expiration_date} | "
            f"Entry:{self._entry_date} "
            f"@${self._entry_premium.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} -> "
            f"Exit:{exit_date_str} @{exit_premium_str} | "
            f"P/L:${self._profit_loss.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} "
            f"({float(self._return_pct):.2f}%) "
            f"vol:{self._entry_volume}"
        )
