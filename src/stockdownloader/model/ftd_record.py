"""Immutable representation of a single SEC Failure-to-Deliver record."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class FtdRecord:
    """Single Failure-to-Deliver record from SEC settlement data."""

    settlement_date: str  # "YYYY-MM-DD"
    symbol: str
    cusip: str
    quantity: int  # fail quantity in shares
    description: str
    price: Decimal  # closing price

    def __post_init__(self) -> None:
        if not self.settlement_date:
            raise ValueError("settlement_date must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
