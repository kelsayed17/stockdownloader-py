"""Immutable representation of an estimated stock borrow rate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BorrowRateRecord:
    """Borrow rate proxy record derived from short interest data."""

    date: str  # "YYYY-MM-DD"
    symbol: str
    estimated_fee_pct: float  # estimated annual borrow fee %
    days_to_cover: float
    utilization_pct: float  # short_interest / shares_outstanding

    def __post_init__(self) -> None:
        if not self.date:
            raise ValueError("date must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
