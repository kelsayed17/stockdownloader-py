"""Immutable representation of a FINRA short interest report."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShortInterestRecord:
    """FINRA short interest data for a single reporting date."""

    settlement_date: str  # "YYYY-MM-DD"
    symbol: str
    short_interest: int  # total shares short
    avg_daily_volume: int  # for days-to-cover
    days_to_cover: float
    short_interest_pct: float  # % of float (0.0 if unknown)

    def __post_init__(self) -> None:
        if not self.settlement_date:
            raise ValueError("settlement_date must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.short_interest < 0:
            raise ValueError("short_interest must be non-negative")
