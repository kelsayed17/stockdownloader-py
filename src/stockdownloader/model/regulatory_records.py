"""Immutable dataclasses for FINRA / SEC regulatory data records.

Includes dark-pool volume, short interest, failure-to-deliver, and
estimated borrow-rate records.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


# ── Dark Pool ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DarkPoolRecord:
    """FINRA OTC/ATS volume record (typically weekly)."""

    week_ending: str  # "YYYY-MM-DD"
    symbol: str
    total_weekly_volume: int
    ats_volume: int  # Dark pool (ATS) volume
    otc_volume: int  # Non-ATS OTC volume
    ats_pct: float  # ats_volume / total_weekly_volume

    def __post_init__(self) -> None:
        if not self.week_ending:
            raise ValueError("week_ending must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.total_weekly_volume < 0:
            raise ValueError("total_weekly_volume must be non-negative")


# ── Short Interest ───────────────────────────────────────────────────────


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


# ── Failure to Deliver ───────────────────────────────────────────────────


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


# ── Borrow Rate ──────────────────────────────────────────────────────────


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
