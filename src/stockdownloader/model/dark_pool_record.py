"""Immutable representation of FINRA OTC/ATS (dark pool) volume data."""

from __future__ import annotations

from dataclasses import dataclass


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
