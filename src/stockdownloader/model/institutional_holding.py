"""Immutable representations of SEC 13F institutional holdings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InstitutionalHolding:
    """A single 13F institutional holding entry."""

    filing_date: str  # "YYYY-MM-DD"
    manager_name: str
    manager_cik: str
    shares: int
    value_usd: int  # in thousands
    share_class: str  # typically "COM"

    def __post_init__(self) -> None:
        # filing_date may be empty when created by _parse_13f_xml
        # (the actual date is tracked on OwnershipSnapshot.quarter_end)
        if not self.manager_name:
            raise ValueError("manager_name must not be empty")
        if self.shares < 0:
            raise ValueError("shares must be non-negative")


@dataclass(frozen=True, slots=True)
class OwnershipSnapshot:
    """Aggregated institutional ownership for a single quarter."""

    quarter_end: str  # "YYYY-MM-DD"
    symbol: str
    total_institutional_shares: int
    num_institutions: int
    top_10_concentration: float  # fraction held by top 10
    holdings: tuple[InstitutionalHolding, ...]

    def __post_init__(self) -> None:
        if not self.quarter_end:
            raise ValueError("quarter_end must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
