"""Immutable dataclasses for FINRA / SEC regulatory data records.

Includes dark-pool volume, short interest, failure-to-deliver,
estimated borrow-rate records, SEC EDGAR filing metadata,
13F institutional ownership snapshots, Form 3/4/5 insider
transactions, and Schedule 13D/13G beneficial ownership.
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


# ── SEC Filings ─────────────────────────────────────────────────────────

# Form types most likely to cause significant price movement.
_MATERIAL_FORMS: frozenset[str] = frozenset({
    "10-K", "10-Q", "8-K",
    "SC 13D", "SC 13D/A",
    "DEF 14A", "DEFA14A",
})


@dataclass(frozen=True, slots=True)
class SecFiling:
    """Immutable representation of a single SEC EDGAR filing's metadata."""

    accession_number: str  # e.g. "0001326380-24-000013"
    filing_date: str       # "YYYY-MM-DD"
    report_date: str       # "YYYY-MM-DD"
    form: str              # e.g. "10-K", "10-Q", "8-K", "4"
    primary_document: str  # filename on EDGAR
    description: str       # human-readable description
    filing_url: str        # full URL to the filing on SEC website

    def __post_init__(self) -> None:
        if not self.accession_number:
            raise ValueError("accession_number must not be empty")
        if not self.filing_date:
            raise ValueError("filing_date must not be empty")
        if not self.form:
            raise ValueError("form must not be empty")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_material(self) -> bool:
        """``True`` for form types most likely to cause price movement."""
        return self.form in _MATERIAL_FORMS

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return f"{self.filing_date} {self.form:>10s}  {self.description}"


# ── Institutional Holdings ──────────────────────────────────────────────


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


# ── Insider Transactions (Form 3/4/5) ────────────────────────────────


@dataclass(frozen=True, slots=True)
class InsiderTransaction:
    """A single insider buy/sell from SEC Form 3, 4, or 5.

    Form 4 is the most common: insiders must report changes in
    beneficial ownership within 2 business days.  Transaction codes:

    * ``P`` — open-market purchase
    * ``S`` — open-market sale
    * ``A`` — grant/award
    * ``M`` — exercise of derivative (conversion)
    * ``G`` — gift
    * ``F`` — tax withholding (disposition)
    * ``D`` — disposition to issuer
    * ``J`` — other acquisition/disposition

    ``shares`` is signed: positive = acquired, negative = disposed.
    """

    filing_date: str           # "YYYY-MM-DD"
    transaction_date: str      # "YYYY-MM-DD"
    owner_name: str
    owner_cik: str
    owner_title: str           # "CEO", "Director", etc.
    is_director: bool
    is_officer: bool
    is_ten_pct_owner: bool
    transaction_code: str      # "P", "S", "A", "M", etc.
    shares: int                # signed: positive=acquire, negative=dispose
    price_per_share: float     # 0.0 if not reported
    shares_owned_after: int    # post-transaction balance
    direct_or_indirect: str    # "D"=direct, "I"=indirect

    def __post_init__(self) -> None:
        if not self.filing_date:
            raise ValueError("filing_date must not be empty")
        if not self.owner_name:
            raise ValueError("owner_name must not be empty")


# ── Beneficial Ownership (Schedule 13D/13G) ──────────────────────────


@dataclass(frozen=True, slots=True)
class BeneficialOwner:
    """A >5% beneficial owner from Schedule 13D or 13G.

    Schedule 13D is filed by activist investors (>5% with intent to
    influence).  Schedule 13G is filed by passive investors (>5%
    without activist intent).  Amendments (``/A``) update prior
    filings with new share counts.
    """

    filing_date: str           # "YYYY-MM-DD"
    owner_name: str
    owner_cik: str
    form_type: str             # "SC 13D", "SC 13D/A", "SC 13G", etc.
    shares_beneficially_owned: int
    percent_of_class: float    # e.g. 11.9
    sole_voting_power: int
    shared_voting_power: int
    sole_dispositive_power: int
    shared_dispositive_power: int
    filing_url: str

    def __post_init__(self) -> None:
        if not self.filing_date:
            raise ValueError("filing_date must not be empty")
        if not self.owner_name:
            raise ValueError("owner_name must not be empty")
        if not self.form_type:
            raise ValueError("form_type must not be empty")


# ── Insider Ownership Snapshot ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InsiderOwnershipSnapshot:
    """Aggregated insider + beneficial ownership for a symbol at a point in time."""

    as_of_date: str
    symbol: str
    total_insider_shares: int
    total_beneficial_owner_shares: int
    num_insiders: int
    num_beneficial_owners: int
    transactions: tuple[InsiderTransaction, ...]
    beneficial_owners: tuple[BeneficialOwner, ...]

    def __post_init__(self) -> None:
        if not self.as_of_date:
            raise ValueError("as_of_date must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
