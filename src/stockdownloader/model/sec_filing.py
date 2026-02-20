"""Immutable representation of a single SEC EDGAR filing's metadata."""

from __future__ import annotations

from dataclasses import dataclass

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
