"""Event calendar for deterministic anchor dates.

Provides hardcoded FOMC meeting dates for 2024-2026 and a bisect-based
lookup API.  The ``anchor_type`` parameter allows future extension to
OpEx, quarter-end, or pivot anchors without changing the interface.
"""

from __future__ import annotations

import bisect
from datetime import date, datetime

# ── FOMC meeting dates (announcement day) ─────────────────────────────
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

FOMC_DATES: tuple[str, ...] = (
    # 2024
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    # 2025
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-17",
    # 2026
    "2026-01-28",
    "2026-03-18",
    "2026-05-06",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-16",
)

_ANCHOR_TABLES: dict[str, tuple[str, ...]] = {
    "fomc": FOMC_DATES,
}


def get_anchor_date(
    trading_date: str,
    anchor_type: str = "fomc",
) -> str | None:
    """Return the most recent anchor date on or before *trading_date*.

    Parameters
    ----------
    trading_date:
        ``"YYYY-MM-DD"`` date string.
    anchor_type:
        Anchor event type.  Currently only ``"fomc"`` is supported.

    Returns
    -------
    The anchor date string, or ``None`` if *trading_date* precedes all
    known anchors.
    """
    dates = _ANCHOR_TABLES.get(anchor_type)
    if dates is None:
        return None
    # bisect_right finds insert point *after* any equal element
    idx = bisect.bisect_right(dates, trading_date)
    if idx == 0:
        return None
    return dates[idx - 1]


def is_anchor_date(
    trading_date: str,
    anchor_type: str = "fomc",
) -> bool:
    """Return ``True`` if *trading_date* is an anchor date."""
    dates = _ANCHOR_TABLES.get(anchor_type)
    if dates is None:
        return False
    idx = bisect.bisect_left(dates, trading_date)
    return idx < len(dates) and dates[idx] == trading_date


def days_since_anchor(
    trading_date: str,
    anchor_type: str = "fomc",
) -> int | None:
    """Return the number of calendar days since the most recent anchor.

    Returns ``None`` if *trading_date* precedes all known anchors.
    Returns ``0`` on the anchor day itself.
    """
    anchor = get_anchor_date(trading_date, anchor_type)
    if anchor is None:
        return None
    td = _parse_date(trading_date)
    ad = _parse_date(anchor)
    return (td - ad).days


def _parse_date(date_str: str) -> date:
    """Parse ``YYYY-MM-DD`` string to :class:`datetime.date`."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()
