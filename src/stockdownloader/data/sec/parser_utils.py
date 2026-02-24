"""Shared parser utilities for SEC filing parsers.

Contains constants, date helpers, and XML extraction functions used by both
``sec_insider_parsers`` and ``sec_ownership_parsers``.  Centralising these
avoids duplicated logic and keeps the two parser modules focused on their
respective filing formats.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

#: Abbreviated month names -> two-digit month number (uppercase keys).
MONTH_ABBREVS: dict[str, str] = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

#: GameStop default CUSIP.
GME_CUSIP = "36467W109"

#: Quarter-end dates for each calendar quarter.
QUARTER_ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}

#: Month number -> lowercase abbreviated name (for URL construction).
MONTH_NAMES: dict[int, str] = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}


# ------------------------------------------------------------------
# Date helpers
# ------------------------------------------------------------------


def normalize_date(raw: str) -> str:
    """Normalize a date string to YYYY-MM-DD format.

    Handles:
    - ``2024-01-15`` (already ISO)
    - ``15-JAN-2024`` (SEC bulk TSV format)
    - ``01/15/2024`` (US slash format)
    - ``20240115`` (compact format)
    """
    raw = raw.strip()
    if not raw:
        return ""
    # Already ISO
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return raw
    # DD-MON-YYYY (e.g. 02-JAN-2024)
    parts = raw.split("-")
    if len(parts) == 3 and len(parts[1]) == 3:
        month = MONTH_ABBREVS.get(parts[1].upper(), "")
        if month:
            try:
                return f"{int(parts[2]):04d}-{month}-{int(parts[0]):02d}"
            except ValueError:
                pass
    # MM/DD/YYYY
    if "/" in raw:
        slash_parts = raw.split("/")
        if len(slash_parts) == 3:
            try:
                m, d, y = int(slash_parts[0]), int(slash_parts[1]), int(slash_parts[2])
                return f"{y:04d}-{m:02d}-{d:02d}"
            except ValueError:
                pass
    # YYYYMMDD
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def quarter_range(year: int, quarter: int) -> tuple[date, date]:
    """Return (start_date, end_date) for a calendar quarter."""
    start_month = (quarter - 1) * 3 + 1
    end_month = quarter * 3
    start = date(year, start_month, 1)
    # End of quarter
    if end_month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, end_month + 1, 1) - timedelta(days=1)
    return start, end


def filing_date_to_quarter_end(filing_date: str) -> str:
    """Map a 13F filing date to its reporting quarter-end date.

    13F filings report holdings as of a quarter-end date, but are filed
    during a window after that date:

    * Jan-Feb filings  -> Q4 of prior year (Dec 31)
    * Mar-May filings  -> Q1 (Mar 31)
    * Jun-Aug filings  -> Q2 (Jun 30)
    * Sep-Nov filings  -> Q3 (Sep 30)
    * Dec filings      -> Q4 (Dec 31)
    """
    if not filing_date or len(filing_date) < 7:
        return filing_date
    try:
        month = int(filing_date[5:7])
        year = int(filing_date[:4])
    except (ValueError, IndexError):
        return filing_date

    if month <= 2:
        return f"{year - 1}-12-31"
    if month <= 5:
        return f"{year}-03-31"
    if month <= 8:
        return f"{year}-06-30"
    if month <= 11:
        return f"{year}-09-30"
    return f"{year}-12-31"


# ------------------------------------------------------------------
# XML helpers
# ------------------------------------------------------------------


def xml_text(content: str, tag: str) -> str | None:
    """Extract text content from a simple XML tag using regex.

    Works on raw XML string content (no parsing required).  Suitable for
    quick extraction from SEC filings where proper namespace handling is
    not needed.
    """
    match = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        content, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def get_xml_text(element: ET.Element, tag: str) -> str | None:
    """Extract text from a child XML element via ElementTree.

    Works on parsed ``ET.Element`` objects — use this when you already
    have a parsed XML tree (e.g. 13F info-tables).
    """
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None
