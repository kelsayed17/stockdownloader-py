"""Shared SEC HTTP helpers.

Standalone functions extracted from SecInsiderClient and SecOwnershipClient
to eliminate duplication.  Each function takes a ``requests.Session`` as the
first argument and an optional *rate_limit_fn* callable that is invoked
before every HTTP request (replacing the per-client ``self._rate_limit()``
call).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable

import requests

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

#: SEC enforces 10 req/sec.  110 ms gap gives comfortable margin.
SEC_RATE_LIMIT_DELAY: float = 0.11

#: EDGAR full-text search index API base URL.
EFTS_BASE_URL: str = "https://efts.sec.gov/LATEST/search-index"

#: Direct archive access for filing documents.
ARCHIVE_BASE: str = "https://www.sec.gov/Archives/edgar/data"


# ------------------------------------------------------------------
# Quarter helper
# ------------------------------------------------------------------


def prev_quarter(year: int, quarter: int) -> tuple[int, int]:
    """Return *(year, quarter)* for the previous calendar quarter."""
    if quarter == 1:
        return year - 1, 4
    return year, quarter - 1


def quarter_iterator(
    num_quarters: int,
    *,
    min_year: int = 2003,
    min_quarter: int = 1,
) -> Iterator[tuple[int, int]]:
    """Yield *(year, quarter)* tuples backwards from current quarter.

    Stops after *num_quarters* or when reaching *min_year*/*min_quarter*
    (whichever comes first).  Skips future quarters automatically.
    """
    today = date.today()
    current_year = today.year
    current_quarter = (today.month - 1) // 3 + 1

    year, quarter = current_year, current_quarter
    yielded = 0

    while yielded < num_quarters:
        if year < min_year or (year == min_year and quarter < min_quarter):
            break
        if (year, quarter) <= (current_year, current_quarter):
            yield year, quarter
            yielded += 1
        year, quarter = prev_quarter(year, quarter)


def ipo_quarter_floor(symbol: str) -> tuple[int, int]:
    """Return the earliest useful *(year, quarter)* for *symbol*.

    Uses the symbol registry IPO date.  Returns ``(2003, 1)`` as the
    default floor (EDGAR data starts around 2003).
    """
    from stockdownloader.model.symbol_info import get_symbol_info

    info = get_symbol_info(symbol.upper())
    if info is not None and info.ipo_date:
        y = info.ipo_date.year
        q = (info.ipo_date.month - 1) // 3 + 1
        if (y, q) > (2003, 1):
            return y, q
    return 2003, 1


# ------------------------------------------------------------------
# Generic split-adjustment support
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SplitAdjustment:
    """Stock split event for adjusting historical FTD data.

    Attributes
    ----------
    symbol:
        Upper-cased ticker symbol (e.g. ``"GME"``).
    split_date:
        Effective date of the split.  Records *before* this date are
        adjusted.
    split_ratio:
        The split multiplier.  For a 4-for-1 split use ``Decimal("4")``.
        Pre-split quantities are **multiplied** and prices **divided**
        by this value.
    """

    symbol: str
    split_date: date
    split_ratio: Decimal


# Registry of well-known stock splits that affect FTD and 13F data.
# Callers can extend this at runtime via the *extra_splits* parameter.
_KNOWN_SPLITS: list[SplitAdjustment] = [
    SplitAdjustment(
        symbol="GME",
        split_date=date(2022, 7, 22),
        split_ratio=Decimal("4"),
    ),
    SplitAdjustment(
        symbol="AAPL",
        split_date=date(2020, 8, 31),
        split_ratio=Decimal("4"),
    ),
    SplitAdjustment(
        symbol="TSLA",
        split_date=date(2022, 8, 25),
        split_ratio=Decimal("3"),
    ),
    SplitAdjustment(
        symbol="TSLA",
        split_date=date(2020, 8, 31),
        split_ratio=Decimal("5"),
    ),
    SplitAdjustment(
        symbol="AMZN",
        split_date=date(2022, 6, 6),
        split_ratio=Decimal("20"),
    ),
    SplitAdjustment(
        symbol="GOOGL",
        split_date=date(2022, 7, 18),
        split_ratio=Decimal("20"),
    ),
    SplitAdjustment(
        symbol="GOOG",
        split_date=date(2022, 7, 18),
        split_ratio=Decimal("20"),
    ),
    SplitAdjustment(
        symbol="NVDA",
        split_date=date(2024, 6, 10),
        split_ratio=Decimal("10"),
    ),
    SplitAdjustment(
        symbol="NVDA",
        split_date=date(2021, 7, 20),
        split_ratio=Decimal("4"),
    ),
]


def splits_for_symbol(
    symbol: str,
    extra_splits: list[SplitAdjustment] | None = None,
) -> list[SplitAdjustment]:
    """Return known + extra splits filtered for *symbol*."""
    all_splits = _KNOWN_SPLITS + (extra_splits or [])
    return [s for s in all_splits if s.symbol == symbol.upper()]


# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------


def download_with_retry(
    session: requests.Session,
    url: str,
    *,
    max_retries: int = 3,
    rate_limit_fn: Callable[[], None] | None = None,
) -> bytes | None:
    """Download binary content with retry logic.

    Parameters
    ----------
    session:
        An active ``requests.Session`` (with appropriate headers already set).
    url:
        The URL to download.
    max_retries:
        Number of attempts before giving up.
    rate_limit_fn:
        Optional callable invoked before each HTTP request to enforce
        rate-limiting.

    Returns
    -------
    bytes | None
        Raw bytes on success, or *None* on persistent failure / 404.
    """
    for attempt in range(max_retries):
        try:
            if rate_limit_fn is not None:
                rate_limit_fn()
            resp = session.get(url, timeout=120)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                logger.debug("Not found (404): %s", url)
                return None
            logger.warning(
                "Download returned %d: %s (attempt %d/%d)",
                resp.status_code,
                url,
                attempt + 1,
                max_retries,
            )
        except (requests.RequestException, OSError) as exc:
            logger.warning(
                "Download failed: %s (attempt %d/%d)",
                exc,
                attempt + 1,
                max_retries,
            )
        if attempt < max_retries - 1:
            time.sleep(2.0)
    return None


def fetch_url_text(
    session: requests.Session,
    url: str,
    *,
    max_retries: int = 3,
    rate_limit_fn: Callable[[], None] | None = None,
) -> str | None:
    """Download text content with retry logic.

    Parameters
    ----------
    session:
        An active ``requests.Session``.
    url:
        The URL to fetch.
    max_retries:
        Number of attempts before giving up.
    rate_limit_fn:
        Optional callable invoked before each HTTP request.

    Returns
    -------
    str | None
        Decoded text on success, or *None* on persistent failure.
    """
    for attempt in range(max_retries):
        try:
            if rate_limit_fn is not None:
                rate_limit_fn()
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                return None
        except (requests.RequestException, OSError):
            pass
        if attempt < max_retries - 1:
            time.sleep(1.0)
    return None


def fetch_efts_page(
    session: requests.Session,
    url: str,
    *,
    max_retries: int = 3,
    rate_limit_fn: Callable[[], None] | None = None,
) -> list[dict] | None:
    """Fetch a single page from the EFTS search-index API.

    Parameters
    ----------
    session:
        An active ``requests.Session``.
    url:
        The full EFTS query URL.
    max_retries:
        Number of attempts before giving up.
    rate_limit_fn:
        Optional callable invoked before each HTTP request.

    Returns
    -------
    list[dict] | None
        The ``hits.hits`` list on success, or *None* after all retries.
    """
    for attempt in range(max_retries):
        try:
            if rate_limit_fn is not None:
                rate_limit_fn()
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("hits", {}).get("hits", [])
            logger.warning(
                "EFTS returned %d (attempt %d/%d): %s",
                resp.status_code,
                attempt + 1,
                max_retries,
                url,
            )
        except (
            requests.RequestException,
            json.JSONDecodeError,
            OSError,
        ) as exc:
            logger.warning(
                "EFTS failed: %s (attempt %d/%d)",
                exc,
                attempt + 1,
                max_retries,
            )
        if attempt < max_retries - 1:
            time.sleep(1.0)
    return None


def fetch_all_efts_hits(
    session: requests.Session,
    base_url: str,
    *,
    rate_limit_fn: Callable[[], None] | None = None,
    page_size: int = 100,
    max_results: int = 5000,
) -> list[dict]:
    """Paginate through EFTS search results, returning all hits."""
    all_hits: list[dict] = []
    offset = 0

    while offset < max_results:
        url = f"{base_url}&from={offset}&size={page_size}"
        page = fetch_efts_page(session, url, rate_limit_fn=rate_limit_fn)
        if page is None or not page:
            break
        all_hits.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    return all_hits
