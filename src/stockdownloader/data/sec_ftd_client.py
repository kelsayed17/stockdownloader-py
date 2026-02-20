"""Downloads and caches SEC Failure-to-Deliver data.

SEC publishes FTD data as pipe-delimited text inside zip files.  The URL
prefix depends on the vintage of the data:

* Jul 2009 -- first-half Jun 2017:
  ``/files/data/frequently-requested-foia-document-fails-deliver-data/``
* Second-half Jun 2017 onward:
  ``/files/data/fails-deliver-data/``
* Feb--Apr 2020 (migration artefact):
  ``/files/node/add/data_distribution/``

File naming convention: ``cnsfails{YYYYMM}{a|b}.zip``
where ``a`` = first half of the month (1st-15th) and ``b`` = second half
(16th-end).

Columns: SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
Date format in file: YYYYMMDD

Rate-limited to 10 req/s (SEC fair-use policy).  Downloaded zip files
are cached locally to avoid redundant downloads.

Usage::

    client = SecFtdClient()
    records = client.fetch_ftd_data("GME", start_year=2020)
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests

from stockdownloader.model.ftd_record import FtdRecord

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
# SEC enforces 10 req/sec.  Use 0.25s to be conservative and avoid 403s.
_RATE_LIMIT_DELAY = 0.25

# ---------------------------------------------------------------------------
# URL templates -- SEC hosts FTD zips under three different prefixes
# depending on the data vintage.  Boundaries were determined from the
# official download page:
#     https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data
# ---------------------------------------------------------------------------

_BASE = "https://www.sec.gov"

_FTD_URL_CURRENT = (
    _BASE + "/files/data/fails-deliver-data/"
    "cnsfails{year}{month:02d}{half}.zip"
)
_FTD_URL_FOIA = (
    _BASE + "/files/data/"
    "frequently-requested-foia-document-fails-deliver-data/"
    "cnsfails{year}{month:02d}{half}.zip"
)
_FTD_URL_NODE = (
    _BASE + "/files/node/add/data_distribution/"
    "cnsfails{year}{month:02d}{half}.zip"
)

# Dates that live exclusively under the /files/node/add/ prefix.
_NODE_DATES: set[tuple[int, int, str]] = {
    (2020, 2, "a"), (2020, 2, "b"),
    (2020, 3, "a"), (2020, 3, "b"),
    (2020, 4, "a"), (2020, 4, "b"),
}

# The single known file with an anomalous ``_0`` suffix.
_ANOMALOUS_SUFFIX: dict[tuple[int, int, str], str] = {
    (2019, 10, "a"): "_0",
}

# First (year, month, half) that uses the "current" prefix.
# Everything before this uses the FOIA prefix.
_FOIA_CUTOFF = (2017, 6, "b")


def _ftd_url(year: int, month: int, half: str) -> str:
    """Return the correct SEC FTD download URL for a given date/half."""
    key = (year, month, half)

    if key in _NODE_DATES:
        return _FTD_URL_NODE.format(year=year, month=month, half=half)

    suffix = _ANOMALOUS_SUFFIX.get(key, "")

    if key < _FOIA_CUTOFF:
        template = _FTD_URL_FOIA
    else:
        template = _FTD_URL_CURRENT

    url = template.format(year=year, month=month, half=half)
    if suffix:
        url = url.replace(".zip", f"{suffix}.zip")
    return url


# ---------------------------------------------------------------------------
# Generic split-adjustment support
# ---------------------------------------------------------------------------


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


# Registry of well-known stock splits that affect FTD data.  Callers can
# extend this at runtime via the *extra_splits* parameter on
# :meth:`SecFtdClient.fetch_ftd_data`.
_KNOWN_SPLITS: list[SplitAdjustment] = [
    SplitAdjustment(
        symbol="GME",
        split_date=date(2022, 7, 22),
        split_ratio=Decimal("4"),
    ),
]


class SecFtdClient:
    """Downloads and caches SEC Failure-to-Deliver data."""

    def __init__(
        self,
        user_agent: str = "StockDownloader admin@example.com",
        cache_dir: str = "data/cache/ftd",
    ) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_time: float = 0.0
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ftd_data(
        self,
        symbol: str,
        start_year: int = 2004,
        end_year: int | None = None,
        extra_splits: list[SplitAdjustment] | None = None,
    ) -> list[FtdRecord]:
        """Fetch FTD records for *symbol* across the given year range.

        Downloads and caches zip files from SEC, parses pipe-delimited
        contents, and filters by *symbol* (case-insensitive).

        Any known stock splits (see :data:`_KNOWN_SPLITS`) are applied
        automatically.  Pass *extra_splits* to supply additional split
        events without modifying the module-level registry.

        Returns records sorted by ``settlement_date`` ascending.
        """
        if end_year is None:
            end_year = datetime.now().year

        symbol_upper = symbol.upper()

        # Build the combined splits list and look up the one for this symbol.
        all_splits = _KNOWN_SPLITS + (extra_splits or [])
        split = next(
            (s for s in all_splits if s.symbol == symbol_upper), None,
        )
        all_records: list[FtdRecord] = []

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Skip future months
                now = datetime.now()
                if year == now.year and month > now.month:
                    break

                for half in ("a", "b"):
                    path = self._download_half_month(year, month, half)
                    if path is None:
                        continue
                    try:
                        content = self._read_zip(path)
                    except (zipfile.BadZipFile, OSError) as exc:
                        logger.warning(
                            "Failed to read zip %s: %s", path, exc,
                        )
                        continue

                    records = self._parse_ftd_file(
                        content, symbol_upper, split=split,
                    )
                    all_records.extend(records)

        all_records.sort(key=lambda r: r.settlement_date)
        return all_records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_half_month(
        self,
        year: int,
        month: int,
        half: str,
    ) -> Path | None:
        """Download and cache a single half-month FTD zip file.

        Returns the local file path on success, ``None`` on failure.
        """
        filename = f"cnsfails{year}{month:02d}{half}.zip"
        cached = self._cache_dir / filename
        if cached.exists():
            return cached

        url = _ftd_url(year, month, half)

        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limit()
                resp = self._session.get(url, timeout=30)
                if resp.status_code == 200:
                    cached.write_bytes(resp.content)
                    return cached
                if resp.status_code == 404:
                    # File does not exist (e.g. future dates) — don't retry
                    logger.debug(
                        "FTD file not found (404): %s", url,
                    )
                    return None
                if resp.status_code == 403:
                    # SEC rate-limited us — back off and retry
                    logger.info(
                        "SEC rate-limited (403): %s (attempt %d/%d)",
                        url, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(2.0 * (attempt + 1))
                    continue
                logger.warning(
                    "SEC returned %d for %s (attempt %d/%d)",
                    resp.status_code, url, attempt + 1, _MAX_RETRIES,
                )
            except (requests.RequestException, OSError) as exc:
                logger.warning(
                    "FTD download failed: %s (attempt %d/%d)",
                    exc, attempt + 1, _MAX_RETRIES,
                )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(1.0)

        return None

    @staticmethod
    def _read_zip(path: Path) -> str:
        """Extract the first text file from a cached zip archive.

        SEC FTD zips contain exactly one pipe-delimited text file.
        Some files use latin-1 encoding rather than UTF-8.
        """
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if not names:
                return ""
            raw = zf.read(names[0])
            # Try UTF-8 first, fall back to latin-1
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1")

    @staticmethod
    def _parse_ftd_file(
        content: str,
        symbol: str,
        *,
        split: SplitAdjustment | None = None,
    ) -> list[FtdRecord]:
        """Parse pipe-delimited FTD text and filter by *symbol*.

        Parameters
        ----------
        content:
            Raw pipe-delimited text from the SEC zip file.
        symbol:
            Ticker to filter for (already upper-cased).
        split:
            Optional :class:`SplitAdjustment` to apply.  Records with a
            settlement date **before** the split date will have their
            quantity multiplied and price divided by the split ratio.

        Returns
        -------
        List of :class:`FtdRecord` instances matching *symbol*.
        """
        records: list[FtdRecord] = []
        lines = content.splitlines()

        # Pre-format the split date for fast string comparison (dates in
        # ISO format are comparable as strings).
        split_date_str = split.split_date.isoformat() if split else None

        for line in lines:
            # Skip header and blank lines
            if not line.strip() or line.startswith("SETTLEMENT"):
                continue

            parts = line.split("|")
            if len(parts) < 6:
                continue

            raw_date = parts[0].strip()
            cusip = parts[1].strip()
            row_symbol = parts[2].strip().upper()
            raw_qty = parts[3].strip()
            description = parts[4].strip()
            raw_price = parts[5].strip()

            if row_symbol != symbol:
                continue

            # Parse settlement date: YYYYMMDD -> YYYY-MM-DD
            try:
                dt = datetime.strptime(raw_date, "%Y%m%d")
                settlement_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                logger.debug("Skipping line with bad date: %s", raw_date)
                continue

            # Parse quantity
            try:
                quantity = int(raw_qty)
            except ValueError:
                logger.debug("Skipping line with bad quantity: %s", raw_qty)
                continue

            # Parse price
            try:
                price = Decimal(raw_price) if raw_price else Decimal("0")
            except InvalidOperation:
                price = Decimal("0")

            # Apply split adjustment for records before the split date
            if split_date_str and settlement_date < split_date_str:
                quantity = int(Decimal(quantity) * split.split_ratio)
                if price > 0:
                    price = price / split.split_ratio

            try:
                records.append(FtdRecord(
                    settlement_date=settlement_date,
                    symbol=row_symbol,
                    cusip=cusip,
                    quantity=quantity,
                    description=description,
                    price=price,
                ))
            except ValueError:
                # Skip records that fail validation
                continue

        return records

    def _rate_limit(self) -> None:
        """Sleep if needed to maintain the SEC 10 req/sec rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()
