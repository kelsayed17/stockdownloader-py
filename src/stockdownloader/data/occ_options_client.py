"""Fetches daily per-symbol options open interest from OCC bulk downloads.

The OCC (Options Clearing Corporation) publishes a daily bulk file
containing volume, exercised contracts, and open interest for every
option symbol across all exchanges and expirations.

Data source::

    https://marketdata.theocc.com/cont-volume-download?reportDate=YYYYMMDD&format=txt

The file is fixed-width text (~202K lines per day, all symbols).
Available from January 2021 to present.  Free, no authentication.

Each record is 52 characters:

    Position  Width  Field
    0-5       6      Symbol (space-padded)
    6-11      6      Underlying (space-padded)
    12        1      Exchange code (single char)
    13-21     9      Volume (zero-padded)
    22-30     9      Exercised (zero-padded)
    31-39     9      Open interest (zero-padded)
    40-43     4      Product kind (OSTK, OIND, etc.)
    44-51     8      Expiration (YYYYMMDD)

Usage::

    client = OccOptionsClient()
    records = client.fetch_open_interest("GME")
    # -> list of OccOpenInterestRecord sorted by date ascending
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from stockdownloader.data.base_client import BaseDataClient
from stockdownloader.model.regulatory_records import OccOpenInterestRecord

logger = logging.getLogger(__name__)

_OCC_URL_TEMPLATE = (
    "https://marketdata.theocc.com/cont-volume-download"
    "?reportDate={date}&format=txt"
)

_BATCH_SIZE = 25
_MAX_CONSECUTIVE_FAILURES = 15
# OCC cont-volume-download keeps a rolling ~6-week window.
# Dates outside this window return 200 with "File requested does not
# exist."  We start 60 days back to capture the full window.
_OCC_LOOKBACK_DAYS = 60


class OccOptionsClient(BaseDataClient):
    """OCC daily options open interest and volume data.

    Downloads the daily bulk file from OCC, parses fixed-width text,
    and filters for the requested symbol.  Caches per-symbol results
    and tracks download progress for incremental backfill.

    Parameters
    ----------
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/occ_open_interest.json``.
    """

    def __init__(self, data_dir: str = "data") -> None:
        super().__init__(
            rate_limit_delay=0.5,
            max_retries=3,
            data_dir=data_dir,
            default_headers={
                "User-Agent": "StockDownloader admin@example.com",
            },
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_bulk_line(
        self, line: str, report_date: str,
    ) -> OccOpenInterestRecord | None:
        """Parse a single fixed-width line from OCC bulk download.

        Returns ``None`` for header lines, blank lines, or lines
        shorter than the expected 52 characters.
        """
        if len(line) < 48:  # Minimum viable length
            return None

        # Header lines start with spaces + "H"
        stripped = line.lstrip()
        if stripped.startswith("H"):
            return None

        try:
            symbol = line[0:6].strip()
            if not symbol:
                return None

            exchange = line[12] if len(line) > 12 else ""
            volume = int(line[13:22]) if len(line) >= 22 else 0
            exercised = int(line[22:31]) if len(line) >= 31 else 0
            oi = int(line[31:40]) if len(line) >= 40 else 0
            product_kind = line[40:44].strip() if len(line) >= 44 else ""
            exp_raw = line[44:52].strip() if len(line) >= 52 else ""

            expiration = ""
            if len(exp_raw) == 8 and exp_raw.isdigit():
                expiration = f"{exp_raw[:4]}-{exp_raw[4:6]}-{exp_raw[6:8]}"

            return OccOpenInterestRecord(
                date=report_date,
                symbol=symbol,
                exchange=exchange,
                volume=volume,
                exercised=exercised,
                open_interest=oi,
                product_kind=product_kind,
                expiration=expiration,
            )
        except (ValueError, IndexError):
            return None

    def _parse_bulk_text(
        self, text: str, symbol: str, report_date: str,
    ) -> list[OccOpenInterestRecord]:
        """Parse full OCC bulk text and filter for *symbol*."""
        symbol_upper = symbol.upper()
        records: list[OccOpenInterestRecord] = []

        for line in text.splitlines():
            line = line.rstrip("\r\n")
            if not line:
                continue
            record = self._parse_bulk_line(line, report_date)
            if record and record.symbol == symbol_upper:
                records.append(record)

        return records

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_day(
        self, date_str: str, symbol: str,
    ) -> list[OccOpenInterestRecord]:
        """Download and parse OCC bulk file for one day.

        Parameters
        ----------
        date_str:
            Date in ``"YYYY-MM-DD"`` format.
        symbol:
            Ticker to filter for.

        Returns empty list if download fails or no data for symbol.
        """
        # OCC expects YYYYMMDD format
        occ_date = date_str.replace("-", "")
        url = _OCC_URL_TEMPLATE.format(date=occ_date)

        resp = self._fetch_with_retry("GET", url, timeout=60)
        if resp is None or not resp.ok:
            return []

        text = resp.text
        if not text or len(text) < 100:
            # OCC returns "File requested does not exist." (200) for
            # dates outside the rolling window
            return []

        return self._parse_bulk_text(text, symbol, date_str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_open_interest(
        self,
        symbol: str,
        start_date: str | None = None,
    ) -> list[OccOpenInterestRecord]:
        """Fetch daily OI for *symbol* from OCC bulk files.

        Downloads one bulk file per trading day, parses and filters
        for the requested symbol.  Saves progress incrementally so
        interrupted backfills can resume.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"GME"``).
        start_date:
            Earliest date to fetch (``"YYYY-MM-DD"``).  Defaults to
            60 days ago (OCC keeps a rolling ~6-week window).

        Returns
        -------
        List of :class:`OccOpenInterestRecord` sorted by date ascending.
        """
        symbol_upper = symbol.upper()

        # OCC keeps a rolling ~6-week window; default to 60 days back
        end = date.today()
        earliest = end - timedelta(days=_OCC_LOOKBACK_DAYS)
        if start_date:
            parsed = date.fromisoformat(start_date)
            earliest = max(parsed, earliest)

        # Generate all weekdays in range
        all_dates: list[str] = []
        current = earliest
        while current <= end:
            if current.weekday() < 5:  # Mon-Fri
                all_dates.append(current.isoformat())
            current += timedelta(days=1)

        # Load progress and cache
        progress = self._load_progress(symbol_upper)
        cached = self._load_cache(symbol_upper) or []
        cached_dates = {r.date for r in cached}

        # Filter to dates not yet downloaded
        skip = progress | cached_dates
        remaining = [d for d in all_dates if d not in skip]

        logger.info(
            "OCC open interest for %s: %d total days, "
            "%d already done, %d remaining",
            symbol_upper, len(all_dates), len(skip), len(remaining),
        )

        if not remaining:
            return cached

        # Download remaining dates
        new_records: list[OccOpenInterestRecord] = []
        consecutive_failures = 0

        for i, date_str in enumerate(remaining):
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Too many consecutive failures (%d), stopping. "
                    "Progress saved for %s.",
                    consecutive_failures, symbol_upper,
                )
                break

            try:
                # _fetch_with_retry already calls _rate_limit internally
                day_records = self._download_day(date_str, symbol_upper)

                progress.add(date_str)
                if day_records:
                    new_records.extend(day_records)
                    consecutive_failures = 0
                else:
                    # Empty response is normal for holidays
                    consecutive_failures = 0

            except Exception as exc:
                consecutive_failures += 1
                logger.debug(
                    "OCC download error for %s on %s: %s",
                    symbol_upper, date_str, exc,
                )
                continue

            # Incremental save
            if (i + 1) % _BATCH_SIZE == 0:
                merged = self._merge_records(cached, new_records)
                self._save_cache(symbol_upper, merged)
                self._save_progress(symbol_upper, progress)
                logger.info(
                    "OCC progress: %d/%d dates, %d records so far",
                    i + 1, len(remaining), len(merged),
                )

        # Final save
        merged = self._merge_records(cached, new_records)
        self._save_cache(symbol_upper, merged)
        self._save_progress(symbol_upper, progress)

        logger.info(
            "OCC open interest for %s: %d new records, %d total",
            symbol_upper, len(new_records), len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_records(
        existing: list[OccOpenInterestRecord],
        new: list[OccOpenInterestRecord],
    ) -> list[OccOpenInterestRecord]:
        """Merge existing and new records, deduplicating by composite key."""
        by_key: dict[tuple, OccOpenInterestRecord] = {}
        for r in existing:
            key = (r.date, r.symbol, r.exchange, r.expiration)
            by_key[key] = r
        for r in new:
            key = (r.date, r.symbol, r.exchange, r.expiration)
            by_key[key] = r  # New overwrites
        merged = list(by_key.values())
        merged.sort(key=lambda r: (r.date, r.exchange, r.expiration))
        return merged

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cache(
        self, symbol: str,
    ) -> list[OccOpenInterestRecord] | None:
        cache_file = self._symbol_dir(symbol) / "occ_open_interest.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            records = [OccOpenInterestRecord(**r) for r in data]
            records.sort(key=lambda r: (r.date, r.exchange, r.expiration))
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to load OCC cache for %s: %s", symbol, exc,
            )
            return None

    def _save_cache(
        self, symbol: str, records: list[OccOpenInterestRecord],
    ) -> None:
        cache_file = self._symbol_dir(symbol) / "occ_open_interest.json"
        try:
            cache_file.write_text(
                json.dumps([asdict(r) for r in records], indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save OCC cache for %s: %s", symbol, exc,
            )

    def _load_progress(self, symbol: str) -> set[str]:
        progress_file = self._symbol_dir(symbol) / "occ_options_progress.json"
        if not progress_file.exists():
            return set()
        try:
            return set(json.loads(progress_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return set()

    def _save_progress(self, symbol: str, dates: set[str]) -> None:
        progress_file = self._symbol_dir(symbol) / "occ_options_progress.json"
        try:
            progress_file.write_text(
                json.dumps(sorted(dates)), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save OCC progress for %s: %s", symbol, exc,
            )
