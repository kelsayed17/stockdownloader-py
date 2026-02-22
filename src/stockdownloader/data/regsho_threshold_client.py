"""Fetches Reg SHO threshold list data from NYSE, OCC, Nasdaq, and CBOE.

The **Regulation SHO threshold list** contains securities where aggregate
failures to deliver (FTDs) have reached or exceeded 10,000 shares and
equal at least 0.5% of the issuer's total shares outstanding for 5
consecutive settlement days.

Being on the threshold list signals extreme short-selling pressure and
may trigger mandatory close-out requirements (forced buy-ins).

Sources (by historical depth):
- **NYSE**: ``https://www.nyse.com/api/regulatory/threshold-securities/download``
  (primary source — covers NYSE, NYSE Arca, NYSE American from ~2010 onward)
- **Nasdaq**: ``https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{YYYYMMDD}.txt``
  (Nasdaq-listed securities from 2006 onward)
- **CBOE BZX**: ``https://www.cboe.com/us/equities/market_statistics/reg_sho_threshold/``
  (BZX-listed ETFs from 2015 onward)
- **OCC**: ``https://marketdata.theocc.com/threshold-securities``
  (combined all-exchange list — only recent ~4-6 weeks available)

Usage::

    client = RegShoThresholdClient()
    records = client.fetch_threshold_status("GME")
    # -> list of dates when GME was on the threshold list
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import requests

from stockdownloader.data.base_client import BaseDataClient

try:
    from curl_cffi import requests as curl_requests

    _HAS_CURL_CFFI = True
except ImportError:  # pragma: no cover – optional dependency
    _HAS_CURL_CFFI = False

logger = logging.getLogger(__name__)

# Retry limits
_MAX_CONSECUTIVE_FAILURES = 15

# Rate-limiting delays (seconds) — randomised to avoid pattern detection
_NYSE_DELAY_RANGE = (1.5, 3.0)     # NYSE — slower to avoid Cloudflare blocks
_NYSE_BATCH_SIZE = 25              # Requests per batch before a long pause
_NYSE_BATCH_PAUSE_RANGE = (30, 60) # Seconds between batches
_NYSE_429_BACKOFF_RANGE = (120, 180)  # Seconds to back off on 429
_OCC_RATE_LIMIT_DELAY = 0.5       # OCC — polite pacing
_CBOE_RATE_LIMIT_DELAY = 0.3      # CBOE
_RATE_LIMIT_DELAY = 0.3           # Nasdaq / general

# NYSE threshold list API — primary source for NYSE/Arca/American stocks
_NYSE_URL_TEMPLATE = (
    "https://www.nyse.com/api/regulatory/threshold-securities/"
    "download?selectedDate={date}"
)
# Nasdaq daily threshold text files — Nasdaq-listed securities only
_NASDAQ_URL_TEMPLATE = (
    "https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{date}.txt"
)
# OCC combined threshold list — all exchanges, but only ~4-6 weeks history
_OCC_URL_TEMPLATE = (
    "https://marketdata.theocc.com/threshold-securities?reportDate={date}"
)
# CBOE BZX threshold list — BZX-listed ETFs, from 2015 onward
_CBOE_URL_TEMPLATE = (
    "https://www.cboe.com/us/equities/market_statistics/"
    "reg_sho_threshold/{date}/csv/"
)

# Browser-like headers for curl_cffi requests (Sec-Fetch-* headers).
# User-Agent and sec-ch-ua are set automatically by curl_cffi impersonation.
_BROWSER_HEADERS = {
    "Accept": "text/plain, text/csv, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


@dataclass(frozen=True, slots=True)
class ThresholdRecord:
    """A single Reg SHO threshold list appearance."""

    date: str             # YYYY-MM-DD
    symbol: str
    market: str           # "NYSE", "NYSE Arca", "NYSE American", "NASDAQ", etc.
    threshold_shares: int  # Shares at threshold level (if available)
    consecutive_days: int  # Days on threshold list (if trackable)

    def __post_init__(self) -> None:
        if not self.date:
            raise ValueError("date must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")


class RegShoThresholdClient(BaseDataClient):
    """Fetches Reg SHO threshold list data from multiple exchange sources.

    Queries **all available sources** and merges results for comprehensive
    coverage.  The NYSE API is the primary deep-history source for
    NYSE-listed securities (works from ~2010 onward).  OCC provides a
    combined all-exchange list but only retains ~4-6 weeks of data.

    Uses ``curl_cffi`` with Chrome TLS impersonation to bypass Cloudflare
    JA3/JA4 TLS fingerprinting on the NYSE API.  Falls back to the system
    ``curl`` binary if ``curl_cffi`` is not installed.

    Parameters
    ----------
    client_id:
        Reserved for future use (FINRA credentials).  Not required.
    client_secret:
        Reserved for future use (FINRA credentials).  Not required.
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/regsho_threshold.json``.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        data_dir: str = "data",
    ) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "StockDownloader admin@example.com",
            "Accept": "text/plain, text/csv, */*",
        })

        # curl_cffi session with Chrome TLS impersonation for NYSE
        if _HAS_CURL_CFFI:
            self._curl_session = curl_requests.Session(impersonate="chrome")
        else:
            self._curl_session = None

        self._last_request_time: float = 0.0
        self._last_nyse_request_time: float = 0.0
        self._last_occ_request_time: float = 0.0
        self._last_cboe_request_time: float = 0.0
        self._nyse_request_count: int = 0
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_threshold_status(
        self, symbol: str, lookback_days: int = 5500
    ) -> list[ThresholdRecord]:
        """Fetch all dates where *symbol* appeared on the Reg SHO threshold list.

        Queries **all** available sources and merges results for maximum
        coverage.  Each source covers different exchanges and time ranges:

        - NYSE API: NYSE, NYSE Arca, NYSE American (2010+)
        - Nasdaq: Nasdaq-listed securities (2006+)
        - CBOE BZX: BZX-listed ETFs (2015+)
        - OCC: All exchanges combined (~recent 4-6 weeks only)

        Results are merged with any previously cached data so that
        historical records are never lost.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"GME"``).
        lookback_days:
            How many calendar days of history to check.
            Default is ~15 years (5500 days).

        Returns
        -------
        List of :class:`ThresholdRecord` sorted by date ascending.
        """
        symbol_upper = symbol.upper()

        # Load cache first — for NYSE we can skip already-cached dates
        cached = self._load_cache(symbol_upper) or []
        cached_dates: set[str] = {r.date for r in cached}

        # Query all sources and merge
        all_records: list[ThresholdRecord] = []

        # 1. NYSE — primary deep-history source for NYSE-listed stocks
        logger.info("Querying NYSE threshold list for %s...", symbol_upper)
        nyse_records = self._query_nyse(
            symbol_upper, lookback_days, cached_dates=cached_dates,
        )
        all_records.extend(nyse_records)

        # 2. Nasdaq — covers Nasdaq-listed securities
        logger.info("Querying Nasdaq threshold list for %s...", symbol_upper)
        nasdaq_records = self._query_nasdaq(symbol_upper, lookback_days)
        all_records.extend(nasdaq_records)

        # 3. CBOE BZX — covers BZX-listed ETFs
        logger.info("Querying CBOE BZX threshold list for %s...", symbol_upper)
        cboe_records = self._query_cboe(symbol_upper, lookback_days)
        all_records.extend(cboe_records)

        # 4. OCC — supplemental, only has recent ~4-6 weeks
        logger.info("Querying OCC threshold list for %s...", symbol_upper)
        occ_records = self._query_occ(symbol_upper, lookback_days)
        all_records.extend(occ_records)

        # Merge with cache — dedup by (date, source_key)
        by_key: dict[tuple[str, str], ThresholdRecord] = {}
        for r in cached:
            by_key[(r.date, r.market)] = r
        for r in all_records:
            by_key[(r.date, r.market)] = r

        merged = sorted(by_key.values(), key=lambda r: r.date)

        if merged:
            self._save_cache(symbol_upper, merged)
            logger.info(
                "Reg SHO: %d total threshold dates for %s "
                "(NYSE=%d, Nasdaq=%d, CBOE=%d, OCC=%d, cached=%d)",
                len(merged), symbol_upper,
                len(nyse_records), len(nasdaq_records),
                len(cboe_records), len(occ_records), len(cached),
            )
        elif cached:
            logger.info(
                "Using cached Reg SHO data for %s (%d records)",
                symbol_upper, len(cached),
            )
            return cached

        return merged

    def fetch_threshold_targeted(
        self,
        symbol: str,
        target_dates: list[str],
    ) -> list[ThresholdRecord]:
        """Fetch threshold data for specific dates only (FTD-guided).

        Instead of scanning every weekday back to 2010, this method accepts
        a pre-computed list of dates where FTD data suggests the security
        *might* have been on the threshold list.  This dramatically reduces
        the number of NYSE API calls needed.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"GME"``).
        target_dates:
            List of ``"YYYY-MM-DD"`` date strings to query.

        Returns
        -------
        List of :class:`ThresholdRecord` sorted by date ascending.
        """
        symbol_upper = symbol.upper()

        # Load cache and progress so we skip already-queried dates
        cached = self._load_cache(symbol_upper) or []
        cached_dates: set[str] = {r.date for r in cached}
        queried_dates: set[str] = self._load_nyse_progress(symbol_upper)

        # Filter to only dates we haven't already queried
        skip = cached_dates | queried_dates
        remaining = sorted(d for d in target_dates if d not in skip)

        logger.info(
            "NYSE targeted query for %s: %d target dates, "
            "%d already queried, %d remaining",
            symbol_upper, len(target_dates), len(skip), len(remaining),
        )

        if not remaining:
            logger.info("All target dates already queried for %s", symbol_upper)
            return cached or []

        records: list[ThresholdRecord] = []
        self._nyse_request_count = 0
        consecutive_429s = 0
        max_429s = 10  # More tolerant for targeted queries

        for date_str in remaining:
            if consecutive_429s >= max_429s:
                logger.warning(
                    "Too many 429s (%d), stopping. Progress saved.",
                    consecutive_429s,
                )
                break

            url = _NYSE_URL_TEMPLATE.format(date=date_str)

            try:
                self._nyse_rate_limit()

                # Incremental save after each batch
                if (
                    self._nyse_request_count > 0
                    and self._nyse_request_count % _NYSE_BATCH_SIZE == 0
                ):
                    self._incremental_nyse_save(
                        symbol_upper, records, cached_dates, queried_dates,
                    )

                text, status = self._cffi_get(url)

                if status == 200 and text:
                    queried_dates.add(date_str)
                    consecutive_429s = 0
                    lines = text.strip().splitlines()
                    if len(lines) <= 2:
                        continue
                    for line in lines:
                        parts = line.split("|")
                        if len(parts) >= 4 and parts[0].strip() == symbol_upper:
                            market = parts[2].strip() or "NYSE"
                            records.append(ThresholdRecord(
                                date=date_str,
                                symbol=symbol_upper,
                                market=market,
                                threshold_shares=0,
                                consecutive_days=0,
                            ))
                            break
                elif status in (404, 204):
                    queried_dates.add(date_str)
                    consecutive_429s = 0
                elif status in (403, 429):
                    consecutive_429s += 1
                    if consecutive_429s in (5, 10):
                        wait_secs = random.uniform(*_NYSE_429_BACKOFF_RANGE)
                        logger.info(
                            "NYSE rate limited — pausing %.0fs (429 #%d)",
                            wait_secs, consecutive_429s,
                        )
                        time.sleep(wait_secs)
                        # Retry
                        text2, status2 = self._cffi_get(url)
                        if status2 == 200:
                            queried_dates.add(date_str)
                            consecutive_429s = 0
                            lines = text2.strip().splitlines()
                            for line in lines:
                                parts = line.split("|")
                                if (
                                    len(parts) >= 4
                                    and parts[0].strip() == symbol_upper
                                ):
                                    market = parts[2].strip() or "NYSE"
                                    records.append(ThresholdRecord(
                                        date=date_str,
                                        symbol=symbol_upper,
                                        market=market,
                                        threshold_shares=0,
                                        consecutive_days=0,
                                    ))
                                    break
                    logger.debug(
                        "NYSE %d on %s (429s: %d)",
                        status, date_str, consecutive_429s,
                    )
                else:
                    pass  # Other errors — skip silently
            except Exception as exc:
                logger.debug("NYSE request error for %s: %s", date_str, exc)

        # Final save
        if queried_dates:
            self._incremental_nyse_save(
                symbol_upper, records, cached_dates, queried_dates,
            )

        # Return merged cache
        merged_cache = self._load_cache(symbol_upper) or []
        logger.info(
            "NYSE targeted: %d new threshold records found, "
            "%d total records, %d requests made",
            len(records), len(merged_cache), self._nyse_request_count,
        )
        return merged_cache

    def is_currently_on_threshold(self, symbol: str) -> bool:
        """Check if *symbol* is currently on the threshold list.

        Looks at the last 5 trading days.
        """
        records = self.fetch_threshold_status(symbol, lookback_days=10)
        if not records:
            return False

        today = date.today()
        cutoff = today - timedelta(days=7)
        recent = [r for r in records if r.date >= cutoff.isoformat()]
        return len(recent) > 0

    # ------------------------------------------------------------------
    # NYSE threshold list (primary deep-history source)
    # ------------------------------------------------------------------

    def _query_nyse(
        self,
        symbol: str,
        lookback_days: int = 5500,
        *,
        cached_dates: set[str] | None = None,
    ) -> list[ThresholdRecord]:
        """Query NYSE daily threshold list API.

        The NYSE API returns pipe-delimited text covering NYSE, NYSE Arca,
        and NYSE American listed securities.  Data is available from ~2010
        onward.  Format::

            Symbol|Security Name|Market Category|Reg SHO Threshold Flag||

        The ``Market Category`` column contains the actual exchange
        (e.g. ``"NYSE"``, ``"NYSE Arca"``, ``"NYSE American"``).

        Uses ``curl_cffi`` with Chrome TLS impersonation to bypass
        Cloudflare JA3/JA4 TLS fingerprint blocking.  Randomised delays
        and batch pausing avoid triggering rate limits.

        Saves incrementally after every batch to preserve progress if the
        process is interrupted.
        """
        if cached_dates is None:
            cached_dates = set()

        # Load dates already queried in prior (interrupted) runs so we
        # can skip them entirely — even dates where GME wasn't found.
        queried_dates: set[str] = self._load_nyse_progress(symbol)
        if queried_dates:
            logger.info(
                "NYSE: resuming — %d dates already queried in prior runs",
                len(queried_dates),
            )

        records: list[ThresholdRecord] = []
        today = date.today()
        consecutive_failures = 0
        self._nyse_request_count = 0

        for offset in range(lookback_days):
            check_date = today - timedelta(days=offset)
            if check_date.weekday() >= 5:
                continue

            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Too many consecutive NYSE failures (%d), "
                    "stopping at %s",
                    consecutive_failures, check_date.isoformat(),
                )
                break

            date_str = check_date.strftime("%Y-%m-%d")

            # Skip dates we already have cached or queried in prior runs
            if date_str in cached_dates or date_str in queried_dates:
                consecutive_failures = 0  # known date = data exists or was checked
                continue

            url = _NYSE_URL_TEMPLATE.format(date=date_str)

            try:
                self._nyse_rate_limit()
                # Incremental save after each batch pause to preserve
                # progress if the process is interrupted.
                if (
                    self._nyse_request_count > 0
                    and self._nyse_request_count % _NYSE_BATCH_SIZE == 0
                ):
                    self._incremental_nyse_save(
                        symbol, records, cached_dates, queried_dates,
                    )

                text, status = self._cffi_get(url)

                if status == 200 and text:
                    queried_dates.add(date_str)
                    lines = text.strip().splitlines()
                    if len(lines) <= 2:
                        # Just header + timestamp — no threshold securities
                        consecutive_failures = 0
                        continue
                    consecutive_failures = 0
                    for line in lines:
                        parts = line.split("|")
                        if len(parts) >= 4 and parts[0].strip() == symbol:
                            market = parts[2].strip() or "NYSE"
                            records.append(ThresholdRecord(
                                date=check_date.isoformat(),
                                symbol=symbol,
                                market=market,
                                threshold_shares=0,
                                consecutive_days=0,
                            ))
                            break
                elif status in (404, 204):
                    queried_dates.add(date_str)
                    consecutive_failures = 0
                elif status in (403, 429):
                    consecutive_failures += 1
                    if consecutive_failures in (5, 10):
                        # Extended backoff on persistent rate limiting
                        wait_secs = random.uniform(*_NYSE_429_BACKOFF_RANGE)
                        logger.info(
                            "NYSE rate limited — pausing %.0fs before retry "
                            "(failure %d)",
                            wait_secs, consecutive_failures,
                        )
                        time.sleep(wait_secs)
                        text2, status2 = self._cffi_get(url)
                        if status2 == 200:
                            queried_dates.add(date_str)
                            consecutive_failures = 0
                            lines = text2.strip().splitlines()
                            for line in lines:
                                parts = line.split("|")
                                if (
                                    len(parts) >= 4
                                    and parts[0].strip() == symbol
                                ):
                                    market = parts[2].strip() or "NYSE"
                                    records.append(ThresholdRecord(
                                        date=check_date.isoformat(),
                                        symbol=symbol,
                                        market=market,
                                        threshold_shares=0,
                                        consecutive_days=0,
                                    ))
                                    break
                            continue
                    logger.info(
                        "NYSE %d on %s (failures: %d)",
                        status, date_str, consecutive_failures,
                    )
                else:
                    consecutive_failures += 1
            except Exception as exc:
                consecutive_failures += 1
                logger.debug("NYSE request error for %s: %s", date_str, exc)

        if records:
            logger.info(
                "NYSE Reg SHO: %d threshold dates for %s", len(records), symbol,
            )
        # Final save to ensure all progress is persisted (even if no
        # new threshold records were found, we still save queried_dates)
        if queried_dates:
            self._incremental_nyse_save(
                symbol, records, cached_dates, queried_dates,
            )
        return records

    def _incremental_nyse_save(
        self,
        symbol: str,
        new_records: list[ThresholdRecord],
        cached_dates: set[str],
        queried_dates: set[str],
    ) -> None:
        """Merge *new_records* with existing cache and save.

        Called periodically during long NYSE backfills so that progress
        is preserved if the process is interrupted.  Also saves the set
        of all *queried_dates* (including dates where the symbol was NOT
        on the threshold list) so that re-runs can skip already-checked
        dates entirely.
        """
        try:
            cached = self._load_cache(symbol) or []
            by_key: dict[tuple[str, str], ThresholdRecord] = {}
            for r in cached:
                by_key[(r.date, r.market)] = r
            for r in new_records:
                by_key[(r.date, r.market)] = r
            merged = sorted(by_key.values(), key=lambda r: r.date)
            self._save_cache(symbol, merged)
            # Update cached_dates so future iterations skip saved dates
            for r in new_records:
                cached_dates.add(r.date)
            # Save queried-dates progress file
            self._save_nyse_progress(symbol, queried_dates)
            logger.info(
                "NYSE incremental save: %d threshold records, %d dates queried for %s",
                len(merged), len(queried_dates), symbol,
            )
        except Exception as exc:
            logger.warning(
                "NYSE incremental save failed for %s: %s", symbol, exc,
            )

    def _save_nyse_progress(
        self, symbol: str, queried_dates: set[str]
    ) -> None:
        """Persist set of NYSE dates already queried for *symbol*."""
        progress_file = self._progress_dir(symbol) / "regsho_nyse.json"
        try:
            existing: set[str] = set()
            if progress_file.exists():
                existing = set(json.loads(
                    progress_file.read_text(encoding="utf-8")
                ))
            combined = sorted(existing | queried_dates)
            progress_file.write_text(
                json.dumps(combined), encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Failed to save NYSE progress for %s: %s", symbol, exc)

    def _load_nyse_progress(self, symbol: str) -> set[str]:
        """Load previously queried NYSE dates for *symbol*."""
        progress_file = self._progress_dir(symbol) / "regsho_nyse.json"
        legacy = self._symbol_dir(symbol) / "regsho_nyse_progress.json"
        if not progress_file.exists() and legacy.exists():
            legacy.rename(progress_file)
            logger.info("Migrated %s → %s", legacy, progress_file)
        if not progress_file.exists():
            return set()
        try:
            return set(json.loads(progress_file.read_text(encoding="utf-8")))
        except Exception:
            return set()

    def _cffi_get(self, url: str, timeout: int = 15) -> tuple[str, int]:
        """HTTP GET via ``curl_cffi`` with Chrome TLS impersonation.

        ``curl_cffi`` uses ``curl-impersonate`` under the hood — a patched
        ``libcurl`` that replaces OpenSSL with Chrome's BoringSSL and mimics
        real browser TLS handshakes, HTTP/2 frames, and cipher suites.  This
        produces JA3/JA4 fingerprints that match actual Chrome browsers,
        bypassing Cloudflare TLS fingerprint detection.

        Falls back to system ``curl`` via subprocess if ``curl_cffi`` is not
        available.

        Returns ``(body_text, http_status_code)``.
        """
        if self._curl_session is not None:
            try:
                resp = self._curl_session.get(
                    url, timeout=timeout, headers=_BROWSER_HEADERS,
                )
                return resp.text, resp.status_code
            except Exception:
                return "", 0
        else:
            # Fallback: subprocess curl
            return self._curl_get(url, timeout=timeout)

    @staticmethod
    def _curl_get(url: str, timeout: int = 15) -> tuple[str, int]:
        """HTTP GET via system ``curl`` binary (fallback).

        Used when ``curl_cffi`` is not installed.  Spawns a subprocess for
        each request, which is slower but still bypasses Cloudflare TLS
        fingerprinting since the system curl has a different TLS
        implementation than Python's ``requests`` library.

        Returns ``(body_text, http_status_code)``.
        """
        import subprocess

        try:
            result = subprocess.run(
                [
                    "curl", "-s",
                    "-o", "-",               # body to stdout
                    "-w", "\n%{http_code}",   # status code after body
                    "--max-time", str(timeout),
                    url,
                ],
                capture_output=True, text=True, timeout=timeout + 5,
            )
            output = result.stdout
            # Last line is the HTTP status code
            lines = output.rsplit("\n", 1)
            if len(lines) == 2:
                body = lines[0]
                try:
                    status = int(lines[1].strip())
                except ValueError:
                    status = 0
            else:
                body = output
                status = 0
            return body, status
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "", 0

    def _nyse_rate_limit(self, delay: float | None = None) -> None:
        """Rate limit for NYSE requests with randomised delay and batch pausing.

        Implements two layers of pacing:

        1. **Intra-batch delay**: randomised 1.5-3.0s between individual
           requests to avoid pattern detection by Cloudflare.
        2. **Batch pause**: every 25 requests, pause 30-60s to let
           Cloudflare's sliding-window rate counter decay.
        """
        self._nyse_request_count += 1

        # Batch pause every _NYSE_BATCH_SIZE requests
        if self._nyse_request_count % _NYSE_BATCH_SIZE == 0:
            pause = random.uniform(*_NYSE_BATCH_PAUSE_RANGE)
            logger.info(
                "NYSE batch pause: %.0fs after %d requests",
                pause, self._nyse_request_count,
            )
            time.sleep(pause)
            self._last_nyse_request_time = time.monotonic()
            return

        # Normal intra-batch delay (randomised)
        actual_delay = delay or random.uniform(*_NYSE_DELAY_RANGE)
        now = time.monotonic()
        elapsed = now - self._last_nyse_request_time
        if elapsed < actual_delay:
            time.sleep(actual_delay - elapsed)
        self._last_nyse_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # OCC combined threshold list (supplemental — recent only)
    # ------------------------------------------------------------------

    def _query_occ(
        self, symbol: str, lookback_days: int = 60
    ) -> list[ThresholdRecord]:
        """Query the OCC combined threshold list.

        The OCC (Options Clearing Corporation) publishes a combined
        threshold list covering NYSE, NASDAQ, NYSE Arca, NYSE American,
        and other exchanges.  Pipe-delimited format::

            Symbol|Security Name|Market Category|Reg SHO Flag|...

        .. note::
            OCC only retains approximately 4-6 weeks of history.
            Older dates return ``"File requested does not exist."``.
            The lookback is capped at 60 days by default.
        """
        # OCC only has ~4-6 weeks of data, so cap lookback
        effective_lookback = min(lookback_days, 60)
        records: list[ThresholdRecord] = []
        today = date.today()
        consecutive_failures = 0

        for offset in range(effective_lookback):
            check_date = today - timedelta(days=offset)
            if check_date.weekday() >= 5:
                continue

            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.debug(
                    "OCC: %d consecutive failures, stopping at %s",
                    consecutive_failures, check_date.isoformat(),
                )
                break

            date_str = check_date.strftime("%Y%m%d")
            url = _OCC_URL_TEMPLATE.format(date=date_str)

            try:
                self._occ_rate_limit()
                resp = self._session.get(url, timeout=15)
                if resp.status_code == 200:
                    text = resp.text.strip()
                    # OCC returns "File requested does not exist." for old dates
                    if "does not exist" in text.lower():
                        consecutive_failures += 1
                        continue
                    consecutive_failures = 0
                    for line in text.splitlines():
                        parts = line.split("|")
                        if len(parts) >= 4 and parts[0].strip() == symbol:
                            market = parts[2].strip() or "OCC"
                            records.append(ThresholdRecord(
                                date=check_date.isoformat(),
                                symbol=symbol,
                                market=market,
                                threshold_shares=0,
                                consecutive_days=0,
                            ))
                            break
                elif resp.status_code in (404, 204):
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except requests.RequestException:
                consecutive_failures += 1

        if records:
            logger.info(
                "OCC Reg SHO: %d threshold dates for %s",
                len(records), symbol,
            )
        return records

    def _occ_rate_limit(self) -> None:
        """Rate limit for OCC requests."""
        now = time.monotonic()
        elapsed = now - self._last_occ_request_time
        if elapsed < _OCC_RATE_LIMIT_DELAY:
            time.sleep(_OCC_RATE_LIMIT_DELAY - elapsed)
        self._last_occ_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Nasdaq text file query
    # ------------------------------------------------------------------

    def _query_nasdaq(
        self, symbol: str, lookback_days: int = 5500
    ) -> list[ThresholdRecord]:
        """Check Nasdaq daily threshold list text files.

        Nasdaq publishes pipe-delimited threshold lists at a predictable
        URL pattern.  Available from ~2006 onward but only covers
        Nasdaq-listed securities (market categories Q, G, S).
        """
        records: list[ThresholdRecord] = []
        today = date.today()
        consecutive_failures = 0

        for offset in range(lookback_days):
            check_date = today - timedelta(days=offset)
            if check_date.weekday() >= 5:
                continue

            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Too many consecutive Nasdaq failures, stopping at %s",
                    check_date.isoformat(),
                )
                break

            date_str = check_date.strftime("%Y%m%d")
            url = _NASDAQ_URL_TEMPLATE.format(date=date_str)

            try:
                self._rate_limit()
                resp = self._session.get(url, timeout=10)
                if resp.status_code == 200 and "text" in resp.headers.get(
                    "Content-Type", ""
                ):
                    consecutive_failures = 0
                    for line in resp.text.splitlines():
                        parts = line.split("|")
                        if len(parts) >= 2 and parts[0].strip() == symbol:
                            # Extract market category if available
                            market = "NASDAQ"
                            if len(parts) >= 3:
                                cat = parts[2].strip()
                                if cat in ("Q", "G", "S"):
                                    market = f"NASDAQ ({cat})"
                            records.append(ThresholdRecord(
                                date=check_date.isoformat(),
                                symbol=symbol,
                                market=market,
                                threshold_shares=0,
                                consecutive_days=0,
                            ))
                            break
                elif resp.status_code in (302, 404):
                    # 302 redirect = file doesn't exist
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except requests.RequestException:
                consecutive_failures += 1

        if records:
            logger.info(
                "Nasdaq Reg SHO: %d threshold dates for %s",
                len(records), symbol,
            )
        return records

    # ------------------------------------------------------------------
    # CBOE BZX threshold list
    # ------------------------------------------------------------------

    def _query_cboe(
        self, symbol: str, lookback_days: int = 5500
    ) -> list[ThresholdRecord]:
        """Query CBOE BZX daily threshold list.

        CBOE publishes pipe-delimited threshold lists for BZX-listed
        securities (mostly leveraged/buffer ETFs).  Available from
        ~2015 onward.  Format::

            Symbol|CompanyName
            MSTU|T-Rex 2X Long MSTR Daily Target ETF

        Last line is a timestamp (``YYYYMMDDHHMMSS``).
        """
        records: list[ThresholdRecord] = []
        today = date.today()
        consecutive_failures = 0

        # CBOE data starts around 2015
        earliest = date(2015, 1, 1)

        for offset in range(lookback_days):
            check_date = today - timedelta(days=offset)
            if check_date < earliest:
                break
            if check_date.weekday() >= 5:
                continue

            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.debug(
                    "CBOE: %d consecutive failures, stopping at %s",
                    consecutive_failures, check_date.isoformat(),
                )
                break

            date_str = check_date.strftime("%Y-%m-%d")
            url = _CBOE_URL_TEMPLATE.format(date=date_str)

            try:
                self._cboe_rate_limit()
                resp = self._session.get(url, timeout=10)
                if resp.status_code == 200:
                    consecutive_failures = 0
                    for line in resp.text.splitlines():
                        parts = line.split("|")
                        if len(parts) >= 2 and parts[0].strip() == symbol:
                            records.append(ThresholdRecord(
                                date=check_date.isoformat(),
                                symbol=symbol,
                                market="CBOE BZX",
                                threshold_shares=0,
                                consecutive_days=0,
                            ))
                            break
                elif resp.status_code in (404, 204):
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except requests.RequestException:
                consecutive_failures += 1

        if records:
            logger.info(
                "CBOE Reg SHO: %d threshold dates for %s",
                len(records), symbol,
            )
        return records

    def _cboe_rate_limit(self) -> None:
        """Rate limit for CBOE requests."""
        now = time.monotonic()
        elapsed = now - self._last_cboe_request_time
        if elapsed < _CBOE_RATE_LIMIT_DELAY:
            time.sleep(_CBOE_RATE_LIMIT_DELAY - elapsed)
        self._last_cboe_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _symbol_dir(self, symbol: str) -> Path:
        """Return per-symbol data directory, creating it if needed."""
        d = self._data_dir / symbol.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_cache(self, symbol: str) -> list[ThresholdRecord] | None:
        records = self._load_csv(symbol, "regsho_threshold.csv", ThresholdRecord)
        if records is not None:
            records.sort(key=lambda r: r.date)
            return records
        # JSON fallback
        sym_dir = self._symbol_dir(symbol)
        json_path = sym_dir / "regsho_threshold.json"
        if not json_path.exists():
            for legacy_path in (
                self._data_dir / "cache" / "regsho" / symbol.upper() / "threshold.json",
                self._data_dir / "cache" / "regsho" / f"{symbol}_threshold.json",
            ):
                if legacy_path.exists():
                    legacy_path.rename(json_path)
                    logger.info("Migrated %s -> %s", legacy_path, json_path)
                    break
        if not json_path.exists():
            return None
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            records = [ThresholdRecord(**r) for r in data]
            records.sort(key=lambda r: r.date)
            self._save_cache(symbol, records)
            logger.info("Migrated %s to CSV", json_path)
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to load Reg SHO cache for %s: %s", symbol, exc)
            return None

    def _save_cache(
        self, symbol: str, records: list[ThresholdRecord]
    ) -> None:
        self._save_csv(symbol, "regsho_threshold.csv", records)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()
