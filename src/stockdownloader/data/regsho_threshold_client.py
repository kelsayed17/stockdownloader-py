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
from stockdownloader.data.regsho_sources import (
    NYSE_429_BACKOFF_RANGE,
    NYSE_BATCH_SIZE,
    NYSE_URL_TEMPLATE,
    cffi_get,
    incremental_nyse_save,
    load_nyse_progress,
    nyse_rate_limit,
    query_cboe,
    query_nasdaq,
    query_nyse,
    query_occ,
    save_nyse_progress,
)

try:
    from curl_cffi import requests as curl_requests

    _HAS_CURL_CFFI = True
except ImportError:  # pragma: no cover – optional dependency
    _HAS_CURL_CFFI = False

logger = logging.getLogger(__name__)

_RATE_LIMIT_DELAY = 0.3           # Nasdaq / general


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
        rate_state: dict = {"request_count": 0, "last_request_time": 0.0}
        consecutive_429s = 0
        max_429s = 10  # More tolerant for targeted queries

        for date_str in remaining:
            if consecutive_429s >= max_429s:
                logger.warning(
                    "Too many 429s (%d), stopping. Progress saved.",
                    consecutive_429s,
                )
                break

            url = NYSE_URL_TEMPLATE.format(date=date_str)

            try:
                nyse_rate_limit(rate_state)

                # Incremental save after each batch
                if (
                    rate_state["request_count"] > 0
                    and rate_state["request_count"] % NYSE_BATCH_SIZE == 0
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
                        wait_secs = random.uniform(*NYSE_429_BACKOFF_RANGE)
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
            len(records), len(merged_cache), rate_state["request_count"],
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
    # Delegation to regsho_sources standalone functions
    # ------------------------------------------------------------------

    def _query_nyse(
        self,
        symbol: str,
        lookback_days: int = 5500,
        *,
        cached_dates: set[str] | None = None,
    ) -> list[ThresholdRecord]:
        """Delegate to :func:`regsho_sources.query_nyse`."""
        return query_nyse(
            symbol,
            lookback_days,
            cached_dates=cached_dates,
            curl_session=self._curl_session,
            session=self._session,
            load_cache_fn=self._load_cache,
            save_cache_fn=self._save_cache,
            progress_dir_fn=self._progress_dir,
            symbol_dir_fn=self._symbol_dir,
        )

    def _query_occ(
        self, symbol: str, lookback_days: int = 60
    ) -> list[ThresholdRecord]:
        """Delegate to :func:`regsho_sources.query_occ`."""
        return query_occ(symbol, lookback_days, session=self._session)

    def _query_nasdaq(
        self, symbol: str, lookback_days: int = 5500
    ) -> list[ThresholdRecord]:
        """Delegate to :func:`regsho_sources.query_nasdaq`."""
        return query_nasdaq(
            symbol, lookback_days,
            session=self._session,
            rate_limit_fn=self._rate_limit,
        )

    def _query_cboe(
        self, symbol: str, lookback_days: int = 5500
    ) -> list[ThresholdRecord]:
        """Delegate to :func:`regsho_sources.query_cboe`."""
        return query_cboe(symbol, lookback_days, session=self._session)

    def _cffi_get(self, url: str, timeout: int = 15) -> tuple[str, int]:
        """Delegate to :func:`regsho_sources.cffi_get`."""
        return cffi_get(self._curl_session, url, timeout)

    def _incremental_nyse_save(
        self,
        symbol: str,
        new_records: list[ThresholdRecord],
        cached_dates: set[str],
        queried_dates: set[str],
    ) -> None:
        """Delegate to :func:`regsho_sources.incremental_nyse_save`."""
        progress_dir = self._progress_dir(symbol)
        incremental_nyse_save(
            symbol, new_records, cached_dates, queried_dates,
            load_cache_fn=self._load_cache,
            save_cache_fn=self._save_cache,
            save_progress_fn=lambda qd: save_nyse_progress(progress_dir, qd),
        )

    def _save_nyse_progress(
        self, symbol: str, queried_dates: set[str]
    ) -> None:
        """Delegate to :func:`regsho_sources.save_nyse_progress`."""
        save_nyse_progress(self._progress_dir(symbol), queried_dates)

    def _load_nyse_progress(self, symbol: str) -> set[str]:
        """Delegate to :func:`regsho_sources.load_nyse_progress`."""
        return load_nyse_progress(
            self._progress_dir(symbol), self._symbol_dir(symbol),
        )

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
