"""Fetches Reg SHO threshold list data from OCC, Nasdaq, and NYSE.

The **Regulation SHO threshold list** contains securities where aggregate
failures to deliver (FTDs) have reached or exceeded 10,000 shares and
equal at least 0.5% of the issuer's total shares outstanding for 5
consecutive settlement days.

Being on the threshold list signals extreme short-selling pressure and
may trigger mandatory close-out requirements (forced buy-ins).

Sources (in priority order):
- **OCC**: ``https://marketdata.theocc.com/threshold-securities``
  (combined list covering ALL exchanges — preferred source)
- **Nasdaq**: ``https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{YYYYMMDD}.txt``
- **NYSE**: ``https://www.nyse.com/api/regulatory/threshold-securities/download``
  (often Cloudflare-blocked — used as last resort)

Usage::

    client = RegShoThresholdClient()
    records = client.fetch_threshold_status("GME")
    # -> list of dates when GME was on the threshold list
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RATE_LIMIT_DELAY = 0.3
_NYSE_RATE_LIMIT_DELAY = 1.0  # NYSE Cloudflare is aggressive
_OCC_RATE_LIMIT_DELAY = 0.5   # OCC is generous but be polite

# OCC combined threshold list (covers ALL exchanges — preferred source)
_OCC_URL_TEMPLATE = (
    "https://marketdata.theocc.com/threshold-securities?reportDate={date}"
)
# Nasdaq publishes daily threshold lists (YYYYMMDD format)
_NASDAQ_URL_TEMPLATE = (
    "https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{date}.txt"
)
# NYSE threshold list API (date as YYYY-MM-DD query param)
_NYSE_URL_TEMPLATE = (
    "https://www.nyse.com/api/regulatory/threshold-securities/"
    "download?selectedDate={date}"
)


@dataclass(frozen=True, slots=True)
class ThresholdRecord:
    """A single Reg SHO threshold list appearance."""

    date: str             # YYYY-MM-DD
    symbol: str
    market: str           # "FINRA", "NYSE", "NASDAQ"
    threshold_shares: int  # Shares at threshold level (if available)
    consecutive_days: int  # Days on threshold list (if trackable)

    def __post_init__(self) -> None:
        if not self.date:
            raise ValueError("date must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")


class RegShoThresholdClient:
    """Fetches Reg SHO threshold list data from OCC, Nasdaq, and NYSE.

    The preferred source is the **OCC combined threshold list** which
    covers all exchanges in a single request without Cloudflare issues.
    Falls back to Nasdaq text files and NYSE API if OCC is unavailable.

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
            "Accept": "text/plain",
        })
        self._last_request_time: float = 0.0
        self._last_nyse_request_time: float = 0.0
        self._last_occ_request_time: float = 0.0
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_threshold_status(
        self, symbol: str, lookback_days: int = 1825
    ) -> list[ThresholdRecord]:
        """Fetch all dates where *symbol* appeared on the Reg SHO threshold list.

        Queries the OCC combined list first (covers all exchanges),
        then falls back to Nasdaq and NYSE individual queries if OCC
        returned nothing.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"GME"``).
        lookback_days:
            How many days of history to check.

        Returns
        -------
        List of :class:`ThresholdRecord` sorted by date ascending.
        """
        symbol_upper = symbol.upper()
        records: list[ThresholdRecord] = []

        # Priority 1: OCC combined list (covers ALL exchanges)
        occ_records = self._query_occ(symbol_upper, lookback_days)
        records.extend(occ_records)

        # Priority 2: Only query individual exchanges if OCC
        # returned no results (avoids redundant network calls)
        if not occ_records:
            # Query NYSE (covers NYSE-listed securities like GME)
            nyse_records = self._query_nyse(symbol_upper, lookback_days)
            records.extend(nyse_records)

            # Query Nasdaq (covers Nasdaq-listed securities)
            nasdaq_records = self._query_nasdaq(symbol_upper, lookback_days)
            # Merge without duplicates
            existing_dates = {(r.date, r.market) for r in records}
            for nr in nasdaq_records:
                if (nr.date, nr.market) not in existing_dates:
                    records.append(nr)

        if records:
            records.sort(key=lambda r: r.date)
            # Merge with cache (preserves historical data)
            cached = self._load_cache(symbol_upper) or []
            by_key: dict[tuple[str, str], ThresholdRecord] = {}
            for r in cached:
                by_key[(r.date, r.market)] = r
            for r in records:
                by_key[(r.date, r.market)] = r
            merged = sorted(by_key.values(), key=lambda r: r.date)
            self._save_cache(symbol_upper, merged)
            return merged

        # Fall back to cache
        cached = self._load_cache(symbol_upper)
        if cached:
            logger.info(
                "Using cached Reg SHO data for %s (%d records)",
                symbol_upper, len(cached),
            )
            return cached

        return []

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
    # OCC combined threshold list (covers all exchanges)
    # ------------------------------------------------------------------

    def _query_occ(
        self, symbol: str, lookback_days: int = 1825
    ) -> list[ThresholdRecord]:
        """Query the OCC combined threshold list.

        The OCC (Options Clearing Corporation) publishes a combined
        threshold list covering NYSE, NASDAQ, NYSE Arca, NYSE American,
        and other exchanges.  Pipe-delimited format::

            Symbol|Security Name|Market Category|Reg SHO Flag|...

        The ``Market Category`` column contains the actual exchange
        (e.g. ``"NYSE"``, ``"NYSE Arca"``, ``"G"`` for OTC).
        """
        records: list[ThresholdRecord] = []
        today = date.today()
        consecutive_failures = 0

        for offset in range(lookback_days):
            check_date = today - timedelta(days=offset)
            if check_date.weekday() >= 5:
                continue

            if consecutive_failures >= _MAX_RETRIES:
                logger.warning(
                    "Too many consecutive OCC failures, stopping at %s",
                    check_date.isoformat(),
                )
                break

            date_str = check_date.strftime("%Y%m%d")
            url = _OCC_URL_TEMPLATE.format(date=date_str)

            try:
                self._occ_rate_limit()
                resp = self._session.get(url, timeout=15)
                if resp.status_code == 200:
                    consecutive_failures = 0
                    for line in resp.text.splitlines():
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
    # NYSE threshold list query
    # ------------------------------------------------------------------

    def _query_nyse(
        self, symbol: str, lookback_days: int = 1825
    ) -> list[ThresholdRecord]:
        """Query NYSE daily threshold list API.

        The NYSE API returns pipe-delimited text with format:
        ``Symbol|Security Name|Market|Reg SHO Threshold Flag||``

        NYSE's Cloudflare protection is aggressive, so we use a
        slower rate limit and back off on 429 responses.
        """
        records: list[ThresholdRecord] = []
        today = date.today()
        consecutive_failures = 0

        for offset in range(lookback_days):
            check_date = today - timedelta(days=offset)
            if check_date.weekday() >= 5:
                continue

            if consecutive_failures >= _MAX_RETRIES:
                logger.warning(
                    "Too many consecutive NYSE failures, stopping at %s",
                    check_date.isoformat(),
                )
                break

            date_str = check_date.strftime("%Y-%m-%d")
            url = _NYSE_URL_TEMPLATE.format(date=date_str)

            try:
                self._nyse_rate_limit()
                resp = self._session.get(url, timeout=10)
                if resp.status_code == 200:
                    consecutive_failures = 0
                    for line in resp.text.splitlines():
                        parts = line.split("|")
                        if len(parts) >= 2 and parts[0].strip() == symbol:
                            records.append(ThresholdRecord(
                                date=check_date.isoformat(),
                                symbol=symbol,
                                market="NYSE",
                                threshold_shares=0,
                                consecutive_days=0,
                            ))
                            break
                elif resp.status_code == 429:
                    consecutive_failures += 1
                    retry_after = int(
                        resp.headers.get("Retry-After", "60")
                    )
                    logger.info(
                        "NYSE rate limited on %s (retry-after: %ds)",
                        date_str, retry_after,
                    )
                elif resp.status_code in (404, 204):
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except requests.RequestException:
                consecutive_failures += 1

        if records:
            logger.info(
                "NYSE Reg SHO: %d threshold dates for %s",
                len(records), symbol,
            )
        return records

    def _nyse_rate_limit(self) -> None:
        """Separate, slower rate limit for NYSE requests."""
        now = time.monotonic()
        elapsed = now - self._last_nyse_request_time
        if elapsed < _NYSE_RATE_LIMIT_DELAY:
            time.sleep(_NYSE_RATE_LIMIT_DELAY - elapsed)
        self._last_nyse_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Nasdaq text file query
    # ------------------------------------------------------------------

    def _query_nasdaq(
        self, symbol: str, lookback_days: int = 1825
    ) -> list[ThresholdRecord]:
        """Check Nasdaq daily threshold list text files."""
        records: list[ThresholdRecord] = []
        today = date.today()

        for offset in range(lookback_days):
            check_date = today - timedelta(days=offset)
            # Skip weekends
            if check_date.weekday() >= 5:
                continue

            date_str = check_date.strftime("%Y%m%d")
            url = _NASDAQ_URL_TEMPLATE.format(date=date_str)

            try:
                self._rate_limit()
                resp = self._session.get(url, timeout=10)
                if resp.status_code == 200 and "text/plain" in resp.headers.get("Content-Type", ""):
                    for line in resp.text.splitlines():
                        parts = line.split("|")
                        if len(parts) >= 2 and parts[0].strip() == symbol:
                            records.append(ThresholdRecord(
                                date=check_date.isoformat(),
                                symbol=symbol,
                                market="NASDAQ",
                                threshold_shares=0,
                                consecutive_days=0,
                            ))
                            break
            except requests.RequestException:
                continue

        if records:
            logger.info(
                "Nasdaq Reg SHO: %d threshold dates for %s",
                len(records), symbol,
            )
        return records

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _symbol_dir(self, symbol: str) -> Path:
        """Return per-symbol data directory, creating it if needed."""
        d = self._data_dir / symbol.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_cache(self, symbol: str) -> list[ThresholdRecord] | None:
        sym_dir = self._symbol_dir(symbol)
        cache_file = sym_dir / "regsho_threshold.json"

        # Legacy migration from old cache paths
        if not cache_file.exists():
            for legacy_path in (
                self._data_dir / "cache" / "regsho" / symbol.upper() / "threshold.json",
                self._data_dir / "cache" / "regsho" / f"{symbol}_threshold.json",
            ):
                if legacy_path.exists():
                    legacy_path.rename(cache_file)
                    logger.info("Migrated %s -> %s", legacy_path, cache_file)
                    break

        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            records = [ThresholdRecord(**r) for r in data]
            records.sort(key=lambda r: r.date)
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to load Reg SHO cache for %s: %s", symbol, exc,
            )
            return None

    def _save_cache(
        self, symbol: str, records: list[ThresholdRecord]
    ) -> None:
        cache_file = self._symbol_dir(symbol) / "regsho_threshold.json"
        try:
            cache_file.write_text(
                json.dumps(
                    [asdict(r) for r in records], indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save Reg SHO cache for %s: %s", symbol, exc,
            )

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()


def _normalize_date(raw: str) -> str:
    """Normalize date to YYYY-MM-DD format."""
    if not raw:
        return ""
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return raw
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3:
            try:
                month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return ""
