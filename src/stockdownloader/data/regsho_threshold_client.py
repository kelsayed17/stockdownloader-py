"""Fetches Reg SHO threshold list data from Nasdaq.

The **Regulation SHO threshold list** contains securities where aggregate
failures to deliver (FTDs) have reached or exceeded 10,000 shares and
equal at least 0.5% of the issuer's total shares outstanding for 5
consecutive settlement days.

Being on the threshold list signals extreme short-selling pressure and
may trigger mandatory close-out requirements (forced buy-ins).

Sources:
- **Nasdaq**: ``https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{MMDDYYYY}.txt``

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
_RATE_LIMIT_DELAY = 0.2

# Nasdaq publishes daily threshold lists
_NASDAQ_URL_TEMPLATE = (
    "https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{date}.txt"
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
    """Fetches Reg SHO threshold list data from Nasdaq text files.

    The FINRA ``regShoThresholdList`` dataset is not available via the
    FINRA API, so this client relies solely on Nasdaq daily text files.

    Parameters
    ----------
    client_id:
        Reserved for future use (FINRA credentials).  Not required.
    client_secret:
        Reserved for future use (FINRA credentials).  Not required.
    cache_dir:
        Directory for JSON cache files.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_dir: str = "data/cache/regsho",
    ) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "StockDownloader admin@example.com",
            "Accept": "text/plain",
        })
        self._last_request_time: float = 0.0
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_threshold_status(
        self, symbol: str, lookback_days: int = 1825
    ) -> list[ThresholdRecord]:
        """Fetch all dates where *symbol* appeared on the Reg SHO threshold list.

        Queries Nasdaq daily text files, then falls back to cache.

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

        # Query Nasdaq text files
        records = self._query_nasdaq(symbol_upper, lookback_days=lookback_days)

        if records:
            records.sort(key=lambda r: r.date)
            self._save_cache(symbol_upper, records)
            return records

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

    def _symbol_cache_dir(self, symbol: str) -> Path:
        """Return per-symbol cache subdirectory, creating it if needed."""
        d = self._cache_dir / symbol.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_cache(self, symbol: str) -> list[ThresholdRecord] | None:
        sym_dir = self._symbol_cache_dir(symbol)
        cache_file = sym_dir / "threshold.json"

        # Legacy migration
        if not cache_file.exists():
            legacy = self._cache_dir / f"{symbol}_threshold.json"
            if legacy.exists():
                legacy.rename(cache_file)
                logger.info("Migrated %s -> %s", legacy, cache_file)

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
        cache_file = self._symbol_cache_dir(symbol) / "threshold.json"
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
