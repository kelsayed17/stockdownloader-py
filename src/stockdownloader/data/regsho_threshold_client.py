"""Fetches Reg SHO threshold list data from FINRA and major exchanges.

The **Regulation SHO threshold list** contains securities where aggregate
failures to deliver (FTDs) have reached or exceeded 10,000 shares and
equal at least 0.5% of the issuer's total shares outstanding for 5
consecutive settlement days.

Being on the threshold list signals extreme short-selling pressure and
may trigger mandatory close-out requirements (forced buy-ins).

Sources:
- **FINRA**: ``https://api.finra.org/data/group/otcMarket/name/regShoThresholdList``
- **NYSE**: ``https://www.nyse.com/regulation/threshold-securities`` (web page)
- **Nasdaq**: ``https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{MMDDYYYY}.txt``

Usage::

    client = RegShoThresholdClient()
    records = client.fetch_threshold_status("GME")
    # -> list of dates when GME was on the threshold list
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RATE_LIMIT_DELAY = 0.5

_FINRA_URL = (
    "https://api.finra.org/data/group/otcMarket/name/regShoThresholdList"
)
_FINRA_TOKEN_URL = (
    "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
    "?grant_type=client_credentials"
)

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
    """Fetches Reg SHO threshold list data.

    Primary source: FINRA API (requires OAuth2 credentials).
    Fallback: Nasdaq daily text files (no auth needed).

    Parameters
    ----------
    client_id:
        FINRA API client ID.  Falls back to ``FINRA_CLIENT_ID`` env var.
    client_secret:
        FINRA API client secret.  Falls back to ``FINRA_CLIENT_SECRET``.
    cache_dir:
        Directory for JSON cache files.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_dir: str = "data/cache/regsho",
    ) -> None:
        self._client_id = client_id or os.environ.get("FINRA_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get(
            "FINRA_CLIENT_SECRET", ""
        )
        self._access_token: str | None = None
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "StockDownloader admin@example.com",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        self._last_request_time: float = 0.0
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # OAuth2 (reuses FINRA pattern from dark pool / SI clients)
    # ------------------------------------------------------------------

    def _authenticate(self) -> bool:
        if not self._client_id or not self._client_secret:
            logger.info(
                "FINRA credentials not configured for Reg SHO — "
                "will try Nasdaq fallback"
            )
            return False

        credentials = f"{self._client_id}:{self._client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        try:
            resp = requests.post(
                _FINRA_TOKEN_URL,
                headers={"Authorization": f"Basic {encoded}"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._access_token = data.get("access_token")
                if self._access_token:
                    self._session.headers["Authorization"] = (
                        f"Bearer {self._access_token}"
                    )
                    logger.info("FINRA OAuth2 authentication successful (Reg SHO)")
                    return True
            logger.warning(
                "FINRA OAuth2 failed for Reg SHO: %d", resp.status_code,
            )
        except Exception as exc:
            logger.warning("FINRA OAuth2 error: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_threshold_status(
        self, symbol: str, lookback_days: int = 365
    ) -> list[ThresholdRecord]:
        """Fetch all dates where *symbol* appeared on the Reg SHO threshold list.

        Tries FINRA API first, falls back to Nasdaq text files,
        then to cache.

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

        # Try FINRA API
        if self._access_token is None:
            self._authenticate()

        if self._access_token:
            finra_records = self._query_finra(symbol_upper, lookback_days)
            if finra_records:
                records.extend(finra_records)

        # Also try Nasdaq (no auth needed) for recent data
        nasdaq_records = self._query_nasdaq(symbol_upper, lookback_days=30)
        if nasdaq_records:
            # Merge without duplicates
            existing_dates = {r.date for r in records}
            for nr in nasdaq_records:
                if nr.date not in existing_dates:
                    records.append(nr)

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
    # FINRA API query
    # ------------------------------------------------------------------

    def _query_finra(
        self, symbol: str, lookback_days: int
    ) -> list[ThresholdRecord]:
        """Query FINRA Reg SHO threshold list API."""
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        all_rows: list[dict] = []
        offset = 0
        page_size = 5000
        failures = 0

        while failures < _MAX_RETRIES:
            payload = {
                "domainFilters": [
                    {
                        "fieldName": "symbolCode",
                        "values": [symbol],
                    },
                ],
                "dateRangeFilters": [
                    {
                        "fieldName": "thresholdListPublishDate",
                        "startDate": start_date.isoformat(),
                        "endDate": end_date.isoformat(),
                    },
                ],
                "limit": page_size,
                "offset": offset,
            }

            try:
                self._rate_limit()
                resp = self._session.post(
                    _FINRA_URL, json=payload, timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if not data:
                        break
                    all_rows.extend(data)
                    failures = 0
                    if len(data) < page_size:
                        break
                    offset += page_size
                elif resp.status_code == 404:
                    break
                else:
                    failures += 1
                    logger.debug(
                        "FINRA Reg SHO returned %d: %s",
                        resp.status_code, resp.text[:200],
                    )
            except (requests.RequestException, json.JSONDecodeError) as exc:
                failures += 1
                logger.debug("FINRA Reg SHO request failed: %s", exc)

        records: list[ThresholdRecord] = []
        for row in all_rows:
            try:
                raw_date = row.get("thresholdListPublishDate", "")
                dt = _normalize_date(raw_date)
                if dt:
                    records.append(ThresholdRecord(
                        date=dt,
                        symbol=symbol,
                        market="FINRA",
                        threshold_shares=int(
                            row.get("thresholdListShareQuantity", 0)
                        ),
                        consecutive_days=0,
                    ))
            except (ValueError, TypeError):
                continue

        logger.info(
            "FINRA Reg SHO: %d threshold records for %s",
            len(records), symbol,
        )
        return records

    # ------------------------------------------------------------------
    # Nasdaq text file fallback
    # ------------------------------------------------------------------

    def _query_nasdaq(
        self, symbol: str, lookback_days: int = 30
    ) -> list[ThresholdRecord]:
        """Check Nasdaq daily threshold list text files."""
        records: list[ThresholdRecord] = []
        today = date.today()

        for offset in range(lookback_days):
            check_date = today - timedelta(days=offset)
            # Skip weekends
            if check_date.weekday() >= 5:
                continue

            date_str = check_date.strftime("%m%d%Y")
            url = _NASDAQ_URL_TEMPLATE.format(date=date_str)

            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
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

            # Be gentle with Nasdaq servers
            time.sleep(0.2)

        if records:
            logger.info(
                "Nasdaq Reg SHO: %d threshold dates for %s",
                len(records), symbol,
            )
        return records

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cache(self, symbol: str) -> list[ThresholdRecord] | None:
        cache_file = self._cache_dir / f"{symbol}_threshold.json"
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
        cache_file = self._cache_dir / f"{symbol}_threshold.json"
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
