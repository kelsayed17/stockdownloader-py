"""Fetches daily short sale volume from FINRA Reg SHO reports.

Unlike the bi-monthly consolidated short interest data, FINRA also
publishes **daily** short sale volume — the number of shares sold short
each day across all trade reporting facilities.  This provides a much
more granular view of short-selling activity.

Data endpoint:
    ``https://api.finra.org/data/group/otcMarket/name/regShoDaily``

The ratio of short volume to total volume (short volume ratio, SVR)
is a useful indicator:
- SVR > 50% is common for liquid stocks (not necessarily bearish)
- SVR > 60% sustained for days may indicate increased short pressure
- SVR changes vs rolling average are more meaningful than absolute levels

Usage::

    client = FinraShortVolumeClient()
    records = client.fetch_short_volume("GME")
    # -> list of ShortVolumeRecord sorted by date ascending
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import requests

from stockdownloader.data.base_client import BaseDataClient

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RATE_LIMIT_DELAY = 0.5

_FINRA_URL = (
    "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
)
_FINRA_TOKEN_URL = (
    "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
    "?grant_type=client_credentials"
)


@dataclass(frozen=True, slots=True)
class ShortVolumeRecord:
    """Daily short sale volume record from FINRA."""

    date: str              # YYYY-MM-DD
    symbol: str
    short_volume: int      # Shares sold short that day
    total_volume: int      # Total shares traded
    short_exempt_volume: int  # Short exempt volume
    short_volume_ratio: float  # short_volume / total_volume

    def __post_init__(self) -> None:
        if not self.date:
            raise ValueError("date must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")


class FinraShortVolumeClient(BaseDataClient):
    """Fetches daily Reg SHO short sale volume from FINRA API.

    Reuses the same FINRA OAuth2 credentials as the short interest
    and dark pool clients.

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
        cache_dir: str = "data/cache/short_volume",
    ) -> None:
        super().__init__(
            rate_limit_delay=_RATE_LIMIT_DELAY,
            max_retries=_MAX_RETRIES,
            cache_dir=cache_dir,
            default_headers={
                "User-Agent": "StockDownloader admin@example.com",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        self._client_id = client_id or os.environ.get("FINRA_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get(
            "FINRA_CLIENT_SECRET", ""
        )
        self._access_token: str | None = None

    # ------------------------------------------------------------------
    # OAuth2
    # ------------------------------------------------------------------

    def _authenticate(self) -> bool:
        if not self._client_id or not self._client_secret:
            logger.info(
                "FINRA credentials not configured for short volume — "
                "set FINRA_CLIENT_ID and FINRA_CLIENT_SECRET"
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
                    logger.info(
                        "FINRA OAuth2 authentication successful (short volume)"
                    )
                    return True
            logger.warning(
                "FINRA OAuth2 failed for short volume: %d",
                resp.status_code,
            )
        except Exception as exc:
            logger.warning("FINRA OAuth2 error: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_short_volume(
        self,
        symbol: str,
        lookback_days: int = 365,
    ) -> list[ShortVolumeRecord]:
        """Fetch daily short sale volume records for *symbol*.

        Tries FINRA API first.  Falls back to cached data on failure.

        Returns records sorted by date ascending.
        """
        symbol_upper = symbol.upper()

        if self._access_token is None:
            self._authenticate()

        if self._access_token:
            raw = self._query_api(symbol_upper, lookback_days)
            if raw:
                records = self._raw_to_records(raw, symbol_upper)
                if records:
                    # Merge with existing cache
                    cached = self._load_cache(symbol_upper) or []
                    merged = self._merge_records(cached, records)
                    self._save_cache(symbol_upper, merged)
                    return merged

        # Fall back to cache
        cached = self._load_cache(symbol_upper)
        if cached:
            logger.info(
                "Using cached short volume data for %s (%d records)",
                symbol_upper, len(cached),
            )
            return cached

        logger.warning("No short volume data available for %s", symbol_upper)
        return []

    def get_recent_stats(
        self, symbol: str, days: int = 20
    ) -> dict:
        """Compute recent short volume statistics.

        Returns a dict with:
        - ``avg_svr``: Average short volume ratio over the period
        - ``latest_svr``: Most recent day's SVR
        - ``svr_trend``: Direction of SVR change
        - ``days_above_50pct``: Days where SVR exceeded 50%
        - ``max_svr``: Highest SVR in the period
        """
        records = self.fetch_short_volume(symbol, lookback_days=days + 30)
        if not records:
            return {
                "avg_svr": 0.0, "latest_svr": 0.0,
                "svr_trend": "unknown", "days_above_50pct": 0,
                "max_svr": 0.0,
            }

        recent = records[-days:]
        svrs = [r.short_volume_ratio for r in recent]

        avg_svr = sum(svrs) / len(svrs) if svrs else 0.0
        latest_svr = svrs[-1] if svrs else 0.0
        max_svr = max(svrs) if svrs else 0.0
        days_above_50 = sum(1 for s in svrs if s > 0.5)

        # Trend: compare first half avg to second half avg
        mid = len(svrs) // 2
        first_half = sum(svrs[:mid]) / max(mid, 1)
        second_half = sum(svrs[mid:]) / max(len(svrs) - mid, 1)
        if second_half > first_half * 1.05:
            trend = "increasing"
        elif second_half < first_half * 0.95:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "avg_svr": round(avg_svr, 4),
            "latest_svr": round(latest_svr, 4),
            "svr_trend": trend,
            "days_above_50pct": days_above_50,
            "max_svr": round(max_svr, 4),
            "total_days": len(recent),
        }

    # ------------------------------------------------------------------
    # API query
    # ------------------------------------------------------------------

    def _query_api(
        self, symbol: str, lookback_days: int
    ) -> list[dict] | None:
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
                        "fieldName": "tradeReportDate",
                        "startDate": start_date.isoformat(),
                        "endDate": end_date.isoformat(),
                    },
                ],
                "limit": page_size,
                "offset": offset,
                "sortFields": ["-tradeReportDate"],
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
                        "FINRA short volume returned %d: %s",
                        resp.status_code, resp.text[:200],
                    )
            except (requests.RequestException, json.JSONDecodeError) as exc:
                failures += 1
                logger.debug("FINRA short volume request failed: %s", exc)

        if all_rows:
            logger.info(
                "FINRA short volume: fetched %d records for %s",
                len(all_rows), symbol,
            )
            return all_rows
        return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_to_records(
        raw: list[dict], symbol: str
    ) -> list[ShortVolumeRecord]:
        records: list[ShortVolumeRecord] = []

        for row in raw:
            try:
                raw_date = (
                    row.get("tradeReportDate", "")
                    or row.get("date", "")
                )
                dt = _normalize_date(raw_date)
                if not dt:
                    continue

                short_vol = int(row.get("shortParQuantity", 0))
                short_exempt = int(row.get("shortExemptParQuantity", 0))
                total_vol = int(row.get("totalParQuantity", 0))

                # Compute SVR
                svr = (short_vol / total_vol) if total_vol > 0 else 0.0

                records.append(ShortVolumeRecord(
                    date=dt,
                    symbol=symbol,
                    short_volume=short_vol,
                    total_volume=total_vol,
                    short_exempt_volume=short_exempt,
                    short_volume_ratio=round(svr, 6),
                ))
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed short volume row: %s", exc)
                continue

        records.sort(key=lambda r: r.date)
        return records

    # ------------------------------------------------------------------
    # Merge & Caching
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_records(
        existing: list[ShortVolumeRecord],
        new: list[ShortVolumeRecord],
    ) -> list[ShortVolumeRecord]:
        """Merge existing and new records, preferring new data."""
        by_date: dict[str, ShortVolumeRecord] = {}
        for r in existing:
            by_date[r.date] = r
        for r in new:
            by_date[r.date] = r  # New overwrites
        merged = list(by_date.values())
        merged.sort(key=lambda r: r.date)
        return merged

    def _symbol_cache_dir(self, symbol: str) -> Path:
        """Return per-symbol cache subdirectory, creating it if needed."""
        d = self._cache_dir / symbol.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_cache(self, symbol: str) -> list[ShortVolumeRecord] | None:
        sym_dir = self._symbol_cache_dir(symbol)
        cache_file = sym_dir / "sv.json"

        # Legacy migration
        if not cache_file.exists():
            legacy = self._cache_dir / f"{symbol}_sv.json"
            if legacy.exists():
                legacy.rename(cache_file)
                logger.info("Migrated %s → %s", legacy, cache_file)

        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            records = [ShortVolumeRecord(**r) for r in data]
            records.sort(key=lambda r: r.date)
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to load short volume cache for %s: %s", symbol, exc,
            )
            return None

    def _save_cache(
        self, symbol: str, records: list[ShortVolumeRecord]
    ) -> None:
        cache_file = self._symbol_cache_dir(symbol) / "sv.json"
        try:
            cache_file.write_text(
                json.dumps([asdict(r) for r in records], indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save short volume cache for %s: %s", symbol, exc,
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
    if not raw:
        return ""
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return raw
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3:
            try:
                m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
                return f"{y:04d}-{m:02d}-{d:02d}"
            except ValueError:
                pass
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return ""
