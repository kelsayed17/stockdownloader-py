"""Fetches daily short sale volume from FINRA Reg SHO reports.

Unlike the bi-monthly consolidated short interest data, FINRA also
publishes **daily** short sale volume — the number of shares sold short
each day across all trade reporting facilities.  This provides a much
more granular view of short-selling activity.

Data sources (in priority order):

1. **FINRA CDN text files** (2019–present, no auth needed)::

       https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt

   The consolidated (CNMS) file contains one row per symbol per day with
   pipe-delimited fields: ``Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market``

2. **FINRA API** (recent data only, requires OAuth2)::

       https://api.finra.org/data/group/otcMarket/name/regShoDaily

   The API only has data from ~March 2025 onward and requires per-date
   queries (``tradeReportDate`` is a partition key).

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

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta

import requests

from stockdownloader.data.finra_base_client import (
    FinraBaseClient,
    _MAX_RETRIES,
    normalize_finra_date,
)

logger = logging.getLogger(__name__)

_FINRA_URL = (
    "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
)
# FINRA CDN text files — consolidated short volume (no auth, 2019–present)
_FINRA_CDN_TEMPLATE = (
    "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
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


class FinraShortVolumeClient(FinraBaseClient):
    """Fetches daily Reg SHO short sale volume from FINRA API.

    Reuses the same FINRA OAuth2 credentials as the short interest
    and dark pool clients.

    Parameters
    ----------
    client_id:
        FINRA API client ID.  Falls back to ``FINRA_CLIENT_ID`` env var.
    client_secret:
        FINRA API client secret.  Falls back to ``FINRA_CLIENT_SECRET``.
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/short_volume.json``.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        data_dir: str = "data",
    ) -> None:
        super().__init__(client_id, client_secret, data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_short_volume(
        self,
        symbol: str,
        lookback_days: int = 2555,
    ) -> list[ShortVolumeRecord]:
        """Fetch daily short sale volume records for *symbol*.

        Tries FINRA CDN text files first (2019–present, no auth needed),
        then the FINRA API for the most recent data, merges with cache.

        Returns records sorted by date ascending.
        """
        symbol_upper = symbol.upper()

        # 1. Try CDN text files (primary source, 2019–present)
        cdn_records = self._query_text_files(symbol_upper, lookback_days)

        # 2. Try API for recent data (may have data not yet in CDN)
        api_records: list[ShortVolumeRecord] = []
        if self._access_token is None:
            self._authenticate()
        if self._access_token:
            raw = self._query_api(symbol_upper, lookback_days=60)
            if raw:
                api_records = self._raw_to_records(raw, symbol_upper)

        # 3. Merge all sources: cache + CDN + API (API newest wins)
        cached = self._load_cache(symbol_upper) or []
        merged = self._merge_records(cached, cdn_records)
        merged = self._merge_records(merged, api_records)

        if merged:
            self._save_cache(symbol_upper, merged)
            return merged

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
    # CDN text file query (primary source, 2019–present)
    # ------------------------------------------------------------------

    def _query_text_files(
        self, symbol: str, lookback_days: int
    ) -> list[ShortVolumeRecord]:
        """Fetch short volume from FINRA CDN text files.

        The consolidated (CNMS) file contains one row per symbol per day:
        ``Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market``

        Returns parsed :class:`ShortVolumeRecord` list sorted by date.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        records: list[ShortVolumeRecord] = []
        consecutive_failures = 0
        current = start_date
        cdn_retries = 10  # more tolerant for CDN (long iteration)

        while current <= end_date:
            if current.weekday() >= 5:  # skip weekends
                current += timedelta(days=1)
                continue

            if consecutive_failures >= cdn_retries:
                logger.warning(
                    "Too many consecutive failures fetching FINRA CDN "
                    "short volume files, stopping at %s",
                    current.isoformat(),
                )
                break

            date_str = current.strftime("%Y%m%d")
            url = _FINRA_CDN_TEMPLATE.format(date=date_str)

            try:
                # CDN is static files — use shorter delay than API
                time.sleep(0.15)
                resp = self._session.get(url, timeout=15)
                if resp.status_code == 200 and "text/plain" in resp.headers.get(
                    "Content-Type", ""
                ):
                    consecutive_failures = 0
                    for line in resp.text.splitlines():
                        parts = line.split("|")
                        if len(parts) >= 5 and parts[1] == symbol:
                            try:
                                short_vol = int(parts[2])
                                exempt_vol = int(parts[3])
                                total_vol = int(parts[4])
                                svr = (
                                    short_vol / total_vol
                                    if total_vol > 0
                                    else 0.0
                                )
                                records.append(ShortVolumeRecord(
                                    date=current.isoformat(),
                                    symbol=symbol,
                                    short_volume=short_vol,
                                    total_volume=total_vol,
                                    short_exempt_volume=exempt_vol,
                                    short_volume_ratio=round(svr, 6),
                                ))
                            except (ValueError, IndexError):
                                pass
                            break
                elif resp.status_code in (404, 204):
                    consecutive_failures = 0  # normal for holidays
                elif resp.status_code == 403:
                    consecutive_failures += 1
                    logger.debug(
                        "FINRA CDN returned 403 for %s",
                        current.isoformat(),
                    )
                else:
                    consecutive_failures += 1
            except requests.RequestException:
                consecutive_failures += 1

            current += timedelta(days=1)

        if records:
            logger.info(
                "FINRA CDN short volume: %d records for %s (%s to %s)",
                len(records), symbol,
                records[0].date, records[-1].date,
            )
        return records

    # ------------------------------------------------------------------
    # API query (recent data only, requires OAuth2)
    # ------------------------------------------------------------------

    def _query_api(
        self, symbol: str, lookback_days: int
    ) -> list[dict] | None:
        """Query FINRA regShoDaily API for *symbol*.

        The ``regShoDaily`` dataset uses ``tradeReportDate`` as a
        partition key, which requires an EQUAL CompareFilter — date
        range filters do not work.  We therefore iterate over each
        business day in the lookback window, issuing one request per
        date.

        Returns raw JSON response rows, or ``None`` on failure.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        # Generate business days (skip weekends)
        query_dates: list[date] = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Mon-Fri
                query_dates.append(current)
            current += timedelta(days=1)

        all_rows: list[dict] = []
        consecutive_failures = 0

        for query_date in query_dates:
            if consecutive_failures >= _MAX_RETRIES:
                logger.warning(
                    "Too many consecutive failures querying FINRA "
                    "short volume for %s, stopping",
                    symbol,
                )
                break

            payload = {
                "compareFilters": [
                    {
                        "fieldName": "tradeReportDate",
                        "fieldValue": query_date.isoformat(),
                        "compareType": "EQUAL",
                    },
                ],
                "domainFilters": [
                    {
                        "fieldName":
                            "securitiesInformationProcessorSymbolIdentifier",
                        "values": [symbol],
                    },
                ],
                "limit": 50,
            }

            try:
                self._rate_limit()
                resp = self._session.post(
                    _FINRA_URL, json=payload, timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        all_rows.extend(data)
                    consecutive_failures = 0
                elif resp.status_code == 204:
                    # No data for this date — normal
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.debug(
                        "FINRA short volume returned %d for %s on %s: %s",
                        resp.status_code, symbol,
                        query_date.isoformat(), resp.text[:200],
                    )
            except (requests.RequestException, json.JSONDecodeError) as exc:
                consecutive_failures += 1
                logger.debug(
                    "FINRA short volume request failed for %s on %s: %s",
                    symbol, query_date.isoformat(), exc,
                )

        if all_rows:
            logger.info(
                "FINRA short volume: fetched %d records for %s "
                "across %d business days",
                len(all_rows), symbol, len(query_dates),
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
                dt = normalize_finra_date(raw_date)
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

    def _load_cache(self, symbol: str) -> list[ShortVolumeRecord] | None:
        # Primary: CSV
        records = self._load_csv(symbol, "short_volume.csv", ShortVolumeRecord)
        if records is not None:
            records.sort(key=lambda r: r.date)
            return records

        # Fallback: JSON (legacy migration)
        sym_dir = self._symbol_dir(symbol)
        json_path = sym_dir / "short_volume.json"

        # Also check legacy paths
        if not json_path.exists():
            for legacy_path in (
                self._data_dir / "cache" / "short_volume" / symbol.upper() / "sv.json",
                self._data_dir / "cache" / "short_volume" / f"{symbol}_sv.json",
            ):
                if legacy_path.exists():
                    legacy_path.rename(json_path)
                    logger.info("Migrated %s → %s", legacy_path, json_path)
                    break

        if not json_path.exists():
            return None
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            records = [ShortVolumeRecord(**r) for r in data]
            records.sort(key=lambda r: r.date)
            # Migrate: write CSV for future loads
            self._save_cache(symbol, records)
            logger.info("Migrated %s to CSV", json_path)
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to load short volume cache for %s: %s", symbol, exc,
            )
            return None

    def _save_cache(
        self, symbol: str, records: list[ShortVolumeRecord]
    ) -> None:
        self._save_csv(symbol, "short_volume.csv", records)
