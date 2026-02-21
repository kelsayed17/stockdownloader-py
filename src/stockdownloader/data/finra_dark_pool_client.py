"""Fetches OTC/ATS (dark pool) volume data from FINRA.

Uses the FINRA data API at
``https://api.finra.org/data/group/otcMarket/name/weeklySummary``
to retrieve weekly OTC and ATS volume data.  Falls back gracefully
to cached results when the API is unavailable.

The FINRA OTC transparency data is available under two dataset names:

* **otcMarket / weeklySummary** -- OTC (non-ATS) weekly summary.
* **otcMarket / ukaTradingActivity** -- ATS (dark pool) weekly data.

Both datasets use ``symbolCode`` as the symbol field and support
``weekStartDate`` range filters.  This client queries both and
merges the results so the caller gets a combined OTC + ATS view.

FINRA API requires OAuth2 authentication.  Provide ``client_id`` and
``client_secret`` (obtained from the FINRA API Console) or set the
``FINRA_CLIENT_ID`` and ``FINRA_CLIENT_SECRET`` environment variables.

Usage::

    client = FinraDarkPoolClient(
        client_id="your_id", client_secret="your_secret",
    )
    records = client.fetch_dark_pool_volume("GME")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

import requests

from stockdownloader.data.base_client import BaseDataClient
from stockdownloader.model.regulatory_records import DarkPoolRecord

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RATE_LIMIT_DELAY = 0.5  # conservative rate for FINRA API

# FINRA OTC transparency dataset.  Both ATS (dark pool) and non-ATS (OTC)
# data live in the same ``weeklySummary`` dataset, distinguished by
# ``summaryTypeCode``:
#   - ``ATS_W_SMBL``  — ATS aggregate per symbol (dark pools)
#   - ``OTC_W_SMBL``  — non-ATS aggregate per symbol (OTC)
_FINRA_URL = (
    "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
)
_FINRA_TOKEN_URL = (
    "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
    "?grant_type=client_credentials"
)


class FinraDarkPoolClient(BaseDataClient):
    """Fetches OTC/ATS (dark pool) volume data from FINRA.

    Parameters
    ----------
    client_id:
        FINRA API client ID.  Falls back to ``FINRA_CLIENT_ID`` env var.
    client_secret:
        FINRA API client secret.  Falls back to ``FINRA_CLIENT_SECRET`` env var.
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/dark_pool.json``.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        data_dir: str = "data",
    ) -> None:
        super().__init__(
            rate_limit_delay=_RATE_LIMIT_DELAY,
            max_retries=_MAX_RETRIES,
            data_dir=data_dir,
            default_headers={
                "User-Agent": "StockDownloader admin@example.com",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        self._client_id = client_id or os.environ.get("FINRA_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("FINRA_CLIENT_SECRET", "")
        self._access_token: str | None = None

    # ------------------------------------------------------------------
    # OAuth2 authentication
    # ------------------------------------------------------------------

    def _authenticate(self) -> bool:
        """Obtain an OAuth2 access token from FINRA FIP.

        Returns ``True`` if authentication succeeds.
        """
        if not self._client_id or not self._client_secret:
            logger.info(
                "FINRA credentials not configured — "
                "set FINRA_CLIENT_ID and FINRA_CLIENT_SECRET env vars"
            )
            return False

        credentials = f"{self._client_id}:{self._client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        try:
            resp = requests.post(
                _FINRA_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {encoded}",
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._access_token = data.get("access_token")
                if self._access_token:
                    self._session.headers["Authorization"] = (
                        f"Bearer {self._access_token}"
                    )
                    logger.info("FINRA OAuth2 authentication successful (dark pool)")
                    return True
            logger.warning(
                "FINRA OAuth2 failed: %d %s",
                resp.status_code, resp.text[:200],
            )
        except Exception as exc:
            logger.warning("FINRA OAuth2 error: %s", exc)

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_dark_pool_volume(
        self,
        symbol: str,
    ) -> list[DarkPoolRecord]:
        """Fetch OTC/ATS volume records for *symbol*.

        Attempts the FINRA API first.  If the API is unavailable or
        returns errors, falls back to cached data.

        Returns records sorted by ``week_ending`` ascending.
        Returns an empty list on complete failure.
        """
        symbol_upper = symbol.upper()

        # Authenticate before querying
        if self._access_token is None:
            self._authenticate()

        # Try API first
        raw = self._query_api(symbol_upper)
        if raw is not None and len(raw) > 0:
            records = self._raw_to_records(raw, symbol_upper)
            if records:
                self._save_cache(symbol_upper, records)
                return records

        # Fall back to cache
        cached = self._load_cache(symbol_upper)
        if cached is not None:
            logger.info(
                "Using cached dark pool data for %s (%d records)",
                symbol_upper, len(cached),
            )
            return cached

        logger.warning(
            "No dark pool data available for %s", symbol_upper,
        )
        return []

    # ------------------------------------------------------------------
    # API query
    # ------------------------------------------------------------------

    def _query_api(self, symbol: str) -> list[dict] | None:
        """Query FINRA OTC/ATS weekly data for *symbol*.

        Fetches both ATS (dark pool) and non-ATS (OTC) aggregate data
        from the single ``weeklySummary`` endpoint.  Rows are tagged
        with ``_source`` ("ats" or "otc") based on ``summaryTypeCode``.

        Returns raw JSON rows, or ``None`` on failure.
        """
        all_rows: list[dict] = []
        offset = 0
        page_size = 5000
        failures = 0

        while failures < _MAX_RETRIES:
            payload = {
                "domainFilters": [
                    {
                        "fieldName": "issueSymbolIdentifier",
                        "values": [symbol],
                    },
                    {
                        "fieldName": "summaryTypeCode",
                        "values": ["ATS_W_SMBL", "OTC_W_SMBL"],
                    },
                ],
                "limit": page_size,
                "offset": offset,
            }

            try:
                self._rate_limit()
                resp = self._session.post(
                    _FINRA_URL,
                    json=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if not data:
                        break  # No more data

                    # Tag each row with source based on summaryTypeCode
                    for row in data:
                        stype = row.get("summaryTypeCode", "")
                        if "ATS" in stype:
                            row["_source"] = "ats"
                        else:
                            row["_source"] = "otc"

                    all_rows.extend(data)
                    failures = 0
                    if len(data) < page_size:
                        break  # Last page
                    offset += page_size
                elif resp.status_code == 404:
                    logger.debug(
                        "FINRA weeklySummary returned 404 for %s", symbol,
                    )
                    break
                else:
                    failures += 1
                    logger.debug(
                        "FINRA weeklySummary returned %d for %s: %s",
                        resp.status_code, symbol, resp.text[:200],
                    )
            except (requests.RequestException, json.JSONDecodeError, OSError) as exc:
                failures += 1
                logger.debug(
                    "FINRA request failed for %s: %s", symbol, exc,
                )

        if failures >= _MAX_RETRIES:
            logger.warning(
                "Too many consecutive failures querying FINRA for %s",
                symbol,
            )

        if all_rows:
            logger.info(
                "FINRA OTC/ATS: fetched %d records for %s",
                len(all_rows), symbol,
            )
            return all_rows

        return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_to_records(
        raw: list[dict],
        symbol: str,
    ) -> list[DarkPoolRecord]:
        """Convert raw FINRA JSON rows to :class:`DarkPoolRecord` objects.

        Rows are tagged with ``_source`` ("otc" or "ats") by
        :meth:`_query_api`.  This method groups them by week and
        computes the ATS percentage.
        """
        # Group by week so we can aggregate ATS vs non-ATS
        weeks: dict[str, dict[str, int]] = {}

        for row in raw:
            try:
                # FINRA uses ``weekStartDate`` in the API response.
                # Fall back to ``weekEndDate`` / ``week_ending`` for
                # compatibility with cached data.
                raw_date = (
                    row.get("weekStartDate", "")
                    or row.get("weekEndDate", "")
                    or row.get("week_ending", "")
                )
                week_ending = _normalize_date(raw_date)
                if not week_ending:
                    continue

                volume = int(row.get("totalWeeklyShareQuantity", 0))
                source = row.get("_source", "otc")

                if week_ending not in weeks:
                    weeks[week_ending] = {
                        "total": 0,
                        "ats": 0,
                        "otc": 0,
                    }

                weeks[week_ending]["total"] += volume
                if source == "ats":
                    weeks[week_ending]["ats"] += volume
                else:
                    weeks[week_ending]["otc"] += volume

            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed FINRA OTC row: %s", exc)
                continue

        records: list[DarkPoolRecord] = []
        for week_ending, data in weeks.items():
            total = data["total"]
            ats = data["ats"]
            otc = data["otc"]
            ats_pct = ats / total if total > 0 else 0.0

            try:
                records.append(DarkPoolRecord(
                    week_ending=week_ending,
                    symbol=symbol,
                    total_weekly_volume=total,
                    ats_volume=ats,
                    otc_volume=otc,
                    ats_pct=ats_pct,
                ))
            except ValueError:
                continue

        records.sort(key=lambda r: r.week_ending)
        return records

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cache(self, symbol: str) -> list[DarkPoolRecord] | None:
        """Load cached dark pool records for *symbol*.

        Returns ``None`` if no cache exists.
        """
        sym_dir = self._symbol_dir(symbol)
        cache_file = sym_dir / "dark_pool.json"

        # Legacy migration from old cache paths
        if not cache_file.exists():
            for legacy_path in (
                self._data_dir / "cache" / "dark_pool" / symbol.upper() / "dp.json",
                self._data_dir / "cache" / "dark_pool" / f"{symbol}_dp.json",
            ):
                if legacy_path.exists():
                    legacy_path.rename(cache_file)
                    logger.info("Migrated %s → %s", legacy_path, cache_file)
                    break

        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            records = [
                DarkPoolRecord(
                    week_ending=r["week_ending"],
                    symbol=r["symbol"],
                    total_weekly_volume=r["total_weekly_volume"],
                    ats_volume=r["ats_volume"],
                    otc_volume=r["otc_volume"],
                    ats_pct=r.get("ats_pct", 0.0),
                )
                for r in data
            ]
            records.sort(key=lambda r: r.week_ending)
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to load dark pool cache for %s: %s", symbol, exc,
            )
            return None

    def _save_cache(
        self,
        symbol: str,
        records: list[DarkPoolRecord],
    ) -> None:
        """Persist dark pool records to JSON cache."""
        cache_file = self._symbol_dir(symbol) / "dark_pool.json"
        data = [
            {
                "week_ending": r.week_ending,
                "symbol": r.symbol,
                "total_weekly_volume": r.total_weekly_volume,
                "ats_volume": r.ats_volume,
                "otc_volume": r.otc_volume,
                "ats_pct": r.ats_pct,
            }
            for r in records
        ]
        try:
            cache_file.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save dark pool cache for %s: %s", symbol, exc,
            )

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if needed to maintain a conservative request rate."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _normalize_date(raw: str) -> str:
    """Normalize a date string to ``YYYY-MM-DD`` format.

    Handles ``YYYY-MM-DD``, ``MM/DD/YYYY``, and ``YYYYMMDD`` formats.
    Returns an empty string for unrecognizable dates.
    """
    if not raw:
        return ""

    # Already in ISO format
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return raw

    # Try MM/DD/YYYY
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3:
            try:
                month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass

    # Try YYYYMMDD
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    return ""
