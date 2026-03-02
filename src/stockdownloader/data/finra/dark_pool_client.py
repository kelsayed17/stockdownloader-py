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

import json
import logging

import requests

from stockdownloader.data.finra.base_client import (
    FinraBaseClient,
    _MAX_RETRIES,
    normalize_finra_date,
)
from stockdownloader.core.models.regulatory import DarkPoolRecord

logger = logging.getLogger(__name__)

# FINRA OTC transparency dataset.  Both ATS (dark pool) and non-ATS (OTC)
# data live in the same ``weeklySummary`` dataset, distinguished by
# ``summaryTypeCode``:
#   - ``ATS_W_SMBL``  — ATS aggregate per symbol (dark pools)
#   - ``OTC_W_SMBL``  — non-ATS aggregate per symbol (OTC)
_FINRA_URL = (
    "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
)


class FinraDarkPoolClient(FinraBaseClient):
    """Fetches OTC/ATS (dark pool) volume data from FINRA.

    Parameters
    ----------
    client_id:
        FINRA API client ID.  Falls back to ``FINRA_CLIENT_ID`` env var.
    client_secret:
        FINRA API client secret.  Falls back to ``FINRA_CLIENT_SECRET`` env var.
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/dark_pool.csv``.
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
                week_ending = normalize_finra_date(raw_date)
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

        Reads from CSV first.  Falls back to legacy JSON files and
        migrates them to CSV on success.

        Returns ``None`` if no cache exists.
        """
        # Primary: CSV
        records = self._load_csv(symbol, "dark_pool.csv", DarkPoolRecord)
        if records is not None:
            records.sort(key=lambda r: r.week_ending)
            return records

        # Fallback: JSON (legacy migration)
        sym_dir = self._symbol_dir(symbol)
        json_path = sym_dir / "dark_pool.json"

        if not json_path.exists():
            for legacy_path in (
                self._data_dir / "cache" / "dark_pool" / symbol.upper() / "dp.json",
                self._data_dir / "cache" / "dark_pool" / f"{symbol}_dp.json",
            ):
                if legacy_path.exists():
                    legacy_path.rename(json_path)
                    logger.info("Migrated %s → %s", legacy_path, json_path)
                    break

        if not json_path.exists():
            return None

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
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
            self._save_cache(symbol, records)
            logger.info("Migrated %s to CSV", json_path)
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
        """Persist dark pool records to CSV cache."""
        self._save_csv(symbol, "dark_pool.csv", records)
