"""Fetches short interest data from FINRA.

Uses the FINRA data API at
``https://api.finra.org/data/group/otcMarket/name/shortInterest``
to retrieve short interest reporting data.  Falls back gracefully
to cached results when the API is unavailable.

FINRA API requires OAuth2 authentication.  Provide ``client_id`` and
``client_secret`` (obtained from the FINRA API Console) or set the
``FINRA_CLIENT_ID`` and ``FINRA_CLIENT_SECRET`` environment variables.

Usage::

    client = FinraShortInterestClient(
        client_id="your_id", client_secret="your_secret",
    )
    records = client.fetch_short_interest("GME")
"""

from __future__ import annotations

import json
import logging

import requests

from stockdownloader.data.finra_base_client import (
    FinraBaseClient,
    _MAX_RETRIES,
    normalize_finra_date,
)
from stockdownloader.core.models.regulatory import ShortInterestRecord

logger = logging.getLogger(__name__)

_FINRA_SI_URL = (
    "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
)


class FinraShortInterestClient(FinraBaseClient):
    """Fetches short interest data from FINRA.

    Parameters
    ----------
    client_id:
        FINRA API client ID.  Falls back to ``FINRA_CLIENT_ID`` env var.
    client_secret:
        FINRA API client secret.  Falls back to ``FINRA_CLIENT_SECRET`` env var.
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/short_interest.csv``.
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

    def fetch_short_interest(
        self,
        symbol: str,
    ) -> list[ShortInterestRecord]:
        """Fetch short interest records for *symbol*.

        Attempts the FINRA API first.  If the API is unavailable or
        returns errors, falls back to cached data.

        Returns records sorted by ``settlement_date`` ascending.
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
                "Using cached short interest data for %s (%d records)",
                symbol_upper, len(cached),
            )
            return cached

        logger.warning(
            "No short interest data available for %s", symbol_upper,
        )
        return []

    # ------------------------------------------------------------------
    # API query
    # ------------------------------------------------------------------

    def _query_api(self, symbol: str) -> list[dict] | None:
        """Query FINRA consolidated short interest API for *symbol*.

        The ``consolidatedShortInterest`` dataset supports querying by
        ``symbolCode`` domain filter without a partition key, returning
        all available records in a single paginated request.

        Returns raw JSON response rows, or ``None`` on failure.
        """
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
                "limit": page_size,
                "offset": offset,
            }

            try:
                self._rate_limit()
                resp = self._session.post(
                    _FINRA_SI_URL,
                    json=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if not data:
                        break  # No more data
                    all_rows.extend(data)
                    failures = 0
                    if len(data) < page_size:
                        break  # Last page
                    offset += page_size
                elif resp.status_code == 404:
                    break
                else:
                    failures += 1
                    logger.debug(
                        "FINRA SI returned %d for %s: %s",
                        resp.status_code, symbol, resp.text[:200],
                    )
            except (requests.RequestException, json.JSONDecodeError, OSError) as exc:
                failures += 1
                logger.debug(
                    "FINRA SI request failed for %s: %s",
                    symbol, exc,
                )

        if failures >= _MAX_RETRIES:
            logger.warning(
                "Too many consecutive failures querying FINRA SI for %s",
                symbol,
            )

        if all_rows:
            logger.info(
                "FINRA SI: fetched %d records for %s",
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
    ) -> list[ShortInterestRecord]:
        """Convert raw FINRA JSON rows to :class:`ShortInterestRecord` objects."""
        records: list[ShortInterestRecord] = []

        for row in raw:
            try:
                raw_date = row.get("settlementDate", "")
                # FINRA dates may come as "YYYY-MM-DD" or "MM/DD/YYYY"
                settlement_date = normalize_finra_date(raw_date)
                if not settlement_date:
                    continue

                short_interest = int(
                    row.get("currentShortPositionQuantity", 0)
                )
                avg_daily_volume = int(
                    row.get("averageDailyVolumeQuantity", 0)
                )
                days_to_cover = float(
                    row.get("daysToCoverQuantity", 0.0)
                )

                records.append(ShortInterestRecord(
                    settlement_date=settlement_date,
                    symbol=symbol,
                    short_interest=short_interest,
                    avg_daily_volume=avg_daily_volume,
                    days_to_cover=days_to_cover,
                    short_interest_pct=0.0,  # not available from FINRA directly
                ))
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed FINRA row: %s", exc)
                continue

        records.sort(key=lambda r: r.settlement_date)
        return records

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cache(self, symbol: str) -> list[ShortInterestRecord] | None:
        """Load cached short interest records for *symbol*.

        Tries CSV first, then falls back to legacy JSON files and
        migrates them to CSV on successful read.

        Returns ``None`` if no cache exists.
        """
        # Primary: CSV
        records = self._load_csv(symbol, "short_interest.csv", ShortInterestRecord)
        if records is not None:
            records.sort(key=lambda r: r.settlement_date)
            return records

        # Fallback: JSON (legacy migration)
        sym_dir = self._symbol_dir(symbol)
        json_path = sym_dir / "short_interest.json"

        # Also check legacy paths
        if not json_path.exists():
            for legacy_path in (
                self._data_dir / "cache" / "short_interest" / symbol.upper() / "si.json",
                self._data_dir / "cache" / "short_interest" / f"{symbol}_si.json",
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
                ShortInterestRecord(
                    settlement_date=r["settlement_date"],
                    symbol=r["symbol"],
                    short_interest=r["short_interest"],
                    avg_daily_volume=r["avg_daily_volume"],
                    days_to_cover=r["days_to_cover"],
                    short_interest_pct=r.get("short_interest_pct", 0.0),
                )
                for r in data
            ]
            records.sort(key=lambda r: r.settlement_date)
            self._save_cache(symbol, records)
            logger.info("Migrated %s to CSV", json_path)
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to load SI cache for %s: %s", symbol, exc)
            return None

    def _save_cache(
        self,
        symbol: str,
        records: list[ShortInterestRecord],
    ) -> None:
        """Persist short interest records to CSV cache."""
        self._save_csv(symbol, "short_interest.csv", records)
