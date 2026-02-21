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

import base64
import json
import logging
import os
import time
from pathlib import Path

import requests

from stockdownloader.data.base_client import BaseDataClient
from stockdownloader.model.regulatory_records import ShortInterestRecord

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RATE_LIMIT_DELAY = 0.5  # FINRA is more conservative than SEC
_FINRA_SI_URL = (
    "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
)
_FINRA_TOKEN_URL = (
    "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
    "?grant_type=client_credentials"
)


class FinraShortInterestClient(BaseDataClient):
    """Fetches short interest data from FINRA.

    Parameters
    ----------
    client_id:
        FINRA API client ID.  Falls back to ``FINRA_CLIENT_ID`` env var.
    client_secret:
        FINRA API client secret.  Falls back to ``FINRA_CLIENT_SECRET`` env var.
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/short_interest.json``.
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
                    logger.info("FINRA OAuth2 authentication successful")
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
                settlement_date = _normalize_date(raw_date)
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

        Returns ``None`` if no cache exists.
        """
        sym_dir = self._symbol_dir(symbol)
        cache_file = sym_dir / "short_interest.json"

        # Legacy migration from old cache paths
        if not cache_file.exists():
            for legacy_path in (
                self._data_dir / "cache" / "short_interest" / symbol.upper() / "si.json",
                self._data_dir / "cache" / "short_interest" / f"{symbol}_si.json",
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
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to load SI cache for %s: %s", symbol, exc)
            return None

    def _save_cache(
        self,
        symbol: str,
        records: list[ShortInterestRecord],
    ) -> None:
        """Persist short interest records to JSON cache."""
        cache_file = self._symbol_dir(symbol) / "short_interest.json"
        data = [
            {
                "settlement_date": r.settlement_date,
                "symbol": r.symbol,
                "short_interest": r.short_interest,
                "avg_daily_volume": r.avg_daily_volume,
                "days_to_cover": r.days_to_cover,
                "short_interest_pct": r.short_interest_pct,
            }
            for r in records
        ]
        try:
            cache_file.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save SI cache for %s: %s", symbol, exc)

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

    Handles both ``YYYY-MM-DD`` and ``MM/DD/YYYY`` formats.
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
