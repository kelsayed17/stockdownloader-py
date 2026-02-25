"""Fetches historical options data from Polygon.io.

Downloads option contract metadata and daily bars for backtesting
options strategies with real market prices.

Usage::

    client = PolygonOptionsClient(api_key="YOUR_KEY")
    contracts = client.fetch_option_contracts("SPY", from_date, to_date)
    bar = client.fetch_option_daily_bar("O:SPY250207P00580000", trade_date)

Set the API key via:
  - Constructor parameter: ``PolygonOptionsClient(api_key="...")``
  - Environment variable: ``POLYGON_API_KEY``
"""
from __future__ import annotations

import json
import logging
import os
import time as _time
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)

_CONTRACTS_URL = "https://api.polygon.io/v3/reference/options/contracts"
_AGGS_URL = (
    "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
)

_DEFAULT_DELAY = 0.2


class PolygonOptionsClient:
    """Fetches option contract metadata and prices from Polygon.io."""

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit_delay: float = _DEFAULT_DELAY,
    ) -> None:
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Polygon API key required. Pass api_key= or set POLYGON_API_KEY env var."
            )
        self._delay = rate_limit_delay
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self._api_key}"}
        )

    def fetch_option_contracts(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        *,
        expired: bool = True,
        fridays_only: bool = False,
        contract_type: str | None = None,
    ) -> list[dict]:
        """Fetch option contract metadata from Polygon reference API.

        Parameters
        ----------
        symbol:
            Underlying ticker (e.g. ``"SPY"``).
        from_date:
            Include contracts expiring on or after this date.
        to_date:
            Include contracts expiring on or before this date.
        expired:
            If ``True``, include expired contracts (needed for backtesting).
        fridays_only:
            If ``True``, only return contracts expiring on a Friday.
        contract_type:
            Filter to ``"call"`` or ``"put"`` only.  ``None`` returns both.

        Returns
        -------
        list[dict]
            Raw contract dicts from Polygon.
        """
        params: dict = {
            "underlying_ticker": symbol.upper(),
            "expiration_date.gte": str(from_date),
            "expiration_date.lte": str(to_date),
            "expired": str(expired).lower(),
            "limit": 1000,
            "order": "asc",
            "sort": "expiration_date",
        }
        if contract_type:
            params["contract_type"] = contract_type

        all_contracts: list[dict] = []

        try:
            resp = self._session.get(_CONTRACTS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            all_contracts.extend(results)

            next_url = data.get("next_url")
            while next_url:
                _time.sleep(self._delay)
                parsed = urlparse(next_url)
                qs = parse_qs(parsed.query, keep_blank_values=True)
                qs.pop("apiKey", None)
                clean_query = urlencode(qs, doseq=True)
                clean_url = urlunparse(parsed._replace(query=clean_query))
                resp = self._session.get(clean_url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                all_contracts.extend(results)
                next_url = data.get("next_url")

        except requests.RequestException as exc:
            logger.warning("Polygon options contracts request failed: %s", exc)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Polygon options contracts parse error: %s", exc)

        if fridays_only:
            all_contracts = [
                c for c in all_contracts
                if _is_friday(c.get("expiration_date", ""))
            ]

        logger.info(
            "Fetched %d option contracts for %s (%s to %s)",
            len(all_contracts), symbol, from_date, to_date,
        )
        return all_contracts

    def fetch_option_daily_bar(
        self,
        option_ticker: str,
        trade_date: date,
    ) -> dict | None:
        """Fetch a single daily bar for an options contract.

        Parameters
        ----------
        option_ticker:
            Polygon options ticker (e.g. ``"O:SPY250207P00580000"``).
        trade_date:
            The date to fetch the bar for.

        Returns
        -------
        dict | None
            Bar dict with keys ``o``, ``h``, ``l``, ``c``, ``v``, ``t``,
            or ``None`` if no data is available.
        """
        url = _AGGS_URL.format(
            ticker=option_ticker,
            from_date=str(trade_date),
            to_date=str(trade_date),
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 1}

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                return results[0]
        except requests.RequestException as exc:
            logger.warning(
                "Polygon option bar request failed for %s: %s",
                option_ticker, exc,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Polygon option bar parse error for %s: %s",
                option_ticker, exc,
            )

        return None


def _is_friday(date_str: str) -> bool:
    """Return True if the date string (YYYY-MM-DD) falls on a Friday."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() == 4
    except (ValueError, TypeError):
        return False
