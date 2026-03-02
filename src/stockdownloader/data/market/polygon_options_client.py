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
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

from stockdownloader.analysis.options.pricing import delta as bs_delta
from stockdownloader.core.models.options import OptionType

logger = logging.getLogger(__name__)

_CONTRACTS_URL = "https://api.polygon.io/v3/reference/options/contracts"
_AGGS_URL = (
    "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
)
_SNAPSHOT_URL = "https://api.polygon.io/v3/snapshot/options/{underlying_asset}"

# 0.2s delay for upgraded Polygon tier (vs 12.5s free tier in polygon_client.py)
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

    def fetch_option_daily_bars_range(
        self,
        option_ticker: str,
        from_date: date,
        to_date: date,
    ) -> list[dict]:
        """Fetch all daily bars for an options contract across a date range.

        Parameters
        ----------
        option_ticker:
            Polygon options ticker (e.g. ``"O:SPY250207P00580000"``).
        from_date:
            Start of the date range.
        to_date:
            End of the date range.

        Returns
        -------
        list[dict]
            List of bar dicts, each with keys ``o``, ``h``, ``l``, ``c``,
            ``v``, ``vw``, ``t``, ``n``.  Empty if no data.
        """
        url = _AGGS_URL.format(
            ticker=option_ticker,
            from_date=str(from_date),
            to_date=str(to_date),
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except requests.RequestException as exc:
            logger.warning(
                "Polygon option bars range request failed for %s: %s",
                option_ticker, exc,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Polygon option bars range parse error for %s: %s",
                option_ticker, exc,
            )

        return []

    def fetch_options_chain_snapshot(
        self,
        symbol: str,
        *,
        limit: int = 250,
    ) -> list[dict]:
        """Fetch snapshot data for all active option contracts.

        Returns the full chain with OI, greeks, and IV for each
        contract via paginated calls to the Polygon snapshot endpoint.

        Parameters
        ----------
        symbol:
            Underlying ticker (e.g. ``"GME"``).
        limit:
            Results per page (max 250).

        Returns
        -------
        list[dict]
            Raw snapshot result dicts from Polygon.
        """
        url = _SNAPSHOT_URL.format(underlying_asset=symbol.upper())
        params: dict = {"limit": limit}
        all_results: list[dict] = []

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(data.get("results", []))

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
                all_results.extend(data.get("results", []))
                next_url = data.get("next_url")

        except requests.RequestException as exc:
            logger.warning(
                "Polygon options snapshot request failed for %s: %s",
                symbol, exc,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Polygon options snapshot parse error for %s: %s",
                symbol, exc,
            )

        logger.info(
            "Fetched %d option snapshots for %s",
            len(all_results), symbol,
        )
        return all_results


def _is_friday(date_str: str) -> bool:
    """Return True if the date string (YYYY-MM-DD) falls on a Friday."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() == 4
    except (ValueError, TypeError):
        return False


def save_chain_cache(cache_dir: Path, expiration_date: str, data: dict) -> None:
    """Save option chain data to disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{expiration_date}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Saved cache: %s", cache_file)


def load_chain_cache(cache_dir: Path, expiration_date: str) -> dict | None:
    """Load option chain data from disk cache. Returns None if not found."""
    cache_file = cache_dir / f"{expiration_date}.json"
    if not cache_file.exists():
        return None
    with open(cache_file) as f:
        return json.load(f)


_DEFAULT_RISK_FREE_RATE = 0.05


def select_strike_by_delta(
    contracts: list[dict],
    *,
    contract_type: str,
    spot: float,
    target_delta: float,
    days_to_expiry: int,
    volatility: float,
    risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
) -> dict | None:
    """Select the contract whose BS delta is closest to the target.

    Parameters
    ----------
    contracts:
        List of contract dicts (must have ``strike_price`` and ``contract_type``).
    contract_type:
        ``"call"`` or ``"put"``.
    spot:
        Current underlying price.
    target_delta:
        Target absolute delta (e.g. 0.30).
    days_to_expiry:
        Days to expiration.
    volatility:
        Annualized volatility estimate.

    Returns
    -------
    dict | None
        The contract dict closest to the target delta, or None.
    """
    filtered = [c for c in contracts if c.get("contract_type") == contract_type]
    if not filtered:
        return None

    opt_type = OptionType.CALL if contract_type == "call" else OptionType.PUT
    time_to_expiry = Decimal(str(days_to_expiry)) / Decimal("365")
    spot_d = Decimal(str(spot))
    vol_d = Decimal(str(volatility))
    rfr_d = Decimal(str(risk_free_rate))

    best_contract = None
    best_diff = float("inf")

    for c in filtered:
        strike_d = Decimal(str(c["strike_price"]))
        d = bs_delta(opt_type, spot_d, strike_d, time_to_expiry, rfr_d, vol_d)
        diff = abs(abs(float(d)) - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_contract = c

    return best_contract
