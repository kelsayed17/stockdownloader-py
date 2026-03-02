"""Fetches historical OHLCV data from Polygon.io (now Massive.com).

Polygon's free tier provides 2 years of minute-level data and full
daily history at 5 API calls per minute.  This client handles rate
limiting, pagination, and chunking to fetch arbitrarily long date
ranges of bar data.

Usage::

    client = PolygonDataClient(api_key="YOUR_KEY")
    bars = client.fetch_intraday_history("SPY", total_days=730)
    daily = client.fetch_daily_history("GME")

Set the API key via:
  - Constructor parameter: ``PolygonDataClient(api_key="...")``
  - Environment variable: ``POLYGON_API_KEY``
"""
from __future__ import annotations

import json
import logging
import os
import time as _time
from datetime import datetime, date, timedelta
from decimal import Decimal
from operator import attrgetter
from zoneinfo import ZoneInfo

import requests

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.price import PriceData

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

_NY = ZoneInfo("America/New_York")
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MINUTE = 30
_MARKET_CLOSE_HOUR = 16
_MARKET_CLOSE_MINUTE = 0

# Polygon free tier: 5 requests/minute => 12 seconds between requests
_FREE_TIER_DELAY = 12.5

# Max results per request
_LIMIT = 50000

# Chunk size: fetch ~30 days at a time to stay well under the 50K limit
# (78 bars/day * 30 days = 2,340 bars, well within 50K)
_CHUNK_DAYS = 30


class PolygonDataClient:
    """Fetches intraday bar data from the Polygon.io REST API.

    Parameters
    ----------
    api_key:
        Polygon.io API key.  If ``None``, reads from the
        ``POLYGON_API_KEY`` environment variable.
    rate_limit_delay:
        Seconds to sleep between API requests (default: 12.5s for
        free tier's 5 calls/min limit).
    """

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit_delay: float = _FREE_TIER_DELAY,
    ) -> None:
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Polygon API key required. Pass api_key= or set POLYGON_API_KEY env var. "
                "Get a free key at https://polygon.io/"
            )
        self._delay = rate_limit_delay
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_intraday_data(
        self,
        symbol: str,
        from_date: str | date,
        to_date: str | date,
        multiplier: int = 5,
        timespan: str = "minute",
    ) -> list[IntradayPriceData]:
        """Fetch intraday bars for a single date range.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"SPY"``).
        from_date:
            Start date (``YYYY-MM-DD`` string or ``date`` object).
        to_date:
            End date (``YYYY-MM-DD`` string or ``date`` object).
        multiplier:
            Bar size multiplier (default: 5 for 5-minute bars).
        timespan:
            Bar type (default: ``"minute"``).

        Returns
        -------
        list[IntradayPriceData]
            Bars filtered to regular market hours (9:30-16:00 ET).
        """
        from_str = str(from_date)
        to_str = str(to_date)

        url = _BASE_URL.format(
            ticker=symbol.upper(),
            multiplier=multiplier,
            timespan=timespan,
            from_date=from_str,
            to_date=to_str,
        )

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": _LIMIT,
        }

        all_bars: list[IntradayPriceData] = []

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "ERROR":
                logger.warning(
                    "Polygon API error for %s: %s",
                    symbol,
                    data.get("error", "unknown"),
                )
                return []

            results = data.get("results", [])
            if not results:
                logger.debug(
                    "No results for %s from %s to %s", symbol, from_str, to_str
                )
                return []

            all_bars = _parse_results(results)
            logger.info(
                "Fetched %d bars for %s (%s to %s)",
                len(all_bars),
                symbol,
                from_str,
                to_str,
            )

            # Handle pagination
            next_url = data.get("next_url")
            while next_url:
                _time.sleep(self._delay)
                # Pass API key via header (already set on session),
                # strip any leaked key from the pagination URL.
                from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

                parsed = urlparse(next_url)
                params = parse_qs(parsed.query, keep_blank_values=True)
                params.pop("apiKey", None)
                clean_query = urlencode(params, doseq=True)
                next_url = urlunparse(parsed._replace(query=clean_query))
                resp = self._session.get(next_url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if results:
                    all_bars.extend(_parse_results(results))
                next_url = data.get("next_url")

        except requests.RequestException as exc:
            logger.warning("Polygon request failed for %s: %s", symbol, exc)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Polygon response parse error for %s: %s", symbol, exc)

        return all_bars

    def fetch_daily_history(
        self,
        symbol: str,
        from_date: str | date | None = None,
        to_date: str | date | None = None,
    ) -> list[PriceData]:
        """Fetch full daily OHLCV history from Polygon.

        Polygon free tier provides the complete daily history for US
        equities.  The data is split-adjusted by default.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"GME"``).
        from_date:
            Start date.  Defaults to ``2000-01-01`` to capture all
            available history for most equities.
        to_date:
            End date.  Defaults to today.

        Returns
        -------
        list[PriceData]
            Daily bars sorted chronologically.
        """
        if from_date is None:
            from_date = date(2000, 1, 1)
        if to_date is None:
            to_date = date.today()

        from_str = str(from_date)
        to_str = str(to_date)

        url = _BASE_URL.format(
            ticker=symbol.upper(),
            multiplier=1,
            timespan="day",
            from_date=from_str,
            to_date=to_str,
        )

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": _LIMIT,
        }

        all_bars: list[PriceData] = []

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "ERROR":
                logger.warning(
                    "Polygon API error for %s daily: %s",
                    symbol,
                    data.get("error", "unknown"),
                )
                return []

            results = data.get("results", [])
            if not results:
                logger.debug(
                    "No daily results for %s from %s to %s",
                    symbol, from_str, to_str,
                )
                return []

            all_bars = _parse_daily_results(results)

            # Handle pagination
            next_url = data.get("next_url")
            while next_url:
                _time.sleep(self._delay)
                from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

                parsed = urlparse(next_url)
                qs = parse_qs(parsed.query, keep_blank_values=True)
                qs.pop("apiKey", None)
                clean_query = urlencode(qs, doseq=True)
                next_url = urlunparse(parsed._replace(query=clean_query))
                resp = self._session.get(next_url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if results:
                    all_bars.extend(_parse_daily_results(results))
                next_url = data.get("next_url")

            logger.info(
                "Fetched %d daily bars for %s (%s to %s)",
                len(all_bars), symbol, from_str, to_str,
            )

        except requests.RequestException as exc:
            logger.warning("Polygon daily request failed for %s: %s", symbol, exc)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Polygon daily parse error for %s: %s", symbol, exc)

        return all_bars

    def fetch_intraday_history(
        self,
        symbol: str,
        total_days: int = 730,
        multiplier: int = 5,
        timespan: str = "minute",
        interval: str | None = None,
    ) -> list[IntradayPriceData]:
        """Fetch intraday bars over a long date range by chunking.

        Splits the request into ~30-day windows, respecting rate limits,
        and deduplicates the merged results.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        total_days:
            Total calendar days to fetch (default: 730 for 2 years).
        multiplier:
            Bar size multiplier (default: 5).
        timespan:
            Bar type (default: ``"minute"``).
        interval:
            Ignored (for API compatibility with YahooDataClient).

        Returns
        -------
        list[IntradayPriceData]
            All bars sorted chronologically, deduplicated.
        """
        today = date.today()
        start = today - timedelta(days=total_days)

        all_bars: dict[str, IntradayPriceData] = {}
        current_start = start
        chunk_num = 0

        while current_start < today:
            current_end = min(current_start + timedelta(days=_CHUNK_DAYS), today)
            chunk_num += 1

            logger.info(
                "Chunk %d: Fetching %s 5m data: %s to %s",
                chunk_num,
                symbol,
                current_start,
                current_end,
            )
            logger.info(
                "Fetching %s 5m data: %s to %s",
                symbol, current_start, current_end,
            )

            bars = self.fetch_intraday_data(
                symbol,
                from_date=current_start,
                to_date=current_end,
                multiplier=multiplier,
                timespan=timespan,
            )

            for bar in bars:
                all_bars[bar.date] = bar

            if bars:
                logger.info("  Got %d bars", len(bars))
            else:
                logger.info("  No data for this window")

            current_start = current_end + timedelta(days=1)

            # Rate limit between chunks
            if current_start < today:
                _time.sleep(self._delay)

        result = sorted(all_bars.values(), key=attrgetter("date"))
        logger.info(
            "Total: %d unique bars for %s over %d days",
            len(result),
            symbol,
            total_days,
        )
        return result


def _parse_daily_results(results: list[dict]) -> list[PriceData]:
    """Parse Polygon API result objects into daily PriceData.

    No market-hours filtering is needed for daily bars.
    """
    bars: list[PriceData] = []

    for r in results:
        try:
            ts_ms = r["t"]
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=_NY)
            date_str = dt.strftime("%Y-%m-%d")

            open_ = Decimal(str(r["o"]))
            high = Decimal(str(r["h"]))
            low = Decimal(str(r["l"]))
            close = Decimal(str(r["c"]))
            volume = int(r.get("v", 0))

            bars.append(
                PriceData(
                    date=date_str,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    adj_close=close,  # Polygon returns adjusted data by default
                    volume=volume,
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Skipping malformed Polygon daily bar: %s", exc)
            continue

    return bars


def _parse_results(results: list[dict]) -> list[IntradayPriceData]:
    """Parse Polygon API result objects into IntradayPriceData.

    Filters to regular market hours (9:30 AM - 4:00 PM ET).
    """
    bars: list[IntradayPriceData] = []

    for r in results:
        try:
            # Polygon timestamps are Unix milliseconds
            ts_ms = r["t"]
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=_NY)

            # Filter to regular market hours
            market_open = dt.replace(
                hour=_MARKET_OPEN_HOUR,
                minute=_MARKET_OPEN_MINUTE,
                second=0,
                microsecond=0,
            )
            market_close = dt.replace(
                hour=_MARKET_CLOSE_HOUR,
                minute=_MARKET_CLOSE_MINUTE,
                second=0,
                microsecond=0,
            )
            if dt < market_open or dt >= market_close:
                continue

            # Format datetime string to match Yahoo format
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S%z")
            # Insert colon in timezone offset: -0500 -> -05:00
            date_str = date_str[:-2] + ":" + date_str[-2:]

            open_ = Decimal(str(r["o"]))
            high = Decimal(str(r["h"]))
            low = Decimal(str(r["l"]))
            close = Decimal(str(r["c"]))
            volume = int(r.get("v", 0))

            bars.append(
                IntradayPriceData(
                    date=date_str,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    adj_close=close,
                    volume=volume,
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Skipping malformed Polygon bar: %s", exc)
            continue

    return bars
