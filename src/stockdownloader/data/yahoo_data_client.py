"""Fetches historical OHLCV price data directly from Yahoo Finance v8 chart API
for any ticker symbol.  Returns data as ``list[PriceData]`` in memory without
intermediate CSV files.

Supports configurable time ranges: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
and intervals: 1d, 1wk, 1mo

Also supports intraday intervals (1m, 2m, 5m, 15m, 30m, 60m, 90m) via
:meth:`fetch_intraday_data` and :meth:`fetch_intraday_history`.
"""
from __future__ import annotations

import json
import logging
import time as _time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from operator import attrgetter
from zoneinfo import ZoneInfo

import requests

from stockdownloader.data.data_parsers import get_decimal_at, get_long_at
from stockdownloader.data.yahoo_base_client import YahooAuthHelper, YahooBaseClient
from stockdownloader.core.models import PriceData
from stockdownloader.core.models.price import IntradayPriceData

logger = logging.getLogger(__name__)
_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?range={range}&interval={interval}"
)
_PERIOD_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={start}&period2={end}&interval={interval}"
)
_INTRADAY_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={start}&period2={end}&interval={interval}"
    "&includePrePost=false"
)
_DATE_FMT = "%Y-%m-%d"
_INTRADAY_FMT = "%Y-%m-%d %H:%M:%S%z"
_NY_TZ = ZoneInfo("America/New_York")
_WINDOW_DAYS = 55  # stay under Yahoo's ~60-day intraday limit


class YahooDataClient(YahooBaseClient):
    """Facade client that fetches historical OHLCV price data from Yahoo
    Finance and returns ``list[PriceData]``.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_price_data(
        self,
        symbol: str,
        range_: str = "5y",
        interval: str = "1d",
    ) -> list[PriceData]:
        """Fetch historical price data for *symbol*.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"AAPL"``, ``"SPY"``, ``"TSLA"``).
        range_:
            Time range -- one of ``1mo``, ``3mo``, ``6mo``, ``1y``, ``2y``,
            ``5y``, ``10y``, ``max``.
        interval:
            Data interval -- one of ``1d``, ``1wk``, ``1mo``.

        Returns
        -------
        list[PriceData]
            Sorted by date ascending; empty if the fetch fails.
        """
        self._ensure_authenticated()

        if range_ == "max":
            # Yahoo's v8 API is unreliable with ``range=max`` -- it often
            # returns only a few hundred bars instead of the full history.
            # Use explicit epoch timestamps (period1=0 means earliest
            # available data, period2=now) which reliably returns full
            # history (e.g. 5000+ bars for GME back to 2002).
            end_epoch = int(datetime.now(tz=timezone.utc).timestamp())
            return self.fetch_price_data_by_epoch(
                symbol, start_epoch=0, end_epoch=end_epoch,
                interval=interval,
            )

        url = (
            _CHART_URL.format(
                symbol=symbol.upper(),
                range=range_,
                interval=interval,
            )
            + f"&crumb={self._auth.crumb}"
        )
        return self._fetch_and_parse(url, symbol)

    def fetch_price_data_by_epoch(
        self,
        symbol: str,
        start_epoch: int,
        end_epoch: int,
        interval: str = "1d",
    ) -> list[PriceData]:
        """Fetch price data using explicit epoch timestamps for precise date
        ranges.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        start_epoch:
            Start time as Unix epoch seconds (``0`` for earliest available).
        end_epoch:
            End time as Unix epoch seconds.
        interval:
            Data interval -- ``"1d"``, ``"1wk"``, or ``"1mo"``
            (default ``"1d"``).
        """
        self._ensure_authenticated()

        url = (
            _PERIOD_URL.format(
                symbol=symbol.upper(),
                start=start_epoch,
                end=end_epoch,
                interval=interval,
            )
            + f"&crumb={self._auth.crumb}"
        )
        return self._fetch_and_parse(url, symbol)

    def fetch_intraday_data(
        self,
        symbol: str,
        start_epoch: int,
        end_epoch: int,
        interval: str = "5m",
    ) -> list[IntradayPriceData]:
        """Fetch intraday OHLCV bars for a single window.

        Yahoo limits intraday data to ~60 calendar days per request.
        Use :meth:`fetch_intraday_history` for longer periods.
        """
        self._ensure_authenticated()

        url = (
            _INTRADAY_URL.format(
                symbol=symbol.upper(),
                start=start_epoch,
                end=end_epoch,
                interval=interval,
            )
            + f"&crumb={self._auth.crumb}"
        )
        return self._fetch_and_parse_intraday(url, symbol)

    def fetch_intraday_history(
        self,
        symbol: str,
        total_days: int = 365,
        interval: str = "5m",
    ) -> list[IntradayPriceData]:
        """Fetch intraday data going back *total_days* calendar days.

        Automatically chunks into ~55-day windows to stay within Yahoo's
        intraday data limits.  Returns bars sorted by datetime ascending.
        """
        now = datetime.now(tz=_NY_TZ)
        end_dt = now
        start_dt = now - timedelta(days=total_days)

        all_bars: list[IntradayPriceData] = []
        seen_dates: set[str] = set()

        window_start = start_dt
        while window_start < end_dt:
            window_end = min(window_start + timedelta(days=_WINDOW_DAYS), end_dt)

            s_epoch = int(window_start.timestamp())
            e_epoch = int(window_end.timestamp())

            logger.info(
                "Fetching %s %s data: %s to %s",
                symbol,
                interval,
                window_start.strftime(_DATE_FMT),
                window_end.strftime(_DATE_FMT),
            )

            bars = self.fetch_intraday_data(
                symbol, s_epoch, e_epoch, interval
            )

            for bar in bars:
                if bar.date not in seen_dates:
                    seen_dates.add(bar.date)
                    all_bars.append(bar)

            window_start = window_end
            # Brief pause between requests to avoid rate limiting
            if window_start < end_dt:
                _time.sleep(0.5)

        all_bars.sort(key=attrgetter("date"))
        return all_bars

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_parse_intraday(
        self, url: str, symbol: str
    ) -> list[IntradayPriceData]:
        result = self._fetch_with_retry(
            url,
            lambda text: self._parse_intraday_chart_response(text, symbol),
            f"intraday fetch for {symbol}",
            timeout=30,
        )
        return result if result is not None else []

    @staticmethod
    def _parse_intraday_chart_response(
        raw: str, symbol: str
    ) -> list[IntradayPriceData]:
        data: list[IntradayPriceData] = []

        try:
            root = json.loads(raw)
            chart = root.get("chart")
            if chart is None:
                logger.warning("No chart data in intraday response for %s", symbol)
                return data

            results = chart.get("result")
            if not results:
                logger.warning("Empty intraday results for %s", symbol)
                return data

            result = results[0]
            timestamps = result.get("timestamp")
            if not timestamps:
                logger.warning("No intraday timestamps for %s", symbol)
                return data

            indicators = result.get("indicators", {})
            quote_array = indicators.get("quote")
            if not quote_array:
                return data

            quote = quote_array[0]
            open_array = quote.get("open", [])
            high_array = quote.get("high", [])
            low_array = quote.get("low", [])
            close_array = quote.get("close", [])
            volume_array = quote.get("volume", [])

            for i, ts in enumerate(timestamps):
                try:
                    if ts is None:
                        continue

                    dt = (
                        datetime.fromtimestamp(int(ts), tz=timezone.utc)
                        .astimezone(_NY_TZ)
                    )

                    # Filter to regular market hours (9:30 - 16:00 ET)
                    hour, minute = dt.hour, dt.minute
                    if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
                        continue

                    date_str = dt.strftime(_INTRADAY_FMT)
                    # strftime %z produces '-0500'; insert colon for
                    # ISO-8601 compliance ('-05:00') matching Polygon.
                    if len(date_str) >= 5 and date_str[-5] in ('+', '-') and ':' not in date_str[-5:]:
                        date_str = date_str[:-2] + ':' + date_str[-2:]

                    open_ = get_decimal_at(open_array, i)
                    high = get_decimal_at(high_array, i)
                    low = get_decimal_at(low_array, i)
                    close = get_decimal_at(close_array, i)
                    volume = get_long_at(volume_array, i)

                    if close == Decimal(0):
                        continue

                    data.append(
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
                except (ValueError, TypeError, IndexError) as exc:
                    logger.debug("Skipping malformed intraday bar %d: %s", i, exc)
        except (json.JSONDecodeError, KeyError, TypeError, IndexError, AttributeError) as exc:
            logger.warning(
                "Error parsing intraday chart data for %s: %s", symbol, exc
            )

        return data

    def _fetch_and_parse(self, url: str, symbol: str) -> list[PriceData]:
        result = self._fetch_with_retry(
            url,
            lambda text: self._parse_chart_response(text, symbol),
            f"price data fetch for {symbol}",
        )
        return result if result is not None else []

    @staticmethod
    def _parse_chart_response(raw: str, symbol: str) -> list[PriceData]:
        data: list[PriceData] = []

        try:
            root = json.loads(raw)
            chart = root.get("chart")
            if chart is None:
                logger.warning("No chart data in response for %s", symbol)
                return data

            results = chart.get("result")
            if not results:
                logger.warning("Empty results for %s", symbol)
                return data

            result = results[0]
            timestamps = result.get("timestamp")
            if not timestamps:
                logger.warning("No timestamps for %s", symbol)
                return data

            indicators = result.get("indicators", {})
            quote_array = indicators.get("quote")
            if not quote_array:
                return data

            quote = quote_array[0]
            open_array = quote.get("open", [])
            high_array = quote.get("high", [])
            low_array = quote.get("low", [])
            close_array = quote.get("close", [])
            volume_array = quote.get("volume", [])

            # Check for adjusted close
            adj_close_array: list | None = None
            adj_close_wrapper = indicators.get("adjclose")
            if adj_close_wrapper:
                adj_close_array = adj_close_wrapper[0].get("adjclose")

            for i, ts in enumerate(timestamps):
                try:
                    if ts is None:
                        continue

                    date_str = (
                        datetime.fromtimestamp(int(ts), tz=timezone.utc)
                        .astimezone(_NY_TZ)
                        .strftime(_DATE_FMT)
                    )

                    open_ = get_decimal_at(open_array, i)
                    high = get_decimal_at(high_array, i)
                    low = get_decimal_at(low_array, i)
                    close = get_decimal_at(close_array, i)
                    volume = get_long_at(volume_array, i)

                    adj_close = (
                        get_decimal_at(adj_close_array, i)
                        if adj_close_array is not None
                        else close
                    )

                    # Skip bars with zero close prices
                    if close == Decimal(0):
                        continue

                    data.append(
                        PriceData(
                            date=date_str,
                            open=open_,
                            high=high,
                            low=low,
                            close=close,
                            adj_close=adj_close,
                            volume=volume,
                        )
                    )
                except (ValueError, TypeError, IndexError) as exc:
                    logger.debug("Skipping malformed daily bar %d: %s", i, exc)
        except (json.JSONDecodeError, KeyError, TypeError, IndexError, AttributeError) as exc:
            logger.warning(
                "Error parsing chart data for %s: %s", symbol, exc
            )

        return data

