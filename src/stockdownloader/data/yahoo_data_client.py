"""Fetches historical OHLCV price data directly from Yahoo Finance v8 chart API
for any ticker symbol.  Returns data as ``list[PriceData]`` in memory without
intermediate CSV files.

Supports configurable time ranges: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
and intervals: 1d, 1wk, 1mo
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from stockdownloader.data.yahoo_auth_helper import YahooAuthHelper
from stockdownloader.model import PriceData

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?range={range}&interval={interval}"
)
_PERIOD_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={start}&period2={end}&interval=1d"
)
_DATE_FMT = "%Y-%m-%d"
_NY_TZ = ZoneInfo("America/New_York")


class YahooDataClient:
    """Facade client that fetches historical OHLCV price data from Yahoo
    Finance and returns ``list[PriceData]``.
    """

    def __init__(self, auth: YahooAuthHelper | None = None) -> None:
        self._auth = auth or YahooAuthHelper()

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
        if self._auth.crumb is None:
            self._auth.authenticate()

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
    ) -> list[PriceData]:
        """Fetch price data using explicit epoch timestamps for precise date
        ranges.
        """
        if self._auth.crumb is None:
            self._auth.authenticate()

        url = (
            _PERIOD_URL.format(
                symbol=symbol.upper(),
                start=start_epoch,
                end=end_epoch,
            )
            + f"&crumb={self._auth.crumb}"
        )
        return self._fetch_and_parse(url, symbol)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def auth(self) -> YahooAuthHelper:
        """Return the shared auth helper."""
        return self._auth

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_parse(self, url: str, symbol: str) -> list[PriceData]:
        result: list[PriceData] = []
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._auth.session.get(url, timeout=15)
                result.extend(self._parse_chart_response(resp.text, symbol))
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.debug(
                        "Retrying price data fetch for %s, attempt %d",
                        symbol,
                        attempt + 1,
                    )
                else:
                    logger.warning(
                        "Failed price data fetch for %s after %d retries: %s",
                        symbol,
                        _MAX_RETRIES,
                        last_exc,
                    )

        return result

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

                    open_ = _get_decimal_at(open_array, i)
                    high = _get_decimal_at(high_array, i)
                    low = _get_decimal_at(low_array, i)
                    close = _get_decimal_at(close_array, i)
                    volume = _get_long_at(volume_array, i)

                    adj_close = (
                        _get_decimal_at(adj_close_array, i)
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
                except Exception:
                    # Skip malformed bars
                    pass
        except Exception as exc:
            logger.warning(
                "Error parsing chart data for %s: %s", symbol, exc
            )

        return data


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _get_decimal_at(arr: list | None, index: int) -> Decimal:
    if arr is None or index >= len(arr):
        return Decimal(0)
    val = arr[index]
    if val is None:
        return Decimal(0)
    return Decimal(str(val))


def _get_long_at(arr: list | None, index: int) -> int:
    if arr is None or index >= len(arr):
        return 0
    val = arr[index]
    if val is None:
        return 0
    return int(val)
