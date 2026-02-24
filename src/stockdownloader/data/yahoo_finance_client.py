"""Yahoo Finance API clients for real-time quotes and historical data.

Contains:
- :class:`YahooFinanceClient` -- real-time quote data via Yahoo v7 quote JSON API.
- :class:`YahooHistoricalClient` -- historical price data and patterns via Yahoo v8
  chart API.

Both extend :class:`YahooBaseClient` for shared authentication and retry logic.
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal, ROUND_CEILING
from typing import Sequence

import requests

from stockdownloader.data.data_parsers import (
    format_market_cap,
    get_decimal,
    get_long,
    get_string,
)
from stockdownloader.data.yahoo_base_client import YahooAuthHelper, YahooBaseClient
from stockdownloader.core.models import HistoricalData, QuoteData

logger = logging.getLogger(__name__)
_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
_DEFAULT_BATCH_SIZE = 100
_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?range=1mo&interval=1d"
)
_PATTERN_DAYS = 7


class YahooFinanceClient(YahooBaseClient):
    """HTTP client that fetches real-time quote data from Yahoo Finance."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, ticker: str) -> QuoteData:
        """Download real-time quote data for *ticker* and return a
        :class:`QuoteData` instance.
        """
        self._ensure_authenticated()

        def _parse(text: str) -> QuoteData:
            root = json.loads(text)
            quote_response = root.get("quoteResponse")
            if not quote_response or not quote_response.get("result"):
                logger.warning("Empty quote response from Yahoo Finance for %s", ticker)
                data = QuoteData()
                data.incomplete = True
                return data
            return self._parse_single_quote(quote_response["result"][0])

        url = _QUOTE_URL.format(symbol=ticker) + f"&crumb={self._auth.crumb}"
        result = self._fetch_with_retry(url, _parse, f"quote download for {ticker}")
        return result if result is not None else QuoteData()

    def download_batch(
        self,
        tickers: Sequence[str],
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> dict[str, QuoteData]:
        """Batch-download quotes for multiple tickers.

        Parameters
        ----------
        tickers:
            Sequence of ticker symbols to fetch.
        batch_size:
            Number of symbols per API call (default 100).

        Returns
        -------
        Dict mapping symbol → :class:`QuoteData`.  Tickers that fail
        to parse are silently omitted from the result.
        """
        if not tickers:
            return {}

        self._ensure_authenticated()

        results: dict[str, QuoteData] = {}
        ticker_list = list(tickers)

        for start in range(0, len(ticker_list), batch_size):
            chunk = ticker_list[start : start + batch_size]
            self._fetch_chunk(chunk, results)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_chunk(
        self,
        chunk: list[str],
        results: dict[str, QuoteData],
    ) -> None:
        """Fetch a single batch chunk and populate *results*."""
        symbols_str = ",".join(chunk)

        def _parse(text: str) -> bool:
            root = json.loads(text)
            quote_response = root.get("quoteResponse")
            if not quote_response or not quote_response.get("result"):
                logger.warning(
                    "Empty batch quote response for %d tickers",
                    len(chunk),
                )
                return True  # success — just empty

            for quote_dict in quote_response["result"]:
                try:
                    symbol = quote_dict.get("symbol", "")
                    if symbol:
                        results[symbol] = self._parse_single_quote(quote_dict)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.debug(
                        "Skipping quote parse for %s: %s",
                        quote_dict.get("symbol", "?"),
                        exc,
                    )
            return True

        url = _QUOTE_URL.format(symbol=symbols_str) + f"&crumb={self._auth.crumb}"
        self._fetch_with_retry(
            url, _parse, f"batch download for {len(chunk)} tickers", timeout=30
        )

    @staticmethod
    def _parse_single_quote(quote: dict) -> QuoteData:
        """Parse a single Yahoo v7 quote dict into a :class:`QuoteData`.

        Parameters
        ----------
        quote:
            A single element from ``quoteResponse.result[]``.

        Returns
        -------
        Populated :class:`QuoteData` instance.
        """
        data = QuoteData()

        data.price_sales = get_decimal(quote, "priceToSalesTrailing12Months")
        data.trailing_annual_dividend_yield = get_decimal(
            quote, "trailingAnnualDividendYield"
        )
        data.diluted_eps = get_decimal(quote, "epsTrailingTwelveMonths")
        data.eps_estimate_next_year = get_decimal(quote, "epsForward")
        data.last_trade_price_only = get_decimal(quote, "regularMarketPrice")
        data.year_high = get_decimal(quote, "fiftyTwoWeekHigh")
        data.year_low = get_decimal(quote, "fiftyTwoWeekLow")
        data.fifty_day_moving_average = get_decimal(quote, "fiftyDayAverage")
        data.two_hundred_day_moving_average = get_decimal(
            quote, "twoHundredDayAverage"
        )
        data.previous_close = get_decimal(quote, "regularMarketPreviousClose")
        data.open = get_decimal(quote, "regularMarketOpen")
        data.days_high = get_decimal(quote, "regularMarketDayHigh")
        data.days_low = get_decimal(quote, "regularMarketDayLow")
        data.volume = get_decimal(quote, "regularMarketVolume")

        # Valuation fields
        data.trailing_pe = get_decimal(quote, "trailingPE")
        data.forward_pe = get_decimal(quote, "forwardPE")
        data.price_to_book = get_decimal(quote, "priceToBook")
        data.book_value = get_decimal(quote, "bookValue")

        data.year_range = get_string(quote, "fiftyTwoWeekRange")

        market_cap = get_long(quote, "marketCap")
        data.market_capitalization = market_cap
        data.market_capitalization_str = format_market_cap(market_cap)

        if data.last_trade_price_only < data.year_low:
            data.year_low = data.last_trade_price_only

        return data


# ---------------------------------------------------------------------------
# Historical price data client
# ---------------------------------------------------------------------------


class YahooHistoricalClient(YahooBaseClient):
    """Fetches historical price data and computes movement patterns.

    Downloads one-month daily close prices from the Yahoo Finance v8 chart API
    and derives up/down patterns.  Replaces the deprecated Google Finance
    historical CSV endpoint (www.google.com/finance/historical) which was shut
    down around 2015.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, ticker: str) -> HistoricalData:
        """Download one-month daily close prices for *ticker* and derive
        up/down patterns.
        """
        data = HistoricalData(ticker)
        self._ensure_authenticated()

        def _parse(text: str) -> HistoricalData:
            self._parse_chart_json(text, data)
            return data

        url = _CHART_URL.format(symbol=ticker) + f"&crumb={self._auth.crumb}"
        result = self._fetch_with_retry(
            url, _parse, f"historical download for {ticker}",
        )
        return result if result is not None else data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_chart_json(self, raw: str, data: HistoricalData) -> None:
        try:
            root = json.loads(raw)
            chart = root.get("chart")
            if chart is None:
                data.incomplete = True
                return

            results = chart.get("result")
            if not results:
                data.incomplete = True
                return

            result = results[0]
            indicators = result.get("indicators", {})
            quote_array = indicators.get("quote")
            if not quote_array:
                data.incomplete = True
                return

            close_array = quote_array[0].get("close")
            if not close_array or len(close_array) < 2:
                data.incomplete = True
                return

            self._parse_patterns(close_array, data)
        except (json.JSONDecodeError, KeyError, TypeError, IndexError, ValueError) as exc:
            logger.warning(
                "%s has incomplete data from Yahoo chart API: %s",
                data.ticker,
                exc,
            )
            data.incomplete = True

    @staticmethod
    def _parse_patterns(close_array: list, data: HistoricalData) -> None:
        up_down_list: list[int] = []
        previous_close = Decimal(0)

        limit = min(len(close_array), _PATTERN_DAYS + 1)

        for i in range(limit):
            val = close_array[i]
            if val is None:
                continue

            close_price = Decimal(str(val))

            if i > 0 and previous_close != Decimal(0):
                close_change = (
                    (close_price - previous_close)
                    / previous_close
                    * Decimal(100)
                ).quantize(Decimal("1"), rounding=ROUND_CEILING)

                if close_change > 0:
                    up_down_list.append(1)
                elif close_change < 0:
                    up_down_list.append(-1)
                else:
                    up_down_list.append(0)

                data.patterns[str(up_down_list)] = data.ticker

            previous_close = close_price
