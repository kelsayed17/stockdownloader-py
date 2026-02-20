"""Downloads real-time stock quote data from Yahoo Finance v7 quote JSON API
and returns a populated QuoteData model.

Replaces the deprecated download.finance.yahoo.com/d/quotes.csv endpoint
which was shut down in 2017.
"""
from __future__ import annotations

import json
import logging
from typing import Sequence

import requests

from stockdownloader.data.data_parsers import (
    format_market_cap,
    get_decimal,
    get_long,
    get_string,
)
from stockdownloader.data.yahoo_base_client import YahooAuthHelper, YahooBaseClient
from stockdownloader.model import QuoteData

logger = logging.getLogger(__name__)
_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
_DEFAULT_BATCH_SIZE = 100


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
