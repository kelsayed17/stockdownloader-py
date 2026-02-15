"""Downloads real-time stock quote data from Yahoo Finance v7 quote JSON API
and returns a populated QuoteData model.

Replaces the deprecated download.finance.yahoo.com/d/quotes.csv endpoint
which was shut down in 2017.
"""
from __future__ import annotations

import json
import logging

import requests

from stockdownloader.data.json_helpers import (
    format_market_cap,
    get_decimal,
    get_long,
    get_string,
)
from stockdownloader.data.yahoo_auth_helper import YahooAuthHelper
from stockdownloader.model import QuoteData

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"


class YahooFinanceClient:
    """HTTP client that fetches real-time quote data from Yahoo Finance."""

    def __init__(self, auth: YahooAuthHelper | None = None) -> None:
        self._auth = auth or YahooAuthHelper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, ticker: str) -> QuoteData:
        """Download real-time quote data for *ticker* and return a
        :class:`QuoteData` instance.
        """
        data = QuoteData()

        if self._auth.crumb is None:
            self._auth.authenticate()

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                url = (
                    _QUOTE_URL.format(symbol=ticker)
                    + f"&crumb={self._auth.crumb}"
                )
                resp = self._auth.session.get(url, timeout=15)
                self._parse_quote_json(resp.text, data)
                return data
            except (requests.RequestException, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.debug(
                        "Retrying Yahoo Finance download for %s, attempt %d",
                        ticker,
                        attempt + 1,
                    )
                else:
                    logger.warning(
                        "Failed Yahoo Finance download for %s after %d retries: %s",
                        ticker,
                        _MAX_RETRIES,
                        last_exc,
                    )

        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_quote_json(self, raw: str, data: QuoteData) -> None:
        try:
            root = json.loads(raw)
            quote_response = root.get("quoteResponse")

            if not quote_response or not quote_response.get("result"):
                logger.warning("Empty quote response from Yahoo Finance")
                data.incomplete = True
                return

            quote = quote_response["result"][0]

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

            data.year_range = get_string(quote, "fiftyTwoWeekRange")

            market_cap = get_long(quote, "marketCap")
            data.market_capitalization = market_cap
            data.market_capitalization_str = format_market_cap(market_cap)

            if data.last_trade_price_only < data.year_low:
                data.year_low = data.last_trade_price_only
        except (json.JSONDecodeError, KeyError, TypeError, IndexError, ValueError) as exc:
            logger.warning(
                "Error parsing Yahoo Finance quote JSON: %s", exc
            )
            data.incomplete = True
