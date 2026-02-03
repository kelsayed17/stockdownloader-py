"""Downloads real-time stock quote data from Yahoo Finance v7 quote JSON API
and returns a populated QuoteData model.

Replaces the deprecated download.finance.yahoo.com/d/quotes.csv endpoint
which was shut down in 2017.
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal

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
            except Exception as exc:
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

            data.price_sales = _get_decimal(quote, "priceToSalesTrailing12Months")
            data.trailing_annual_dividend_yield = _get_decimal(
                quote, "trailingAnnualDividendYield"
            )
            data.diluted_eps = _get_decimal(quote, "epsTrailingTwelveMonths")
            data.eps_estimate_next_year = _get_decimal(quote, "epsForward")
            data.last_trade_price_only = _get_decimal(quote, "regularMarketPrice")
            data.year_high = _get_decimal(quote, "fiftyTwoWeekHigh")
            data.year_low = _get_decimal(quote, "fiftyTwoWeekLow")
            data.fifty_day_moving_average = _get_decimal(quote, "fiftyDayAverage")
            data.two_hundred_day_moving_average = _get_decimal(
                quote, "twoHundredDayAverage"
            )
            data.previous_close = _get_decimal(quote, "regularMarketPreviousClose")
            data.open = _get_decimal(quote, "regularMarketOpen")
            data.days_high = _get_decimal(quote, "regularMarketDayHigh")
            data.days_low = _get_decimal(quote, "regularMarketDayLow")
            data.volume = _get_decimal(quote, "regularMarketVolume")

            data.year_range = _get_string(quote, "fiftyTwoWeekRange")

            market_cap = _get_long(quote, "marketCap")
            data.market_capitalization = market_cap
            data.market_capitalization_str = _format_market_cap(market_cap)

            if data.last_trade_price_only < data.year_low:
                data.year_low = data.last_trade_price_only
        except Exception as exc:
            logger.warning(
                "Error parsing Yahoo Finance quote JSON: %s", exc
            )
            data.incomplete = True


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _get_decimal(obj: dict, field: str) -> Decimal:
    val = obj.get(field)
    if val is None:
        return Decimal(0)
    try:
        return Decimal(str(val))
    except Exception:
        return Decimal(0)


def _get_long(obj: dict, field: str) -> int:
    val = obj.get(field)
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _get_string(obj: dict, field: str) -> str:
    val = obj.get(field)
    if val is None:
        return ""
    return str(val)


def _format_market_cap(market_cap: int) -> str:
    if market_cap >= 1_000_000_000:
        return f"{market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"{market_cap / 1_000_000:.2f}M"
    return str(market_cap)
