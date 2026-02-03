"""Downloads historical price data from Yahoo Finance v8 chart API
and computes price movement patterns.

Replaces the deprecated Google Finance historical CSV endpoint
(www.google.com/finance/historical) which was shut down around 2015.
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal, ROUND_CEILING

from stockdownloader.data.yahoo_auth_helper import YahooAuthHelper
from stockdownloader.model import HistoricalData

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_PATTERN_DAYS = 7
_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?range=1mo&interval=1d"
)


class YahooHistoricalClient:
    """Fetches historical price data and computes movement patterns."""

    def __init__(self, auth: YahooAuthHelper | None = None) -> None:
        self._auth = auth or YahooAuthHelper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, ticker: str) -> HistoricalData:
        """Download one-month daily close prices for *ticker* and derive
        up/down patterns.
        """
        data = HistoricalData(ticker)

        if self._auth.crumb is None:
            self._auth.authenticate()

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                url = (
                    _CHART_URL.format(symbol=ticker)
                    + f"&crumb={self._auth.crumb}"
                )
                resp = self._auth.session.get(url, timeout=15)
                self._parse_chart_json(resp.text, data)
                return data
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.debug(
                        "Retrying historical download for %s, attempt %d",
                        ticker,
                        attempt + 1,
                    )
                else:
                    logger.warning(
                        "Failed historical download for %s after %d retries: %s",
                        ticker,
                        _MAX_RETRIES,
                        last_exc,
                    )

        return data

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
        except Exception as exc:
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
