"""Downloads and parses fundamental financial data using Yahoo Finance
quoteSummary API, returning a populated FinancialData model.

Replaces the deprecated Morningstar ReportProcess4CSV endpoint
(financials.morningstar.com) which is no longer functional.
Now uses Yahoo Finance's incomeStatementHistory and defaultKeyStatistics
modules to obtain revenue and shares outstanding data.
"""
from __future__ import annotations

import json
import logging

from stockdownloader.data.yahoo_auth_helper import YahooAuthHelper
from stockdownloader.model import FinancialData

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_QUOTE_SUMMARY_URL = (
    "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    "?modules=incomeStatementHistory,incomeStatementHistoryQuarterly,"
    "defaultKeyStatistics&crumb={crumb}"
)


class MorningstarClient:
    """Fetches fundamental financial data via the Yahoo Finance quoteSummary
    API (replaces the defunct Morningstar CSV endpoint).
    """

    def __init__(self, auth: YahooAuthHelper | None = None) -> None:
        self._auth = auth or YahooAuthHelper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, ticker: str) -> FinancialData:
        """Download financial data for *ticker* and return a
        :class:`FinancialData` instance.
        """
        data = FinancialData()

        if self._auth.crumb is None:
            self._auth.authenticate()

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                url = _QUOTE_SUMMARY_URL.format(
                    symbol=ticker, crumb=self._auth.crumb
                )
                resp = self._auth.session.get(url, timeout=15)
                self._parse_data(resp.text, data)
                data.compute_revenue_per_share()
                return data
            except (ArithmeticError, TypeError) as exc:
                logger.warning(
                    "%s has incomplete data from Yahoo Finance.", ticker
                )
                data.incomplete = True
                return data
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.debug(
                        "Retrying financial data download for %s, attempt %d",
                        ticker,
                        attempt + 1,
                    )
                else:
                    logger.warning(
                        "Failed financial data download for %s after %d retries: %s",
                        ticker,
                        _MAX_RETRIES,
                        last_exc,
                    )

        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_data(self, raw: str, data: FinancialData) -> None:
        root = json.loads(raw)
        quote_summary = root.get("quoteSummary")

        if quote_summary is None:
            data.incomplete = True
            return

        results = quote_summary.get("result")
        if not results:
            data.incomplete = True
            return

        result = results[0]

        # Parse quarterly income statements for revenue
        self._parse_quarterly_income(result, data)

        # Parse annual income statements for additional revenue data
        self._parse_annual_income(result, data)

        # Parse shares outstanding from defaultKeyStatistics
        self._parse_key_statistics(result, data)

    @staticmethod
    def _parse_quarterly_income(result: dict, data: FinancialData) -> None:
        quarterly = result.get("incomeStatementHistoryQuarterly")
        if quarterly is None:
            return

        statements = quarterly.get("incomeStatementHistory")
        if statements is None:
            return

        # Yahoo returns most recent quarters first
        count = min(len(statements), 5)
        for i in range(count):
            stmt = statements[i]

            revenue = _get_raw_long(stmt, "totalRevenue")
            data.set_revenue(i, revenue)

            end_date = _get_string(stmt, "endDate")
            if end_date:
                data.set_fiscal_quarter(i, end_date)

    @staticmethod
    def _parse_annual_income(result: dict, data: FinancialData) -> None:
        annual = result.get("incomeStatementHistory")
        if annual is None:
            return

        statements = annual.get("incomeStatementHistory")
        if not statements:
            return

        # Use most recent annual as TTM approximation (index 5)
        latest_annual = statements[0]
        annual_revenue = _get_raw_long(latest_annual, "totalRevenue")
        if annual_revenue > 0:
            data.set_revenue(5, annual_revenue)

    @staticmethod
    def _parse_key_statistics(result: dict, data: FinancialData) -> None:
        stats = result.get("defaultKeyStatistics")
        if stats is None:
            return

        shares_outstanding = _get_raw_long(stats, "sharesOutstanding")
        float_shares = _get_raw_long(stats, "floatShares")

        # Use shares outstanding as basic, float as diluted approximation
        for i in range(6):
            if shares_outstanding > 0:
                data.set_basic_shares(i, shares_outstanding)
            if float_shares > 0:
                data.set_diluted_shares(i, float_shares)
            elif shares_outstanding > 0:
                data.set_diluted_shares(i, shares_outstanding)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _get_raw_long(obj: dict, field: str) -> int:
    """Extract a long integer from a Yahoo Finance JSON field.

    Yahoo wraps numeric values as ``{"raw": 123, "fmt": "123"}``.
    This helper handles both wrapped and direct numeric values.
    """
    val = obj.get(field)
    if val is None:
        return 0

    # Yahoo Finance wraps numeric values in {"raw": ..., "fmt": ...}
    if isinstance(val, dict):
        raw = val.get("raw")
        if raw is not None:
            try:
                return int(raw)
            except (ValueError, TypeError):
                return 0

    # Direct numeric value
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _get_string(obj: dict, field: str) -> str:
    """Extract a string from a Yahoo Finance JSON field.

    Handles the ``{"raw": ..., "fmt": "..."}`` wrapper format.
    """
    val = obj.get(field)
    if val is None:
        return ""

    if isinstance(val, dict):
        fmt = val.get("fmt")
        if fmt is not None:
            return str(fmt)

    return str(val)
