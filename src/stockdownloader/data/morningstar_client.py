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

import requests

from stockdownloader.data.json_helpers import get_raw_decimal, get_raw_long, get_raw_string
from stockdownloader.data.yahoo_base_client import YahooBaseClient
from stockdownloader.data.yahoo_auth_helper import YahooAuthHelper
from stockdownloader.model import DetailedFinancialData, FinancialData

logger = logging.getLogger(__name__)
_QUOTE_SUMMARY_URL = (
    "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    "?modules=incomeStatementHistory,incomeStatementHistoryQuarterly,"
    "defaultKeyStatistics&crumb={crumb}"
)
_DETAILED_QUOTE_SUMMARY_URL = (
    "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    "?modules=incomeStatementHistory,incomeStatementHistoryQuarterly,"
    "defaultKeyStatistics,financialData,earningsTrend,"
    "balanceSheetHistory&crumb={crumb}"
)


class YahooFundamentalsClient(YahooBaseClient):
    """Fetches fundamental financial data via the Yahoo Finance quoteSummary
    API (replaces the defunct Morningstar CSV endpoint).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, ticker: str) -> FinancialData:
        """Download financial data for *ticker* and return a
        :class:`FinancialData` instance.
        """
        data = FinancialData()
        self._ensure_authenticated()

        def _parse(text: str) -> FinancialData:
            try:
                self._parse_data(text, data)
                data.compute_revenue_per_share()
                return data
            except (ArithmeticError, TypeError):
                logger.warning(
                    "%s has incomplete data from Yahoo Finance.", ticker
                )
                data.incomplete = True
                return data

        url = _QUOTE_SUMMARY_URL.format(
            symbol=ticker, crumb=self._auth.crumb
        )
        result = self._fetch_with_retry(
            url, _parse, f"financial data download for {ticker}",
        )
        return result if result is not None else data

    def download_detailed(self, ticker: str) -> DetailedFinancialData:
        """Download expanded financial data for deep value analysis.

        Uses the ``financialData``, ``earningsTrend``, and
        ``balanceSheetHistory`` quoteSummary modules in addition to
        the standard modules.

        Parameters
        ----------
        ticker:
            Stock symbol to fetch.

        Returns
        -------
        Populated :class:`DetailedFinancialData` instance.
        """
        data = DetailedFinancialData(symbol=ticker)
        self._ensure_authenticated()

        def _parse(text: str) -> DetailedFinancialData:
            try:
                self._parse_detailed(text, data)
                return data
            except (ArithmeticError, TypeError):
                logger.warning(
                    "%s has incomplete detailed data from Yahoo Finance.",
                    ticker,
                )
                data.incomplete = True
                return data

        url = _DETAILED_QUOTE_SUMMARY_URL.format(
            symbol=ticker, crumb=self._auth.crumb
        )
        result = self._fetch_with_retry(
            url, _parse, f"detailed download for {ticker}",
        )
        if result is None:
            data.incomplete = True
        return result if result is not None else data

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

            revenue = get_raw_long(stmt, "totalRevenue")
            data.revenue[i] = revenue

            end_date = get_raw_string(stmt, "endDate")
            if end_date:
                data.fiscal_quarters[i] = end_date

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
        annual_revenue = get_raw_long(latest_annual, "totalRevenue")
        if annual_revenue > 0:
            data.revenue[5] = annual_revenue

    @staticmethod
    def _parse_key_statistics(result: dict, data: FinancialData) -> None:
        stats = result.get("defaultKeyStatistics")
        if stats is None:
            return

        shares_outstanding = get_raw_long(stats, "sharesOutstanding")
        float_shares = get_raw_long(stats, "floatShares")

        # Use shares outstanding as basic, float as diluted approximation
        for i in range(6):
            if shares_outstanding > 0:
                data.basic_shares[i] = shares_outstanding
            if float_shares > 0:
                data.diluted_shares[i] = float_shares
            elif shares_outstanding > 0:
                data.diluted_shares[i] = shares_outstanding

    # ------------------------------------------------------------------
    # Detailed financial data parsing
    # ------------------------------------------------------------------

    def _parse_detailed(self, raw: str, data: DetailedFinancialData) -> None:
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

        self._parse_financial_data(result, data)
        self._parse_earnings_trend(result, data)
        self._parse_balance_sheet(result, data)
        self._parse_key_statistics_detailed(result, data)

    @staticmethod
    def _parse_financial_data(
        result: dict, data: DetailedFinancialData
    ) -> None:
        fin = result.get("financialData")
        if fin is None:
            return

        data.profit_margin = get_raw_decimal(fin, "profitMargins")
        data.operating_margin = get_raw_decimal(fin, "operatingMargins")
        data.return_on_equity = get_raw_decimal(fin, "returnOnEquity")
        data.return_on_assets = get_raw_decimal(fin, "returnOnAssets")
        data.debt_to_equity = get_raw_decimal(fin, "debtToEquity")
        data.current_ratio = get_raw_decimal(fin, "currentRatio")
        data.revenue_growth = get_raw_decimal(fin, "revenueGrowth")
        data.earnings_growth = get_raw_decimal(fin, "earningsGrowth")
        data.free_cash_flow = get_raw_long(fin, "freeCashflow")
        data.operating_cash_flow = get_raw_long(fin, "operatingCashflow")
        data.total_debt = get_raw_long(fin, "totalDebt")
        data.ebitda = get_raw_long(fin, "ebitda")
        data.total_cash = get_raw_long(fin, "totalCash")
        data.gross_margins = get_raw_decimal(fin, "grossMargins")
        data.gross_profits = get_raw_long(fin, "grossProfits")
        data.revenue_per_share = get_raw_decimal(fin, "revenuePerShare")
        data.target_mean_price = get_raw_decimal(fin, "targetMeanPrice")

    @staticmethod
    def _parse_earnings_trend(
        result: dict, data: DetailedFinancialData
    ) -> None:
        trend = result.get("earningsTrend")
        if trend is None:
            return

        trends = trend.get("trend")
        if not trends:
            return

        # trends[0] = current quarter, trends[1] = next quarter,
        # trends[2] = current year, trends[3] = next year
        for entry in trends:
            period = entry.get("period", "")
            earnings_est = entry.get("earningsEstimate", {})
            avg_estimate = get_raw_decimal(earnings_est, "avg")

            if period == "+1q":
                data.earnings_estimate_next_qtr = avg_estimate
            elif period == "+1y":
                data.earnings_estimate_next_year = avg_estimate

    @staticmethod
    def _parse_balance_sheet(
        result: dict, data: DetailedFinancialData
    ) -> None:
        bs = result.get("balanceSheetHistory")
        if bs is None:
            return

        statements = bs.get("balanceSheetStatements")
        if not statements:
            return

        # Most recent statement first
        latest = statements[0]
        data.total_equity = get_raw_long(latest, "totalStockholderEquity")

        # total_debt may also come from balance sheet if not set
        if data.total_debt == 0:
            long_term = get_raw_long(latest, "longTermDebt")
            short_term = get_raw_long(latest, "shortLongTermDebt")
            data.total_debt = long_term + short_term

    @staticmethod
    def _parse_key_statistics_detailed(
        result: dict, data: DetailedFinancialData
    ) -> None:
        """Parse ``defaultKeyStatistics`` module for enterprise valuation metrics."""
        stats = result.get("defaultKeyStatistics")
        if stats is None:
            return

        data.enterprise_value = get_raw_long(stats, "enterpriseValue")
        data.enterprise_to_revenue = get_raw_decimal(stats, "enterpriseToRevenue")
        data.enterprise_to_ebitda = get_raw_decimal(stats, "enterpriseToEbitda")
        data.peg_ratio = get_raw_decimal(stats, "pegRatio")
        data.beta = get_raw_decimal(stats, "beta")
        data.shares_outstanding = get_raw_long(stats, "sharesOutstanding")
        data.payout_ratio = get_raw_decimal(stats, "payoutRatio")
        data.forward_eps = get_raw_decimal(stats, "forwardEps")


# Backward-compatible alias — the class was renamed from MorningstarClient
# to YahooFundamentalsClient to reflect that it exclusively uses the Yahoo
# Finance quoteSummary API (the Morningstar endpoint is defunct).
MorningstarClient = YahooFundamentalsClient
