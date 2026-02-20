"""Unit tests for MorningstarClient — download_detailed and parsing."""
from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock

from stockdownloader.data.morningstar_client import MorningstarClient


def _mock_auth() -> MagicMock:
    auth = MagicMock()
    auth.crumb = "test_crumb"
    auth.session = MagicMock()
    return auth


_DETAILED_RESPONSE = {
    "quoteSummary": {
        "result": [
            {
                "financialData": {
                    "profitMargins": {"raw": 0.2576, "fmt": "25.76%"},
                    "operatingMargins": {"raw": 0.30, "fmt": "30.00%"},
                    "returnOnEquity": {"raw": 0.175, "fmt": "17.50%"},
                    "returnOnAssets": {"raw": 0.21, "fmt": "21.00%"},
                    "debtToEquity": {"raw": 1.73, "fmt": "173.00"},
                    "currentRatio": {"raw": 0.98, "fmt": "0.98"},
                    "revenueGrowth": {"raw": 0.049, "fmt": "4.90%"},
                    "earningsGrowth": {"raw": 0.109, "fmt": "10.90%"},
                    "freeCashflow": {"raw": 111443000000, "fmt": "111.44B"},
                    "operatingCashflow": {"raw": 122151000000, "fmt": "122.15B"},
                    "totalDebt": {"raw": 108040000000, "fmt": "108.04B"},
                    "ebitda": {"raw": 137000000000, "fmt": "137.00B"},
                    "totalCash": {"raw": 62000000000, "fmt": "62.00B"},
                    "grossMargins": {"raw": 0.4618, "fmt": "46.18%"},
                    "grossProfits": {"raw": 176000000000, "fmt": "176.00B"},
                    "revenuePerShare": {"raw": 25.15, "fmt": "25.15"},
                    "targetMeanPrice": {"raw": 235.50, "fmt": "235.50"},
                },
                "earningsTrend": {
                    "trend": [
                        {
                            "period": "0q",
                            "earningsEstimate": {"avg": {"raw": 1.60, "fmt": "1.60"}},
                        },
                        {
                            "period": "+1q",
                            "earningsEstimate": {"avg": {"raw": 2.36, "fmt": "2.36"}},
                        },
                        {
                            "period": "0y",
                            "earningsEstimate": {"avg": {"raw": 7.10, "fmt": "7.10"}},
                        },
                        {
                            "period": "+1y",
                            "earningsEstimate": {"avg": {"raw": 7.70, "fmt": "7.70"}},
                        },
                    ]
                },
                "balanceSheetHistory": {
                    "balanceSheetStatements": [
                        {
                            "totalStockholderEquity": {
                                "raw": 62146000000,
                                "fmt": "62.15B",
                            },
                            "longTermDebt": {"raw": 95281000000, "fmt": "95.28B"},
                            "shortLongTermDebt": {"raw": 12759000000, "fmt": "12.76B"},
                        },
                    ]
                },
                "defaultKeyStatistics": {
                    "enterpriseValue": {"raw": 2800000000000, "fmt": "2.80T"},
                    "enterpriseToRevenue": {"raw": 7.35, "fmt": "7.35"},
                    "enterpriseToEbitda": {"raw": 20.44, "fmt": "20.44"},
                    "pegRatio": {"raw": 1.85, "fmt": "1.85"},
                    "beta": {"raw": 1.24, "fmt": "1.24"},
                    "sharesOutstanding": {"raw": 15204100000, "fmt": "15.20B"},
                    "payoutRatio": {"raw": 0.1546, "fmt": "15.46%"},
                    "forwardEps": {"raw": 7.70, "fmt": "7.70"},
                },
            }
        ]
    }
}


class TestDownloadDetailed:
    """Tests for MorningstarClient.download_detailed."""

    def test_parses_financial_data(self) -> None:
        auth = _mock_auth()
        client = MorningstarClient(auth=auth)

        resp = MagicMock()
        resp.text = json.dumps(_DETAILED_RESPONSE)
        auth.session.get.return_value = resp

        data = client.download_detailed("AAPL")

        assert data.symbol == "AAPL"
        assert data.profit_margin == Decimal("0.2576")
        assert data.operating_margin == Decimal("0.30")
        assert data.return_on_equity == Decimal("0.175")
        assert data.return_on_assets == Decimal("0.21")
        assert data.debt_to_equity == Decimal("1.73")
        assert data.current_ratio == Decimal("0.98")
        assert data.revenue_growth == Decimal("0.049")
        assert data.earnings_growth == Decimal("0.109")
        assert data.free_cash_flow == 111443000000
        assert data.operating_cash_flow == 122151000000
        assert data.total_debt == 108040000000
        assert data.incomplete is False

    def test_parses_new_financial_data_fields(self) -> None:
        auth = _mock_auth()
        client = MorningstarClient(auth=auth)

        resp = MagicMock()
        resp.text = json.dumps(_DETAILED_RESPONSE)
        auth.session.get.return_value = resp

        data = client.download_detailed("AAPL")

        assert data.ebitda == 137000000000
        assert data.total_cash == 62000000000
        assert data.gross_margins == Decimal("0.4618")
        assert data.gross_profits == 176000000000
        assert data.revenue_per_share == Decimal("25.15")
        assert data.target_mean_price == Decimal("235.50")

    def test_parses_earnings_trend(self) -> None:
        auth = _mock_auth()
        client = MorningstarClient(auth=auth)

        resp = MagicMock()
        resp.text = json.dumps(_DETAILED_RESPONSE)
        auth.session.get.return_value = resp

        data = client.download_detailed("AAPL")

        assert data.earnings_estimate_next_qtr == Decimal("2.36")
        assert data.earnings_estimate_next_year == Decimal("7.70")

    def test_parses_balance_sheet(self) -> None:
        auth = _mock_auth()
        client = MorningstarClient(auth=auth)

        resp = MagicMock()
        resp.text = json.dumps(_DETAILED_RESPONSE)
        auth.session.get.return_value = resp

        data = client.download_detailed("AAPL")

        assert data.total_equity == 62146000000

    def test_parses_key_statistics(self) -> None:
        auth = _mock_auth()
        client = MorningstarClient(auth=auth)

        resp = MagicMock()
        resp.text = json.dumps(_DETAILED_RESPONSE)
        auth.session.get.return_value = resp

        data = client.download_detailed("AAPL")

        assert data.enterprise_value == 2800000000000
        assert data.enterprise_to_revenue == Decimal("7.35")
        assert data.enterprise_to_ebitda == Decimal("20.44")
        assert data.peg_ratio == Decimal("1.85")
        assert data.beta == Decimal("1.24")
        assert data.shares_outstanding == 15204100000
        assert data.payout_ratio == Decimal("0.1546")
        assert data.forward_eps == Decimal("7.70")

    def test_handles_missing_modules(self) -> None:
        auth = _mock_auth()
        client = MorningstarClient(auth=auth)

        # Response with empty result
        resp = MagicMock()
        resp.text = json.dumps({"quoteSummary": {"result": [{}]}})
        auth.session.get.return_value = resp

        data = client.download_detailed("UNKNOWN")

        assert data.symbol == "UNKNOWN"
        assert data.profit_margin == Decimal("0")
        assert data.return_on_equity == Decimal("0")
        assert data.free_cash_flow == 0
        assert data.enterprise_value == 0
        assert data.peg_ratio == Decimal("0")
        assert data.beta == Decimal("0")
        assert data.ebitda == 0
        assert data.gross_margins == Decimal("0")
        assert data.incomplete is False  # parsed OK, just no data

    def test_handles_empty_response(self) -> None:
        auth = _mock_auth()
        client = MorningstarClient(auth=auth)

        resp = MagicMock()
        resp.text = json.dumps({"quoteSummary": {"result": []}})
        auth.session.get.return_value = resp

        data = client.download_detailed("EMPTY")

        assert data.incomplete is True
