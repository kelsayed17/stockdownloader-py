"""Live tests for MorningstarClient (Yahoo v10 quoteSummary API).

Uses AAPL instead of SPY because SPY is an ETF with sparse income
statement data, while AAPL has rich quarterly financial data.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.data.morningstar_client import MorningstarClient
from stockdownloader.model.unified_market_data import FinancialData

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def financial_data(yahoo_auth) -> FinancialData:
    client = MorningstarClient(auth=yahoo_auth)
    return client.download("AAPL")


class TestMorningstarClient:

    def test_returns_financial_data_instance(self, financial_data):
        assert isinstance(financial_data, FinancialData)

    def test_not_marked_incomplete(self, financial_data):
        assert financial_data.incomplete is False

    def test_has_revenue_data(self, financial_data):
        assert any(r > 0 for r in financial_data.revenue)

    def test_has_shares_outstanding(self, financial_data):
        assert any(s > 0 for s in financial_data.diluted_shares)

    def test_revenue_per_share_computed(self, financial_data):
        assert any(
            rps > Decimal("0") for rps in financial_data.revenue_per_share
        )
