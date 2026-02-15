"""Live tests for YahooFinanceClient (v7 quote API)."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.data.yahoo_finance_client import YahooFinanceClient
from stockdownloader.model.quote_data import QuoteData

pytestmark = pytest.mark.live

_MIN_PRICE = Decimal("50")
_MAX_PRICE = Decimal("1500")


@pytest.fixture(scope="module")
def quote(yahoo_auth) -> QuoteData:
    client = YahooFinanceClient(auth=yahoo_auth)
    return client.download("SPY")


class TestYahooFinanceClient:

    def test_returns_quote_data_instance(self, quote):
        assert isinstance(quote, QuoteData)

    def test_not_marked_incomplete(self, quote):
        assert quote.incomplete is False

    def test_last_trade_price_is_positive_and_reasonable(self, quote):
        assert quote.last_trade_price_only > _MIN_PRICE
        assert quote.last_trade_price_only < _MAX_PRICE

    def test_year_high_gte_year_low(self, quote):
        assert quote.year_high >= quote.year_low
        assert quote.year_low > Decimal("0")

    def test_moving_averages_are_positive(self, quote):
        assert quote.fifty_day_moving_average > Decimal("0")
        assert quote.two_hundred_day_moving_average > Decimal("0")

    def test_volume_is_positive(self, quote):
        assert quote.volume > Decimal("0")

    def test_market_cap_is_large(self, quote):
        assert quote.market_capitalization > 100_000_000_000

    def test_market_cap_string_not_empty(self, quote):
        assert len(quote.market_capitalization_str) > 0

    def test_year_range_string_contains_dash(self, quote):
        assert " - " in quote.year_range
