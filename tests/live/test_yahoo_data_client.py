"""Live tests for YahooDataClient (v8 chart API, historical OHLCV)."""
from __future__ import annotations

import re
from decimal import Decimal

import pytest

from stockdownloader.data.yahoo_data_client import YahooDataClient
from stockdownloader.model.price_data import PriceData

pytestmark = pytest.mark.live

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@pytest.fixture(scope="module")
def price_data_1y(yahoo_auth) -> list[PriceData]:
    client = YahooDataClient(auth=yahoo_auth)
    return client.fetch_price_data("SPY", range_="1y", interval="1d")


@pytest.fixture(scope="module")
def price_data_1mo(yahoo_auth) -> list[PriceData]:
    client = YahooDataClient(auth=yahoo_auth)
    return client.fetch_price_data("SPY", range_="1mo", interval="1d")


class TestYahooDataClient:

    def test_returns_non_empty_list(self, price_data_1y):
        assert len(price_data_1y) > 0

    def test_1y_returns_roughly_250_trading_days(self, price_data_1y):
        assert 180 < len(price_data_1y) < 280

    def test_1mo_returns_roughly_20_trading_days(self, price_data_1mo):
        assert 10 < len(price_data_1mo) < 30

    def test_all_elements_are_price_data(self, price_data_1y):
        for item in price_data_1y:
            assert isinstance(item, PriceData)

    def test_ohlcv_values_are_positive(self, price_data_1y):
        for bar in price_data_1y:
            assert bar.open > Decimal("0"), f"Zero open on {bar.date}"
            assert bar.high > Decimal("0"), f"Zero high on {bar.date}"
            assert bar.low > Decimal("0"), f"Zero low on {bar.date}"
            assert bar.close > Decimal("0"), f"Zero close on {bar.date}"
            assert bar.volume >= 0, f"Negative volume on {bar.date}"

    def test_high_gte_low_for_every_bar(self, price_data_1y):
        for bar in price_data_1y:
            assert bar.high >= bar.low, f"high < low on {bar.date}"

    def test_dates_are_in_ascending_order(self, price_data_1y):
        dates = [bar.date for bar in price_data_1y]
        assert dates == sorted(dates)

    def test_date_format_is_yyyy_mm_dd(self, price_data_1y):
        for bar in price_data_1y[:5]:
            assert _DATE_PATTERN.match(bar.date), f"Bad date format: {bar.date}"
