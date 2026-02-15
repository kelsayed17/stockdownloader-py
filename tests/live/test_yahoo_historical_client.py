"""Live tests for YahooHistoricalClient (1-month patterns)."""
from __future__ import annotations

import pytest

from stockdownloader.data.yahoo_historical_client import YahooHistoricalClient
from stockdownloader.model.unified_market_data import HistoricalData

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def historical(yahoo_auth) -> HistoricalData:
    client = YahooHistoricalClient(auth=yahoo_auth)
    return client.download("SPY")


class TestYahooHistoricalClient:

    def test_returns_historical_data_instance(self, historical):
        assert isinstance(historical, HistoricalData)

    def test_ticker_is_spy(self, historical):
        assert historical.ticker == "SPY"

    def test_not_marked_incomplete(self, historical):
        assert historical.incomplete is False

    def test_patterns_dict_is_non_empty(self, historical):
        assert len(historical.patterns) > 0

    def test_pattern_keys_are_list_like_strings(self, historical):
        for key in historical.patterns:
            assert key.startswith("["), f"Unexpected pattern key: {key}"
            assert key.endswith("]"), f"Unexpected pattern key: {key}"
