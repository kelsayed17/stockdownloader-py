"""Tests for PolygonDataClient.

Uses mocked HTTP responses so tests run offline without an API key.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.polygon_data_client import (
    PolygonDataClient,
    _parse_results,
)
from stockdownloader.model.price_data import IntradayPriceData


# ---------------------------------------------------------------
# Sample Polygon API response
# ---------------------------------------------------------------

# 2025-01-15 09:30:00 EST = Unix ms 1736951400000
_SAMPLE_RESULTS = [
    {
        "t": 1736951400000,  # 2025-01-15 09:30:00 EST
        "o": 590.50,
        "h": 591.00,
        "l": 590.00,
        "c": 590.75,
        "v": 1500000,
        "vw": 590.60,
        "n": 5000,
    },
    {
        "t": 1736951700000,  # 2025-01-15 09:35:00 EST
        "o": 590.75,
        "h": 591.50,
        "l": 590.25,
        "c": 591.00,
        "v": 1200000,
        "vw": 590.80,
        "n": 4500,
    },
    {
        "t": 1736928000000,  # Pre-market: 2025-01-15 03:00:00 EST
        "o": 589.00,
        "h": 589.50,
        "l": 588.50,
        "c": 589.25,
        "v": 50000,
        "vw": 589.10,
        "n": 200,
    },
    {
        "t": 1736974800000,  # After-hours: 2025-01-15 16:00:00 EST
        "o": 591.00,
        "h": 591.25,
        "l": 590.50,
        "c": 590.75,
        "v": 80000,
        "vw": 590.90,
        "n": 300,
    },
]


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


class TestParseResults:

    def test_parse_valid_bars(self):
        bars = _parse_results(_SAMPLE_RESULTS)
        # Should filter out pre-market and after-hours
        assert len(bars) == 2

    def test_parse_bar_fields(self):
        bars = _parse_results(_SAMPLE_RESULTS[:1])
        assert len(bars) == 1
        bar = bars[0]
        assert isinstance(bar, IntradayPriceData)
        assert bar.open == Decimal("590.50")
        assert bar.high == Decimal("591.00")
        assert bar.low == Decimal("590.00")
        assert bar.close == Decimal("590.75")
        assert bar.volume == 1500000

    def test_parse_datetime_format(self):
        bars = _parse_results(_SAMPLE_RESULTS[:1])
        bar = bars[0]
        # Should have ISO format with timezone
        assert "2025-01-15" in bar.date
        assert "09:30:00" in bar.date
        assert "-05:00" in bar.date

    def test_parse_filters_premarket(self):
        # Only pre-market bar
        bars = _parse_results([_SAMPLE_RESULTS[2]])
        assert len(bars) == 0

    def test_parse_filters_afterhours(self):
        # Only after-hours bar
        bars = _parse_results([_SAMPLE_RESULTS[3]])
        assert len(bars) == 0

    def test_parse_empty_results(self):
        assert _parse_results([]) == []

    def test_parse_skips_malformed(self):
        bad = [{"t": 1736951400000}]  # missing o, h, l, c
        bars = _parse_results(bad)
        assert len(bars) == 0

    def test_adj_close_equals_close(self):
        bars = _parse_results(_SAMPLE_RESULTS[:1])
        assert bars[0].adj_close == bars[0].close


class TestPolygonDataClientInit:

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="API key required"):
            PolygonDataClient(api_key="")

    def test_accepts_api_key(self):
        client = PolygonDataClient(api_key="test_key_123")
        assert client._api_key == "test_key_123"

    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "env_key_456")
        client = PolygonDataClient()
        assert client._api_key == "env_key_456"


class TestFetchIntradayData:

    def test_fetch_returns_bars(self):
        client = PolygonDataClient(api_key="test_key")
        mock_resp = _mock_response({
            "status": "OK",
            "results": _SAMPLE_RESULTS[:2],
            "resultsCount": 2,
            "queryCount": 2,
        })

        with patch.object(client._session, "get", return_value=mock_resp):
            bars = client.fetch_intraday_data("SPY", "2025-01-15", "2025-01-15")

        assert len(bars) == 2
        assert bars[0].close == Decimal("590.75")

    def test_fetch_handles_empty_results(self):
        client = PolygonDataClient(api_key="test_key")
        mock_resp = _mock_response({
            "status": "OK",
            "results": [],
            "resultsCount": 0,
        })

        with patch.object(client._session, "get", return_value=mock_resp):
            bars = client.fetch_intraday_data("SPY", "2020-01-01", "2020-01-02")

        assert bars == []

    def test_fetch_handles_api_error(self):
        client = PolygonDataClient(api_key="test_key")
        mock_resp = _mock_response({
            "status": "ERROR",
            "error": "Not authorized",
        })

        with patch.object(client._session, "get", return_value=mock_resp):
            bars = client.fetch_intraday_data("SPY", "2025-01-15", "2025-01-15")

        assert bars == []

    def test_fetch_handles_network_error(self):
        import requests as req
        client = PolygonDataClient(api_key="test_key")

        with patch.object(
            client._session, "get",
            side_effect=req.ConnectionError("Network error"),
        ):
            bars = client.fetch_intraday_data("SPY", "2025-01-15", "2025-01-15")

        assert bars == []


class TestFetchIntradayHistory:

    def test_history_chunks_date_range(self):
        client = PolygonDataClient(api_key="test_key", rate_limit_delay=0)
        mock_resp = _mock_response({
            "status": "OK",
            "results": _SAMPLE_RESULTS[:2],
            "resultsCount": 2,
        })

        mock_get = MagicMock(return_value=mock_resp)
        client._session.get = mock_get
        bars = client.fetch_intraday_history("SPY", total_days=90)

        # Should have made multiple API calls (90 days / 30 day chunks = 3)
        assert mock_get.call_count == 3
        # Bars should be deduplicated (same timestamps returned each call)
        assert len(bars) == 2

    def test_history_returns_sorted(self):
        client = PolygonDataClient(api_key="test_key", rate_limit_delay=0)
        mock_resp = _mock_response({
            "status": "OK",
            "results": _SAMPLE_RESULTS[:2],
            "resultsCount": 2,
        })

        with patch.object(client._session, "get", return_value=mock_resp):
            bars = client.fetch_intraday_history("SPY", total_days=30)

        for i in range(1, len(bars)):
            assert bars[i].date >= bars[i - 1].date

    def test_history_handles_empty_chunks(self):
        client = PolygonDataClient(api_key="test_key", rate_limit_delay=0)
        mock_resp = _mock_response({
            "status": "OK",
            "results": [],
            "resultsCount": 0,
        })

        with patch.object(client._session, "get", return_value=mock_resp):
            bars = client.fetch_intraday_history("SPY", total_days=60)

        assert bars == []
