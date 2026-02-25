"""Tests for the Polygon options data client."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient


_CONTRACTS_RESPONSE = {
    "status": "OK",
    "request_id": "test",
    "results": [
        {
            "ticker": "O:SPY250207C00590000",
            "underlying_ticker": "SPY",
            "contract_type": "call",
            "exercise_style": "american",
            "expiration_date": "2025-02-07",
            "strike_price": 590.0,
            "shares_per_contract": 100,
        },
        {
            "ticker": "O:SPY250207C00600000",
            "underlying_ticker": "SPY",
            "contract_type": "call",
            "exercise_style": "american",
            "expiration_date": "2025-02-07",
            "strike_price": 600.0,
            "shares_per_contract": 100,
        },
        {
            "ticker": "O:SPY250207P00590000",
            "underlying_ticker": "SPY",
            "contract_type": "put",
            "exercise_style": "american",
            "expiration_date": "2025-02-07",
            "strike_price": 590.0,
            "shares_per_contract": 100,
        },
        {
            "ticker": "O:SPY250207P00580000",
            "underlying_ticker": "SPY",
            "contract_type": "put",
            "exercise_style": "american",
            "expiration_date": "2025-02-07",
            "strike_price": 580.0,
            "shares_per_contract": 100,
        },
    ],
    "next_url": None,
}

_BAR_RESPONSE = {
    "status": "OK",
    "ticker": "O:SPY250207P00580000",
    "resultsCount": 1,
    "results": [
        {"o": 3.50, "h": 3.80, "l": 3.20, "c": 3.55, "v": 1500, "t": 1738886400000}
    ],
}


class TestFetchOptionContracts:

    def test_returns_contracts_for_expiration(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _CONTRACTS_RESPONSE
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        contracts = client.fetch_option_contracts(
            "SPY", from_date=date(2025, 2, 1), to_date=date(2025, 2, 28),
        )
        assert len(contracts) == 4
        assert contracts[0]["ticker"] == "O:SPY250207C00590000"
        assert contracts[0]["strike_price"] == 590.0
        assert contracts[0]["contract_type"] == "call"

    def test_handles_pagination(self):
        page1 = {
            "status": "OK",
            "results": _CONTRACTS_RESPONSE["results"][:2],
            "next_url": "https://api.polygon.io/v3/reference/options/contracts?cursor=abc",
        }
        page2 = {
            "status": "OK",
            "results": _CONTRACTS_RESPONSE["results"][2:],
            "next_url": None,
        }

        session = MagicMock()
        resp1 = MagicMock()
        resp1.json.return_value = page1
        resp1.raise_for_status = MagicMock()
        resp2 = MagicMock()
        resp2.json.return_value = page2
        resp2.raise_for_status = MagicMock()
        session.get.side_effect = [resp1, resp2]

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        contracts = client.fetch_option_contracts(
            "SPY", from_date=date(2025, 2, 1), to_date=date(2025, 2, 28),
        )
        assert len(contracts) == 4
        assert session.get.call_count == 2

    def test_empty_results(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"status": "OK", "results": [], "next_url": None}
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        contracts = client.fetch_option_contracts(
            "SPY", from_date=date(2025, 2, 1), to_date=date(2025, 2, 28),
        )
        assert contracts == []

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                PolygonOptionsClient(api_key="")

    def test_filters_to_friday_expirations(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _CONTRACTS_RESPONSE
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        contracts = client.fetch_option_contracts(
            "SPY", from_date=date(2025, 2, 1), to_date=date(2025, 2, 28),
            fridays_only=True,
        )
        # 2025-02-07 is a Friday
        assert len(contracts) == 4


class TestFetchOptionDailyBar:
    def test_returns_bar_data(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _BAR_RESPONSE
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        bar = client.fetch_option_daily_bar("O:SPY250207P00580000", date(2025, 2, 3))
        assert bar is not None
        assert bar["c"] == 3.55
        assert bar["v"] == 1500

    def test_returns_none_for_no_data(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"status": "OK", "results": []}
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        bar = client.fetch_option_daily_bar("O:SPY250207P00580000", date(2025, 2, 3))
        assert bar is None


class TestChainCache:
    def test_save_and_load_cache(self, tmp_path):
        from stockdownloader.data.market.polygon_options_client import (
            save_chain_cache,
            load_chain_cache,
        )

        cache_data = {
            "expiration_date": "2025-02-07",
            "contracts": _CONTRACTS_RESPONSE["results"],
            "bars": {"O:SPY250207P00580000": _BAR_RESPONSE["results"][0]},
        }
        save_chain_cache(tmp_path, "2025-02-07", cache_data)
        loaded = load_chain_cache(tmp_path, "2025-02-07")

        assert loaded is not None
        assert loaded["expiration_date"] == "2025-02-07"
        assert len(loaded["contracts"]) == 4

    def test_load_returns_none_for_missing(self, tmp_path):
        from stockdownloader.data.market.polygon_options_client import load_chain_cache
        loaded = load_chain_cache(tmp_path, "2025-02-07")
        assert loaded is None
