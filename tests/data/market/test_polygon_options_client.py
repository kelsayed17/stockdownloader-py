"""Tests for the Polygon options data client."""
from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
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
