"""Tests for Polygon snapshot collector and OI enrichment."""
from __future__ import annotations

import pytest
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient


def _make_snapshot_result(
    ticker: str = "O:GME260227C00025000",
    strike: float = 25.0,
    contract_type: str = "call",
    expiration: str = "2026-02-27",
    close: float = 0.04,
    volume: int = 9493,
    open_interest: int = 8921,
    iv: float = 0.57,
    delta: float = 0.11,
    gamma: float = 0.24,
    theta: float = -0.06,
    vega: float = 0.003,
    underlying_price: float = 23.975,
) -> dict:
    """Build a snapshot result dict matching Polygon's response shape."""
    return {
        "details": {
            "ticker": ticker,
            "strike_price": strike,
            "contract_type": contract_type,
            "expiration_date": expiration,
        },
        "day": {"close": close, "volume": volume},
        "open_interest": open_interest,
        "implied_volatility": iv,
        "greeks": {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
        },
        "underlying_asset": {"price": underlying_price},
    }


class TestFetchOptionsChainSnapshot:
    def test_returns_all_results_single_page(self):
        """Single page of results with no next_url."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                _make_snapshot_result("O:GME260227C00025000", open_interest=100),
                _make_snapshot_result("O:GME260227P00025000", contract_type="put", open_interest=200),
            ],
            "status": "OK",
        }
        mock_resp.raise_for_status = MagicMock()

        client = PolygonOptionsClient.__new__(PolygonOptionsClient)
        client._delay = 0.0
        client._session = MagicMock()
        client._session.get.return_value = mock_resp

        results = client.fetch_options_chain_snapshot("GME")
        assert len(results) == 2
        assert results[0]["open_interest"] == 100
        assert results[1]["open_interest"] == 200

    def test_paginates_through_next_url(self):
        """Follows next_url to collect all pages."""
        page1_resp = MagicMock()
        page1_resp.json.return_value = {
            "results": [_make_snapshot_result(open_interest=100)],
            "next_url": "https://api.polygon.io/v3/snapshot/options/GME?cursor=abc&apiKey=SECRET",
            "status": "OK",
        }
        page1_resp.raise_for_status = MagicMock()

        page2_resp = MagicMock()
        page2_resp.json.return_value = {
            "results": [_make_snapshot_result(open_interest=200)],
            "status": "OK",
        }
        page2_resp.raise_for_status = MagicMock()

        client = PolygonOptionsClient.__new__(PolygonOptionsClient)
        client._delay = 0.0
        client._session = MagicMock()
        client._session.get.side_effect = [page1_resp, page2_resp]

        results = client.fetch_options_chain_snapshot("GME")
        assert len(results) == 2
        assert results[0]["open_interest"] == 100
        assert results[1]["open_interest"] == 200

        # Verify apiKey was stripped from the next_url
        second_call_url = client._session.get.call_args_list[1][0][0]
        assert "apiKey" not in second_call_url
        assert "SECRET" not in second_call_url

    def test_handles_request_failure_gracefully(self):
        """Returns empty list on network failure."""
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")

        client = PolygonOptionsClient.__new__(PolygonOptionsClient)
        client._delay = 0.0
        client._session = MagicMock()
        client._session.get.return_value = mock_resp

        results = client.fetch_options_chain_snapshot("GME")
        assert results == []

    def test_handles_missing_greeks(self):
        """Contracts with empty greeks are still included."""
        result = _make_snapshot_result()
        result["greeks"] = {}
        result.pop("implied_volatility", None)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [result], "status": "OK"}
        mock_resp.raise_for_status = MagicMock()

        client = PolygonOptionsClient.__new__(PolygonOptionsClient)
        client._delay = 0.0
        client._session = MagicMock()
        client._session.get.return_value = mock_resp

        results = client.fetch_options_chain_snapshot("GME")
        assert len(results) == 1
        assert results[0]["open_interest"] == 8921


from stockdownloader.gme.options.snapshot import SnapshotCollector
from stockdownloader.gme.options.config import GMEOptionsConfig


class TestSnapshotCollector:
    def test_collect_produces_dataframe(self, tmp_path):
        """collect() returns a DataFrame with correct schema."""
        mock_client = MagicMock()
        mock_client.fetch_options_chain_snapshot.return_value = [
            _make_snapshot_result("O:GME260227C00025000", open_interest=100, volume=500),
            _make_snapshot_result("O:GME260227P00020000", contract_type="put",
                                  strike=20.0, open_interest=200, volume=300),
        ]

        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, mock_client)
        df = collector.collect()

        assert len(df) == 2
        required_cols = {
            "date", "option_ticker", "strike", "option_type", "expiration",
            "close", "volume", "open_interest", "implied_volatility",
            "delta", "gamma", "theta", "vega", "underlying_price",
        }
        assert required_cols.issubset(set(df.columns))
        assert df.iloc[0]["open_interest"] == 100
        assert df.iloc[1]["option_type"] == "put"

    def test_collect_handles_missing_greeks(self, tmp_path):
        """Contracts with empty greeks get NaN for greek columns."""
        import math

        result = _make_snapshot_result()
        result["greeks"] = {}
        result.pop("implied_volatility", None)

        mock_client = MagicMock()
        mock_client.fetch_options_chain_snapshot.return_value = [result]

        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, mock_client)
        df = collector.collect()

        assert len(df) == 1
        assert math.isnan(df.iloc[0]["delta"])
        assert math.isnan(df.iloc[0]["implied_volatility"])

    def test_save_and_load(self, tmp_path):
        """save() writes Parquet, load() reads it back."""
        import pandas as pd

        mock_client = MagicMock()
        mock_client.fetch_options_chain_snapshot.return_value = [
            _make_snapshot_result(open_interest=42),
        ]

        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, mock_client)
        df = collector.collect()

        path = collector.save(df, date(2026, 2, 26))
        assert path.exists()
        assert "2026-02-26" in path.name

        loaded = collector.load(date(2026, 2, 26))
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded.iloc[0]["open_interest"] == 42

    def test_load_returns_none_when_missing(self, tmp_path):
        """load() returns None for dates with no snapshot."""
        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, MagicMock())
        assert collector.load(date(1999, 1, 1)) is None

    def test_run_collects_and_saves(self, tmp_path):
        """run() collects and saves in one call."""
        mock_client = MagicMock()
        mock_client.fetch_options_chain_snapshot.return_value = [
            _make_snapshot_result(open_interest=77),
        ]

        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, mock_client)
        df = collector.run()

        assert len(df) == 1
        # Verify file was saved
        snapshots_dir = tmp_path / "snapshots"
        assert any(snapshots_dir.glob("*.parquet"))
