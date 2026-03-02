"""Tests for resumable Polygon options fetcher."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import date

from stockdownloader.gme.options.fetcher import (
    FetchCheckpoint,
    OptionsDataFetcher,
)
from stockdownloader.gme.options.config import GMEOptionsConfig


class TestFetchCheckpoint:
    def test_empty_checkpoint(self, tmp_path):
        cp = FetchCheckpoint(tmp_path / "checkpoint.json")
        assert cp.completed_months == set()
        assert cp.last_contract is None

    def test_mark_month_complete(self, tmp_path):
        cp = FetchCheckpoint(tmp_path / "checkpoint.json")
        cp.mark_month_complete("2022-01")
        assert "2022-01" in cp.completed_months

    def test_persist_and_reload(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        cp = FetchCheckpoint(path)
        cp.mark_month_complete("2022-01")
        cp.mark_month_complete("2022-02")
        cp.save()
        cp2 = FetchCheckpoint(path)
        assert cp2.completed_months == {"2022-01", "2022-02"}

    def test_is_month_done(self, tmp_path):
        cp = FetchCheckpoint(tmp_path / "checkpoint.json")
        cp.mark_month_complete("2022-03")
        assert cp.is_month_done("2022-03") is True
        assert cp.is_month_done("2022-04") is False

    def test_update_last_contract(self, tmp_path):
        cp = FetchCheckpoint(tmp_path / "checkpoint.json")
        cp.last_contract = "O:GME220121C00020000"
        cp.save()
        cp2 = FetchCheckpoint(tmp_path / "checkpoint.json")
        assert cp2.last_contract == "O:GME220121C00020000"


class TestOptionsDataFetcherMonthList:
    def test_generates_month_range(self):
        cfg = GMEOptionsConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 4, 15),
        )
        fetcher = OptionsDataFetcher(cfg, client=MagicMock())
        months = fetcher._expiration_months()
        assert months == ["2022-01", "2022-02", "2022-03", "2022-04"]

    def test_skips_completed_months(self, tmp_path):
        cfg = GMEOptionsConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 3, 31),
            data_dir=tmp_path,
        )
        fetcher = OptionsDataFetcher(cfg, client=MagicMock())
        fetcher.checkpoint.mark_month_complete("2022-01")
        fetcher.checkpoint.mark_month_complete("2022-02")
        remaining = fetcher._remaining_months()
        assert remaining == ["2022-03"]


class TestOptionsDataFetcherBars:
    def test_fetch_bars_for_month(self, tmp_path):
        mock_client = MagicMock()
        mock_client.fetch_option_contracts.return_value = [
            {
                "ticker": "O:GME230120C00020000",
                "listed_date": "2023-01-03",
                "expiration_date": "2023-01-20",
                "strike_price": 20.0,
                "contract_type": "call",
            },
        ]
        # Range API returns list of bar dicts with Unix-ms timestamps
        mock_client.fetch_option_daily_bars_range.return_value = [
            {
                "o": 5.0, "h": 6.0, "l": 4.5, "c": 5.5, "v": 100, "vw": 5.3,
                "t": 1674172800000,  # 2023-01-20 UTC
            },
            {
                "o": 5.2, "h": 6.1, "l": 4.8, "c": 5.8, "v": 80, "vw": 5.5,
                "t": 1674086400000,  # 2023-01-19 UTC
            },
        ]
        cfg = GMEOptionsConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            data_dir=tmp_path,
            rate_limit_delay=0.0,
        )
        fetcher = OptionsDataFetcher(cfg, client=mock_client)
        df = fetcher.fetch_bars_for_month("2023-01")
        assert len(df) == 2
        assert df.iloc[0]["option_ticker"] == "O:GME230120C00020000"
        assert df.iloc[0]["close"] == 5.5
        assert df.iloc[0]["option_type"] == "call"
        # Verify column names match StateEngine schema
        assert "date" in df.columns
        assert "strike" in df.columns
        assert "expiration" in df.columns
        assert "open_interest" in df.columns

    def test_save_and_load_month(self, tmp_path):
        import pandas as pd
        cfg = GMEOptionsConfig(data_dir=tmp_path)
        df = pd.DataFrame([{
            "option_ticker": "O:GME220121C00020000",
            "close": 5.5,
        }])
        fetcher = OptionsDataFetcher(cfg, client=MagicMock())
        path = fetcher.save_month("2022-01", df)
        loaded = fetcher.load_month(path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["close"] == 5.5
