"""Tests for Stage 1: Data Acquisition."""

from __future__ import annotations

import json
import random
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from stockdownloader.ml.pipeline.config import DataConfig
from stockdownloader.ml.pipeline.stage_data import DataStage
from stockdownloader.core.models.price import PriceData


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_data(n: int = 100, seed: int = 42) -> list[PriceData]:
    rng = random.Random(seed)
    data: list[PriceData] = []
    price = 100.0
    for i in range(n):
        price += (rng.random() - 0.48) * 2
        price = max(50, price)
        h = price + rng.random() * 2
        low = price - rng.random() * 2
        data.append(PriceData(
            date=f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            open=Decimal(str(round(price - 0.5, 2))),
            high=Decimal(str(round(h, 2))),
            low=Decimal(str(round(low, 2))),
            close=Decimal(str(round(price, 2))),
            adj_close=Decimal(str(round(price, 2))),
            volume=int(1_000_000 + rng.random() * 5_000_000),
        ))
    return data


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestDataStage:
    @patch("stockdownloader.app.helpers.fetch_daily_data")
    def test_fetch_returns_data_result(
        self, mock_fetch: object, tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = _make_data(50)  # type: ignore[attr-defined]
        cfg = DataConfig(
            symbol="SPY",
            range_="5y",
            use_cache=False,
        )
        stage = DataStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run()

        assert result.symbol == "SPY"
        assert result.bar_count == 50
        assert not result.cached
        assert result.date_range[0] == result.data[0].date
        assert result.date_range[1] == result.data[-1].date

    @patch("stockdownloader.app.helpers.fetch_daily_data")
    def test_fetch_empty_raises(self, mock_fetch: object) -> None:
        mock_fetch.return_value = []  # type: ignore[attr-defined]
        cfg = DataConfig(symbol="BAD", use_cache=False)
        stage = DataStage(cfg, print_fn=lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="Could not fetch"):
            stage.run()

    @patch("stockdownloader.app.helpers.fetch_daily_data")
    def test_cache_round_trip(
        self, mock_fetch: object, tmp_path: Path,
    ) -> None:
        original_data = _make_data(30)
        mock_fetch.return_value = original_data  # type: ignore[attr-defined]

        cfg = DataConfig(
            symbol="SPY",
            range_="1y",
            cache_dir=str(tmp_path),
            use_cache=True,
        )

        # First run — fetches and caches
        stage = DataStage(cfg, print_fn=lambda *a, **k: None)
        r1 = stage.run()
        assert not r1.cached
        assert r1.bar_count == 30

        # Second run — loads from cache
        r2 = stage.run()
        assert r2.cached
        assert r2.bar_count == 30
        assert r2.data[0].close == original_data[0].close
        assert r2.data[-1].date == original_data[-1].date

    def test_load_cache_corrupt(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        result = DataStage._load_cache(bad_file)
        assert result == []

    def test_save_and_load_cache(self, tmp_path: Path) -> None:
        data = _make_data(10)
        path = tmp_path / "test.json"
        DataStage._save_cache(data, path)
        loaded = DataStage._load_cache(path)
        assert len(loaded) == 10
        assert loaded[0].close == data[0].close
        assert loaded[-1].volume == data[-1].volume
