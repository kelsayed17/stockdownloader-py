"""Tests for the ml-pipeline CLI app."""

from __future__ import annotations

import random
from decimal import Decimal
from unittest.mock import patch

import pytest

from stockdownloader.app.spy_ml_pipeline_app import (
    _build_generic_parser as _build_parser,
    main_generic as main,
)
from stockdownloader.model.price_data import PriceData


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_data(n: int = 300, seed: int = 42) -> list[PriceData]:
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
# Parser tests
# ------------------------------------------------------------------


class TestMLPipelineParser:
    def test_default_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.symbol == "SPY"
        assert args.range_ == "10y"
        assert args.quick is False
        assert args.no_pine is False
        assert args.capital == 100_000.0
        assert args.top_n == 5

    def test_custom_symbol(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["AAPL"])
        assert args.symbol == "AAPL"

    def test_quick_mode(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--quick"])
        assert args.quick is True

    def test_no_pine(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--no-pine"])
        assert args.no_pine is True

    def test_custom_grid(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "SPY",
            "--forward-periods", "5,10",
            "--profit-thresholds", "0.005,0.01",
            "--model-types", "gradient_boosting",
        ])
        assert args.forward_periods == "5,10"
        assert args.profit_thresholds == "0.005,0.01"
        assert args.model_types == "gradient_boosting"

    def test_ml_threshold(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--ml-threshold", "0.55"])
        assert args.ml_threshold == 0.55

    def test_output_dir(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--output-dir", "/tmp/out"])
        assert args.output_dir == "/tmp/out"

    def test_modes(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--modes", "confirmed,override"])
        assert args.modes == "confirmed,override"


# ------------------------------------------------------------------
# Main function tests
# ------------------------------------------------------------------


class TestMLPipelineMain:
    @patch("stockdownloader.app.app_helpers.fetch_daily_data")
    def test_quick_mode_runs(
        self, mock_fetch: object, tmp_path: object, capsys: object,
    ) -> None:
        mock_fetch.return_value = _make_data(300)  # type: ignore[attr-defined]

        main([
            "SPY",
            "--quick",
            "--no-cache",
            "--no-pine",
            "--output-dir", str(tmp_path),
            "--n-estimators", "10",
            "--max-depth", "2",
        ])

        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "PIPELINE COMPLETE" in captured.out
        assert "STAGE 1" in captured.out

    @patch("stockdownloader.app.app_helpers.fetch_daily_data")
    def test_no_data_exits(self, mock_fetch: object) -> None:
        mock_fetch.return_value = []  # type: ignore[attr-defined]

        with pytest.raises((RuntimeError, SystemExit)):
            main(["SPY", "--no-cache", "--quick"])
