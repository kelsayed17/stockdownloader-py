"""Tests for ml_train_app — ML training CLI."""

from __future__ import annotations

import random
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.app.ml_train_app import _build_parser, main
from stockdownloader.model.price_data import PriceData
from stockdownloader.util.config import DEFAULT_MODELS_DIR


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
        l = price - rng.random() * 2
        data.append(PriceData(
            date=f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            open=Decimal(str(round(price - 0.5, 2))),
            high=Decimal(str(round(h, 2))),
            low=Decimal(str(round(l, 2))),
            close=Decimal(str(round(price, 2))),
            adj_close=Decimal(str(round(price, 2))),
            volume=int(1_000_000 + rng.random() * 5_000_000),
        ))
    return data


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------


class TestMLTrainParser:
    def test_default_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.symbol == "SPY"
        assert args.forward_period == 10
        assert args.profit_threshold == 0.005
        assert args.model_type == "gradient_boosting"
        assert args.output_dir == str(DEFAULT_MODELS_DIR / "spy")

    def test_custom_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "AAPL",
            "--forward-period", "10",
            "--model-type", "logistic_regression",
            "--n-estimators", "100",
            "--max-depth", "3",
            "--output-dir", "/tmp/models",
        ])
        assert args.symbol == "AAPL"
        assert args.forward_period == 10
        assert args.model_type == "logistic_regression"
        assert args.n_estimators == 100
        assert args.max_depth == 3
        assert args.output_dir == "/tmp/models"

    def test_profit_threshold(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["SPY", "--profit-threshold", "0.01"])
        assert args.profit_threshold == 0.01

    def test_tune_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["SPY", "--tune"])
        assert args.tune is True

    def test_feature_select_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["SPY", "--feature-select"])
        assert args.feature_select is True

    def test_balanced_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["SPY", "--balanced"])
        assert args.balanced is True

    def test_export_pine_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["SPY", "--export-pine"])
        assert args.export_pine is True

    def test_use_atr_labels_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["SPY", "--use-atr-labels"])
        assert args.use_atr_labels is True

    def test_pine_depth_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.pine_depth == 6
        assert args.pine_top_features == 15


# ------------------------------------------------------------------
# Main function (mocked fetch)
# ------------------------------------------------------------------


class TestMLTrainMain:
    @patch("stockdownloader.app.app_helpers.fetch_daily_data")
    def test_main_with_mocked_data(
        self, mock_fetch: MagicMock, tmp_path: object, capsys: object,
    ) -> None:
        mock_fetch.return_value = _make_data(300)

        main([
            "SPY",
            "--forward-period", "5",
            "--profit-threshold", "0.0",
            "--n-estimators", "10",
            "--max-depth", "2",
            "--output-dir", str(tmp_path),
        ])

        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "TRAINING RESULTS" in captured.out
        assert "OOS Accuracy" in captured.out
        assert "Model saved to" in captured.out

    @patch("stockdownloader.app.app_helpers.fetch_daily_data")
    def test_main_no_data_exits(
        self, mock_fetch: MagicMock,
    ) -> None:
        mock_fetch.return_value = []

        with pytest.raises(SystemExit) as exc_info:
            main(["SPY"])
        assert exc_info.value.code == 1

    @patch("stockdownloader.app.app_helpers.fetch_daily_data")
    def test_main_logistic_regression(
        self, mock_fetch: MagicMock, tmp_path: object, capsys: object,
    ) -> None:
        mock_fetch.return_value = _make_data(300)

        main([
            "SPY",
            "--forward-period", "5",
            "--profit-threshold", "0.0",
            "--model-type", "logistic_regression",
            "--output-dir", str(tmp_path),
        ])

        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "TRAINING RESULTS" in captured.out

    @patch("stockdownloader.app.app_helpers.fetch_daily_data")
    def test_main_with_balanced_and_selection(
        self, mock_fetch: MagicMock, tmp_path: object, capsys: object,
    ) -> None:
        mock_fetch.return_value = _make_data(300)

        main([
            "SPY",
            "--forward-period", "5",
            "--profit-threshold", "0.0",
            "--n-estimators", "10",
            "--max-depth", "2",
            "--balanced",
            "--feature-select",
            "--output-dir", str(tmp_path),
        ])

        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "TRAINING RESULTS" in captured.out
        assert "Selected feats" in captured.out

    @patch("stockdownloader.app.app_helpers.fetch_daily_data")
    def test_main_with_export_pine(
        self, mock_fetch: MagicMock, tmp_path: object, capsys: object,
    ) -> None:
        mock_fetch.return_value = _make_data(300)

        main([
            "SPY",
            "--forward-period", "5",
            "--profit-threshold", "0.0",
            "--n-estimators", "10",
            "--max-depth", "2",
            "--export-pine",
            "--pine-depth", "3",
            "--pine-top-features", "5",
            "--output-dir", str(tmp_path),
        ])

        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "TRAINING RESULTS" in captured.out
        assert "Pine Script saved to" in captured.out


# ------------------------------------------------------------------
# Monitor app --ml-model flag
# ------------------------------------------------------------------


class TestMonitorMLFlag:
    def test_ml_model_arg_parsed(self) -> None:
        from stockdownloader.app.monitor_app import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["SPY", "--ml-model", "/path/to/model.joblib"])
        assert args.ml_model == "/path/to/model.joblib"

    def test_ml_model_default_none(self) -> None:
        from stockdownloader.app.monitor_app import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["SPY"])
        assert args.ml_model is None
