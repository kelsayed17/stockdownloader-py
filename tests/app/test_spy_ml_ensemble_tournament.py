"""Tests for walk-forward tournament backtest in spy_ml_ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stockdownloader.app.spy_ml_ensemble import (
    _build_parser,
    _run_portfolio_backtest,
)


# ======================================================================
# Synthetic data helpers
# ======================================================================


@dataclass
class FakeBar:
    """Minimal price bar for testing."""

    date: str
    close: float


def _make_daily_data(n: int = 50, base_price: float = 100.0) -> list[FakeBar]:
    """Create synthetic daily price bars with mild uptrend."""
    bars = []
    price = base_price
    for i in range(n):
        date = f"2024-01-{i + 1:02d}" if i < 31 else f"2024-02-{i - 30:02d}"
        bars.append(FakeBar(date=date, close=round(price, 2)))
        price += np.random.default_rng(42 + i).uniform(-1, 1.5)
    return bars


def _make_dates_from_bars(bars: list[FakeBar]) -> tuple[str, ...]:
    """Extract date tuple from fake bars."""
    return tuple(b.date for b in bars)


# ======================================================================
# TestRunPortfolioBacktest
# ======================================================================


class TestRunPortfolioBacktest:
    """Tests for _run_portfolio_backtest()."""

    def test_returns_dict_with_expected_keys(self) -> None:
        bars = _make_daily_data(30)
        dates = _make_dates_from_bars(bars)
        preds = np.full(len(dates), 0.6)  # always above buy_thresh

        result = _run_portfolio_backtest(
            preds, dates, bars,
            buy_thresh=0.55, sell_thresh=0.45,
            initial_capital=100_000.0,
            label="Test",
        )
        expected_keys = {
            "n_trades", "win_rate", "initial_capital", "final_equity",
            "total_return_pct", "total_return_dollar",
            "max_drawdown_pct", "max_drawdown_dollar",
            "sharpe", "avg_hold_bars", "avg_trade_pnl",
            "best_trade", "worst_trade",
        }
        assert set(result.keys()) == expected_keys

    def test_no_trades_when_predictions_neutral(self) -> None:
        """Predictions between thresholds should generate no trades."""
        bars = _make_daily_data(20)
        dates = _make_dates_from_bars(bars)
        preds = np.full(len(dates), 0.50)  # neutral — between 0.45 and 0.55

        result = _run_portfolio_backtest(
            preds, dates, bars,
            buy_thresh=0.55, sell_thresh=0.45,
        )
        assert result["n_trades"] == 0
        assert result["final_equity"] == 100_000.0

    def test_generates_trades_with_varied_predictions(self) -> None:
        """Alternating high/low predictions should generate trades."""
        bars = _make_daily_data(40)
        dates = _make_dates_from_bars(bars)
        # Alternate between buy and sell signals every 5 bars
        preds = np.array([
            0.7 if (i // 5) % 2 == 0 else 0.3
            for i in range(len(dates))
        ])

        result = _run_portfolio_backtest(
            preds, dates, bars,
            buy_thresh=0.55, sell_thresh=0.45,
            initial_capital=50_000.0,
        )
        assert result["n_trades"] > 0
        assert result["initial_capital"] == 50_000.0

    def test_initial_capital_preserved_in_result(self) -> None:
        bars = _make_daily_data(20)
        dates = _make_dates_from_bars(bars)
        preds = np.full(len(dates), 0.50)

        result = _run_portfolio_backtest(
            preds, dates, bars,
            buy_thresh=0.55, sell_thresh=0.45,
            initial_capital=200_000.0,
        )
        assert result["initial_capital"] == 200_000.0

    def test_win_rate_between_zero_and_one(self) -> None:
        bars = _make_daily_data(40)
        dates = _make_dates_from_bars(bars)
        preds = np.array([
            0.7 if (i // 5) % 2 == 0 else 0.3
            for i in range(len(dates))
        ])

        result = _run_portfolio_backtest(
            preds, dates, bars,
            buy_thresh=0.55, sell_thresh=0.45,
        )
        if result["n_trades"] > 0:
            assert 0.0 <= result["win_rate"] <= 1.0

    def test_max_drawdown_non_positive(self) -> None:
        bars = _make_daily_data(40)
        dates = _make_dates_from_bars(bars)
        preds = np.array([
            0.7 if (i // 5) % 2 == 0 else 0.3
            for i in range(len(dates))
        ])

        result = _run_portfolio_backtest(
            preds, dates, bars,
            buy_thresh=0.55, sell_thresh=0.45,
        )
        # Max drawdown % should be <= 0 (negative or zero)
        assert result["max_drawdown_pct"] <= 0.0

    def test_custom_label_printed(self, capsys) -> None:
        bars = _make_daily_data(10)
        dates = _make_dates_from_bars(bars)
        preds = np.full(len(dates), 0.50)

        _run_portfolio_backtest(
            preds, dates, bars,
            buy_thresh=0.55, sell_thresh=0.45,
            label="My Custom Label",
        )
        captured = capsys.readouterr()
        assert "My Custom Label" in captured.out

    def test_empty_predictions(self) -> None:
        """Should handle zero-length predictions gracefully."""
        bars = _make_daily_data(10)
        preds = np.array([])
        dates: tuple[str, ...] = ()

        result = _run_portfolio_backtest(
            preds, dates, bars,
            buy_thresh=0.55, sell_thresh=0.45,
        )
        assert result["n_trades"] == 0


# ======================================================================
# TestWalkForwardPredictions
# ======================================================================


class TestWalkForwardPredictions:
    """Tests for _generate_walk_forward_predictions()."""

    @pytest.fixture
    def _patch_ml_pipeline(self):
        """Mock the ML pipeline so walk-forward doesn't do real training."""
        from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset

        # Create a small fake dataset
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 10
        X = rng.standard_normal((n_samples, n_features))
        y = rng.integers(0, 2, size=n_samples)
        dates = tuple(f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}" for i in range(n_samples))
        feature_names = tuple(f"feat_{i}" for i in range(n_features))
        label_config = LabelConfig(forward_period=10, profit_threshold=0.005)

        dataset = MLDataset(
            X=X, y=y, dates=dates,
            feature_names=feature_names,
            label_config=label_config,
        )
        bars = [FakeBar(date=d, close=100.0 + i * 0.1) for i, d in enumerate(dates)]

        # Mock training result
        mock_result = MagicMock()
        mock_result.oos_accuracy = 0.55
        mock_result.oos_roc_auc = 0.58
        mock_result.config.model_type = "gradient_boosting"
        mock_result.config.use_class_balance = False
        mock_result.feature_importances = {f"feat_{i}": rng.uniform(0, 1) for i in range(n_features)}
        mock_result.model.predict_proba.return_value = rng.uniform(0, 1, (n_samples, 2))

        return dataset, bars, mock_result

    def test_walk_forward_produces_oos_predictions(self, _patch_ml_pipeline) -> None:
        """Walk-forward should produce predictions without error."""
        dataset, bars, mock_result = _patch_ml_pipeline

        with patch("stockdownloader.ml.trainer.MLTrainer") as MockTrainer, \
             patch("stockdownloader.ml.deep_surrogate.DeepSurrogateExporter") as MockExporter:
            MockTrainer.return_value.train.return_value = mock_result
            mock_exporter_inst = MockExporter.return_value
            mock_exporter_inst.predict.side_effect = lambda X: np.full(X.shape[0], 0.55)

            from stockdownloader.app.spy_ml_ensemble import (
                _generate_walk_forward_predictions,
            )

            preds, dates, n_used = _generate_walk_forward_predictions(
                dataset, bars,
                grid=[("gradient_boosting", False)],
                n_estimators=50, max_depth=3,
                n_windows=3, min_train_ratio=0.5,
            )

        assert len(preds) > 0
        assert len(dates) == len(preds)
        assert n_used > 0

    def test_walk_forward_no_train_test_overlap(self, _patch_ml_pipeline) -> None:
        """Each window's OOS dates should not appear in any prior training set."""
        dataset, bars, mock_result = _patch_ml_pipeline

        # Track what ranges are used for training vs testing
        train_ranges_seen = []
        test_ranges_seen = []

        from stockdownloader.ml.trainer_tuning import TimeSeriesExpandingCV
        original_split = TimeSeriesExpandingCV.split

        def tracking_split(self, n_samples):
            folds = original_split(self, n_samples)
            for train_range, test_range in folds:
                train_ranges_seen.append(set(train_range))
                test_ranges_seen.append(set(test_range))
            return folds

        with patch("stockdownloader.ml.trainer.MLTrainer") as MockTrainer, \
             patch("stockdownloader.ml.deep_surrogate.DeepSurrogateExporter") as MockExporter, \
             patch.object(TimeSeriesExpandingCV, "split", tracking_split):
            MockTrainer.return_value.train.return_value = mock_result
            MockExporter.return_value.predict.side_effect = lambda X: np.full(X.shape[0], 0.55)

            from stockdownloader.app.spy_ml_ensemble import (
                _generate_walk_forward_predictions,
            )
            _generate_walk_forward_predictions(
                dataset, bars,
                grid=[("gradient_boosting", False)],
                n_estimators=50, max_depth=3,
                n_windows=3, min_train_ratio=0.5,
            )

        # Verify: each test range should have no overlap with its corresponding train range
        for i, (train_set, test_set) in enumerate(zip(train_ranges_seen, test_ranges_seen)):
            overlap = train_set & test_set
            assert len(overlap) == 0, (
                f"Window {i}: train/test overlap at indices {overlap}"
            )

    def test_walk_forward_returns_correct_tuple(self, _patch_ml_pipeline) -> None:
        """Return value should be (ndarray, tuple[str], int)."""
        dataset, bars, mock_result = _patch_ml_pipeline

        with patch("stockdownloader.ml.trainer.MLTrainer") as MockTrainer, \
             patch("stockdownloader.ml.deep_surrogate.DeepSurrogateExporter") as MockExporter:
            MockTrainer.return_value.train.return_value = mock_result
            MockExporter.return_value.predict.side_effect = lambda X: np.full(X.shape[0], 0.55)

            from stockdownloader.app.spy_ml_ensemble import (
                _generate_walk_forward_predictions,
            )
            preds, dates, n_used = _generate_walk_forward_predictions(
                dataset, bars,
                grid=[("gradient_boosting", False)],
                n_estimators=50, max_depth=3,
                n_windows=2, min_train_ratio=0.5,
            )

        assert isinstance(preds, np.ndarray)
        assert isinstance(dates, tuple)
        assert isinstance(n_used, int)
        assert preds.ndim == 1


# ======================================================================
# TestEnsembleParserWalkForward
# ======================================================================


class TestEnsembleParserWalkForward:
    """Tests for walk-forward CLI args in spy_ml_ensemble parser."""

    def test_walk_forward_windows_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.walk_forward_windows == 5

    def test_walk_forward_windows_custom(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--walk-forward-windows", "3"])
        assert args.walk_forward_windows == 3

    def test_no_walk_forward_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.no_walk_forward is False

    def test_no_walk_forward_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--no-walk-forward"])
        assert args.no_walk_forward is True

    def test_walk_forward_with_no_tournament(self) -> None:
        """Walk-forward args should coexist with no-tournament."""
        parser = _build_parser()
        args = parser.parse_args([
            "--no-tournament", "--walk-forward-windows", "2",
        ])
        assert args.no_tournament is True
        assert args.walk_forward_windows == 2
