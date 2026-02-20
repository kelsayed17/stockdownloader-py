"""Tests for pipeline configuration dataclasses."""

from __future__ import annotations

from stockdownloader.ml.pipeline.config import (
    BacktestConfig,
    ConvergenceConfig,
    DataConfig,
    HybridConfig,
    PipelineConfig,
    SelectionConfig,
    TrainingGridConfig,
)


class TestDataConfig:
    def test_defaults(self) -> None:
        cfg = DataConfig()
        assert cfg.symbol == "SPY"
        assert cfg.range_ == "10y"
        assert cfg.interval == "1d"
        assert cfg.use_cache is True

    def test_custom(self) -> None:
        cfg = DataConfig(symbol="AAPL", range_="5y")
        assert cfg.symbol == "AAPL"
        assert cfg.range_ == "5y"


class TestTrainingGridConfig:
    def test_defaults(self) -> None:
        cfg = TrainingGridConfig()
        assert cfg.forward_periods == (5, 10, 20)
        assert cfg.profit_thresholds == (0.003, 0.005, 0.01)
        assert cfg.model_types == ("gradient_boosting", "logistic_regression")
        assert cfg.n_estimators == 200
        assert cfg.max_depth == 4

    def test_grid_size(self) -> None:
        cfg = TrainingGridConfig()
        combos = (
            len(cfg.forward_periods)
            * len(cfg.profit_thresholds)
            * len(cfg.model_types)
            * len(cfg.use_class_balance_options)
            * len(cfg.use_atr_labels_options)
        )
        # 3 * 3 * 2 * 2 * 2 = 72
        assert combos == 72


class TestConvergenceConfig:
    def test_defaults(self) -> None:
        cfg = ConvergenceConfig()
        assert cfg.strategy_categories == ("daily",)
        assert cfg.top_models == 10


class TestHybridConfig:
    def test_defaults(self) -> None:
        cfg = HybridConfig()
        assert cfg.modes == ("confirmed", "weighted", "override")
        assert cfg.confirmed_threshold == 0.6
        assert cfg.override_threshold == 0.55


class TestBacktestConfig:
    def test_defaults(self) -> None:
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.commission == 10.0
        assert cfg.walk_forward_windows == 5


class TestSelectionConfig:
    def test_defaults(self) -> None:
        cfg = SelectionConfig()
        assert cfg.top_n == 5
        assert cfg.pine_depth == 6
        assert cfg.export_pine is True


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.data.symbol == "SPY"
        assert cfg.training.forward_periods == (5, 10, 20)
        assert cfg.verbose is True

    def test_custom_nested(self) -> None:
        cfg = PipelineConfig(
            data=DataConfig(symbol="AAPL"),
            training=TrainingGridConfig(forward_periods=(10,)),
        )
        assert cfg.data.symbol == "AAPL"
        assert cfg.training.forward_periods == (10,)
