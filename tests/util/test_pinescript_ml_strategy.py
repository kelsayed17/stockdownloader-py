"""Tests for PineScript ML signal strategy export."""

from __future__ import annotations

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
from stockdownloader.util.pinescript.ml_export import (
    FEATURE_PINE_MAP,
    DecisionTreeExporter,
    ml_signal_strategy,
)
from stockdownloader.util.pinescript.models import StrategyDefinition


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_FEATURE_NAMES_10 = tuple(f"feat_{i}" for i in range(10))


def _make_dummy_dataset(n: int = 200, n_feats: int = 10) -> MLDataset:
    """Create a simple separable dataset."""
    rng = np.random.RandomState(42)
    half = n // 2
    X_0 = rng.randn(half, n_feats) - 1.0
    X_1 = rng.randn(n - half, n_feats) + 1.0
    X = np.vstack([X_0, X_1])
    y = np.array([0] * half + [1] * (n - half), dtype=np.int64)
    dates = tuple(f"2024-01-{i + 1:03d}" for i in range(n))
    names = _FEATURE_NAMES_10 if n_feats == 10 else tuple(f"feat_{i}" for i in range(n_feats))
    return MLDataset(X=X, y=y, dates=dates, feature_names=names, label_config=LabelConfig())


def _make_real_feature_dataset(n: int = 200) -> MLDataset:
    """Create a dataset with real feature names from the ML pipeline."""
    from stockdownloader.ml.feature_extractor import _ALL_FEATURE_NAMES

    rng = np.random.RandomState(42)
    n_feats = len(_ALL_FEATURE_NAMES)
    half = n // 2
    X_0 = rng.randn(half, n_feats) - 0.5
    X_1 = rng.randn(n - half, n_feats) + 0.5
    X = np.vstack([X_0, X_1])
    y = np.array([0] * half + [1] * (n - half), dtype=np.int64)
    dates = tuple(f"2024-01-{i + 1:03d}" for i in range(n))
    return MLDataset(
        X=X, y=y, dates=dates,
        feature_names=_ALL_FEATURE_NAMES, label_config=LabelConfig(),
    )


# ------------------------------------------------------------------
# Feature map coverage
# ------------------------------------------------------------------


class TestFeaturePineMap:
    def test_all_real_features_mapped(self) -> None:
        """Every feature in _ALL_FEATURE_NAMES should have a Pine mapping."""
        from stockdownloader.ml.feature_extractor import _ALL_FEATURE_NAMES

        missing = [n for n in _ALL_FEATURE_NAMES if n not in FEATURE_PINE_MAP]
        assert missing == [], f"Missing Pine mappings: {missing}"

    def test_expressions_are_strings(self) -> None:
        for name, (expr, deps) in FEATURE_PINE_MAP.items():
            assert isinstance(expr, str), f"{name} expression not a string"
            assert isinstance(deps, list), f"{name} deps not a list"
            for dep in deps:
                assert isinstance(dep, str), f"{name} dep item not a string"


# ------------------------------------------------------------------
# Surrogate tree training
# ------------------------------------------------------------------


class TestDecisionTreeExporter:
    def test_not_trained_initially(self) -> None:
        exp = DecisionTreeExporter()
        assert not exp.is_trained

    def test_train_surrogate(self) -> None:
        ds = _make_dummy_dataset(200, 10)
        importances = {f"feat_{i}": float(10 - i) / 55 for i in range(10)}
        exp = DecisionTreeExporter(max_depth=4, min_samples_leaf=5)
        exp.train_surrogate(ds, importances, top_n=5)
        assert exp.is_trained
        assert len(exp.feature_names) == 5

    def test_tree_to_pine_raises_before_training(self) -> None:
        exp = DecisionTreeExporter()
        with pytest.raises(RuntimeError, match="train_surrogate"):
            exp.tree_to_pine()

    def test_tree_to_pine_produces_expression(self) -> None:
        ds = _make_dummy_dataset(200, 10)
        importances = {f"feat_{i}": float(10 - i) / 55 for i in range(10)}
        exp = DecisionTreeExporter(max_depth=3, min_samples_leaf=10)
        exp.train_surrogate(ds, importances, top_n=5)
        pine = exp.tree_to_pine()
        assert isinstance(pine, str)
        assert len(pine) > 10
        # Should contain ternary expressions or leaf probabilities
        assert "?" in pine or "." in pine  # either ternary or float leaf

    def test_get_pine_dependencies(self) -> None:
        ds = _make_real_feature_dataset(200)
        importances = {n: 1.0 / len(ds.feature_names) for n in ds.feature_names}
        exp = DecisionTreeExporter(max_depth=3, min_samples_leaf=10)
        exp.train_surrogate(ds, importances, top_n=10)
        deps = exp.get_pine_dependencies()
        assert isinstance(deps, list)
        # Dependencies should be unique
        assert len(deps) == len(set(deps))


# ------------------------------------------------------------------
# Strategy factory
# ------------------------------------------------------------------


class TestMLSignalStrategy:
    def test_factory_raises_if_not_trained(self) -> None:
        exp = DecisionTreeExporter()
        with pytest.raises(RuntimeError, match="trained"):
            ml_signal_strategy("SPY", exp)

    def test_factory_produces_strategy(self) -> None:
        ds = _make_real_feature_dataset(200)
        importances = {n: 1.0 / len(ds.feature_names) for n in ds.feature_names}
        exp = DecisionTreeExporter(max_depth=4, min_samples_leaf=10)
        exp.train_surrogate(ds, importances, top_n=10)

        strat = ml_signal_strategy("SPY", exp)
        assert isinstance(strat, StrategyDefinition)
        assert "SPY" in strat.name
        assert strat.long_entry is not None
        assert strat.short_entry is not None
        assert strat.strategy_mode is False

    def test_strategy_has_plots(self) -> None:
        ds = _make_real_feature_dataset(200)
        importances = {n: 1.0 / len(ds.feature_names) for n in ds.feature_names}
        exp = DecisionTreeExporter(max_depth=3, min_samples_leaf=10)
        exp.train_surrogate(ds, importances, top_n=8)

        strat = ml_signal_strategy("SPY", exp)
        # Should have plot for mlProb in data window (overlay-friendly)
        plot_text = " ".join(strat.extra_plots)
        assert "mlProb" in plot_text
        assert "display.data_window" in plot_text

    def test_strategy_generates_valid_pine(self) -> None:
        """Full pipeline: train → export → generate Pine Script."""
        from stockdownloader.util.pinescript import PineScriptGenerator

        ds = _make_real_feature_dataset(200)
        importances = {n: 1.0 / len(ds.feature_names) for n in ds.feature_names}
        exp = DecisionTreeExporter(max_depth=3, min_samples_leaf=10)
        exp.train_surrogate(ds, importances, top_n=8)

        strat = ml_signal_strategy("SPY", exp)
        gen = PineScriptGenerator()
        pine_code = gen.generate(strat)

        assert isinstance(pine_code, str)
        assert "//@version=6" in pine_code
        assert "mlProb" in pine_code
        assert "alertcondition" in pine_code

    def test_custom_thresholds(self) -> None:
        ds = _make_real_feature_dataset(200)
        importances = {n: 1.0 / len(ds.feature_names) for n in ds.feature_names}
        exp = DecisionTreeExporter(max_depth=3, min_samples_leaf=10)
        exp.train_surrogate(ds, importances, top_n=5)

        strat = ml_signal_strategy("AAPL", exp, buy_threshold=0.65, sell_threshold=0.35)
        assert "AAPL" in strat.name
        # Check inputs have custom defaults
        thresh_inputs = [inp for inp in strat.inputs if "Thresh" in inp.name]
        assert len(thresh_inputs) == 2
