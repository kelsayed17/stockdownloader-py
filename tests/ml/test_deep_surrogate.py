"""Tests for DeepSurrogateExporter -- regression surrogate for ensemble probs."""

from __future__ import annotations

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter
from stockdownloader.pinescript.ml_export import FEATURE_PINE_MAP


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Use real feature names that exist in FEATURE_PINE_MAP so Pine
# dependency tests work correctly.
_FEATURE_NAMES = (
    "rsi14_norm",
    "stoch_k_norm",
    "stoch_d_norm",
    "adx14_norm",
    "plus_di_norm",
    "minus_di_norm",
    "mfi14_norm",
    "williams_r_norm",
    "cci20_norm",
    "bb_percent_b",
    "bb_width_norm",
    "roc12_clamped",
    "macd_hist_atr",
    "macd_signal_spread_atr",
    "price_vs_sma20_atr",
    "price_vs_sma50_atr",
    "price_vs_sma200_atr",
    "volume_ratio",
    "atr_pct",
    "fib_position",
    "rsi_score",
    "macd_score",
    "stochastic_score",
    "cci_score",
    "williams_r_score",
    "sma_cross_score",
    "adx_score",
    "ema_trend_score",
    "obv_score",
    "mfi_score",
)

_N_FEATURES = len(_FEATURE_NAMES)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _make_dataset_and_probs(
    n: int = 1000,
    seed: int = 42,
) -> tuple[MLDataset, np.ndarray]:
    """Create a synthetic dataset with known ensemble probabilities.

    The 'ensemble probabilities' are a logistic function of the first
    3 features, providing a known ground truth the surrogate can learn.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, _N_FEATURES)
    y = (rng.rand(n) > 0.5).astype(np.int64)
    dates = tuple(f"2024-01-{i + 1:04d}" for i in range(n))

    # Ensemble probs = sigmoid(2*f0 + 1.5*f1 - f2 + noise)
    logit = 2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2]
    noise = rng.randn(n) * 0.3
    ensemble_probs = _sigmoid(logit + noise)

    dataset = MLDataset(
        X=X,
        y=y,
        dates=dates,
        feature_names=_FEATURE_NAMES,
        label_config=LabelConfig(),
    )
    return dataset, ensemble_probs


def _make_importances() -> dict[str, float]:
    """Create feature importances with first 3 features dominating."""
    importances: dict[str, float] = {}
    for i, name in enumerate(_FEATURE_NAMES):
        if i == 0:
            importances[name] = 0.35
        elif i == 1:
            importances[name] = 0.25
        elif i == 2:
            importances[name] = 0.15
        else:
            importances[name] = 0.01
    return importances


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestDeepSurrogateExporter:
    """Core tests for the DeepSurrogateExporter."""

    def test_init_defaults(self) -> None:
        """Constructor sets expected defaults."""
        exporter = DeepSurrogateExporter()
        assert exporter.max_depth == 10
        assert exporter.min_samples_leaf == 10
        assert exporter.top_n == 25
        assert not exporter.is_trained
        assert exporter.feature_names == ()

    def test_init_custom(self) -> None:
        """Constructor accepts custom parameters."""
        exporter = DeepSurrogateExporter(max_depth=5, min_samples_leaf=20, top_n=10)
        assert exporter.max_depth == 5
        assert exporter.min_samples_leaf == 20
        assert exporter.top_n == 10

    def test_train_sets_is_trained(self) -> None:
        """After training, is_trained is True."""
        dataset, probs = _make_dataset_and_probs()
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=4, top_n=10)
        assert not exporter.is_trained

        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)
        assert exporter.is_trained

    def test_feature_names_correct_count(self) -> None:
        """feature_names has correct count (top_n or fewer)."""
        dataset, probs = _make_dataset_and_probs()
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=4, top_n=10)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)
        assert len(exporter.feature_names) == 10

    def test_feature_names_capped_by_available(self) -> None:
        """When top_n > available features, use all available."""
        dataset, probs = _make_dataset_and_probs()
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=4, top_n=100)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)
        assert len(exporter.feature_names) == _N_FEATURES

    def test_feature_names_are_real_map_keys(self) -> None:
        """All selected feature names exist in FEATURE_PINE_MAP."""
        dataset, probs = _make_dataset_and_probs()
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=4, top_n=10)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)
        for name in exporter.feature_names:
            assert name in FEATURE_PINE_MAP, f"{name} not in FEATURE_PINE_MAP"


class TestDeepSurrogatePredict:
    """Tests for predict() and fidelity."""

    def test_predict_shape(self) -> None:
        """predict() returns correct shape."""
        dataset, probs = _make_dataset_and_probs(n=500)
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=6, top_n=15)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        preds = exporter.predict(dataset.X)
        assert preds.shape == (500,)

    def test_predict_range(self) -> None:
        """predict() returns values clipped to [0, 1]."""
        dataset, probs = _make_dataset_and_probs(n=500)
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=6, top_n=15)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        preds = exporter.predict(dataset.X)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_predict_raises_before_training(self) -> None:
        """predict() raises RuntimeError before training."""
        exporter = DeepSurrogateExporter()
        X = np.zeros((10, _N_FEATURES))
        with pytest.raises(RuntimeError, match="train_surrogate"):
            exporter.predict(X)

    def test_fidelity_r_squared_high(self) -> None:
        """R-squared > 0.70 for clean signal with depth 8."""
        dataset, probs = _make_dataset_and_probs(n=2000, seed=123)
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=8, min_samples_leaf=5, top_n=15)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        r2 = exporter.fidelity_r_squared(dataset.X, probs)
        assert r2 > 0.70, f"R-squared {r2:.4f} should be > 0.70"

    def test_fidelity_r_squared_type(self) -> None:
        """fidelity_r_squared returns a float."""
        dataset, probs = _make_dataset_and_probs(n=300)
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=4, top_n=10)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        r2 = exporter.fidelity_r_squared(dataset.X, probs)
        assert isinstance(r2, float)


class TestDeepSurrogatePineExport:
    """Tests for Pine Script export methods."""

    def test_tree_to_pine_lines_has_tn0(self) -> None:
        """tree_to_pine_lines produces output containing tn0."""
        dataset, probs = _make_dataset_and_probs(n=300)
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=4, top_n=10)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        lines = exporter.tree_to_pine_lines()
        assert len(lines) > 0
        # The last line should define tn0
        assert "tn0" in lines[-1]

    def test_tree_to_pine_lines_format(self) -> None:
        """Each pine line declares a float tnX variable."""
        dataset, probs = _make_dataset_and_probs(n=300)
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=3, top_n=10)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        lines = exporter.tree_to_pine_lines()
        for line in lines:
            assert line.startswith("float tn"), f"Bad line format: {line}"

    def test_tree_to_pine_lines_raises_before_training(self) -> None:
        """tree_to_pine_lines raises RuntimeError before training."""
        exporter = DeepSurrogateExporter()
        with pytest.raises(RuntimeError, match="train_surrogate"):
            exporter.tree_to_pine_lines()

    def test_tree_to_pine_ternary(self) -> None:
        """tree_to_pine returns a ternary expression string."""
        dataset, probs = _make_dataset_and_probs(n=300)
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=3, top_n=10)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        expr = exporter.tree_to_pine()
        assert isinstance(expr, str)
        assert "?" in expr  # contains ternary operator
        assert "<=" in expr  # contains threshold comparison

    def test_tree_to_pine_raises_before_training(self) -> None:
        """tree_to_pine raises RuntimeError before training."""
        exporter = DeepSurrogateExporter()
        with pytest.raises(RuntimeError, match="train_surrogate"):
            exporter.tree_to_pine()

    def test_pine_leaf_values_in_range(self) -> None:
        """Leaf values in Pine output are between 0.0 and 1.0."""
        dataset, probs = _make_dataset_and_probs(n=300)
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=3, top_n=10)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        import re

        expr = exporter.tree_to_pine()
        # Extract all float literals from the expression
        floats = re.findall(r"(?<!\w)\d+\.\d{4}(?!\w)", expr)
        for f in floats:
            val = float(f)
            assert 0.0 <= val <= 1.0, f"Leaf value {val} out of [0,1] range"

    def test_get_pine_dependencies_deduped(self) -> None:
        """get_pine_dependencies returns a de-duplicated list."""
        dataset, probs = _make_dataset_and_probs(n=300)
        importances = _make_importances()

        exporter = DeepSurrogateExporter(max_depth=4, top_n=15)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        deps = exporter.get_pine_dependencies()
        assert isinstance(deps, list)
        # Must be de-duplicated
        assert len(deps) == len(set(deps)), "Dependencies contain duplicates"

    def test_get_pine_dependencies_contains_expected(self) -> None:
        """Dependencies include lines for features that have them."""
        dataset, probs = _make_dataset_and_probs(n=300)
        importances = _make_importances()

        # Use top_n=10 to ensure adx14_norm (which has deps) is included
        exporter = DeepSurrogateExporter(max_depth=4, top_n=10)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        deps = exporter.get_pine_dependencies()
        # adx14_norm depends on "[pDiPlus, pDiMinus, pAdxVal] = ta.dmi(14, 14)"
        if "adx14_norm" in exporter.feature_names:
            assert any("ta.dmi" in d for d in deps), (
                "Expected DMI dependency for adx14_norm"
            )

    def test_get_pine_dependencies_empty_before_training(self) -> None:
        """get_pine_dependencies returns empty list before training."""
        exporter = DeepSurrogateExporter()
        assert exporter.get_pine_dependencies() == []

    def test_deeper_tree_more_pine_lines(self) -> None:
        """A deeper tree produces more pine lines than a shallow one."""
        dataset, probs = _make_dataset_and_probs(n=1000, seed=99)
        importances = _make_importances()

        shallow = DeepSurrogateExporter(max_depth=2, min_samples_leaf=50, top_n=10)
        shallow.train_surrogate(dataset, importances, ensemble_probs=probs)
        shallow_lines = shallow.tree_to_pine_lines()

        deep = DeepSurrogateExporter(max_depth=8, min_samples_leaf=5, top_n=10)
        deep.train_surrogate(dataset, importances, ensemble_probs=probs)
        deep_lines = deep.tree_to_pine_lines()

        assert len(deep_lines) > len(shallow_lines), (
            f"Deep tree ({len(deep_lines)} lines) should have more lines "
            f"than shallow tree ({len(shallow_lines)} lines)"
        )

    def test_safe_pine_expr_wraps_ternary(self) -> None:
        """_safe_pine_expr wraps ternary expressions in parentheses."""
        assert DeepSurrogateExporter._safe_pine_expr("x") == "x"
        assert DeepSurrogateExporter._safe_pine_expr("a ? b : c") == "(a ? b : c)"

    def test_pine_lines_use_feature_map_expressions(self) -> None:
        """Pine lines reference FEATURE_PINE_MAP expressions, not raw names."""
        dataset, probs = _make_dataset_and_probs(n=500)
        # Give high importance to rsi14_norm which maps to "ta.rsi(close, 14) / 100.0"
        importances = {name: 0.01 for name in _FEATURE_NAMES}
        importances["rsi14_norm"] = 0.90

        exporter = DeepSurrogateExporter(max_depth=3, top_n=5)
        exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

        lines = exporter.tree_to_pine_lines()
        all_text = " ".join(lines)
        # If rsi14_norm is used in splits, the Pine expression should appear
        if "ta.rsi" in all_text or "rsi14_norm" not in all_text:
            pass  # Correct: uses pine expression or feature not in splits
        else:
            pytest.fail("Pine lines reference raw feature name instead of Pine expression")
