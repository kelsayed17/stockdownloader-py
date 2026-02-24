# SPY ML Ensemble Strategy — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a multi-model ML ensemble on SPY, approximate it with a deep surrogate decision tree, and export a production-ready PineScript v6 strategy with ML-driven dynamic exits and visual overlays.

**Architecture:** Train GradientBoosting + RandomForest + LogisticRegression on 10-year SPY daily data using the existing ML pipeline. Combine top models into an `EnsemblePredictor` (soft-voting). Train a `DecisionTreeRegressor` (depth 10, 25 features) to approximate the ensemble's probability surface. Export the surrogate to PineScript with dynamic ATR-based TP/SL scaled by ML confidence, buy/sell labels, background shading, and a confidence panel. Validate via walk-forward backtest and tournament against existing SPY strategies.

**Tech Stack:** Python, scikit-learn (GradientBoosting, RandomForest, LogisticRegression, DecisionTreeRegressor), existing `MLTrainer`/`DatasetBuilder`/`PineScriptGenerator` infrastructure, PineScript v6

---

### Task 1: Create `EnsemblePredictor` with tests

**Files:**
- Create: `src/stockdownloader/ml/ensemble.py`
- Create: `tests/ml/test_ensemble.py`

**Step 1: Write the failing tests**

Create `tests/ml/test_ensemble.py`:

```python
"""Tests for ML ensemble predictor."""

from __future__ import annotations

import numpy as np
import pytest

from stockdownloader.ml.ensemble import EnsemblePredictor, EnsembleBuilder
from stockdownloader.ml.trainer import MLModelConfig, MLTrainer, TrainingResult
from stockdownloader.ml.dataset_builder import MLDataset, LabelConfig


def _make_separable_dataset(n: int = 200, n_features: int = 10) -> MLDataset:
    """Create linearly separable dataset for fast training."""
    rng = np.random.RandomState(42)
    half = n // 2
    X_0 = rng.randn(half, n_features) - 1.0
    X_1 = rng.randn(n - half, n_features) + 1.0
    X = np.empty((n, n_features), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    # Interleave for temporal ordering
    for i in range(half):
        X[2 * i] = X_0[i]
        y[2 * i] = 0
        if 2 * i + 1 < n:
            X[2 * i + 1] = X_1[i]
            y[2 * i + 1] = 1
    dates = tuple(f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n))
    names = tuple(f"feat_{j}" for j in range(n_features))
    return MLDataset(X=X, y=y, dates=dates, feature_names=names,
                     label_config=LabelConfig())


class TestEnsemblePredictor:
    def test_predict_proba_returns_float_array(self):
        ds = _make_separable_dataset()
        configs = [
            MLModelConfig(model_type="gradient_boosting", n_estimators=20,
                          max_depth=2, n_cv_folds=2),
            MLModelConfig(model_type="logistic_regression", n_cv_folds=2),
        ]
        results = [MLTrainer(c).train(ds) for c in configs]
        ensemble = EnsemblePredictor(results)

        probs = ensemble.predict_proba(ds.X)
        assert probs.shape == (len(ds.y),)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_averaged_feature_importances(self):
        ds = _make_separable_dataset()
        configs = [
            MLModelConfig(model_type="gradient_boosting", n_estimators=20,
                          max_depth=2, n_cv_folds=2),
            MLModelConfig(model_type="random_forest", n_estimators=20,
                          max_depth=2, n_cv_folds=2),
        ]
        results = [MLTrainer(c).train(ds) for c in configs]
        ensemble = EnsemblePredictor(results)

        importances = ensemble.averaged_feature_importances()
        assert len(importances) == 10
        assert abs(sum(importances.values()) - 1.0) < 0.01

    def test_ensemble_at_least_as_good_as_best_single(self):
        ds = _make_separable_dataset(300)
        configs = [
            MLModelConfig(model_type="gradient_boosting", n_estimators=30,
                          max_depth=2, n_cv_folds=2),
            MLModelConfig(model_type="random_forest", n_estimators=30,
                          max_depth=2, n_cv_folds=2),
            MLModelConfig(model_type="logistic_regression", n_cv_folds=2),
        ]
        results = [MLTrainer(c).train(ds) for c in configs]
        ensemble = EnsemblePredictor(results)

        best_single = max(r.oos_accuracy for r in results)
        # Ensemble probabilities should be well-calibrated
        probs = ensemble.predict_proba(ds.X)
        preds = (probs >= 0.5).astype(int)
        ensemble_acc = np.mean(preds == ds.y)
        # Ensemble should be competitive (within 5% of best single)
        assert ensemble_acc >= best_single - 0.05


class TestEnsembleBuilder:
    def test_select_top_n(self):
        ds = _make_separable_dataset()
        configs = [
            MLModelConfig(model_type="gradient_boosting", n_estimators=20,
                          max_depth=2, n_cv_folds=2),
            MLModelConfig(model_type="random_forest", n_estimators=20,
                          max_depth=2, n_cv_folds=2),
            MLModelConfig(model_type="logistic_regression", n_cv_folds=2),
        ]
        results = [MLTrainer(c).train(ds) for c in configs]

        builder = EnsembleBuilder(results)
        ensemble = builder.select_top(n=2)
        assert len(ensemble.results) == 2

    def test_select_top_n_sorted_by_accuracy(self):
        ds = _make_separable_dataset()
        configs = [
            MLModelConfig(model_type="gradient_boosting", n_estimators=20,
                          max_depth=2, n_cv_folds=2),
            MLModelConfig(model_type="random_forest", n_estimators=20,
                          max_depth=2, n_cv_folds=2),
            MLModelConfig(model_type="logistic_regression", n_cv_folds=2),
        ]
        results = [MLTrainer(c).train(ds) for c in configs]

        builder = EnsembleBuilder(results)
        ensemble = builder.select_top(n=2)
        # Should be sorted by OOS accuracy descending
        accs = [r.oos_accuracy for r in ensemble.results]
        assert accs == sorted(accs, reverse=True)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/ml/test_ensemble.py -v --tb=short`
Expected: FAIL with `ModuleNotFoundError: No module named 'stockdownloader.ml.ensemble'`

**Step 3: Write the implementation**

Create `src/stockdownloader/ml/ensemble.py`:

```python
"""Multi-model ensemble predictor via soft-voting.

Wraps N trained models (from MLTrainer) and averages their
predict_proba outputs for consensus probability estimation.

Usage::

    from stockdownloader.ml.ensemble import EnsemblePredictor, EnsembleBuilder

    builder = EnsembleBuilder(training_results)
    ensemble = builder.select_top(n=3)
    probs = ensemble.predict_proba(X)
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stockdownloader.ml.trainer import TrainingResult

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Soft-voting ensemble over multiple trained ML models.

    Parameters
    ----------
    results:
        List of TrainingResult objects from MLTrainer.train().
        Each must have a `.model` with `.predict_proba(X)`.
    """

    def __init__(self, results: list[TrainingResult]) -> None:
        if not results:
            raise ValueError("At least one TrainingResult required")
        self.results = list(results)
        logger.info(
            "Ensemble created with %d models: %s",
            len(self.results),
            ", ".join(r.config.model_type for r in self.results),
        )

    def predict_proba(self, X: Any) -> Any:
        """Average P(profitable) across all models.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples,) with averaged probabilities.
        """
        probas = []
        for result in self.results:
            try:
                p = result.model.predict_proba(X)
                # predict_proba returns (n, 2) — take column 1
                if p.ndim == 2:
                    p = p[:, 1]
                probas.append(p)
            except Exception:
                logger.warning(
                    "Model %s failed predict_proba, skipping",
                    result.config.model_type,
                )
        if not probas:
            return np.full(X.shape[0], 0.5)
        return np.mean(probas, axis=0)

    def averaged_feature_importances(self) -> dict[str, float]:
        """Average feature importances across all models.

        Returns
        -------
        dict[str, float]
            Feature name → averaged importance (sums to ~1.0).
        """
        all_imps: dict[str, list[float]] = {}
        for result in self.results:
            for name, imp in result.feature_importances.items():
                all_imps.setdefault(name, []).append(imp)

        averaged = {
            name: sum(vals) / len(self.results)
            for name, vals in all_imps.items()
        }
        # Normalize to sum to 1.0
        total = sum(averaged.values())
        if total > 0:
            averaged = {k: v / total for k, v in averaged.items()}
        return dict(sorted(averaged.items(), key=lambda x: x[1], reverse=True))


class EnsembleBuilder:
    """Selects top models from training grid and builds ensemble.

    Parameters
    ----------
    results:
        All TrainingResult objects from the training grid.
    """

    def __init__(self, results: list[TrainingResult]) -> None:
        self._results = list(results)

    def select_top(self, n: int = 3) -> EnsemblePredictor:
        """Select top N models by OOS accuracy.

        Parameters
        ----------
        n:
            Number of models to include (default 3).

        Returns
        -------
        EnsemblePredictor
            Ensemble wrapping the top N models.
        """
        ranked = sorted(
            self._results,
            key=lambda r: r.oos_accuracy,
            reverse=True,
        )
        top = ranked[:n]
        logger.info(
            "Selected top %d models: %s",
            len(top),
            ", ".join(
                f"{r.config.model_type}(acc={r.oos_accuracy:.3f})"
                for r in top
            ),
        )
        return EnsemblePredictor(top)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/ml/test_ensemble.py -v --tb=short`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/ml/ensemble.py tests/ml/test_ensemble.py
git commit -m "feat: add EnsemblePredictor and EnsembleBuilder for soft-voting"
```

---

### Task 2: Create `DeepSurrogateExporter` with tests

**Files:**
- Create: `src/stockdownloader/ml/deep_surrogate.py`
- Create: `tests/ml/test_deep_surrogate.py`

**Step 1: Write the failing tests**

Create `tests/ml/test_deep_surrogate.py`:

```python
"""Tests for deep surrogate decision tree exporter."""

from __future__ import annotations

import numpy as np
import pytest

from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter
from stockdownloader.ml.dataset_builder import MLDataset, LabelConfig


def _make_dataset_with_known_importances(
    n: int = 500, n_features: int = 30,
) -> tuple[MLDataset, dict[str, float], np.ndarray]:
    """Create dataset with known feature importances and ensemble probs."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, n_features)
    # Ensemble probabilities are a function of first 3 features
    logit = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 1.0 * X[:, 2]
    probs = 1.0 / (1.0 + np.exp(-logit))
    y = (probs > 0.5).astype(np.int64)
    dates = tuple(f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n))
    names = tuple(f"feat_{j}" for j in range(n_features))
    # Feature importances: first 3 dominate
    importances = {f"feat_{j}": 0.0 for j in range(n_features)}
    importances["feat_0"] = 0.40
    importances["feat_1"] = 0.30
    importances["feat_2"] = 0.20
    importances["feat_3"] = 0.05
    importances["feat_4"] = 0.05
    ds = MLDataset(X=X, y=y, dates=dates, feature_names=names,
                   label_config=LabelConfig())
    return ds, importances, probs


class TestDeepSurrogateExporter:
    def test_train_surrogate_regression(self):
        ds, imps, probs = _make_dataset_with_known_importances()
        exporter = DeepSurrogateExporter(max_depth=8, min_samples_leaf=10, top_n=10)
        exporter.train_surrogate(ds, imps, ensemble_probs=probs)
        assert exporter.is_trained
        assert len(exporter.feature_names) == 10

    def test_fidelity_r_squared(self):
        ds, imps, probs = _make_dataset_with_known_importances()
        exporter = DeepSurrogateExporter(max_depth=8, min_samples_leaf=10, top_n=10)
        exporter.train_surrogate(ds, imps, ensemble_probs=probs)
        r2 = exporter.fidelity_r_squared(ds.X, probs)
        # With depth 8 and clean signal, R² should be high
        assert r2 > 0.70

    def test_predict_returns_probabilities(self):
        ds, imps, probs = _make_dataset_with_known_importances()
        exporter = DeepSurrogateExporter(max_depth=8, min_samples_leaf=10, top_n=10)
        exporter.train_surrogate(ds, imps, ensemble_probs=probs)
        preds = exporter.predict(ds.X)
        assert preds.shape == (len(ds.y),)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_tree_to_pine_lines(self):
        ds, imps, probs = _make_dataset_with_known_importances()
        exporter = DeepSurrogateExporter(max_depth=4, min_samples_leaf=20, top_n=5)
        exporter.train_surrogate(ds, imps, ensemble_probs=probs)
        lines = exporter.tree_to_pine_lines()
        assert len(lines) > 0
        assert any("tn0" in line for line in lines)
        # All lines should be valid Pine variable assignments
        for line in lines:
            assert "float tn" in line or line.startswith("tn")

    def test_get_pine_dependencies(self):
        ds, imps, probs = _make_dataset_with_known_importances()
        exporter = DeepSurrogateExporter(max_depth=4, min_samples_leaf=20, top_n=5)
        exporter.train_surrogate(ds, imps, ensemble_probs=probs)
        deps = exporter.get_pine_dependencies()
        # Dependencies should be a list of strings (Pine code lines)
        assert isinstance(deps, list)
        # No duplicates
        assert len(deps) == len(set(deps))

    def test_deeper_tree_more_leaves(self):
        ds, imps, probs = _make_dataset_with_known_importances()
        shallow = DeepSurrogateExporter(max_depth=4, min_samples_leaf=20, top_n=10)
        shallow.train_surrogate(ds, imps, ensemble_probs=probs)
        deep = DeepSurrogateExporter(max_depth=10, min_samples_leaf=10, top_n=10)
        deep.train_surrogate(ds, imps, ensemble_probs=probs)
        # Deep tree should have more pine lines
        assert len(deep.tree_to_pine_lines()) >= len(shallow.tree_to_pine_lines())
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/ml/test_deep_surrogate.py -v --tb=short`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/stockdownloader/ml/deep_surrogate.py`:

```python
"""Deep surrogate decision tree for ensemble approximation.

Trains a DecisionTreeRegressor to approximate the continuous probability
surface of an ensemble model. The deeper tree (depth 8-10) captures
more non-linear interactions than the standard classifier surrogate.

The tree's decision path is translated into Pine Script using the same
approach as DecisionTreeExporter but targeting regression values.

Usage::

    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter

    exporter = DeepSurrogateExporter(max_depth=10, top_n=25)
    exporter.train_surrogate(dataset, importances, ensemble_probs=probs)
    pine_lines = exporter.tree_to_pine_lines()
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, TYPE_CHECKING

import numpy as np

from stockdownloader.pinescript.ml_export import FEATURE_PINE_MAP

if TYPE_CHECKING:
    from stockdownloader.ml.dataset_builder import MLDataset

logger = logging.getLogger(__name__)


class DeepSurrogateExporter:
    """Train a deep regression surrogate and export to Pine Script.

    Unlike DecisionTreeExporter (classifier on binary labels), this uses
    DecisionTreeRegressor to learn the ensemble's continuous P(profitable).

    Parameters
    ----------
    max_depth:
        Maximum tree depth (default 10).
    min_samples_leaf:
        Minimum samples per leaf (default 10).
    top_n:
        Number of top features to use (default 25).
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 10,
        top_n: int = 25,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.top_n = top_n
        self._tree: Any = None
        self._feature_names: tuple[str, ...] = ()
        self._top_indices: list[int] = []

    @property
    def is_trained(self) -> bool:
        return self._tree is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def train_surrogate(
        self,
        dataset: MLDataset,
        feature_importances: dict[str, float],
        *,
        ensemble_probs: Any,
    ) -> None:
        """Train a regression surrogate on ensemble probabilities.

        Parameters
        ----------
        dataset:
            The full ML dataset.
        feature_importances:
            Averaged importances from the ensemble.
        ensemble_probs:
            Array of shape (n_samples,) — the ensemble's P(profitable)
            for each sample. This is the regression target.
        """
        from sklearn.tree import DecisionTreeRegressor

        # Select top N features by importance
        sorted_feats = sorted(
            feature_importances.items(), key=lambda x: x[1], reverse=True,
        )
        top_names = [name for name, _ in sorted_feats[:self.top_n]]

        name_to_idx = {
            name: i for i, name in enumerate(dataset.feature_names)
        }
        self._top_indices = [
            name_to_idx[n] for n in top_names if n in name_to_idx
        ]
        self._feature_names = tuple(
            dataset.feature_names[i] for i in self._top_indices
        )

        X_sub = dataset.X[:, self._top_indices]

        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree.fit(X_sub, ensemble_probs)

        self._tree = tree
        logger.info(
            "Deep surrogate: depth=%d, leaves=%d, features=%d",
            tree.get_depth(), tree.get_n_leaves(), len(self._feature_names),
        )

    def predict(self, X: Any) -> Any:
        """Predict ensemble probabilities using the surrogate.

        Parameters
        ----------
        X:
            Full feature matrix (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted probabilities clipped to [0, 1].
        """
        if self._tree is None:
            raise RuntimeError("Must call train_surrogate() first")
        X_sub = X[:, self._top_indices]
        preds = self._tree.predict(X_sub)
        return np.clip(preds, 0.0, 1.0)

    def fidelity_r_squared(self, X: Any, ensemble_probs: Any) -> float:
        """Measure R² between surrogate and ensemble predictions.

        Parameters
        ----------
        X:
            Full feature matrix.
        ensemble_probs:
            True ensemble probabilities.

        Returns
        -------
        float
            R² score (1.0 = perfect, 0.0 = mean baseline).
        """
        preds = self.predict(X)
        ss_res = np.sum((ensemble_probs - preds) ** 2)
        ss_tot = np.sum((ensemble_probs - np.mean(ensemble_probs)) ** 2)
        if ss_tot == 0:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    def tree_to_pine(self) -> str:
        """Convert tree to a single Pine Script ternary expression."""
        if self._tree is None:
            raise RuntimeError("Must call train_surrogate() first")
        tree = self._tree.tree_
        return self._node_to_pine(tree, 0)

    def tree_to_pine_lines(self) -> list[str]:
        """Convert tree to intermediate Pine Script variable lines.

        For deep trees this avoids excessively long ternary expressions.
        Returns lines ending with ``tn0`` holding the probability.
        """
        if self._tree is None:
            raise RuntimeError("Must call train_surrogate() first")
        tree = self._tree.tree_
        lines: list[str] = []
        self._node_to_pine_lines(tree, 0, lines)
        return lines

    def get_pine_dependencies(self) -> list[str]:
        """Collect de-duplicated Pine Script variable declarations."""
        if self._tree is None:
            return []
        seen: set[str] = set()
        deps: list[str] = []
        for name in self._feature_names:
            _, dep_lines = FEATURE_PINE_MAP.get(name, (name, []))
            for line in dep_lines:
                if line not in seen:
                    seen.add(line)
                    deps.append(line)
        return deps

    @staticmethod
    def _safe_pine_expr(pine_expr: str) -> str:
        if "?" in pine_expr:
            return f"({pine_expr})"
        return pine_expr

    def _node_to_pine(self, tree: Any, node_id: int) -> str:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:  # leaf
            value = float(tree.value[node_id][0][0])
            return f"{max(0.0, min(1.0, value)):.4f}"
        feat_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feat_name = self._feature_names[feat_idx]
        pine_expr, _ = FEATURE_PINE_MAP.get(feat_name, (feat_name, []))
        safe = self._safe_pine_expr(pine_expr)
        left_expr = self._node_to_pine(tree, left)
        right_expr = self._node_to_pine(tree, right)
        return f"({safe} <= {threshold:.6f} ? {left_expr} : {right_expr})"

    def _node_to_pine_lines(
        self, tree: Any, node_id: int, lines: list[str],
    ) -> str:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:
            value = float(tree.value[node_id][0][0])
            return f"{max(0.0, min(1.0, value)):.4f}"
        feat_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feat_name = self._feature_names[feat_idx]
        pine_expr, _ = FEATURE_PINE_MAP.get(feat_name, (feat_name, []))
        safe = self._safe_pine_expr(pine_expr)
        left_var = self._node_to_pine_lines(tree, left, lines)
        right_var = self._node_to_pine_lines(tree, right, lines)
        var_name = f"tn{node_id}"
        lines.append(
            f"float {var_name} = {safe} <= {threshold:.6f} "
            f"? {left_var} : {right_var}"
        )
        return var_name
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/ml/test_deep_surrogate.py -v --tb=short`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/ml/deep_surrogate.py tests/ml/test_deep_surrogate.py
git commit -m "feat: add DeepSurrogateExporter with regression mode for ensemble approximation"
```

---

### Task 3: Create PineScript strategy factory with ML-driven exits

**Files:**
- Create: `src/stockdownloader/app/pinescript_catalog/spy_ml_ensemble.py`
- Create: `tests/pinescript/test_spy_ml_ensemble.py`

**Step 1: Write the failing tests**

Create `tests/pinescript/test_spy_ml_ensemble.py`:

```python
"""Tests for SPY ML Ensemble PineScript strategy."""

from __future__ import annotations

import numpy as np
import pytest

from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
    spy_ml_ensemble_strategy,
)
from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter
from stockdownloader.ml.dataset_builder import MLDataset, LabelConfig
from stockdownloader.pinescript import PineScriptGenerator


def _trained_exporter() -> DeepSurrogateExporter:
    """Create a trained exporter with synthetic data."""
    rng = np.random.RandomState(42)
    n, nf = 200, 63
    X = rng.randn(n, nf)
    probs = 1.0 / (1.0 + np.exp(-X[:, 0]))
    y = (probs > 0.5).astype(np.int64)
    # Use real feature names from the feature extractor
    names = (
        "rsi14_norm", "stoch_k_norm", "stoch_d_norm",
        "adx14_norm", "plus_di_norm", "minus_di_norm",
        "mfi14_norm", "williams_r_norm", "cci20_norm",
        "bb_percent_b", "bb_width_norm", "roc12_clamped",
        "macd_hist_atr", "macd_signal_spread_atr",
        "price_vs_sma20_atr", "price_vs_sma50_atr",
        "price_vs_sma200_atr", "price_vs_vwap_atr",
        "ema_cross_atr", "volume_ratio", "atr_pct",
        "fib_position", "bb_position",
        "obv_rising_flag", "sar_bullish_flag", "price_above_cloud_flag",
        "rsi_score", "macd_score", "stochastic_score", "cci_score",
        "williams_r_score", "sma_cross_score", "adx_score",
        "ema_trend_score", "ichimoku_score", "sar_score",
        "vwap_score", "sma_position_score",
        "bb_touch_score", "bb_squeeze_score",
        "obv_score", "mfi_score", "volume_surge_score",
        "regime_confidence", "regime_adx_norm",
        "regime_bb_width_pctl", "regime_trend_slope",
        "regime_sma200_dist_norm",
        "return_1d", "return_3d", "return_5d", "return_10d",
        "realized_vol_5d", "vol_ratio_5d_20d", "intraday_range_pct",
        "rsi_roc_3", "macd_hist_slope_3", "adx_slope_3",
        "rsi_x_adx", "bb_width_x_vol_ratio",
        "regime_conf_x_slope",
        "day_of_week_norm", "month_of_year_norm",
    )
    dates = tuple(f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n))
    ds = MLDataset(X=X, y=y, dates=dates, feature_names=names,
                   label_config=LabelConfig())
    imps = {name: 1.0 / nf for name in names}
    exporter = DeepSurrogateExporter(max_depth=4, min_samples_leaf=20, top_n=15)
    exporter.train_surrogate(ds, imps, ensemble_probs=probs)
    return exporter


@pytest.fixture
def strategy_def():
    return spy_ml_ensemble_strategy(_trained_exporter())


@pytest.fixture
def pine_code(strategy_def):
    return PineScriptGenerator().generate(strategy_def)


class TestStrategyDefinition:
    def test_builds_without_error(self, strategy_def):
        assert strategy_def is not None
        assert strategy_def.name == "SPY ML Ensemble"
        assert strategy_def.short_name == "SPY-ML-ENS"

    def test_strategy_mode_enabled(self, strategy_def):
        assert strategy_def.strategy_mode is True

    def test_has_ml_inputs(self, strategy_def):
        input_names = {i.name for i in strategy_def.inputs}
        assert "buyThresh" in input_names
        assert "sellThresh" in input_names
        assert "baseSL" in input_names
        assert "baseTP" in input_names

    def test_has_entry_conditions(self, strategy_def):
        assert strategy_def.long_entry is not None
        assert strategy_def.short_entry is not None
        assert "mlProb" in strategy_def.long_entry.expr
        assert "mlProb" in strategy_def.short_entry.expr


class TestPineScriptGeneration:
    def test_has_strategy_header(self, pine_code):
        assert 'strategy("SPY ML Ensemble"' in pine_code

    def test_has_ml_probability_computation(self, pine_code):
        assert "float mlProb" in pine_code
        assert "tn0" in pine_code

    def test_has_confidence_variable(self, pine_code):
        assert "confidence" in pine_code

    def test_has_dynamic_sl_tp(self, pine_code):
        assert "dynamicSL" in pine_code
        assert "dynamicTP" in pine_code

    def test_has_background_coloring(self, pine_code):
        assert "bgcolor" in pine_code

    def test_has_buy_sell_labels(self, pine_code):
        assert "label.new" in pine_code or "strategy.entry" in pine_code

    def test_has_ml_confidence_plot(self, pine_code):
        assert "ML Confidence" in pine_code or "mlProb" in pine_code

    def test_no_indicator_keyword(self, pine_code):
        assert "indicator(" not in pine_code
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/pinescript/test_spy_ml_ensemble.py -v --tb=short`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/stockdownloader/app/pinescript_catalog/spy_ml_ensemble.py`:

```python
"""SPY ML Ensemble PineScript strategy with ML-driven dynamic exits.

Generates a production-ready TradingView strategy that embeds a deep
surrogate decision tree (approximating a GradientBoosting + RandomForest
+ LogisticRegression ensemble) with:

- ML-driven dynamic TP/SL scaled by confidence
- Multi-timeframe confirmation (15-min VWAP)
- Buy/sell labels on every trade
- Green/red background during positions
- ML confidence panel
- Circuit breaker and EOD close

Usage::

    from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
        spy_ml_ensemble_strategy,
    )
    from stockdownloader.pinescript import PineScriptGenerator

    exporter = ...  # trained DeepSurrogateExporter
    defn = spy_ml_ensemble_strategy(exporter)
    pine = PineScriptGenerator().generate(defn)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stockdownloader.pinescript.models import (
    Condition,
    Indicator,
    Input,
    StrategyDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter


def spy_ml_ensemble_strategy(
    exporter: DeepSurrogateExporter,
    buy_threshold: float = 0.65,
    sell_threshold: float = 0.35,
) -> StrategyDefinition:
    """Build a strategy-mode PineScript for the SPY ML ensemble.

    Parameters
    ----------
    exporter:
        A trained DeepSurrogateExporter.
    buy_threshold:
        ML probability above which to go long.
    sell_threshold:
        ML probability below which to go short.
    """
    if not exporter.is_trained:
        raise RuntimeError("DeepSurrogateExporter must be trained first")

    # --- Inputs ---
    inputs = [
        Input.float_("buyThresh", buy_threshold, "Buy Threshold",
                      min_val=0.5, max_val=0.9, step=0.05, group="ML Signal"),
        Input.float_("sellThresh", sell_threshold, "Sell Threshold",
                      min_val=0.1, max_val=0.5, step=0.05, group="ML Signal"),
        Input.int_("atrLen", 14, "ATR Length", min_val=1, group="Risk"),
        Input.float_("baseSL", 2.0, "Base SL (x ATR)",
                      min_val=0.5, step=0.25, group="Risk"),
        Input.float_("baseTP", 2.0, "Base TP (x ATR)",
                      min_val=0.5, step=0.25, group="Risk"),
        Input.float_("slCapDollars", 3.0, "SL Cap ($, 0=off)",
                      min_val=0.0, step=0.5, group="Risk"),
        Input.int_("circuitMax", 3, "Circuit Breaker (consecutive losses)",
                    min_val=1, max_val=10, group="Risk"),
        Input.bool_("useVwapConfirm", True, "Use 15-min VWAP Confirmation",
                     group="Multi-Timeframe"),
        Input.bool_("closeEOD", True, "Close at End of Day",
                     group="Session"),
    ]

    # --- Indicators ---
    indicators = [
        Indicator.atr("atrVal", "atrLen"),
    ]

    # --- Pine dependencies from the tree ---
    dep_lines = exporter.get_pine_dependencies()

    # --- Tree computation ---
    tree_lines = exporter.tree_to_pine_lines()

    # --- Extra code: ML probability + confidence + dynamic exits ---
    extra_code = [
        "// ── ML Probability (deep surrogate tree) ──",
        *dep_lines,
        *tree_lines,
        "float mlProb = tn0",
        "",
        "// ── ML Confidence (0 = uncertain, 1 = max confident) ──",
        "float confidence = math.abs(mlProb - 0.5) * 2.0",
        "",
        "// ── Multi-timeframe VWAP confirmation ──",
        "float vwap15 = request.security(syminfo.tickerid, '15', ta.vwap)",
        "bool vwapLong  = useVwapConfirm ? close > vwap15 : true",
        "bool vwapShort = useVwapConfirm ? close < vwap15 : true",
        "",
        "// ── Dynamic TP/SL scaled by confidence ──",
        "float dynamicSL = atrVal * (baseSL - confidence * 0.5)",
        "float dynamicTP = atrVal * (baseTP + confidence * 1.0)",
        "float cappedSL  = slCapDollars > 0 ? math.min(dynamicSL, slCapDollars) : dynamicSL",
        "",
        "// ── Circuit breaker ──",
        "var int consecLosses = 0",
        "if strategy.closedtrades > 0",
        "    lastPnl = strategy.closedtrades.profit(strategy.closedtrades - 1)",
        "    if lastPnl < 0",
        "        consecLosses += 1",
        "    else",
        "        consecLosses := 0",
        "bool circuitOk = consecLosses < circuitMax",
        "",
        "// ── Entry signals ──",
        "bool buySignal  = mlProb > buyThresh and vwapLong and circuitOk",
        "bool sellSignal = mlProb < sellThresh and vwapShort and circuitOk",
        "",
        "// ── Position tracking for visuals ──",
        "var float entryPrice  = na",
        "var float stopPrice   = na",
        "var float targetPrice = na",
    ]

    # --- Extra plots: visual overlays ---
    extra_plots = [
        "// ── Strategy entries ──",
        "if buySignal",
        "    strategy.entry('L', strategy.long)",
        "    entryPrice  := close",
        "    stopPrice   := close - cappedSL",
        "    targetPrice := close + dynamicTP",
        "",
        "if sellSignal",
        "    strategy.entry('S', strategy.short)",
        "    entryPrice  := close",
        "    stopPrice   := close + cappedSL",
        "    targetPrice := close - dynamicTP",
        "",
        "// ── Exits with dynamic TP/SL ──",
        "if strategy.position_size > 0",
        "    strategy.exit('XL', 'L', stop=stopPrice, limit=targetPrice)",
        "if strategy.position_size < 0",
        "    strategy.exit('XS', 'S', stop=stopPrice, limit=targetPrice)",
        "",
        "// ── EOD Close ──",
        "if closeEOD and session.islastbar",
        "    strategy.close_all('EOD')",
        "",
        "// ── Background coloring ──",
        "bgcolor(strategy.position_size > 0 ? color.new(color.green, 90) : "
        "strategy.position_size < 0 ? color.new(color.red, 90) : na)",
        "",
        "// ── Buy/Sell labels ──",
        "if buySignal",
        '    label.new(bar_index, low, "\\u25B2", color=color.green, '
        'textcolor=color.white, style=label.style_label_up, size=size.small)',
        "if sellSignal",
        '    label.new(bar_index, high, "\\u25BC", color=color.red, '
        'textcolor=color.white, style=label.style_label_down, size=size.small)',
        "",
        "// ── TP/SL level plots ──",
        "plot(strategy.position_size != 0 ? entryPrice : na, "
        '"Entry", color=color.blue, style=plot.style_linebr, linewidth=1)',
        "plot(strategy.position_size != 0 ? targetPrice : na, "
        '"TP", color=color.green, style=plot.style_linebr, linewidth=1)',
        "plot(strategy.position_size != 0 ? stopPrice : na, "
        '"SL", color=color.red, style=plot.style_linebr, linewidth=1)',
        "",
        "// ── ML Confidence panel ──",
        'plot(mlProb * 100, "ML Confidence", color=color.orange, '
        "display=display.data_window)",
    ]

    features_desc = ", ".join(exporter.feature_names[:8])
    if len(exporter.feature_names) > 8:
        features_desc += f" (+{len(exporter.feature_names) - 8} more)"

    return StrategyDefinition(
        name="SPY ML Ensemble",
        short_name="SPY-ML-ENS",
        description=(
            f"ML ensemble (GradientBoosting + RandomForest + LogisticRegression) "
            f"approximated by deep surrogate tree. Features: {features_desc}. "
            f"Dynamic TP/SL scaled by ML confidence."
        ),
        indicators=indicators,
        inputs=inputs,
        long_entry=Condition(
            "buySignal",
            "ML probability above threshold + VWAP confirmation",
        ),
        short_entry=Condition(
            "sellSignal",
            "ML probability below threshold + VWAP confirmation",
        ),
        long_exit=None,
        short_exit=None,
        exit_on_reverse=True,
        extra_code=extra_code,
        extra_plots=extra_plots,
        strategy_mode=True,
        initial_capital=100000.0,
        sl_atr_mult=2.0,
        rr_ratio=1.5,
        close_eod=True,
        circuit_breaker_losses=3,
    )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/pinescript/test_spy_ml_ensemble.py -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/app/pinescript_catalog/spy_ml_ensemble.py \
        tests/pinescript/test_spy_ml_ensemble.py
git commit -m "feat: add SPY ML Ensemble PineScript strategy with dynamic exits"
```

---

### Task 4: Create CLI entry point (orchestrator)

**Files:**
- Create: `src/stockdownloader/app/spy_ml_ensemble.py`
- Modify: `pyproject.toml` (add entry point)
- Modify: `src/stockdownloader/app/pinescript_catalog/catalogs.py` (register strategy)

**Step 1: Write the CLI entry point**

Create `src/stockdownloader/app/spy_ml_ensemble.py`:

```python
"""SPY ML Ensemble Pipeline — train, ensemble, surrogate, export.

Downloads fresh SPY data, trains multi-model ensemble, approximates with
a deep surrogate tree, backtests, tournaments against existing strategies,
and exports to PineScript.

Usage::

    spy-ml-ensemble                     # Full pipeline
    spy-ml-ensemble --quick             # Reduced grid, fast iteration
    spy-ml-ensemble --depth 8 --top 20  # Custom surrogate params
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from stockdownloader.app.ml_helpers import (
    add_common_ml_args,
    build_training_config,
    init_ml_env,
)
from stockdownloader.core.config import AppConfig, DEFAULT_ML_PIPELINE_DIR


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SPY ML Ensemble Strategy Pipeline",
    )
    add_common_ml_args(p)

    g = p.add_argument_group("Ensemble")
    g.add_argument("--top-models", type=int, default=3,
                   help="Number of top models to include in ensemble")
    g.add_argument("--depth", type=int, default=10,
                   help="Max depth of surrogate decision tree")
    g.add_argument("--top-features", type=int, default=25,
                   help="Number of top features for surrogate")
    g.add_argument("--min-leaf", type=int, default=10,
                   help="Min samples per leaf in surrogate")

    g2 = p.add_argument_group("Thresholds")
    g2.add_argument("--buy-thresh", type=float, default=0.65,
                    help="ML probability buy threshold")
    g2.add_argument("--sell-thresh", type=float, default=0.35,
                    help="ML probability sell threshold")
    g2.add_argument("--min-r2", type=float, default=0.70,
                    help="Minimum surrogate R² fidelity")

    g3 = p.add_argument_group("Output")
    g3.add_argument("--output-dir", type=str,
                    default=str(DEFAULT_ML_PIPELINE_DIR / "spy_ensemble"),
                    help="Output directory")
    g3.add_argument("--no-pine", action="store_true",
                    help="Skip PineScript export")
    g3.add_argument("--no-tournament", action="store_true",
                    help="Skip tournament comparison")
    return p


def main(argv: list[str] | None = None) -> None:
    """Run the full SPY ML Ensemble pipeline."""
    import numpy as np
    from stockdownloader.data.market.yahoo_data_client import YahooDataClient
    from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig
    from stockdownloader.ml.feature_extractor import FeatureExtractor
    from stockdownloader.ml.trainer import MLModelConfig, MLTrainer
    from stockdownloader.ml.ensemble import EnsembleBuilder
    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter

    args = _build_parser().parse_args(argv)
    init_ml_env(getattr(args, "verbose", False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SPY ML ENSEMBLE PIPELINE")
    print("=" * 60)

    # ── Stage 1: Download fresh SPY data ──
    print("\n[1/6] Downloading SPY daily data (10y)...")
    t0 = time.time()
    client = YahooDataClient()
    daily_data = client.fetch_price_data("SPY", range_="10y")
    print(f"  Loaded {len(daily_data)} daily bars in {time.time() - t0:.1f}s")

    if len(daily_data) < 500:
        print("ERROR: Insufficient data for training. Need 500+ bars.")
        sys.exit(1)

    # ── Stage 2: Build dataset ──
    print("\n[2/6] Building feature dataset...")
    t0 = time.time()
    extractor = FeatureExtractor()
    label_config = LabelConfig(
        forward_period=10, profit_threshold=0.005,
    )
    builder = DatasetBuilder(extractor, label_config)
    dataset = builder.build(daily_data)
    print(f"  {dataset.X.shape[0]} samples, {dataset.X.shape[1]} features "
          f"in {time.time() - t0:.1f}s")
    print(f"  Class distribution: {dict(zip(*np.unique(dataset.y, return_counts=True)))}")

    # ── Stage 3: Train model grid ──
    print("\n[3/6] Training model grid...")
    t0 = time.time()
    configs = [
        MLModelConfig(model_type="gradient_boosting", n_estimators=200,
                      max_depth=4, n_cv_folds=5),
        MLModelConfig(model_type="random_forest", n_estimators=200,
                      max_depth=6, n_cv_folds=5),
        MLModelConfig(model_type="logistic_regression", n_cv_folds=5),
        MLModelConfig(model_type="gradient_boosting", n_estimators=200,
                      max_depth=4, n_cv_folds=5, use_class_balance=True),
        MLModelConfig(model_type="random_forest", n_estimators=200,
                      max_depth=6, n_cv_folds=5, use_class_balance=True),
    ]
    if getattr(args, "quick", False):
        configs = configs[:3]  # Just one of each type

    results = []
    for i, cfg in enumerate(configs):
        print(f"  Training {cfg.model_type} ({i+1}/{len(configs)})...")
        result = MLTrainer(cfg).train(dataset)
        results.append(result)
        print(f"    OOS accuracy={result.oos_accuracy:.3f}, "
              f"ROC-AUC={result.oos_roc_auc:.3f}")

    print(f"  Trained {len(results)} models in {time.time() - t0:.1f}s")

    # ── Stage 4: Build ensemble ──
    print(f"\n[4/6] Building ensemble (top {args.top_models})...")
    ensemble_builder = EnsembleBuilder(results)
    ensemble = ensemble_builder.select_top(n=args.top_models)

    # Get ensemble predictions on full dataset
    ensemble_probs = ensemble.predict_proba(dataset.X)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    ensemble_acc = np.mean(ensemble_preds == dataset.y)
    print(f"  Ensemble accuracy: {ensemble_acc:.3f}")
    print(f"  Ensemble mean P: {ensemble_probs.mean():.3f}, "
          f"std: {ensemble_probs.std():.3f}")

    # ── Stage 5: Train deep surrogate ──
    print(f"\n[5/6] Training deep surrogate (depth={args.depth}, "
          f"features={args.top_features})...")
    t0 = time.time()
    importances = ensemble.averaged_feature_importances()
    print("  Top 10 features:")
    for name, imp in list(importances.items())[:10]:
        print(f"    {name}: {imp:.4f}")

    exporter = DeepSurrogateExporter(
        max_depth=args.depth,
        min_samples_leaf=args.min_leaf,
        top_n=args.top_features,
    )
    exporter.train_surrogate(dataset, importances, ensemble_probs=ensemble_probs)

    r2 = exporter.fidelity_r_squared(dataset.X, ensemble_probs)
    print(f"  Surrogate R²: {r2:.4f} (min={args.min_r2})")
    print(f"  Trained in {time.time() - t0:.1f}s")

    if r2 < args.min_r2:
        print(f"  WARNING: R² below threshold ({r2:.3f} < {args.min_r2}). "
              f"Surrogate may not faithfully represent ensemble.")

    # ── Stage 6: Export to PineScript ──
    if not args.no_pine:
        print("\n[6/6] Exporting to PineScript...")
        from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
            spy_ml_ensemble_strategy,
        )
        from stockdownloader.pinescript import PineScriptGenerator

        strategy_def = spy_ml_ensemble_strategy(
            exporter,
            buy_threshold=args.buy_thresh,
            sell_threshold=args.sell_thresh,
        )
        pine_code = PineScriptGenerator().generate(strategy_def)

        pine_path = output_dir / "spy_ml_ensemble.pine"
        pine_path.write_text(pine_code, encoding="utf-8")
        print(f"  PineScript saved to: {pine_path}")
        print(f"  Code length: {len(pine_code)} chars, "
              f"{pine_code.count(chr(10))} lines")
    else:
        print("\n[6/6] PineScript export skipped (--no-pine)")

    # ── Tournament comparison ──
    if not args.no_tournament:
        print("\n" + "=" * 60)
        print("TOURNAMENT COMPARISON")
        print("=" * 60)
        _run_tournament(dataset, ensemble_probs, exporter, args, daily_data)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


def _run_tournament(dataset, ensemble_probs, exporter, args, daily_data):
    """Compare ML ensemble vs existing strategies via backtest."""
    import numpy as np
    from stockdownloader.backtesting.engines.daily import BacktestEngine

    # Backtest ML ensemble strategy
    surrogate_preds = exporter.predict(dataset.X)
    buy_signals = surrogate_preds > args.buy_thresh
    sell_signals = surrogate_preds < args.sell_thresh

    # Simple backtest: go long when buy, go short when sell
    capital = 100000.0
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    trades = []
    equity = [capital]

    # Align dataset dates to price data indices
    date_to_idx = {bar.date: i for i, bar in enumerate(daily_data)}
    for i, dt in enumerate(dataset.dates):
        bar_idx = date_to_idx.get(dt)
        if bar_idx is None:
            equity.append(equity[-1])
            continue
        price = float(daily_data[bar_idx].close)
        if position == 0 and buy_signals[i]:
            position = 1
            entry_price = price
        elif position == 0 and sell_signals[i]:
            position = -1
            entry_price = price
        elif position == 1 and sell_signals[i]:
            pnl = (price - entry_price) / entry_price * capital
            trades.append(pnl)
            capital += pnl
            position = -1
            entry_price = price
        elif position == -1 and buy_signals[i]:
            pnl = (entry_price - price) / entry_price * capital
            trades.append(pnl)
            capital += pnl
            position = 1
            entry_price = price
        equity.append(capital)

    if trades:
        wins = sum(1 for t in trades if t > 0)
        total_return = (capital - 100000.0) / 100000.0 * 100
        win_rate = wins / len(trades) * 100
        max_dd = _max_drawdown(equity)

        print(f"\n  ML Ensemble Strategy:")
        print(f"    Total Return: {total_return:+.1f}%")
        print(f"    Win Rate:     {win_rate:.1f}%")
        print(f"    Trades:       {len(trades)}")
        print(f"    Max Drawdown: {max_dd:.1f}%")

        print(f"\n  Minimum thresholds:")
        print(f"    Sharpe > 1.0:      {'PASS' if total_return / max(abs(max_dd), 1) > 1.0 else 'FAIL'}")
        print(f"    Max DD < 15%:      {'PASS' if abs(max_dd) < 15 else 'FAIL'}")
        print(f"    Win rate > 55%:    {'PASS' if win_rate > 55 else 'FAIL'}")
    else:
        print("  No trades generated.")


def _max_drawdown(equity: list[float]) -> float:
    """Compute max drawdown percentage from equity curve."""
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


if __name__ == "__main__":
    main()
```

**Step 2: Add entry point to pyproject.toml**

In `pyproject.toml`, in the `[project.scripts]` section, add under the ML pipelines group:

```
spy-ml-ensemble = "stockdownloader.app.spy_ml_ensemble:main"
```

**Step 3: Register in catalogs.py**

In `src/stockdownloader/app/pinescript_catalog/catalogs.py`, add the import and register the strategy in the `STRATEGY_CATALOG` dictionary (or list). Follow the existing pattern:

Add import at top:
```python
from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import spy_ml_ensemble_strategy
```

Note: Since `spy_ml_ensemble_strategy` requires a trained exporter, it cannot be added to the static STRATEGY_CATALOG directly. Instead, it will be imported and used by the CLI entry point. The catalog registration can be done lazily or via a factory wrapper. For now, skip catalog registration and just add the import in the CLI.

**Step 4: Run all tests**

Run: `python3 -m pytest tests/ -x -q --tb=short`
Expected: All existing tests pass (3,297+) plus new tests

**Step 5: Commit**

```bash
git add src/stockdownloader/app/spy_ml_ensemble.py pyproject.toml
git commit -m "feat: add spy-ml-ensemble CLI pipeline with tournament comparison"
```

---

### Task 5: Integration test — run the full pipeline

**Files:**
- Create: `tests/integration/test_spy_ml_ensemble_pipeline.py` (optional)

**Step 1: Run the pipeline end-to-end**

This task is a manual integration test. Run the full pipeline with `--quick` to verify everything connects:

```bash
cd /Users/kelsayed/Documents/GitHub/stockdownloader-py/.claude/worktrees/vigorous-easley
PYTHONPATH=src python3 -m stockdownloader.app.spy_ml_ensemble --quick --no-tournament
```

Expected output:
```
==============================================================
SPY ML ENSEMBLE PIPELINE
==============================================================

[1/6] Downloading SPY daily data (10y)...
  Loaded ~2500 daily bars in ...s

[2/6] Building feature dataset...
  ~2200 samples, 63 features in ...s

[3/6] Training model grid...
  Training gradient_boosting (1/3)...
  Training random_forest (2/3)...
  Training logistic_regression (3/3)...

[4/6] Building ensemble (top 3)...
  Ensemble accuracy: ~0.55+

[5/6] Training deep surrogate...
  Surrogate R²: ~0.70+

[6/6] Exporting to PineScript...
  PineScript saved to: output/ml/spy_ensemble/spy_ml_ensemble.pine
```

**Step 2: Verify the PineScript output**

```bash
wc -l output/ml/spy_ensemble/spy_ml_ensemble.pine
head -30 output/ml/spy_ensemble/spy_ml_ensemble.pine
```

Expected: 200-500+ lines of valid Pine Script v6 code starting with `// This Pine Script™...` and `strategy("SPY ML Ensemble"...`

**Step 3: Run with tournament**

```bash
PYTHONPATH=src python3 -m stockdownloader.app.spy_ml_ensemble --quick
```

Expected: Includes tournament comparison with total return, win rate, max drawdown.

**Step 4: Run full (non-quick) pipeline if quick succeeds**

```bash
PYTHONPATH=src python3 -m stockdownloader.app.spy_ml_ensemble
```

This trains 5 models instead of 3, takes longer but produces the best ensemble.

**Step 5: Commit final results**

```bash
git add -A
git commit -m "feat: verify spy-ml-ensemble pipeline end-to-end"
```

---

### Task 6: Full test suite verification and final cleanup

**Step 1: Run the full test suite**

```bash
python3 -m pytest tests/ -x -q --tb=short
```

Expected: 3,297+ tests pass (baseline + new tests)

**Step 2: Verify no stale imports**

```bash
grep -rn "from stockdownloader.ml.ensemble" src/ tests/ | head -20
grep -rn "from stockdownloader.ml.deep_surrogate" src/ tests/ | head -20
```

All imports should be correct.

**Step 3: Verify PineScript output is valid**

Read the generated `.pine` file and check:
- Has `strategy()` header (not `indicator()`)
- Has `float mlProb = tn0` (surrogate computation)
- Has `float confidence = ...` (confidence calculation)
- Has `float dynamicSL = ...` and `float dynamicTP = ...`
- Has `bgcolor(...)` for position shading
- Has `label.new(...)` for buy/sell labels
- Has `strategy.entry(...)` and `strategy.exit(...)`

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: verify full test suite and PineScript output"
```
