# ML Algorithm Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add XGBoost, LightGBM, CatBoost, SVM, MLP, and KNN to `MLTrainer`, implement diversity-aware ensemble selection with stacking, and create `spy-ml-mega` pipeline.

**Architecture:** Dual-track. Track A adds 6 new model types to `MLTrainer._build_model()` with graceful `ImportError` handling. Track B adds `select_diverse()` and stacking to `EnsembleBuilder`/`EnsemblePredictor`, then wires everything into a new `spy-ml-mega` CLI pipeline.

**Tech Stack:** scikit-learn, xgboost, lightgbm, catboost, numpy, pytest

---

### Task 1: Add Dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml:20-28`

**Context:** Currently `[project.optional-dependencies]` has `ml = ["scikit-learn>=1.4.0", "joblib>=1.3.0"]`. We need to add xgboost, lightgbm, catboost.

**Step 1: Add new ML dependencies**

In `pyproject.toml`, update the `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=6.0.0",
]
ml = [
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
    "xgboost>=2.0",
    "lightgbm>=4.0",
    "catboost>=1.2",
]
```

**Step 2: Install the new dependencies**

Run: `pip install -e '.[ml]'`
Expected: Successfully installs xgboost, lightgbm, catboost

**Step 3: Verify imports work**

Run: `python3 -c "import xgboost; import lightgbm; import catboost; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add xgboost, lightgbm, catboost to ml dependencies"
```

---

### Task 2: Add 6 New Model Types to MLTrainer

**Files:**
- Modify: `src/stockdownloader/ml/trainer.py:251-314` (`_build_model` method)
- Modify: `src/stockdownloader/ml/trainer.py:359-379` (`_extract_importances` method)
- Test: `tests/ml/test_trainer.py`

**Context:** `_build_model()` currently supports 6 model types via an if-chain (lines 251-314). Each returns a sklearn-compatible estimator. The method ends with a `ValueError` listing all valid types. `_extract_importances()` (lines 359-379) checks for `feature_importances_` then `coef_` then returns zeros. The `_compute_sample_weights()` static method (lines 385-391) creates inverse-frequency weights passed to `model.fit(..., sample_weight=...)`.

**Step 1: Write failing tests for all 6 new model types**

Add to `tests/ml/test_trainer.py`. Follow the existing pattern: create a config, train, assert accuracy > 0.5 and config.model_type matches.

```python
# At the bottom of tests/ml/test_trainer.py, after existing test classes.

import pytest


class TestMLTrainerXGBoost:
    """Tests for XGBoost model type."""

    def test_xgboost_trains(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(
            model_type="xgboost",
            n_estimators=50,
            max_depth=3,
            n_cv_folds=2,
        )
        result = MLTrainer(cfg).train(separable_dataset)
        assert result.oos_accuracy >= 0.5
        assert result.config.model_type == "xgboost"

    def test_xgboost_predict_proba(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="xgboost", n_estimators=50, max_depth=3, n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        probs = result.model.predict_proba(separable_dataset.X[:5])
        assert probs.shape == (5, 2)

    def test_xgboost_feature_importances(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="xgboost", n_estimators=50, max_depth=3, n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        assert len(result.feature_importances) == separable_dataset.X.shape[1]

    def test_xgboost_balanced(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(
            model_type="xgboost",
            n_estimators=50,
            max_depth=3,
            n_cv_folds=2,
            use_class_balance=True,
        )
        result = MLTrainer(cfg).train(separable_dataset)
        assert result.oos_accuracy >= 0.5


class TestMLTrainerLightGBM:
    """Tests for LightGBM model type."""

    def test_lightgbm_trains(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(
            model_type="lightgbm",
            n_estimators=50,
            max_depth=3,
            n_cv_folds=2,
        )
        result = MLTrainer(cfg).train(separable_dataset)
        assert result.oos_accuracy >= 0.5
        assert result.config.model_type == "lightgbm"

    def test_lightgbm_predict_proba(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="lightgbm", n_estimators=50, max_depth=3, n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        probs = result.model.predict_proba(separable_dataset.X[:5])
        assert probs.shape == (5, 2)

    def test_lightgbm_balanced(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(
            model_type="lightgbm",
            n_estimators=50,
            max_depth=3,
            n_cv_folds=2,
            use_class_balance=True,
        )
        result = MLTrainer(cfg).train(separable_dataset)
        assert result.oos_accuracy >= 0.5


class TestMLTrainerCatBoost:
    """Tests for CatBoost model type."""

    def test_catboost_trains(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(
            model_type="catboost",
            n_estimators=50,
            max_depth=3,
            n_cv_folds=2,
        )
        result = MLTrainer(cfg).train(separable_dataset)
        assert result.oos_accuracy >= 0.5
        assert result.config.model_type == "catboost"

    def test_catboost_predict_proba(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="catboost", n_estimators=50, max_depth=3, n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        probs = result.model.predict_proba(separable_dataset.X[:5])
        assert probs.shape == (5, 2)

    def test_catboost_balanced(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(
            model_type="catboost",
            n_estimators=50,
            max_depth=3,
            n_cv_folds=2,
            use_class_balance=True,
        )
        result = MLTrainer(cfg).train(separable_dataset)
        assert result.oos_accuracy >= 0.5


class TestMLTrainerSVM:
    """Tests for SVM model type."""

    def test_svm_trains(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="svm", n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        assert result.oos_accuracy >= 0.5
        assert result.config.model_type == "svm"

    def test_svm_predict_proba(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="svm", n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        probs = result.model.predict_proba(separable_dataset.X[:5])
        assert probs.shape == (5, 2)

    def test_svm_feature_importances_uniform(self, separable_dataset: MLDataset) -> None:
        """SVM with RBF kernel has no native importances, should be uniform."""
        cfg = MLModelConfig(model_type="svm", n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        # RBF kernel = no coef_ => all zeros
        vals = list(result.feature_importances.values())
        assert all(v == 0.0 for v in vals)


class TestMLTrainerMLP:
    """Tests for MLP (neural network) model type."""

    def test_mlp_trains(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="mlp", n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        assert result.oos_accuracy >= 0.5
        assert result.config.model_type == "mlp"

    def test_mlp_predict_proba(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="mlp", n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        probs = result.model.predict_proba(separable_dataset.X[:5])
        assert probs.shape == (5, 2)

    def test_mlp_feature_importances_nonzero(self, separable_dataset: MLDataset) -> None:
        """MLP uses input layer weights as proxy importances."""
        cfg = MLModelConfig(model_type="mlp", n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        assert len(result.feature_importances) > 0
        assert any(v > 0 for v in result.feature_importances.values())


class TestMLTrainerKNN:
    """Tests for KNN model type."""

    def test_knn_trains(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="knn", n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        assert result.oos_accuracy >= 0.5
        assert result.config.model_type == "knn"

    def test_knn_predict_proba(self, separable_dataset: MLDataset) -> None:
        cfg = MLModelConfig(model_type="knn", n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        probs = result.model.predict_proba(separable_dataset.X[:5])
        assert probs.shape == (5, 2)

    def test_knn_feature_importances_uniform(self, separable_dataset: MLDataset) -> None:
        """KNN has no native importances, should be all zeros."""
        cfg = MLModelConfig(model_type="knn", n_cv_folds=2)
        result = MLTrainer(cfg).train(separable_dataset)
        vals = list(result.feature_importances.values())
        assert all(v == 0.0 for v in vals)
```

Note: The `separable_dataset` fixture already exists in the test file (function `_make_separable_dataset`). If it's not a fixture, create one:

```python
@pytest.fixture
def separable_dataset() -> MLDataset:
    return _make_separable_dataset()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/ml/test_trainer.py::TestMLTrainerXGBoost -x -v`
Expected: FAIL with `ValueError: Unknown model_type: 'xgboost'`

**Step 3: Implement new model types in `_build_model()`**

Add 6 new if-blocks before the final `raise ValueError` in `_build_model()` (currently line 309):

```python
        # -- External boosting libraries (graceful ImportError) --
        if mt == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError(
                    "model_type='xgboost' requires: pip install 'xgboost>=2.0'"
                ) from None
            return XGBClassifier(
                n_estimators=self._config.n_estimators,
                max_depth=self._config.max_depth,
                learning_rate=self._config.learning_rate,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
            )
        if mt == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError:
                raise ImportError(
                    "model_type='lightgbm' requires: pip install 'lightgbm>=4.0'"
                ) from None
            return LGBMClassifier(
                n_estimators=self._config.n_estimators,
                max_depth=self._config.max_depth,
                learning_rate=self._config.learning_rate,
                min_child_samples=self._config.min_samples_leaf,
                random_state=42,
                verbose=-1,
            )
        if mt == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError:
                raise ImportError(
                    "model_type='catboost' requires: pip install 'catboost>=1.2'"
                ) from None
            return CatBoostClassifier(
                iterations=self._config.n_estimators,
                depth=self._config.max_depth,
                learning_rate=self._config.learning_rate,
                random_seed=42,
                verbose=0,
            )

        # -- sklearn non-tree models --
        if mt == "svm":
            from sklearn.svm import SVC
            return SVC(
                kernel="rbf",
                probability=True,
                C=1.0,
                gamma="scale",
                random_state=42,
            )
        if mt == "mlp":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                random_state=42,
            )
        if mt == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                n_jobs=-1,
            )
```

Also update the final `raise ValueError` message to include all model types:

```python
        raise ValueError(
            f"Unknown model_type: {mt!r}. "
            f"Expected one of: 'gradient_boosting', 'random_forest', "
            f"'extra_trees', 'hist_gradient_boosting', 'adaboost', "
            f"'logistic_regression', 'xgboost', 'lightgbm', 'catboost', "
            f"'svm', 'mlp', 'knn'."
        )
```

**Step 4: Update `_extract_importances()` for MLP**

The existing method already handles `feature_importances_` (tree models, XGBoost, LightGBM, CatBoost all have this), `coef_` (LogisticRegression), and falls back to zeros (SVM-RBF, KNN). For MLP, we need to add support for `coefs_` (note: plural, not `coef_`):

After the `elif hasattr(model, "coef_"):` block, add:

```python
        elif hasattr(model, "coefs_"):
            # MLP: use absolute mean of input layer weights as proxy importances
            raw = np.abs(model.coefs_[0]).mean(axis=1)
```

**Step 5: Handle class balance for external boosters**

The `_compute_sample_weights()` approach works for models that accept `sample_weight` in `fit()`. XGBoost, LightGBM, and CatBoost all support `sample_weight`, so the existing code at lines 171-183 works unchanged.

However, some models do NOT support `sample_weight` in `fit()`:
- **KNN**: `fit()` does not accept `sample_weight`
- **MLP**: `fit()` does not accept `sample_weight`

Wrap the `model.fit()` call at line 183 in a try/except to handle models that reject sample_weight:

```python
        try:
            model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        except TypeError:
            # Model doesn't support sample_weight (e.g., KNN, MLP)
            model.fit(X_train_scaled, y_train)
```

Apply the same pattern to the `_cross_validate()` method where `model.fit()` is called with sample_weight.

**Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/ml/test_trainer.py -x -v`
Expected: All tests pass (existing + new)

**Step 7: Also update the unknown model type test**

The existing test `test_unknown_model_type` in `TestMLTrainerErrors` uses `model_type="xgboost"` to test the error case. Since `xgboost` is now valid, change it to `model_type="nonexistent_model"`.

**Step 8: Run full test suite**

Run: `PYTHONPATH=src python3 -m pytest tests/ -x -q`
Expected: All tests pass

**Step 9: Commit**

```bash
git add src/stockdownloader/ml/trainer.py tests/ml/test_trainer.py
git commit -m "feat: add xgboost, lightgbm, catboost, svm, mlp, knn to MLTrainer"
```

---

### Task 3: Diversity-Aware Ensemble Selection

**Files:**
- Modify: `src/stockdownloader/ml/ensemble.py:139-185`
- Test: `tests/ml/test_ensemble.py`

**Context:** `EnsembleBuilder` (lines 139-185) currently has only `select_top(n)` which ranks by `oos_accuracy`. We need to add `select_diverse(n, diversity_weight)` that balances accuracy with prediction diversity.

**Step 1: Write failing tests for `select_diverse()`**

Add new test class to `tests/ml/test_ensemble.py`:

```python
class TestEnsembleBuilderDiversity:
    """Tests for diversity-aware ensemble selection."""

    def test_select_diverse_returns_ensemble(self) -> None:
        results = _train_multiple_models()
        builder = EnsembleBuilder(results)
        ensemble = builder.select_diverse(n=2)
        assert isinstance(ensemble, EnsemblePredictor)
        assert ensemble.n_models == 2

    def test_select_diverse_default_n(self) -> None:
        results = _train_multiple_models()
        builder = EnsembleBuilder(results)
        ensemble = builder.select_diverse()
        assert ensemble.n_models == 3  # default n=3

    def test_select_diverse_includes_best_model(self) -> None:
        """The best model by accuracy should always be in the ensemble."""
        results = _train_multiple_models()
        builder = EnsembleBuilder(results)
        ensemble = builder.select_diverse(n=2)
        # Best model by accuracy
        best = sorted(results, key=lambda r: r.oos_accuracy, reverse=True)[0]
        ensemble_types = [r.config.model_type for r in ensemble.results]
        assert best.config.model_type in ensemble_types

    def test_select_diverse_n_exceeds_available(self) -> None:
        results = _train_multiple_models()
        builder = EnsembleBuilder(results)
        ensemble = builder.select_diverse(n=100)
        assert ensemble.n_models == len(results)

    def test_select_diverse_single_model(self) -> None:
        results = _train_multiple_models()
        builder = EnsembleBuilder(results)
        ensemble = builder.select_diverse(n=1)
        assert ensemble.n_models == 1

    def test_select_diverse_diversity_weight_zero_matches_top(self) -> None:
        """With diversity_weight=0, should behave like select_top."""
        results = _train_multiple_models()
        builder = EnsembleBuilder(results)
        diverse = builder.select_diverse(n=2, diversity_weight=0.0)
        top = builder.select_top(n=2)
        # Same model types selected (order might differ)
        diverse_types = sorted(r.config.model_type for r in diverse.results)
        top_types = sorted(r.config.model_type for r in top.results)
        assert diverse_types == top_types

    def test_select_diverse_empty_raises(self) -> None:
        builder = EnsembleBuilder([_train_multiple_models()[0]])
        ensemble = builder.select_diverse(n=1)
        assert ensemble.n_models == 1
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/ml/test_ensemble.py::TestEnsembleBuilderDiversity -x -v`
Expected: FAIL with `AttributeError: 'EnsembleBuilder' object has no attribute 'select_diverse'`

**Step 3: Implement `select_diverse()` in `EnsembleBuilder`**

Add to `EnsembleBuilder` class in `ensemble.py`, after the existing `select_top()` method:

```python
    def select_diverse(
        self,
        n: int = 3,
        diversity_weight: float = 0.4,
    ) -> EnsemblePredictor:
        """Select top *n* models balancing accuracy with prediction diversity.

        Uses greedy selection: seeds with the best model, then iteratively
        adds the candidate that maximises
        ``(1 - w) * accuracy + w * (1 - max_corr)``
        where *max_corr* is the maximum Pearson correlation between the
        candidate's OOS predictions and any existing member.

        Parameters
        ----------
        n:
            Number of models to select (default 3).
        diversity_weight:
            Weight for diversity vs accuracy (default 0.4).
            0.0 = pure accuracy (same as select_top).
            1.0 = pure diversity.
        """
        import numpy as np

        n = min(n, len(self._results))
        if n <= 0:
            raise ValueError("n must be >= 1")

        # Sort by accuracy descending
        ranked = sorted(
            self._results,
            key=lambda r: r.oos_accuracy,
            reverse=True,
        )

        # Seed with best model
        selected: list[TrainingResult] = [ranked[0]]
        remaining = list(ranked[1:])

        # We need OOS predictions for correlation.
        # Since we don't store raw predictions in TrainingResult,
        # we use accuracy as the proxy signal and model type diversity
        # as the diversity signal. For a lightweight approach:
        # correlation = 1.0 if same model_type family, 0.5 otherwise.
        # For proper correlation, we'd need stored predictions.

        while len(selected) < n and remaining:
            best_score = -1.0
            best_idx = 0

            for i, candidate in enumerate(remaining):
                acc = candidate.oos_accuracy

                # Compute diversity: max correlation with existing members
                # Use feature importance overlap as proxy for prediction correlation
                max_corr = 0.0
                for member in selected:
                    corr = self._importance_correlation(
                        candidate.feature_importances,
                        member.feature_importances,
                    )
                    max_corr = max(max_corr, corr)

                score = (1.0 - diversity_weight) * acc + diversity_weight * (1.0 - max_corr)
                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        logger.info(
            "EnsembleBuilder selected %d diverse models (accuracies: %s)",
            len(selected),
            [round(r.oos_accuracy, 4) for r in selected],
        )
        return EnsemblePredictor(selected)

    @staticmethod
    def _importance_correlation(
        imp_a: dict[str, float],
        imp_b: dict[str, float],
    ) -> float:
        """Pearson correlation between two feature importance vectors."""
        import numpy as np

        keys = sorted(set(imp_a) | set(imp_b))
        if not keys:
            return 0.0
        a = np.array([imp_a.get(k, 0.0) for k in keys])
        b = np.array([imp_b.get(k, 0.0) for k in keys])
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/ml/test_ensemble.py -x -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/stockdownloader/ml/ensemble.py tests/ml/test_ensemble.py
git commit -m "feat: add diversity-aware ensemble selection (select_diverse)"
```

---

### Task 4: Stacking Ensemble Method

**Files:**
- Modify: `src/stockdownloader/ml/ensemble.py:32-136` (`EnsemblePredictor`)
- Create: `tests/ml/test_ensemble_stacking.py`

**Context:** `EnsemblePredictor.predict_proba()` (lines 66-100) currently averages `predict_proba[:, 1]` across all models (soft voting). We need to add a `"stacking"` mode where a `LogisticRegression` meta-learner is trained on base model OOS predictions to learn optimal per-model weights.

**Step 1: Write failing tests for stacking**

Create `tests/ml/test_ensemble_stacking.py`:

```python
"""Tests for stacking ensemble method."""

from __future__ import annotations

import numpy as np
import pytest

from stockdownloader.ml.ensemble import EnsemblePredictor, EnsembleBuilder
from stockdownloader.ml.trainer import MLModelConfig, MLTrainer
from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig, MLDataset
from stockdownloader.ml.feature_extractor import FeatureExtractor


def _make_separable_dataset(n: int = 200) -> MLDataset:
    """Create a dataset with separable classes for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, 10)
    # Class determined by first two features
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    dates = tuple(f"2020-01-{i+1:03d}" for i in range(n))
    feature_names = tuple(f"feat_{i}" for i in range(10))
    label_config = LabelConfig(forward_period=10, profit_threshold=0.005)
    return MLDataset(X=X, y=y, dates=dates, feature_names=feature_names, label_config=label_config)


def _train_models() -> list:
    ds = _make_separable_dataset()
    results = []
    for mt in ["gradient_boosting", "random_forest", "extra_trees"]:
        cfg = MLModelConfig(model_type=mt, n_estimators=20, max_depth=2, n_cv_folds=2)
        results.append(MLTrainer(cfg).train(ds))
    return results


class TestStackingEnsemble:
    """Tests for stacking ensemble method."""

    def test_stacking_predict_proba_shape(self) -> None:
        results = _train_models()
        ds = _make_separable_dataset()
        ensemble = EnsemblePredictor(results, ensemble_method="stacking")
        ensemble.fit_stacking(ds.X, ds.y)
        probs = ensemble.predict_proba(ds.X)
        assert probs.shape == (ds.X.shape[0],)

    def test_stacking_predict_proba_range(self) -> None:
        results = _train_models()
        ds = _make_separable_dataset()
        ensemble = EnsemblePredictor(results, ensemble_method="stacking")
        ensemble.fit_stacking(ds.X, ds.y)
        probs = ensemble.predict_proba(ds.X)
        assert float(np.min(probs)) >= 0.0
        assert float(np.max(probs)) <= 1.0

    def test_soft_vote_default(self) -> None:
        results = _train_models()
        ensemble = EnsemblePredictor(results)
        assert ensemble.ensemble_method == "soft_vote"

    def test_stacking_method_stored(self) -> None:
        results = _train_models()
        ensemble = EnsemblePredictor(results, ensemble_method="stacking")
        assert ensemble.ensemble_method == "stacking"

    def test_stacking_without_fit_raises(self) -> None:
        results = _train_models()
        ds = _make_separable_dataset()
        ensemble = EnsemblePredictor(results, ensemble_method="stacking")
        with pytest.raises(RuntimeError, match="fit_stacking"):
            ensemble.predict_proba(ds.X)

    def test_invalid_ensemble_method_raises(self) -> None:
        results = _train_models()
        with pytest.raises(ValueError, match="ensemble_method"):
            EnsemblePredictor(results, ensemble_method="invalid")

    def test_soft_vote_still_works(self) -> None:
        results = _train_models()
        ds = _make_separable_dataset()
        ensemble = EnsemblePredictor(results, ensemble_method="soft_vote")
        probs = ensemble.predict_proba(ds.X)
        assert probs.shape == (ds.X.shape[0],)
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/ml/test_ensemble_stacking.py -x -v`
Expected: FAIL (constructor doesn't accept `ensemble_method`)

**Step 3: Implement stacking in `EnsemblePredictor`**

Modify `EnsemblePredictor.__init__()` to accept `ensemble_method` parameter:

```python
    def __init__(
        self,
        results: list[TrainingResult],
        ensemble_method: str = "soft_vote",
    ) -> None:
        if not results:
            raise ValueError("results must not be empty")
        if ensemble_method not in ("soft_vote", "stacking"):
            raise ValueError(
                f"ensemble_method must be 'soft_vote' or 'stacking', "
                f"got {ensemble_method!r}"
            )
        self._results = list(results)
        self._ensemble_method = ensemble_method
        self._meta_learner = None  # fitted by fit_stacking()
```

Add `ensemble_method` property:

```python
    @property
    def ensemble_method(self) -> str:
        return self._ensemble_method
```

Add `fit_stacking()` method:

```python
    def fit_stacking(self, X: NDArray, y: NDArray) -> None:
        """Train the stacking meta-learner on base model predictions.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features).
        y:
            Binary labels (n_samples,).
        """
        from sklearn.linear_model import LogisticRegression

        # Collect base model predictions as meta-features
        meta_X = self._base_predictions(X)
        self._meta_learner = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        )
        self._meta_learner.fit(meta_X, y)

    def _base_predictions(self, X: NDArray) -> NDArray:
        """Get base model probability predictions as a matrix."""
        import numpy as np

        preds = []
        for result in self._results:
            try:
                model = result.model
                proba = model.predict_proba(X)
                if proba.ndim == 2:
                    preds.append(proba[:, 1])
                else:
                    preds.append(proba)
            except Exception:
                continue
        return np.column_stack(preds)
```

Modify `predict_proba()` to branch on method:

```python
    def predict_proba(self, X: NDArray) -> NDArray:
        if self._ensemble_method == "stacking":
            if self._meta_learner is None:
                raise RuntimeError(
                    "Stacking ensemble requires calling fit_stacking() first"
                )
            meta_X = self._base_predictions(X)
            return self._meta_learner.predict_proba(meta_X)[:, 1]
        # Soft-vote (existing logic)
        # ... existing averaging code ...
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/ml/test_ensemble_stacking.py tests/ml/test_ensemble.py -x -v`
Expected: All tests pass (new + existing)

**Step 5: Commit**

```bash
git add src/stockdownloader/ml/ensemble.py tests/ml/test_ensemble_stacking.py
git commit -m "feat: add stacking ensemble method to EnsemblePredictor"
```

---

### Task 5: spy-ml-mega CLI Pipeline

**Files:**
- Create: `src/stockdownloader/app/spy_ml_mega.py`
- Modify: `pyproject.toml` (add entry point)
- Test: `tests/app/test_spy_ml_mega.py`

**Context:** Copy the pattern from `spy_ml_ensemble.py` (same 8-stage pipeline). The mega pipeline uses a larger grid (18 configs), diversity-aware selection, and optional stacking. Reuses the same `_run_tournament()` from `spy_ml_ensemble.py` and the same `spy_ml_ensemble_strategy()` PineScript factory.

**Step 1: Create the mega pipeline file**

Create `src/stockdownloader/app/spy_ml_mega.py`:

```python
"""SPY ML Mega pipeline — exhaustive algorithm search.

Trains all available ML algorithms (12 model types, 18 configs with
class balance variants), selects the best ensemble via diversity-aware
selection, optionally compares soft-voting vs. stacking, then exports
to PineScript through the deep surrogate.

Usage::

    spy-ml-mega                        # Full pipeline (18 configs)
    spy-ml-mega --quick                # Quick mode (6 configs)
    spy-ml-mega --ensemble-method both # Compare soft-vote vs stacking
    spy-ml-mega --diversity-weight 0.6 # Heavier diversity weighting
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from stockdownloader.app.ml_helpers import add_common_ml_args, init_ml_env
from stockdownloader.core.config import DEFAULT_ML_PIPELINE_DIR


# ======================================================================
# Model grid definitions
# ======================================================================

_FULL_GRID = [
    # (model_type, use_class_balance)
    # -- sklearn trees --
    ("gradient_boosting", False),
    ("gradient_boosting", True),
    ("random_forest", False),
    ("random_forest", True),
    ("extra_trees", False),
    ("extra_trees", True),
    ("hist_gradient_boosting", False),
    ("hist_gradient_boosting", True),
    # -- external boosters --
    ("xgboost", False),
    ("xgboost", True),
    ("lightgbm", False),
    ("lightgbm", True),
    ("catboost", False),
    ("catboost", True),
    # -- linear / non-tree --
    ("logistic_regression", False),  # always balanced internally
    ("svm", False),
    ("mlp", False),
    ("knn", False),
]

_QUICK_GRID = [
    ("xgboost", False),
    ("lightgbm", False),
    ("catboost", False),
    ("random_forest", False),
    ("logistic_regression", False),
    ("svm", False),
]


# ======================================================================
# Argparse
# ======================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``spy-ml-mega``."""
    parser = argparse.ArgumentParser(
        prog="spy-ml-mega",
        description=(
            "SPY ML Mega pipeline: exhaustive algorithm search -> "
            "diversity-aware ensemble -> deep surrogate -> PineScript."
        ),
    )

    add_common_ml_args(parser)

    # Ensemble args
    parser.add_argument(
        "--top-models", type=int, default=5,
        help="Number of top models for ensemble (default: 5)",
    )
    parser.add_argument(
        "--ensemble-method",
        choices=["soft_vote", "stacking", "both"],
        default="both",
        help="Ensemble method (default: both — compares and picks best)",
    )
    parser.add_argument(
        "--diversity-weight", type=float, default=0.4,
        help="Weight for diversity vs accuracy in selection (default: 0.4)",
    )

    # Surrogate args
    parser.add_argument(
        "--depth", type=int, default=10,
        help="Max depth for deep surrogate tree (default: 10)",
    )
    parser.add_argument(
        "--top-features", type=int, default=25,
        help="Number of top features for surrogate (default: 25)",
    )
    parser.add_argument(
        "--min-leaf", type=int, default=10,
        help="Minimum samples per leaf in surrogate (default: 10)",
    )

    # Trading args
    parser.add_argument(
        "--buy-thresh", type=float, default=0.55,
        help="Buy threshold (default: 0.55)",
    )
    parser.add_argument(
        "--sell-thresh", type=float, default=0.45,
        help="Sell threshold (default: 0.45)",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="Initial capital for backtest (default: 100000)",
    )
    parser.add_argument(
        "--min-r2", type=float, default=0.70,
        help="Minimum R-squared for surrogate (default: 0.70)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Output directory "
            f"(default: {DEFAULT_ML_PIPELINE_DIR / 'spy_mega'})"
        ),
    )
    parser.add_argument(
        "--no-tournament", action="store_true",
        help="Skip tournament backtest",
    )

    return parser


# ======================================================================
# Main pipeline
# ======================================================================


def main(argv: list[str] | None = None) -> None:
    """Run the SPY ML Mega pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_ML_PIPELINE_DIR / "spy_mega"
    )
    init_ml_env(args.verbose)

    import numpy as np

    from stockdownloader.data.market.yahoo_data_client import YahooDataClient
    from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig
    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter
    from stockdownloader.ml.ensemble import EnsembleBuilder
    from stockdownloader.ml.feature_extractor import FeatureExtractor
    from stockdownloader.ml.trainer import MLModelConfig, MLTrainer
    from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
        spy_ml_ensemble_strategy,
    )
    from stockdownloader.pinescript.generator import PineScriptGenerator
    from stockdownloader.app.spy_ml_ensemble import _run_tournament

    t0 = time.time()
    grid = _QUICK_GRID if args.quick else _FULL_GRID

    print("=" * 60)
    print("SPY ML MEGA PIPELINE")
    print("=" * 60)
    print(f"  Mode:            {'quick' if args.quick else 'full'}")
    print(f"  Model configs:   {len(grid)}")
    print(f"  Top models:      {args.top_models}")
    print(f"  Ensemble method: {args.ensemble_method}")
    print(f"  Diversity weight:{args.diversity_weight}")
    print(f"  Surrogate:       depth={args.depth}, "
          f"features={args.top_features}, min_leaf={args.min_leaf}")
    print(f"  Thresholds:      buy={args.buy_thresh}, sell={args.sell_thresh}")
    print(f"  Capital:         ${args.initial_capital:,.0f}")
    print(f"  Output:          {output_dir}")
    print()

    # [1/8] Download
    print("[1/8] Downloading SPY daily data (10y)...")
    t1 = time.time()
    client = YahooDataClient()
    daily_data = client.fetch_price_data("SPY", range_="10y")
    if len(daily_data) < 500:
        print(f"\nERROR: Insufficient data — got {len(daily_data)} bars.", file=sys.stderr)
        sys.exit(1)
    print(f"  Downloaded {len(daily_data)} bars "
          f"({daily_data[0].date} to {daily_data[-1].date})")
    print(f"  [{time.time() - t1:.1f}s]")

    # [2/8] Features
    print("\n[2/8] Building feature dataset...")
    t2 = time.time()
    label_config = LabelConfig(
        forward_period=int(args.forward_periods.split(",")[0]) if hasattr(args, "forward_periods") else 10,
        profit_threshold=float(args.profit_thresholds.split(",")[0]) if hasattr(args, "profit_thresholds") else 0.005,
    )
    extractor = FeatureExtractor()
    builder = DatasetBuilder(extractor, label_config)
    dataset = builder.build(daily_data)
    print(f"  Samples: {dataset.X.shape[0]}, Features: {dataset.X.shape[1]}")
    print(f"  [{time.time() - t2:.1f}s]")

    # [3/8] Train all models
    print(f"\n[3/8] Training model grid ({len(grid)} configs)...")
    t3 = time.time()
    results = []
    for idx, (model_type, use_balance) in enumerate(grid, 1):
        label = f"{model_type}" + ("+balanced" if use_balance else "")
        print(f"  [{idx}/{len(grid)}] {label}...", end=" ", flush=True)
        try:
            config = MLModelConfig(
                model_type=model_type,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                use_class_balance=use_balance,
            )
            trainer = MLTrainer(config)
            result = trainer.train(dataset)
            results.append(result)
            print(f"acc={result.oos_accuracy:.4f}  auc={result.oos_roc_auc:.4f}")
        except (ImportError, Exception) as exc:
            print(f"SKIP ({exc})")
    print(f"  [{time.time() - t3:.1f}s]")

    if len(results) < 2:
        print("\nERROR: Need at least 2 successful models.", file=sys.stderr)
        sys.exit(1)

    # [4/8] Rank & report
    print(f"\n[4/8] Model leaderboard ({len(results)} models)...")
    ranked = sorted(results, key=lambda r: r.oos_accuracy, reverse=True)
    for i, r in enumerate(ranked, 1):
        bal = "+bal" if r.config.use_class_balance else ""
        print(f"  #{i:2d}  {r.config.model_type}{bal:5s}  "
              f"acc={r.oos_accuracy:.4f}  auc={r.oos_roc_auc:.4f}")

    # [5/8] Build ensemble (diversity-aware)
    print(f"\n[5/8] Building diverse ensemble (top {args.top_models})...")
    t5 = time.time()
    ensemble_builder = EnsembleBuilder(results)
    ensemble = ensemble_builder.select_diverse(
        n=args.top_models,
        diversity_weight=args.diversity_weight,
    )
    probs = ensemble.predict_proba(dataset.X)
    importances = ensemble.averaged_feature_importances()
    print(f"  Ensemble models: {ensemble.n_models}")
    print(f"  Prob range: [{float(np.min(probs)):.4f}, {float(np.max(probs)):.4f}]")
    selected_types = [r.config.model_type for r in ensemble.results]
    print(f"  Selected: {selected_types}")
    print(f"  [{time.time() - t5:.1f}s]")

    # [6/8] Optional stacking comparison
    if args.ensemble_method in ("stacking", "both"):
        print("\n[6/8] Stacking ensemble comparison...")
        t6 = time.time()
        stacking = EnsemblePredictor(
            list(ensemble.results),
            ensemble_method="stacking",
        )
        stacking.fit_stacking(dataset.X, dataset.y)
        stacking_probs = stacking.predict_proba(dataset.X)
        print(f"  Stacking prob range: [{float(np.min(stacking_probs)):.4f}, "
              f"{float(np.max(stacking_probs)):.4f}]")

        if args.ensemble_method == "both":
            # Use the one with wider prob spread (more decisive)
            sv_spread = float(np.std(probs))
            st_spread = float(np.std(stacking_probs))
            if st_spread > sv_spread:
                print(f"  Stacking wins (spread {st_spread:.4f} > {sv_spread:.4f})")
                probs = stacking_probs
            else:
                print(f"  Soft-vote wins (spread {sv_spread:.4f} >= {st_spread:.4f})")
        elif args.ensemble_method == "stacking":
            probs = stacking_probs
        print(f"  [{time.time() - t6:.1f}s]")
    else:
        print("\n[6/8] Stacking skipped (--ensemble-method=soft_vote).")

    # [7/8] Deep surrogate
    print(f"\n[7/8] Training deep surrogate (depth={args.depth})...")
    t7 = time.time()
    exporter = DeepSurrogateExporter(
        max_depth=args.depth,
        min_samples_leaf=args.min_leaf,
        top_n=args.top_features,
    )
    exporter.train_surrogate(dataset, importances, ensemble_probs=probs)
    r_squared = exporter.fidelity_r_squared(dataset.X, probs)
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  Features used: {len(exporter.feature_names)}")
    if r_squared < args.min_r2:
        print(f"\n  WARNING: R² ({r_squared:.4f}) < {args.min_r2:.2f}", file=sys.stderr)
    print(f"  [{time.time() - t7:.1f}s]")

    # [8/8] PineScript export
    if args.no_pine:
        print("\n[8/8] PineScript export skipped.")
    else:
        print("\n[8/8] Exporting PineScript strategy...")
        t8 = time.time()
        strategy_def = spy_ml_ensemble_strategy(
            exporter,
            buy_threshold=args.buy_thresh,
            sell_threshold=args.sell_thresh,
        )
        pine_code = PineScriptGenerator().generate(strategy_def)
        output_dir.mkdir(parents=True, exist_ok=True)
        pine_path = output_dir / "spy_ml_mega.pine"
        pine_path.write_text(pine_code, encoding="utf-8")
        print(f"  PineScript: {pine_path} ({len(pine_code.splitlines())} lines)")
        print(f"  [{time.time() - t8:.1f}s]")

    # Tournament
    if not args.no_tournament:
        _run_tournament(
            dataset.X, dataset.dates, daily_data, exporter,
            args.buy_thresh, args.sell_thresh,
            initial_capital=args.initial_capital,
        )

    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("MEGA PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time:     {elapsed:.1f}s")
    print(f"  Models trained:  {len(results)}")
    print(f"  Ensemble size:   {ensemble.n_models}")
    print(f"  Surrogate R²:    {r_squared:.4f}")
    if not args.no_pine:
        print(f"  PineScript:      {output_dir / 'spy_ml_mega.pine'}")
    print()


if __name__ == "__main__":
    main()
```

**Step 2: Add entry point to pyproject.toml**

In `[project.scripts]`, after the `spy-ml-ensemble` line:

```toml
spy-ml-mega = "stockdownloader.app.spy_ml_mega:main"
```

**Step 3: Write basic tests**

Create `tests/app/test_spy_ml_mega.py`:

```python
"""Tests for spy-ml-mega CLI pipeline."""

from __future__ import annotations

import pytest

from stockdownloader.app.spy_ml_mega import _build_parser, _FULL_GRID, _QUICK_GRID


class TestMegaGrid:
    """Tests for model grid definitions."""

    def test_full_grid_has_18_configs(self) -> None:
        assert len(_FULL_GRID) == 18

    def test_quick_grid_has_6_configs(self) -> None:
        assert len(_QUICK_GRID) == 6

    def test_full_grid_tuples(self) -> None:
        for item in _FULL_GRID:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], bool)

    def test_full_grid_model_types(self) -> None:
        types = {mt for mt, _ in _FULL_GRID}
        expected = {
            "gradient_boosting", "random_forest", "extra_trees",
            "hist_gradient_boosting", "xgboost", "lightgbm",
            "catboost", "logistic_regression", "svm", "mlp", "knn",
        }
        assert types == expected

    def test_quick_grid_model_types(self) -> None:
        types = {mt for mt, _ in _QUICK_GRID}
        expected = {
            "xgboost", "lightgbm", "catboost",
            "random_forest", "logistic_regression", "svm",
        }
        assert types == expected


class TestMegaParser:
    """Tests for argument parser."""

    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.top_models == 5
        assert args.ensemble_method == "both"
        assert args.diversity_weight == 0.4
        assert args.depth == 10
        assert args.buy_thresh == 0.55
        assert args.sell_thresh == 0.45
        assert args.initial_capital == 100_000.0

    def test_ensemble_method_choices(self) -> None:
        parser = _build_parser()
        for method in ["soft_vote", "stacking", "both"]:
            args = parser.parse_args(["--ensemble-method", method])
            assert args.ensemble_method == method

    def test_quick_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--quick"])
        assert args.quick is True

    def test_custom_diversity_weight(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--diversity-weight", "0.7"])
        assert args.diversity_weight == 0.7
```

**Step 4: Run tests**

Run: `PYTHONPATH=src python3 -m pytest tests/app/test_spy_ml_mega.py -x -v`
Expected: All pass

**Step 5: Run full test suite**

Run: `PYTHONPATH=src python3 -m pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/stockdownloader/app/spy_ml_mega.py tests/app/test_spy_ml_mega.py pyproject.toml
git commit -m "feat: add spy-ml-mega pipeline with exhaustive algorithm search"
```

---

### Task 6: Integration Test — Run the Mega Pipeline

**Files:** None (runtime test only)

**Step 1: Run mega pipeline in quick mode**

Run: `PYTHONPATH=src python3 -m stockdownloader.app.spy_ml_mega --quick --no-pine --no-tournament`
Expected: Trains 6 models, builds diverse ensemble, reports R². All models complete (or gracefully skip).

**Step 2: Run mega pipeline full mode**

Run: `PYTHONPATH=src python3 -m stockdownloader.app.spy_ml_mega`
Expected: Trains 18 configs, leaderboard shows all model types, diverse ensemble selected, stacking compared, PineScript exported, tournament run with $100K capital.

**Step 3: Verify all tests still pass**

Run: `PYTHONPATH=src python3 -m pytest tests/ -x -q`
Expected: All tests pass

**Step 4: Commit any fixes discovered during integration**

If any fixes were needed, commit them. Otherwise, no commit needed.
