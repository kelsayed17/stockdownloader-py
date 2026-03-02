"""Tests for stacking ensemble method."""

from __future__ import annotations

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
from stockdownloader.ml.ensemble import EnsemblePredictor
from stockdownloader.ml.trainer import MLModelConfig, MLTrainer


_FEATURE_NAMES = tuple(f"feat_{i}" for i in range(10))


def _make_separable_dataset(n: int = 200) -> MLDataset:
    rng = np.random.RandomState(42)
    half = n // 2
    X_0 = rng.randn(half, 10) - 1.0
    X_1 = rng.randn(n - half, 10) + 1.0
    X = np.empty((n, 10), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    idx_0, idx_1 = 0, 0
    for i in range(n):
        if i % 2 == 0 and idx_0 < half:
            X[i] = X_0[idx_0]; y[i] = 0; idx_0 += 1
        elif idx_1 < (n - half):
            X[i] = X_1[idx_1]; y[i] = 1; idx_1 += 1
        elif idx_0 < half:
            X[i] = X_0[idx_0]; y[i] = 0; idx_0 += 1
        else:
            X[i] = X_1[idx_1]; y[i] = 1; idx_1 += 1
    dates = tuple(f"2024-01-{i+1:03d}" for i in range(n))
    return MLDataset(X=X, y=y, dates=dates, feature_names=_FEATURE_NAMES, label_config=LabelConfig())


def _train_models() -> list:
    ds = _make_separable_dataset()
    results = []
    for mt in ["gradient_boosting", "random_forest", "extra_trees"]:
        cfg = MLModelConfig(model_type=mt, n_estimators=20, max_depth=2, n_cv_folds=2)
        results.append(MLTrainer(cfg).train(ds))
    return results


class TestStackingEnsemble:

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
