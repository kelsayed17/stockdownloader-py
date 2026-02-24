"""Tests for EnsemblePredictor and EnsembleBuilder."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
from stockdownloader.ml.ensemble import EnsembleBuilder, EnsemblePredictor
from stockdownloader.ml.trainer import MLModelConfig, MLTrainer, TrainingResult


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_FEATURE_NAMES = tuple(f"feat_{i}" for i in range(10))


def _make_separable_dataset(n: int = 200) -> MLDataset:
    """Create a dataset where class 1 has higher feature values."""
    rng = np.random.RandomState(42)
    half = n // 2

    X_0 = rng.randn(half, 10) - 1.0  # class 0: shifted left
    X_1 = rng.randn(n - half, 10) + 1.0  # class 1: shifted right

    # Interleave to simulate temporal ordering
    X = np.empty((n, 10), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    idx_0, idx_1 = 0, 0
    for i in range(n):
        if i % 2 == 0 and idx_0 < half:
            X[i] = X_0[idx_0]
            y[i] = 0
            idx_0 += 1
        elif idx_1 < (n - half):
            X[i] = X_1[idx_1]
            y[i] = 1
            idx_1 += 1
        elif idx_0 < half:
            X[i] = X_0[idx_0]
            y[i] = 0
            idx_0 += 1
        else:
            X[i] = X_1[idx_1]
            y[i] = 1
            idx_1 += 1

    dates = tuple(f"2024-01-{i + 1:03d}" for i in range(n))
    return MLDataset(
        X=X,
        y=y,
        dates=dates,
        feature_names=_FEATURE_NAMES,
        label_config=LabelConfig(),
    )


def _train_multiple_models(dataset: MLDataset) -> list[TrainingResult]:
    """Train several small models for ensemble testing."""
    configs = [
        MLModelConfig(
            model_type="gradient_boosting",
            n_estimators=20,
            max_depth=2,
            n_cv_folds=2,
        ),
        MLModelConfig(
            model_type="random_forest",
            n_estimators=20,
            max_depth=2,
            n_cv_folds=2,
        ),
        MLModelConfig(
            model_type="extra_trees",
            n_estimators=20,
            max_depth=2,
            n_cv_folds=2,
        ),
    ]
    results = []
    for cfg in configs:
        trainer = MLTrainer(cfg)
        results.append(trainer.train(dataset))
    return results


# ------------------------------------------------------------------
# EnsemblePredictor — construction
# ------------------------------------------------------------------


class TestEnsemblePredictorConstruction:
    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            EnsemblePredictor([])

    def test_single_model(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)[:1]
        ep = EnsemblePredictor(results)
        assert ep.n_models == 1

    def test_multiple_models(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)
        assert ep.n_models == 3

    def test_results_property_returns_copy(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)[:1]
        ep = EnsemblePredictor(results)
        returned = ep.results
        returned.clear()
        # Internal list should be unaffected
        assert ep.n_models == 1


# ------------------------------------------------------------------
# EnsemblePredictor — predict_proba
# ------------------------------------------------------------------


class TestEnsemblePredictProba:
    def test_shape(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)
        proba = ep.predict_proba(ds.X)
        assert proba.shape == (ds.X.shape[0],)

    def test_range_zero_to_one(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)
        proba = ep.predict_proba(ds.X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_single_sample(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)
        proba = ep.predict_proba(ds.X[:1])
        assert proba.shape == (1,)
        assert 0.0 <= proba[0] <= 1.0

    def test_matches_manual_average(self) -> None:
        """Ensemble proba should be the mean of individual model probas."""
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)

        X_test = ds.X[:10]
        ensemble_proba = ep.predict_proba(X_test)

        individual_probas = []
        for r in results:
            individual_probas.append(r.model.predict_proba(X_test)[:, 1])
        expected = np.mean(np.stack(individual_probas, axis=0), axis=0)

        np.testing.assert_allclose(ensemble_proba, expected, atol=1e-10)

    def test_failed_model_skipped(self) -> None:
        """If a model raises during predict_proba, it is skipped gracefully."""
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)

        # Create a mock result whose model raises on predict_proba
        bad_model = MagicMock()
        bad_model.predict_proba.side_effect = RuntimeError("broken model")
        bad_result = TrainingResult(
            model=bad_model,
            oos_accuracy=0.5,
            oos_roc_auc=0.5,
            feature_importances={name: 0.1 for name in _FEATURE_NAMES},
            train_size=100,
            test_size=50,
            class_distribution={0: 100, 1: 100},
            cv_scores=[0.5],
            config=MLModelConfig(n_estimators=20, max_depth=2, n_cv_folds=2),
        )

        mixed = results + [bad_result]
        ep = EnsemblePredictor(mixed)
        # Should still work — the broken model is skipped
        proba = ep.predict_proba(ds.X[:5])
        assert proba.shape == (5,)

    def test_all_models_fail_raises(self) -> None:
        """If every model fails, RuntimeError is raised."""
        bad_model = MagicMock()
        bad_model.predict_proba.side_effect = RuntimeError("broken")
        bad_result = TrainingResult(
            model=bad_model,
            oos_accuracy=0.5,
            oos_roc_auc=0.5,
            feature_importances={},
            train_size=100,
            test_size=50,
            class_distribution={0: 50, 1: 50},
            cv_scores=[0.5],
            config=MLModelConfig(n_estimators=20, max_depth=2, n_cv_folds=2),
        )
        ep = EnsemblePredictor([bad_result])
        with pytest.raises(RuntimeError, match="All models failed"):
            ep.predict_proba(np.random.randn(5, 10))


# ------------------------------------------------------------------
# EnsemblePredictor — averaged_feature_importances
# ------------------------------------------------------------------


class TestEnsembleFeatureImportances:
    def test_correct_feature_count(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)
        importances = ep.averaged_feature_importances()
        assert len(importances) == len(_FEATURE_NAMES)

    def test_sums_to_one(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)
        importances = ep.averaged_feature_importances()
        total = sum(importances.values())
        assert abs(total - 1.0) < 1e-6

    def test_sorted_descending(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)
        importances = ep.averaged_feature_importances()
        values = list(importances.values())
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

    def test_all_names_present(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)
        importances = ep.averaged_feature_importances()
        assert set(importances.keys()) == set(_FEATURE_NAMES)

    def test_empty_importances(self) -> None:
        """Models with no feature importances produce empty dict."""
        bad_result = TrainingResult(
            model=MagicMock(),
            oos_accuracy=0.5,
            oos_roc_auc=0.5,
            feature_importances={},
            train_size=100,
            test_size=50,
            class_distribution={0: 50, 1: 50},
            cv_scores=[0.5],
            config=MLModelConfig(n_estimators=20, max_depth=2, n_cv_folds=2),
        )
        ep = EnsemblePredictor([bad_result])
        importances = ep.averaged_feature_importances()
        assert importances == {}


# ------------------------------------------------------------------
# EnsemblePredictor — accuracy
# ------------------------------------------------------------------


class TestEnsembleAccuracy:
    def test_competitive_with_best_single(self) -> None:
        """Ensemble accuracy should be within 5% of the best single model."""
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        ep = EnsemblePredictor(results)

        # Use the held-out portion (last 20%) to evaluate
        split_idx = int(ds.X.shape[0] * 0.8)
        X_test = ds.X[split_idx:]
        y_test = ds.y[split_idx:]

        # Ensemble predictions
        ensemble_proba = ep.predict_proba(X_test)
        ensemble_preds = (ensemble_proba >= 0.5).astype(int)
        ensemble_acc = float(np.mean(ensemble_preds == y_test))

        # Best single model accuracy on same test set
        best_single_acc = max(r.oos_accuracy for r in results)

        # Ensemble should be competitive (within 5 percentage points)
        assert ensemble_acc >= best_single_acc - 0.05, (
            f"Ensemble accuracy {ensemble_acc:.4f} is more than 5% below "
            f"best single model {best_single_acc:.4f}"
        )


# ------------------------------------------------------------------
# EnsembleBuilder
# ------------------------------------------------------------------


class TestEnsembleBuilder:
    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            EnsembleBuilder([])

    def test_select_top_returns_ensemble_predictor(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        builder = EnsembleBuilder(results)
        ep = builder.select_top(n=2)
        assert isinstance(ep, EnsemblePredictor)

    def test_select_top_correct_count(self) -> None:
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        builder = EnsembleBuilder(results)
        ep = builder.select_top(n=2)
        assert ep.n_models == 2

    def test_select_top_more_than_available(self) -> None:
        """Requesting more models than available returns all."""
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        builder = EnsembleBuilder(results)
        ep = builder.select_top(n=10)
        assert ep.n_models == len(results)

    def test_select_top_sorted_by_accuracy(self) -> None:
        """Selected models should be the ones with highest oos_accuracy."""
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        builder = EnsembleBuilder(results)
        ep = builder.select_top(n=2)

        selected_accuracies = [r.oos_accuracy for r in ep.results]
        all_accuracies = sorted(
            [r.oos_accuracy for r in results], reverse=True,
        )
        expected_top2 = all_accuracies[:2]

        assert sorted(selected_accuracies, reverse=True) == pytest.approx(
            expected_top2
        )

    def test_select_top_one(self) -> None:
        """Selecting top 1 gives us the best model."""
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        builder = EnsembleBuilder(results)
        ep = builder.select_top(n=1)
        assert ep.n_models == 1

        best_accuracy = max(r.oos_accuracy for r in results)
        assert ep.results[0].oos_accuracy == best_accuracy

    def test_select_top_default_n(self) -> None:
        """Default n=3 should take all 3 models."""
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        builder = EnsembleBuilder(results)
        ep = builder.select_top()
        assert ep.n_models == 3

    def test_select_top_preserves_results(self) -> None:
        """Selected ensemble's results should be valid TrainingResult objects."""
        ds = _make_separable_dataset()
        results = _train_multiple_models(ds)
        builder = EnsembleBuilder(results)
        ep = builder.select_top(n=2)
        for r in ep.results:
            assert isinstance(r, TrainingResult)
            assert r.model is not None
            assert 0.0 <= r.oos_accuracy <= 1.0
