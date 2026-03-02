"""Tests for MLTrainer — temporal CV and model training."""

from __future__ import annotations

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
from stockdownloader.ml.trainer import (
    MLModelConfig,
    MLTrainer,
    TimeSeriesExpandingCV,
    TrainingResult,
)


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


def _make_random_dataset(n: int = 200) -> MLDataset:
    """Create a dataset with random labels (no signal)."""
    rng = np.random.RandomState(99)
    X = rng.randn(n, 10)
    y = rng.randint(0, 2, size=n).astype(np.int64)
    dates = tuple(f"2024-01-{i + 1:03d}" for i in range(n))
    return MLDataset(
        X=X,
        y=y,
        dates=dates,
        feature_names=_FEATURE_NAMES,
        label_config=LabelConfig(),
    )


# ------------------------------------------------------------------
# TimeSeriesExpandingCV
# ------------------------------------------------------------------


class TestTimeSeriesExpandingCV:
    def test_splits_are_temporal(self) -> None:
        """All train indices < all test indices in every fold."""
        cv = TimeSeriesExpandingCV(n_splits=5, min_train_ratio=0.5)
        folds = cv.split(200)

        assert len(folds) >= 2
        for train_range, test_range in folds:
            assert max(train_range) < min(test_range)

    def test_train_grows(self) -> None:
        """Training set expands across folds."""
        cv = TimeSeriesExpandingCV(n_splits=5, min_train_ratio=0.5)
        folds = cv.split(200)

        train_sizes = [len(tr) for tr, _ in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_no_data_leakage(self) -> None:
        """Train and test sets don't overlap."""
        cv = TimeSeriesExpandingCV(n_splits=3, min_train_ratio=0.5)
        folds = cv.split(100)

        for train_range, test_range in folds:
            train_set = set(train_range)
            test_set = set(test_range)
            assert train_set.isdisjoint(test_set)

    def test_min_splits_enforced(self) -> None:
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            TimeSeriesExpandingCV(n_splits=1)

    def test_insufficient_samples(self) -> None:
        cv = TimeSeriesExpandingCV(n_splits=5)
        with pytest.raises(ValueError, match="Need at least"):
            cv.split(3)


# ------------------------------------------------------------------
# MLTrainer — GradientBoosting
# ------------------------------------------------------------------


class TestMLTrainerGB:
    def test_separable_data_good_accuracy(self) -> None:
        """With clearly separable data, accuracy should be decent."""
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)

        assert result.oos_accuracy > 0.55

    def test_result_populated(self) -> None:
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)

        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert result.train_size > 0
        assert result.test_size > 0
        assert result.train_size + result.test_size == 200
        assert 0.0 <= result.oos_accuracy <= 1.0
        assert 0.0 <= result.oos_roc_auc <= 1.0

    def test_feature_importances_sum(self) -> None:
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)

        total = sum(result.feature_importances.values())
        assert abs(total - 1.0) < 0.01

    def test_feature_importances_names(self) -> None:
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)

        assert set(result.feature_importances.keys()) == set(_FEATURE_NAMES)

    def test_class_distribution(self) -> None:
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)

        assert sum(result.class_distribution.values()) == 200

    def test_cv_scores_returned(self) -> None:
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)

        assert len(result.cv_scores) >= 1
        for score in result.cv_scores:
            assert 0.0 <= score <= 1.0

    def test_model_can_predict(self) -> None:
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)

        preds = result.model.predict(ds.X[:5])
        assert len(preds) == 5
        assert all(p in (0, 1) for p in preds)

    def test_model_can_predict_proba(self) -> None:
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)

        proba = result.model.predict_proba(ds.X[:5])
        assert proba.shape == (5, 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


# ------------------------------------------------------------------
# MLTrainer — LogisticRegression
# ------------------------------------------------------------------


class TestMLTrainerLR:
    def test_logistic_regression(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(model_type="logistic_regression", n_cv_folds=3)
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)

        assert result.oos_accuracy > 0.5
        assert result.config.model_type == "logistic_regression"

    def test_lr_feature_importances(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(model_type="logistic_regression", n_cv_folds=3)
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)

        total = sum(result.feature_importances.values())
        assert abs(total - 1.0) < 0.01


# ------------------------------------------------------------------
# MLTrainer — XGBoost
# ------------------------------------------------------------------


class TestMLTrainerXGBoost:
    def test_xgboost_trains(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="xgboost", n_estimators=50, max_depth=3, n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert result.oos_accuracy > 0.5

    def test_xgboost_predict_proba_shape(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="xgboost", n_estimators=50, max_depth=3, n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        proba = result.model.predict_proba(ds.X[:5])
        assert proba.shape == (5, 2)

    def test_xgboost_feature_importances(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="xgboost", n_estimators=50, max_depth=3, n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert len(result.feature_importances) == 10
        assert all(v >= 0.0 for v in result.feature_importances.values())


# ------------------------------------------------------------------
# MLTrainer — LightGBM
# ------------------------------------------------------------------


class TestMLTrainerLightGBM:
    def test_lightgbm_trains(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="lightgbm", n_estimators=50, max_depth=3, n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert result.oos_accuracy > 0.5

    def test_lightgbm_predict_proba_shape(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="lightgbm", n_estimators=50, max_depth=3, n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        proba = result.model.predict_proba(ds.X[:5])
        assert proba.shape == (5, 2)

    def test_lightgbm_feature_importances(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="lightgbm", n_estimators=50, max_depth=3, n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert len(result.feature_importances) == 10
        assert all(v >= 0.0 for v in result.feature_importances.values())


# ------------------------------------------------------------------
# MLTrainer — CatBoost
# ------------------------------------------------------------------


class TestMLTrainerCatBoost:
    def test_catboost_trains(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="catboost", n_estimators=50, max_depth=3, n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert result.oos_accuracy > 0.5

    def test_catboost_predict_proba_shape(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="catboost", n_estimators=50, max_depth=3, n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        proba = result.model.predict_proba(ds.X[:5])
        assert proba.shape == (5, 2)

    def test_catboost_feature_importances(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="catboost", n_estimators=50, max_depth=3, n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert len(result.feature_importances) == 10
        assert all(v >= 0.0 for v in result.feature_importances.values())


# ------------------------------------------------------------------
# MLTrainer — SVM
# ------------------------------------------------------------------


class TestMLTrainerSVM:
    def test_svm_trains(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="svm", n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert result.oos_accuracy > 0.5

    def test_svm_predict_proba_shape(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="svm", n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        proba = result.model.predict_proba(ds.X[:5])
        assert proba.shape == (5, 2)

    def test_svm_feature_importances(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="svm", n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        # SVM doesn't have feature_importances_ or coef_ (rbf kernel),
        # so all importances should be 0.0
        assert len(result.feature_importances) == 10


# ------------------------------------------------------------------
# MLTrainer — MLP
# ------------------------------------------------------------------


class TestMLTrainerMLP:
    def test_mlp_trains(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="mlp", n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert result.oos_accuracy > 0.5

    def test_mlp_predict_proba_shape(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="mlp", n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        proba = result.model.predict_proba(ds.X[:5])
        assert proba.shape == (5, 2)

    def test_mlp_feature_importances(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="mlp", n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        # MLP uses coefs_ for importances
        assert len(result.feature_importances) == 10
        total = sum(result.feature_importances.values())
        assert abs(total - 1.0) < 0.01


# ------------------------------------------------------------------
# MLTrainer — KNN
# ------------------------------------------------------------------


class TestMLTrainerKNN:
    def test_knn_trains(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="knn", n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert result.oos_accuracy > 0.5

    def test_knn_predict_proba_shape(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="knn", n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        proba = result.model.predict_proba(ds.X[:5])
        assert proba.shape == (5, 2)

    def test_knn_feature_importances(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            model_type="knn", n_cv_folds=3,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        # KNN doesn't have feature importances, so all should be 0.0
        assert len(result.feature_importances) == 10
        assert all(v == 0.0 for v in result.feature_importances.values())


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestMLTrainerErrors:
    def test_too_small_dataset(self) -> None:
        rng = np.random.RandomState(0)
        ds = MLDataset(
            X=rng.randn(5, 10),
            y=np.array([0, 1, 0, 1, 0]),
            dates=tuple(f"d{i}" for i in range(5)),
            feature_names=_FEATURE_NAMES,
            label_config=LabelConfig(),
        )
        trainer = MLTrainer()
        with pytest.raises(ValueError, match="too small"):
            trainer.train(ds)

    def test_unknown_model_type(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(model_type="nonexistent_model")
        trainer = MLTrainer(cfg)
        with pytest.raises(ValueError, match="Unknown model_type"):
            trainer.train(ds)

    def test_config_stored(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(n_estimators=50, max_depth=3, n_cv_folds=3)
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)

        assert result.config.n_estimators == 50
        assert result.config.max_depth == 3


# ------------------------------------------------------------------
# Sample weights (class balancing)
# ------------------------------------------------------------------


class TestSampleWeights:
    def test_compute_sample_weights(self) -> None:
        y = np.array([0, 0, 0, 0, 1])  # 4:1 imbalance
        weights = MLTrainer._compute_sample_weights(y)
        assert len(weights) == 5
        # Minority class should get higher weight
        assert weights[4] > weights[0]

    def test_balanced_weights_sum(self) -> None:
        y = np.array([0, 0, 1, 1])
        weights = MLTrainer._compute_sample_weights(y)
        # With balanced classes, all weights should be equal
        assert abs(weights[0] - weights[2]) < 1e-6

    def test_class_balance_flag(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
            use_class_balance=True,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert result.oos_accuracy > 0.5


# ------------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------------


class TestFeatureSelection:
    def test_select_features_reduces_dims(self) -> None:
        ds = _make_separable_dataset(200)
        X_sel, names_sel, kept = MLTrainer._select_features(
            ds.X, ds.y, ds.feature_names, threshold=0.01,
        )
        # Should keep at least 5 features
        assert X_sel.shape[1] >= 5
        assert X_sel.shape[1] == len(names_sel)
        assert X_sel.shape[1] == len(kept)

    def test_select_features_keeps_minimum(self) -> None:
        ds = _make_separable_dataset(200)
        # Very high threshold — should still keep at least 5
        X_sel, names_sel, kept = MLTrainer._select_features(
            ds.X, ds.y, ds.feature_names, threshold=0.99,
        )
        assert X_sel.shape[1] >= 5

    def test_feature_selection_config_flag(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
            use_feature_selection=True,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert result.selected_features is not None
        assert len(result.selected_features) >= 5


# ------------------------------------------------------------------
# Ensemble
# ------------------------------------------------------------------


class TestEnsemble:
    def test_ensemble_fields_populated(self) -> None:
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)
        # Should always try ensemble for GB
        assert result.ensemble_accuracy is not None
        assert result.ensemble_roc_auc is not None
        assert 0.0 <= result.ensemble_accuracy <= 1.0
        assert 0.0 <= result.ensemble_roc_auc <= 1.0

    def test_ensemble_not_for_lr(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(model_type="logistic_regression", n_cv_folds=3)
        trainer = MLTrainer(cfg)
        result = trainer.train(ds)
        assert result.ensemble_accuracy is None
        assert result.ensemble_roc_auc is None


# ------------------------------------------------------------------
# Threshold optimization
# ------------------------------------------------------------------


class TestThresholdOptimization:
    def test_optimal_threshold_in_range(self) -> None:
        ds = _make_separable_dataset(200)
        trainer = MLTrainer(MLModelConfig(
            n_estimators=50, max_depth=3, n_cv_folds=3,
        ))
        result = trainer.train(ds)
        assert 0.3 <= result.optimal_threshold <= 0.7

    def test_optimize_threshold_with_perfect_data(self) -> None:
        y_test = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thresh = MLTrainer._optimize_threshold(y_test, y_proba)
        assert 0.3 <= thresh <= 0.7


# ------------------------------------------------------------------
# Hyperparameter tuning
# ------------------------------------------------------------------


class TestMLTrainerTuning:
    def test_train_with_tuning(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(n_estimators=50, max_depth=3, n_cv_folds=3)
        trainer = MLTrainer(cfg)
        result = trainer.train_with_tuning(
            ds,
            param_grid={
                "n_estimators": [30, 50],
                "max_depth": [2, 3],
                "learning_rate": [0.1],
                "min_samples_leaf": [10],
            },
        )
        assert isinstance(result, TrainingResult)
        assert result.oos_accuracy > 0.4
        assert result.model is not None

    def test_tuning_with_balance_and_selection(self) -> None:
        ds = _make_separable_dataset(200)
        cfg = MLModelConfig(
            n_cv_folds=3,
            use_class_balance=True,
            use_feature_selection=True,
        )
        trainer = MLTrainer(cfg)
        result = trainer.train_with_tuning(
            ds,
            param_grid={
                "n_estimators": [30],
                "max_depth": [2],
                "learning_rate": [0.1],
                "min_samples_leaf": [10],
            },
        )
        assert result.selected_features is not None
        assert result.oos_accuracy > 0.4
