"""Train sklearn classifiers on ML datasets with temporal cross-validation.

Provides expanding-window cross-validation (no random shuffle) and a
high-level :class:`MLTrainer` that trains a GradientBoosting or
LogisticRegression classifier on an :class:`MLDataset`.

Usage::

    from stockdownloader.ml.trainer import MLTrainer, MLModelConfig

    trainer = MLTrainer()
    result = trainer.train(dataset)
    print(result.oos_accuracy, result.oos_roc_auc)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

from stockdownloader.ml.dataset_builder import MLDataset

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MLModelConfig:
    """Hyperparameters and training settings."""

    model_type: str = "gradient_boosting"
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    n_cv_folds: int = 5
    test_size_ratio: float = 0.2
    use_class_balance: bool = False
    use_feature_selection: bool = False
    feature_selection_threshold: float = 0.01


# ------------------------------------------------------------------
# Training result
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Holds the trained model and evaluation metrics."""

    model: Any
    oos_accuracy: float
    oos_roc_auc: float
    feature_importances: dict[str, float]
    train_size: int
    test_size: int
    class_distribution: dict[int, int]
    cv_scores: list[float]
    config: MLModelConfig
    # Ensemble fields (populated when ensemble is tried)
    ensemble_accuracy: float | None = None
    ensemble_roc_auc: float | None = None
    # Threshold optimization
    optimal_threshold: float = 0.5
    # Feature selection info
    selected_features: tuple[str, ...] | None = None


# ------------------------------------------------------------------
# Temporal cross-validation
# ------------------------------------------------------------------


class TimeSeriesExpandingCV:
    """Expanding-window cross-validator for time-series data.

    Each fold trains on data **before** the test window — no look-ahead.

    Parameters
    ----------
    n_splits:
        Number of CV folds.
    min_train_ratio:
        Minimum fraction of data used for the first training window.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_ratio: float = 0.5,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio

    def split(self, n_samples: int) -> list[tuple[range, range]]:
        """Return ``(train_range, test_range)`` pairs.

        All train indices are strictly less than all test indices in
        every fold.
        """
        if n_samples < self.n_splits + 1:
            raise ValueError(
                f"Need at least {self.n_splits + 1} samples for "
                f"{self.n_splits} folds, got {n_samples}"
            )

        min_train = max(int(n_samples * self.min_train_ratio), 1)
        remaining = n_samples - min_train
        step = remaining // self.n_splits

        if step < 1:
            step = 1

        folds: list[tuple[range, range]] = []
        for i in range(self.n_splits):
            train_end = min_train + i * step
            test_start = train_end
            test_end = min(train_end + step, n_samples)

            if test_start >= n_samples:
                break
            if test_end <= test_start:
                break

            folds.append((range(0, train_end), range(test_start, test_end)))

        return folds


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------


class MLTrainer:
    """Train a classifier on an :class:`MLDataset`.

    Parameters
    ----------
    config:
        Model hyperparameters.  Defaults are reasonable for daily
        stock data with ~200+ training samples.
    """

    def __init__(self, config: MLModelConfig | None = None) -> None:
        if not _HAS_SKLEARN:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for ML training. "
                "Install ML extras: pip install -e '.[ml]'"
            )
        self._config = config or MLModelConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, dataset: MLDataset) -> TrainingResult:
        """Train model and evaluate with temporal cross-validation.

        Raises :class:`ValueError` if the dataset is too small.
        """
        X, y = dataset.X, dataset.y
        feature_names = dataset.feature_names
        n_samples = X.shape[0]

        if n_samples < 20:
            raise ValueError(
                f"Dataset too small for training: {n_samples} samples "
                f"(need at least 20)"
            )

        # -- Feature selection (optional) --
        selected_features: tuple[str, ...] | None = None
        if self._config.use_feature_selection:
            X, feature_names, _kept = self._select_features(
                X, y, feature_names, self._config.feature_selection_threshold,
            )
            selected_features = feature_names

        # -- Temporal train/test split (last test_size_ratio for final eval) --
        split_idx = int(n_samples * (1 - self._config.test_size_ratio))
        split_idx = max(split_idx, 10)  # at least 10 training samples

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_test) == 0:
            raise ValueError(
                f"Test set is empty after split at {split_idx}/{n_samples}"
            )

        # -- Scale features --
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -- Class balancing (optional) --
        sample_weight = None
        if self._config.use_class_balance:
            sample_weight = self._compute_sample_weights(y_train)

        # -- Cross-validation on training set --
        cv_scores = self._cross_validate(
            X_train_scaled, y_train, sample_weight=sample_weight,
        )

        # -- Train final model on full training set --
        model = self._build_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

        # -- Evaluate on held-out test set --
        y_pred = model.predict(X_test_scaled)
        oos_accuracy = float(accuracy_score(y_test, y_pred))

        # ROC-AUC requires both classes in test set
        y_proba = None
        if len(np.unique(y_test)) > 1 and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            oos_roc_auc = float(roc_auc_score(y_test, y_proba))
        else:
            oos_roc_auc = 0.5  # uninformative

        # -- Threshold optimization --
        optimal_threshold = 0.5
        if y_proba is not None:
            optimal_threshold = self._optimize_threshold(y_test, y_proba)

        # -- Feature importances --
        importances = self._extract_importances(model, feature_names)

        # -- Class distribution --
        unique, counts = np.unique(dataset.y, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}

        # -- Ensemble --
        ensemble_acc, ensemble_auc = self._try_ensemble(
            X_train_scaled, y_train, X_test_scaled, y_test,
            sample_weight=sample_weight,
        )

        # -- Wrap the model with scaler for inference --
        wrapped = _ScaledModel(scaler=scaler, model=model)

        # -- Baseline comparison --
        self._log_baseline_comparison(
            X_train_scaled, y_train, X_test_scaled, y_test, oos_accuracy,
        )

        return TrainingResult(
            model=wrapped,
            oos_accuracy=oos_accuracy,
            oos_roc_auc=oos_roc_auc,
            feature_importances=importances,
            train_size=len(X_train),
            test_size=len(X_test),
            class_distribution=class_dist,
            cv_scores=cv_scores,
            config=self._config,
            ensemble_accuracy=ensemble_acc,
            ensemble_roc_auc=ensemble_auc,
            optimal_threshold=optimal_threshold,
            selected_features=selected_features,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        """Create an sklearn estimator from config.

        Supported model types:
        - ``gradient_boosting``: Scikit-learn GradientBoostingClassifier
        - ``random_forest``: RandomForestClassifier (parallel, fast)
        - ``extra_trees``: ExtraTreesClassifier (faster, more randomised)
        - ``hist_gradient_boosting``: HistGradientBoostingClassifier
          (histogram-based, much faster on large datasets, native NaN support)
        - ``adaboost``: AdaBoostClassifier (sequential boosting)
        - ``logistic_regression``: LogisticRegression (linear baseline)
        """
        mt = self._config.model_type

        if mt == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self._config.n_estimators,
                max_depth=self._config.max_depth,
                learning_rate=self._config.learning_rate,
                min_samples_leaf=self._config.min_samples_leaf,
            )
        if mt == "random_forest":
            return RandomForestClassifier(
                n_estimators=self._config.n_estimators,
                max_depth=self._config.max_depth,
                min_samples_leaf=self._config.min_samples_leaf,
                n_jobs=-1,
                random_state=42,
            )
        if mt == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=self._config.n_estimators,
                max_depth=self._config.max_depth,
                min_samples_leaf=self._config.min_samples_leaf,
                n_jobs=-1,
                random_state=42,
            )
        if mt == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=self._config.n_estimators,
                max_depth=self._config.max_depth,
                learning_rate=self._config.learning_rate,
                min_samples_leaf=self._config.min_samples_leaf,
                random_state=42,
            )
        if mt == "adaboost":
            return AdaBoostClassifier(
                n_estimators=min(self._config.n_estimators, 100),
                learning_rate=max(self._config.learning_rate, 0.5),
                random_state=42,
                algorithm="SAMME",
            )
        if mt == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )
        raise ValueError(
            f"Unknown model_type: {mt!r}. "
            f"Expected one of: 'gradient_boosting', 'random_forest', "
            f"'extra_trees', 'hist_gradient_boosting', 'adaboost', "
            f"'logistic_regression'."
        )

    def _cross_validate(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
    ) -> list[float]:
        """Temporal expanding-window CV on the training portion."""
        cv = TimeSeriesExpandingCV(
            n_splits=self._config.n_cv_folds,
            min_train_ratio=0.5,
        )
        n_samples = X.shape[0]

        if n_samples < self._config.n_cv_folds + 1:
            logger.warning(
                "Too few training samples (%d) for %d-fold CV — skipping",
                n_samples,
                self._config.n_cv_folds,
            )
            return []

        scores: list[float] = []
        for train_idx, test_idx in cv.split(n_samples):
            idx_tr = list(train_idx)
            idx_te = list(test_idx)
            X_tr, X_te = X[idx_tr], X[idx_te]
            y_tr, y_te = y[idx_tr], y[idx_te]

            if len(np.unique(y_tr)) < 2:
                continue  # skip folds with single class

            sw_tr = sample_weight[idx_tr] if sample_weight is not None else None

            fold_model = self._build_model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_model.fit(X_tr, y_tr, sample_weight=sw_tr)

            fold_acc = float(accuracy_score(y_te, fold_model.predict(X_te)))
            scores.append(fold_acc)

        return scores

    def _extract_importances(
        self,
        model: Any,
        feature_names: tuple[str, ...],
    ) -> dict[str, float]:
        """Extract feature importances from the model."""
        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
        elif hasattr(model, "coef_"):
            raw = np.abs(model.coef_[0])
        else:
            return {name: 0.0 for name in feature_names}

        total = float(np.sum(raw))
        if total == 0:
            return {name: 0.0 for name in feature_names}

        return {
            name: float(val / total)
            for name, val in zip(feature_names, raw)
        }

    # ------------------------------------------------------------------
    # Class balancing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sample_weights(y: Any) -> Any:
        """Compute sample weights inversely proportional to class frequency."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        class_weight = {cls: total / (len(unique) * cnt) for cls, cnt in zip(unique, counts)}
        return np.array([class_weight[label] for label in y], dtype=np.float64)

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_features(
        X: Any,
        y: Any,
        feature_names: tuple[str, ...],
        threshold: float = 0.01,
    ) -> tuple[Any, tuple[str, ...], list[int]]:
        """Drop features below *threshold* importance using a quick GB probe.

        Returns ``(X_filtered, filtered_names, kept_indices)``.
        Always keeps at least 5 features.
        """
        probe = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, min_samples_leaf=20,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe.fit(X, y)

        raw = probe.feature_importances_
        total = float(np.sum(raw))
        if total == 0:
            return X, feature_names, list(range(X.shape[1]))

        importances = raw / total
        kept: list[int] = [
            i for i, imp in enumerate(importances) if imp >= threshold
        ]

        # Ensure at least 5 features
        if len(kept) < 5:
            top_indices = np.argsort(importances)[::-1][:5].tolist()
            kept = sorted(set(kept) | set(top_indices))

        filtered_names = tuple(feature_names[i] for i in kept)
        logger.info(
            "Feature selection: %d → %d features (threshold=%.3f)",
            len(feature_names), len(kept), threshold,
        )
        return X[:, kept], filtered_names, kept

    # ------------------------------------------------------------------
    # Ensemble (GB + LR average)
    # ------------------------------------------------------------------

    def _try_ensemble(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        sample_weight: Any | None = None,
    ) -> tuple[float | None, float | None]:
        """Train primary model + LR, average predict_proba, return (accuracy, roc_auc)."""
        if self._config.model_type == "logistic_regression":
            return None, None

        try:
            primary = self._build_model()
            lr = LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="lbfgs",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Some models (RF, ET, HistGB) don't accept sample_weight via fit
                try:
                    primary.fit(X_train, y_train, sample_weight=sample_weight)
                except TypeError:
                    primary.fit(X_train, y_train)
                lr.fit(X_train, y_train)

            primary_proba = primary.predict_proba(X_test)[:, 1]
            lr_proba = lr.predict_proba(X_test)[:, 1]
            avg_proba = (primary_proba + lr_proba) / 2.0

            y_pred = (avg_proba >= 0.5).astype(int)
            acc = float(accuracy_score(y_test, y_pred))

            if len(np.unique(y_test)) > 1:
                auc = float(roc_auc_score(y_test, avg_proba))
            else:
                auc = 0.5

            mt = self._config.model_type
            logger.info("Ensemble (%s+LR) accuracy=%.3f, AUC=%.3f", mt, acc, auc)
            return acc, auc
        except Exception:
            logger.debug("Ensemble failed", exc_info=True)
            return None, None

    # ------------------------------------------------------------------
    # Threshold optimization
    # ------------------------------------------------------------------

    @staticmethod
    def _optimize_threshold(y_test: Any, y_proba: Any) -> float:
        """Find the threshold that maximizes F1 score on the test set."""
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.30, 0.71, 0.01):
            preds = (y_proba >= thresh).astype(int)
            tp = int(np.sum((preds == 1) & (y_test == 1)))
            fp = int(np.sum((preds == 1) & (y_test == 0)))
            fn = int(np.sum((preds == 0) & (y_test == 1)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
        logger.info("Optimal threshold=%.2f (F1=%.3f)", best_thresh, best_f1)
        return best_thresh

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def train_with_tuning(
        self,
        dataset: MLDataset,
        param_grid: dict[str, list] | None = None,
    ) -> TrainingResult:
        """Train with hyperparameter tuning via temporal CV grid search.

        Parameters
        ----------
        dataset:
            The ML dataset.
        param_grid:
            Dict of parameter lists.  Defaults to a sensible grid.

        Returns the :class:`TrainingResult` from the best parameter combo.
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "min_samples_leaf": [10, 20, 50],
            }

        X, y = dataset.X, dataset.y
        feature_names = dataset.feature_names

        # Feature selection first if enabled
        selected_features: tuple[str, ...] | None = None
        if self._config.use_feature_selection:
            X, feature_names, _kept = self._select_features(
                X, y, feature_names, self._config.feature_selection_threshold,
            )
            selected_features = feature_names

        n_samples = X.shape[0]
        split_idx = int(n_samples * (1 - self._config.test_size_ratio))
        split_idx = max(split_idx, 10)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        sample_weight = None
        if self._config.use_class_balance:
            sample_weight = self._compute_sample_weights(y_train)

        # Grid search with temporal CV
        cv = TimeSeriesExpandingCV(n_splits=3, min_train_ratio=0.5)
        if X_train_scaled.shape[0] < 4:
            raise ValueError(
                f"Dataset too small for tuning: {X_train_scaled.shape[0]} "
                f"training samples"
            )

        best_score = -1.0
        best_params: dict[str, Any] = {}
        combos = _grid_combos(param_grid)

        logger.info("Tuning over %d parameter combos", len(combos))

        for combo in combos:
            scores: list[float] = []
            for train_idx, test_idx in cv.split(X_train_scaled.shape[0]):
                idx_tr = list(train_idx)
                idx_te = list(test_idx)
                X_tr, X_te = X_train_scaled[idx_tr], X_train_scaled[idx_te]
                y_tr, y_te = y_train[idx_tr], y_train[idx_te]

                if len(np.unique(y_tr)) < 2:
                    continue

                sw_tr = sample_weight[idx_tr] if sample_weight is not None else None

                model = GradientBoostingClassifier(**combo)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, y_tr, sample_weight=sw_tr)

                if hasattr(model, "predict_proba") and len(np.unique(y_te)) > 1:
                    proba = model.predict_proba(X_te)[:, 1]
                    score = float(roc_auc_score(y_te, proba))
                else:
                    score = float(accuracy_score(y_te, model.predict(X_te)))
                scores.append(score)

            if scores:
                mean_score = sum(scores) / len(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = combo

        if not best_params:
            best_params = {
                "n_estimators": self._config.n_estimators,
                "max_depth": self._config.max_depth,
                "learning_rate": self._config.learning_rate,
                "min_samples_leaf": self._config.min_samples_leaf,
            }

        logger.info("Best params: %s (CV score=%.3f)", best_params, best_score)

        # Retrain with best params
        best_config = MLModelConfig(
            model_type="gradient_boosting",
            n_estimators=best_params.get("n_estimators", self._config.n_estimators),
            max_depth=best_params.get("max_depth", self._config.max_depth),
            learning_rate=best_params.get("learning_rate", self._config.learning_rate),
            min_samples_leaf=best_params.get("min_samples_leaf", self._config.min_samples_leaf),
            n_cv_folds=self._config.n_cv_folds,
            test_size_ratio=self._config.test_size_ratio,
            use_class_balance=self._config.use_class_balance,
            use_feature_selection=False,  # already done
        )

        # Build a temporary dataset with possibly filtered features
        tuned_dataset = MLDataset(
            X=X, y=y,
            dates=dataset.dates,
            feature_names=feature_names,
            label_config=dataset.label_config,
        )
        tuned_trainer = MLTrainer(best_config)
        result = tuned_trainer.train(tuned_dataset)

        # Patch selected_features into result if applicable
        if selected_features is not None:
            result = TrainingResult(
                model=result.model,
                oos_accuracy=result.oos_accuracy,
                oos_roc_auc=result.oos_roc_auc,
                feature_importances=result.feature_importances,
                train_size=result.train_size,
                test_size=result.test_size,
                class_distribution=result.class_distribution,
                cv_scores=result.cv_scores,
                config=best_config,
                ensemble_accuracy=result.ensemble_accuracy,
                ensemble_roc_auc=result.ensemble_roc_auc,
                optimal_threshold=result.optimal_threshold,
                selected_features=selected_features,
            )

        return result

    def _log_baseline_comparison(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        model_accuracy: float,
    ) -> None:
        """Train a LogisticRegression baseline and compare."""
        if self._config.model_type == "logistic_regression":
            return  # No point comparing LR to LR

        try:
            lr = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr.fit(X_train, y_train)
            lr_acc = float(accuracy_score(y_test, lr.predict(X_test)))

            mt = self._config.model_type
            improvement = model_accuracy - lr_acc
            logger.info(
                "%s accuracy=%.3f, LR baseline=%.3f (Δ=%.3f)",
                mt.upper(), model_accuracy, lr_acc, improvement,
            )
            if improvement < 0.01:
                logger.warning(
                    "%s does not substantially beat "
                    "LogisticRegression (Δ=%.3f). Signal may be weak.",
                    mt, improvement,
                )
        except Exception:
            logger.debug("Baseline comparison failed", exc_info=True)


# ------------------------------------------------------------------
# Scaled model wrapper
# ------------------------------------------------------------------


class _ScaledModel:
    """Wraps a scaler + model so inference can use raw features."""

    def __init__(self, scaler: Any, model: Any) -> None:
        self.scaler = scaler
        self.model = model

    def predict(self, X: Any) -> Any:
        """Predict class labels from raw features."""
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities from raw features."""
        return self.model.predict_proba(self.scaler.transform(X))

    @property
    def feature_importances_(self) -> Any:
        """Delegate to inner model if available."""
        return getattr(self.model, "feature_importances_", None)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def _grid_combos(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of dicts (cartesian product)."""
    from itertools import product

    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]
