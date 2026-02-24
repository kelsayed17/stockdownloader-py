"""Hyperparameter tuning, temporal CV, and training helpers.

Extracted from :mod:`stockdownloader.ml.trainer` to keep that module
focused on core training logic.  Functions that need trainer state
accept the :class:`MLTrainer` instance as their first argument.
"""

from __future__ import annotations

import logging
import warnings
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

if TYPE_CHECKING:
    from stockdownloader.ml.dataset_builder import MLDataset
    from stockdownloader.ml.trainer import MLModelConfig, MLTrainer, TrainingResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Temporal cross-validation
# ------------------------------------------------------------------


class TimeSeriesExpandingCV:
    """Expanding-window cross-validator for time-series data.

    Each fold trains on data **before** the test window -- no look-ahead.

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
# Scaled model wrapper
# ------------------------------------------------------------------


class ScaledModel:
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


def grid_combos(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of dicts (cartesian product)."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ------------------------------------------------------------------
# Ensemble (GB + LR average)
# ------------------------------------------------------------------


def try_ensemble(
    trainer: MLTrainer,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    sample_weight: Any | None = None,
) -> tuple[float | None, float | None]:
    """Train primary model + LR, average predict_proba, return (accuracy, roc_auc)."""
    if trainer._config.model_type == "logistic_regression":
        return None, None

    try:
        primary = trainer._build_model()
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

        mt = trainer._config.model_type
        logger.info("Ensemble (%s+LR) accuracy=%.3f, AUC=%.3f", mt, acc, auc)
        return acc, auc
    except Exception:
        logger.debug("Ensemble failed", exc_info=True)
        return None, None


# ------------------------------------------------------------------
# Baseline comparison
# ------------------------------------------------------------------


def log_baseline_comparison(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    model_accuracy: float,
    model_type: str = "gradient_boosting",
) -> None:
    """Train a LogisticRegression baseline and compare."""
    if model_type == "logistic_regression":
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

        improvement = model_accuracy - lr_acc
        logger.info(
            "%s accuracy=%.3f, LR baseline=%.3f (\u0394=%.3f)",
            model_type.upper(), model_accuracy, lr_acc, improvement,
        )
        if improvement < 0.01:
            logger.warning(
                "%s does not substantially beat "
                "LogisticRegression (\u0394=%.3f). Signal may be weak.",
                model_type, improvement,
            )
    except Exception:
        logger.debug("Baseline comparison failed", exc_info=True)


# ------------------------------------------------------------------
# Hyperparameter tuning
# ------------------------------------------------------------------


def run_tuning(
    trainer: MLTrainer,
    dataset: MLDataset,
    param_grid: dict[str, list] | None = None,
) -> TrainingResult:
    """Train with hyperparameter tuning via temporal CV grid search.

    Parameters
    ----------
    trainer:
        The :class:`MLTrainer` instance (provides config, model building,
        feature selection, and retrain capability).
    dataset:
        The ML dataset.
    param_grid:
        Dict of parameter lists.  Defaults to a sensible grid.

    Returns the :class:`TrainingResult` from the best parameter combo.
    """
    # Deferred imports to avoid circular dependency at module level
    from stockdownloader.ml.trainer import MLModelConfig, MLTrainer, TrainingResult

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
    if trainer._config.use_feature_selection:
        X, feature_names, _kept = trainer._select_features(
            X, y, feature_names, trainer._config.feature_selection_threshold,
        )
        selected_features = feature_names

    n_samples = X.shape[0]
    split_idx = int(n_samples * (1 - trainer._config.test_size_ratio))
    split_idx = max(split_idx, 10)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sample_weight = None
    if trainer._config.use_class_balance:
        sample_weight = trainer._compute_sample_weights(y_train)

    # Grid search with temporal CV
    cv = TimeSeriesExpandingCV(n_splits=3, min_train_ratio=0.5)
    if X_train_scaled.shape[0] < 4:
        raise ValueError(
            f"Dataset too small for tuning: {X_train_scaled.shape[0]} "
            f"training samples"
        )

    best_score = -1.0
    best_params: dict[str, Any] = {}
    combos = grid_combos(param_grid)

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
            "n_estimators": trainer._config.n_estimators,
            "max_depth": trainer._config.max_depth,
            "learning_rate": trainer._config.learning_rate,
            "min_samples_leaf": trainer._config.min_samples_leaf,
        }

    logger.info("Best params: %s (CV score=%.3f)", best_params, best_score)

    # Retrain with best params
    best_config = MLModelConfig(
        model_type="gradient_boosting",
        n_estimators=best_params.get("n_estimators", trainer._config.n_estimators),
        max_depth=best_params.get("max_depth", trainer._config.max_depth),
        learning_rate=best_params.get("learning_rate", trainer._config.learning_rate),
        min_samples_leaf=best_params.get(
            "min_samples_leaf", trainer._config.min_samples_leaf,
        ),
        n_cv_folds=trainer._config.n_cv_folds,
        test_size_ratio=trainer._config.test_size_ratio,
        use_class_balance=trainer._config.use_class_balance,
        use_feature_selection=False,  # already done
    )

    # Build a temporary dataset with possibly filtered features
    from stockdownloader.ml.dataset_builder import MLDataset as _MLDataset

    tuned_dataset = _MLDataset(
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
