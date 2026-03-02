"""Ensemble prediction via soft-voting or stacking over multiple trained models.

Provides :class:`EnsemblePredictor` for combining ``predict_proba`` from
N trained models (soft-voting or stacking), and :class:`EnsembleBuilder`
for selecting the top-performing models from a collection of
:class:`TrainingResult`.

Usage::

    from stockdownloader.ml.ensemble import EnsembleBuilder
    from stockdownloader.ml.trainer import MLTrainer, MLModelConfig

    results = [MLTrainer(cfg).train(dataset) for cfg in configs]
    ensemble = EnsembleBuilder(results).select_top(n=3)
    proba = ensemble.predict_proba(X_new)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from stockdownloader.ml.trainer import TrainingResult

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble that combines ``predict_proba`` across models.

    Supports two combination methods:

    * ``"soft_vote"`` (default): averages ``predict_proba[:, 1]`` across
      all base models.
    * ``"stacking"``: feeds base model predictions into a
      :class:`~sklearn.linear_model.LogisticRegression` meta-learner
      trained via :meth:`fit_stacking`.

    Parameters
    ----------
    results:
        Non-empty list of :class:`TrainingResult` objects whose ``.model``
        supports ``predict_proba(X)``.
    ensemble_method:
        Combination strategy — ``"soft_vote"`` or ``"stacking"``.

    Raises
    ------
    ValueError
        If *results* is empty or *ensemble_method* is invalid.
    """

    def __init__(
        self,
        results: list[TrainingResult],
        ensemble_method: str = "soft_vote",
    ) -> None:
        if not results:
            raise ValueError("EnsemblePredictor requires at least one TrainingResult")
        if ensemble_method not in ("soft_vote", "stacking"):
            raise ValueError(
                f"ensemble_method must be 'soft_vote' or 'stacking', "
                f"got {ensemble_method!r}"
            )
        self._results = list(results)
        self._ensemble_method = ensemble_method
        self._meta_learner = None  # fitted by fit_stacking()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_models(self) -> int:
        """Number of models in the ensemble."""
        return len(self._results)

    @property
    def results(self) -> list[TrainingResult]:
        """The underlying training results."""
        return list(self._results)

    @property
    def ensemble_method(self) -> str:
        """The ensemble combination method."""
        return self._ensemble_method

    # ------------------------------------------------------------------
    # Stacking
    # ------------------------------------------------------------------

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

        meta_X = self._base_predictions(X)
        self._meta_learner = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        )
        self._meta_learner.fit(meta_X, y)

    def _base_predictions(self, X: NDArray) -> NDArray:
        """Get base model probability predictions as a matrix.

        Returns shape (n_samples, n_models).
        """
        preds: list[NDArray] = []
        for result in self._results:
            try:
                proba_2d = result.model.predict_proba(X)
                preds.append(proba_2d[:, 1])
            except Exception:
                logger.warning(
                    "Model with config %s failed predict_proba — skipping",
                    result.config,
                )
                continue
        if not preds:
            raise RuntimeError(
                "All models failed predict_proba; cannot produce base predictions"
            )
        return np.column_stack(preds)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: NDArray) -> NDArray:
        """Compute ensemble probability of the positive class.

        For ``"soft_vote"``: averages ``predict_proba[:, 1]`` across models.
        For ``"stacking"``: passes base model predictions through a
        trained LogisticRegression meta-learner.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_samples,)`` with ensemble
            probability of the positive class (profitable).
        """
        if self._ensemble_method == "stacking":
            if self._meta_learner is None:
                raise RuntimeError(
                    "Stacking ensemble requires calling fit_stacking() first"
                )
            meta_X = self._base_predictions(X)
            return self._meta_learner.predict_proba(meta_X)[:, 1]

        # Soft-vote: existing averaging logic
        probas: list[NDArray] = []
        for result in self._results:
            try:
                proba_2d = result.model.predict_proba(X)
                # Column 1 is P(profitable)
                probas.append(proba_2d[:, 1])
            except Exception:
                logger.warning(
                    "Model with config %s failed predict_proba — skipping",
                    result.config,
                )
                continue

        if not probas:
            raise RuntimeError(
                "All models failed predict_proba; cannot produce ensemble prediction"
            )

        stacked = np.stack(probas, axis=0)  # (n_models, n_samples)
        averaged = np.mean(stacked, axis=0)  # (n_samples,)
        return averaged

    def averaged_feature_importances(self) -> dict[str, float]:
        """Average feature importances across all models, normalized to sum=1.

        Returns
        -------
        dict[str, float]
            Feature names mapped to importance scores, sorted descending.
        """
        all_names: set[str] = set()
        for result in self._results:
            all_names.update(result.feature_importances.keys())

        if not all_names:
            return {}

        aggregated: dict[str, float] = {name: 0.0 for name in all_names}
        for result in self._results:
            for name, importance in result.feature_importances.items():
                aggregated[name] += importance

        n_models = len(self._results)
        for name in aggregated:
            aggregated[name] /= n_models

        # Normalize to sum=1.0
        total = sum(aggregated.values())
        if total > 0:
            for name in aggregated:
                aggregated[name] /= total

        # Sort descending by importance
        sorted_importances = dict(
            sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_importances


class EnsembleBuilder:
    """Selects the top-N models by out-of-sample accuracy.

    Parameters
    ----------
    results:
        List of :class:`TrainingResult` objects to select from.

    Raises
    ------
    ValueError
        If *results* is empty.
    """

    def __init__(self, results: list[TrainingResult]) -> None:
        if not results:
            raise ValueError("EnsembleBuilder requires at least one TrainingResult")
        self._results = list(results)

    def select_top(self, n: int = 3) -> EnsemblePredictor:
        """Return an :class:`EnsemblePredictor` with the top *n* models.

        Models are ranked by ``oos_accuracy`` (descending).  If fewer than
        *n* results are available, all results are used.

        Parameters
        ----------
        n:
            Maximum number of models to include.

        Returns
        -------
        EnsemblePredictor
        """
        sorted_results = sorted(
            self._results,
            key=lambda r: r.oos_accuracy,
            reverse=True,
        )
        selected = sorted_results[:n]
        logger.info(
            "EnsembleBuilder selected top %d/%d models (accuracies: %s)",
            len(selected),
            len(self._results),
            [round(r.oos_accuracy, 4) for r in selected],
        )
        return EnsemblePredictor(selected)

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
        candidate's feature importance vector and any existing member's.

        Parameters
        ----------
        n:
            Number of models to select (default 3).
        diversity_weight:
            Weight for diversity vs accuracy (default 0.4).
            0.0 = pure accuracy (same as select_top).
            1.0 = pure diversity.
        """
        n = min(n, len(self._results))

        # Sort by accuracy descending
        ranked = sorted(
            self._results,
            key=lambda r: r.oos_accuracy,
            reverse=True,
        )

        # Seed with best model
        selected: list[TrainingResult] = [ranked[0]]
        remaining = list(ranked[1:])

        while len(selected) < n and remaining:
            best_score = -1.0
            best_idx = 0

            for i, candidate in enumerate(remaining):
                acc = candidate.oos_accuracy

                # Compute diversity: max correlation with existing members
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
        keys = sorted(set(imp_a) | set(imp_b))
        if not keys:
            return 0.0
        a = np.array([imp_a.get(k, 0.0) for k in keys])
        b = np.array([imp_b.get(k, 0.0) for k in keys])
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
