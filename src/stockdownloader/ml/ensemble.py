"""Ensemble prediction via soft-voting over multiple trained models.

Provides :class:`EnsemblePredictor` for averaging ``predict_proba`` from
N trained models (soft-voting), and :class:`EnsembleBuilder` for selecting
the top-performing models from a collection of :class:`TrainingResult`.

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
    """Soft-voting ensemble that averages ``predict_proba`` across models.

    Parameters
    ----------
    results:
        Non-empty list of :class:`TrainingResult` objects whose ``.model``
        supports ``predict_proba(X)``.

    Raises
    ------
    ValueError
        If *results* is empty.
    """

    def __init__(self, results: list[TrainingResult]) -> None:
        if not results:
            raise ValueError("EnsemblePredictor requires at least one TrainingResult")
        self._results = list(results)

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

    def predict_proba(self, X: NDArray) -> NDArray:
        """Average P(profitable) across all models.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_samples,)`` with averaged
            probability of the positive class (profitable).
        """
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
