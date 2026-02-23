"""Run inference with a trained ML model on live price data.

Usage::

    from stockdownloader.ml.predictor import MLPredictor

    predictor = MLPredictor.from_path("output/models/spy/gradient_boosting_latest.joblib")
    prob = predictor.predict_proba(daily_data, index=len(daily_data) - 1)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from stockdownloader.ml.feature_extractor import FeatureExtractor
from stockdownloader.ml.model_store import ModelMetadata, ModelStore
from stockdownloader.util.indicators.hub import IndicatorHub

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

logger = logging.getLogger(__name__)


class MLPredictor:
    """Predict P(profitable) using a trained model.

    Parameters
    ----------
    model:
        A trained model (or ``_ScaledModel`` wrapper) with
        ``predict_proba(X)`` method.
    metadata:
        Model metadata (feature names, symbol, etc.).
    extractor:
        Feature extraction engine.
    """

    def __init__(
        self,
        model: Any,
        metadata: ModelMetadata,
        extractor: FeatureExtractor,
    ) -> None:
        self._model = model
        self._metadata = metadata
        self._extractor = extractor

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        path: str,
        hub: IndicatorHub | None = None,
    ) -> MLPredictor:
        """Load a model from disk and create a ready-to-use predictor.

        Parameters
        ----------
        path:
            Path to a ``.joblib`` model file.
        hub:
            Optional shared :class:`IndicatorHub`.
        """
        store = ModelStore()
        model, metadata = store.load(path)
        extractor = FeatureExtractor(hub=hub)
        return cls(model=model, metadata=metadata, extractor=extractor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> float:
        """Return P(profitable) in ``[0, 1]``.

        Returns ``0.5`` (neutral) on any error to avoid corrupting
        upstream confidence calculations.
        """
        try:
            fv = self._extractor.extract(data, index)
            X = np.array([fv.values], dtype=np.float64)

            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(X)
                return float(proba[0, 1])
            else:
                # Fallback: binary prediction → 0.0 or 1.0
                pred = self._model.predict(X)
                return float(pred[0])
        except Exception:
            logger.debug("ML prediction failed, returning 0.5", exc_info=True)
            return 0.5

    @property
    def is_available(self) -> bool:
        """Whether this predictor has a usable model."""
        return self._model is not None

    @property
    def metadata(self) -> ModelMetadata:
        """Model metadata."""
        return self._metadata
