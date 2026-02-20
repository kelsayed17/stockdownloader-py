"""Tests for MLPredictor — inference wrapper."""

from __future__ import annotations

import random
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pytest

from stockdownloader.ml.feature_extractor import FeatureExtractor
from stockdownloader.ml.model_store import ModelMetadata
from stockdownloader.ml.predictor import MLPredictor
from stockdownloader.model.price_data import PriceData


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_data(n: int = 250, seed: int = 42) -> list[PriceData]:
    rng = random.Random(seed)
    data: list[PriceData] = []
    price = 100.0
    for i in range(n):
        price += (rng.random() - 0.48) * 2
        price = max(50, price)
        h = price + rng.random() * 2
        l = price - rng.random() * 2
        data.append(PriceData(
            date=f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            open=Decimal(str(round(price - 0.5, 2))),
            high=Decimal(str(round(h, 2))),
            low=Decimal(str(round(l, 2))),
            close=Decimal(str(round(price, 2))),
            adj_close=Decimal(str(round(price, 2))),
            volume=int(1_000_000 + rng.random() * 5_000_000),
        ))
    return data


DATA_250 = _make_data(250)


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        symbol="SPY",
        model_type="gradient_boosting",
        feature_names=FeatureExtractor.FEATURE_NAMES,
        training_date_range=("2024-01-01", "2024-12-31"),
        oos_accuracy=0.62,
        oos_roc_auc=0.65,
        class_distribution={"0": 120, "1": 80},
        trained_at="2024-12-31T23:59:59+00:00",
        config={"n_estimators": 50},
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestMLPredictor:
    def test_predict_proba_returns_float(self) -> None:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        predictor = MLPredictor(
            model=mock_model,
            metadata=_metadata(),
            extractor=FeatureExtractor(),
        )

        prob = predictor.predict_proba(DATA_250, index=220)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_proba_in_range(self) -> None:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])

        predictor = MLPredictor(
            model=mock_model,
            metadata=_metadata(),
            extractor=FeatureExtractor(),
        )

        prob = predictor.predict_proba(DATA_250, index=220)
        assert prob == pytest.approx(0.6)

    def test_predict_proba_fallback_on_error(self) -> None:
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = RuntimeError("broken")

        predictor = MLPredictor(
            model=mock_model,
            metadata=_metadata(),
            extractor=FeatureExtractor(),
        )

        prob = predictor.predict_proba(DATA_250, index=220)
        assert prob == 0.5  # neutral fallback

    def test_predict_without_predict_proba(self) -> None:
        """Falls back to predict() if predict_proba not available."""
        mock_model = MagicMock(spec=["predict"])
        mock_model.predict.return_value = np.array([1])

        predictor = MLPredictor(
            model=mock_model,
            metadata=_metadata(),
            extractor=FeatureExtractor(),
        )

        prob = predictor.predict_proba(DATA_250, index=220)
        assert prob == 1.0

    def test_is_available(self) -> None:
        predictor = MLPredictor(
            model=MagicMock(),
            metadata=_metadata(),
            extractor=FeatureExtractor(),
        )
        assert predictor.is_available is True

    def test_metadata_accessible(self) -> None:
        meta = _metadata()
        predictor = MLPredictor(
            model=MagicMock(),
            metadata=meta,
            extractor=FeatureExtractor(),
        )
        assert predictor.metadata.symbol == "SPY"

    def test_predict_proba_index_too_low(self) -> None:
        """Index < 201 triggers ValueError in extractor → returns 0.5."""
        mock_model = MagicMock()
        predictor = MLPredictor(
            model=mock_model,
            metadata=_metadata(),
            extractor=FeatureExtractor(),
        )

        prob = predictor.predict_proba(DATA_250, index=50)
        assert prob == 0.5  # fallback
