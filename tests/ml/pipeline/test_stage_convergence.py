"""Tests for Stage 3: Strategy-ML Convergence Analysis."""

from __future__ import annotations

import random
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
from stockdownloader.ml.pipeline.config import ConvergenceConfig
from stockdownloader.ml.pipeline.results import (
    ConvergencePair,
    DataResult,
    ModelCandidate,
    TrainingStageResult,
)
from stockdownloader.ml.pipeline.stage_convergence import ConvergenceStage
from stockdownloader.ml.trainer import MLModelConfig
from stockdownloader.model.price_data import PriceData
from stockdownloader.strategy.trading_strategy import Signal


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_data(n: int = 300, seed: int = 42) -> list[PriceData]:
    rng = random.Random(seed)
    data: list[PriceData] = []
    price = 100.0
    for i in range(n):
        price += (rng.random() - 0.48) * 2
        price = max(50, price)
        h = price + rng.random() * 2
        low = price - rng.random() * 2
        data.append(PriceData(
            date=f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            open=Decimal(str(round(price - 0.5, 2))),
            high=Decimal(str(round(h, 2))),
            low=Decimal(str(round(low, 2))),
            close=Decimal(str(round(price, 2))),
            adj_close=Decimal(str(round(price, 2))),
            volume=int(1_000_000 + rng.random() * 5_000_000),
        ))
    return data


def _make_mock_candidate(
    n_samples: int = 50,
    all_bullish: bool = False,
) -> ModelCandidate:
    """Build a mock ModelCandidate with controllable ML predictions."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, 63)
    y = rng.randint(0, 2, size=n_samples)

    # Mock model that returns controllable probabilities
    model = MagicMock()
    if all_bullish:
        model.predict_proba.return_value = np.array([[0.2, 0.8]])
    else:
        # Alternate bullish/bearish
        call_count = [0]

        def _proba(x: object) -> object:
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return np.array([[0.7, 0.3]])  # bearish
            return np.array([[0.3, 0.7]])  # bullish

        model.predict_proba.side_effect = _proba

    # Build a mock TrainingResult
    training_result = MagicMock()
    training_result.model = model
    training_result.oos_accuracy = 0.55
    training_result.oos_roc_auc = 0.56
    training_result.optimal_threshold = 0.5
    training_result.feature_importances = {f"feat_{i}": 1.0 / 63 for i in range(63)}

    dataset = MLDataset(
        X=X,
        y=y,
        dates=tuple(f"2024-01-{i + 1:03d}" for i in range(n_samples)),
        feature_names=tuple(f"feat_{i}" for i in range(63)),
        label_config=LabelConfig(forward_period=5, profit_threshold=0.0),
    )

    return ModelCandidate(
        model_id="test_model",
        training_result=training_result,
        label_config=LabelConfig(forward_period=5, profit_threshold=0.0),
        model_config=MLModelConfig(),
        dataset=dataset,
    )


def _data_result(n: int = 300) -> DataResult:
    data = _make_data(n)
    return DataResult(
        symbol="SPY",
        data=data,
        date_range=(data[0].date, data[-1].date),
        bar_count=len(data),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestGetMLPredictions:
    def test_returns_predictions(self) -> None:
        candidate = _make_mock_candidate(50)
        data = _make_data(300)
        preds = ConvergenceStage._get_ml_predictions(candidate, data)
        assert len(preds) > 0
        for bar_i, prob in preds.items():
            assert 0 <= prob <= 1
            assert bar_i >= 201  # warmup


class TestAnalyzePair:
    def test_returns_pair_with_enough_data(self) -> None:
        candidate = _make_mock_candidate(50)
        data = _make_data(300)
        ml_preds = ConvergenceStage._get_ml_predictions(candidate, data)

        # Mock strategy that always says BUY
        mock_strat = MagicMock()
        mock_strat.evaluate.return_value = Signal.BUY

        pair = ConvergenceStage._analyze_pair(
            candidate, "mock_buy", mock_strat, data, ml_preds,
        )
        # Should get a pair since we have 50 predictions and strategy never HOLDs
        assert pair is not None
        assert pair.model_id == "test_model"
        assert pair.strategy_name == "mock_buy"
        assert pair.agreement_rate >= 0
        assert pair.agreement_rate <= 1

    def test_returns_none_with_few_bars(self) -> None:
        candidate = _make_mock_candidate(5)
        data = _make_data(300)
        ml_preds = {210: 0.6, 211: 0.4}

        mock_strat = MagicMock()
        mock_strat.evaluate.return_value = Signal.BUY

        pair = ConvergenceStage._analyze_pair(
            candidate, "mock_buy", mock_strat, data, ml_preds,
        )
        # Too few bars (< 20)
        assert pair is None

    def test_hold_bars_excluded(self) -> None:
        candidate = _make_mock_candidate(50)
        data = _make_data(300)
        ml_preds = ConvergenceStage._get_ml_predictions(candidate, data)

        # Strategy always HOLDs
        mock_strat = MagicMock()
        mock_strat.evaluate.return_value = Signal.HOLD

        pair = ConvergenceStage._analyze_pair(
            candidate, "hold", mock_strat, data, ml_preds,
        )
        assert pair is None  # No non-HOLD bars

    def test_lift_computed(self) -> None:
        candidate = _make_mock_candidate(50, all_bullish=True)
        data = _make_data(300)
        ml_preds = ConvergenceStage._get_ml_predictions(candidate, data)

        # Strategy alternates BUY/SELL
        call_count = [0]

        def _eval(d: object, i: int) -> Signal:
            call_count[0] += 1
            return Signal.BUY if call_count[0] % 2 == 0 else Signal.SELL

        mock_strat = MagicMock()
        mock_strat.evaluate.side_effect = _eval

        pair = ConvergenceStage._analyze_pair(
            candidate, "alt", mock_strat, data, ml_preds,
        )
        if pair is not None:
            assert isinstance(pair.lift, float)
            assert pair.n_agreement_bars >= 0
            assert pair.n_disagreement_bars >= 0


class TestConvergenceStageRun:
    def test_full_run_with_mock(self) -> None:
        candidate = _make_mock_candidate(50)
        data_r = _data_result(300)
        training_r = TrainingStageResult(
            candidates=[candidate],
            best_by_auc=candidate,
            best_by_accuracy=candidate,
        )
        cfg = ConvergenceConfig(strategy_categories=("daily",))
        stage = ConvergenceStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(data_r, training_r)

        assert isinstance(result.pairs, list)
        # Some pairs should exist (7 daily strategies × 1 model)
        # Some may be None due to too few non-HOLD bars
        for pair in result.pairs:
            assert isinstance(pair, ConvergencePair)
            assert pair.model_id == "test_model"
