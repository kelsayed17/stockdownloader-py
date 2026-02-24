"""Tests for Stage 4: ML-Informed Hybrid Strategies."""

from __future__ import annotations

import random
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
from stockdownloader.ml.pipeline.config import HybridConfig
from stockdownloader.ml.pipeline.results import (
    ConvergencePair,
    ConvergenceResult,
    DataResult,
    ModelCandidate,
    TrainingStageResult,
)
from stockdownloader.ml.pipeline.stage_hybrid_strategies import (
    HybridStage,
    MLConfirmedStrategy,
    MLOverrideStrategy,
    MLWeightedStrategy,
)
from stockdownloader.ml.trainer import MLModelConfig
from stockdownloader.core.models.price import PriceData
from stockdownloader.strategies.base import Signal, TradingStrategy


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


class _AlwaysBuyStrategy(TradingStrategy):
    @property
    def name(self) -> str:
        return "always_buy"

    @property
    def warmup_period(self) -> int:
        return 0

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        return Signal.BUY


class _AlwaysSellStrategy(TradingStrategy):
    @property
    def name(self) -> str:
        return "always_sell"

    @property
    def warmup_period(self) -> int:
        return 0

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        return Signal.SELL


class _AlwaysHoldStrategy(TradingStrategy):
    @property
    def name(self) -> str:
        return "always_hold"

    @property
    def warmup_period(self) -> int:
        return 0

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        return Signal.HOLD


def _bullish_model() -> MagicMock:
    """Model that always predicts bullish (prob=0.8)."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.2, 0.8]])
    return m


def _bearish_model() -> MagicMock:
    """Model that always predicts bearish (prob=0.2)."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.8, 0.2]])
    return m


def _neutral_model() -> MagicMock:
    """Model that predicts 50/50."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.5, 0.5]])
    return m


_FEAT_NAMES = tuple(f"feat_{i}" for i in range(63))


# ------------------------------------------------------------------
# MLConfirmedStrategy tests
# ------------------------------------------------------------------


class TestMLConfirmedStrategy:
    def test_buy_confirmed_by_bullish_ml(self) -> None:
        strat = MLConfirmedStrategy(
            _AlwaysBuyStrategy(), _bullish_model(), _FEAT_NAMES,
            threshold=0.6,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        assert signal == Signal.BUY

    def test_buy_rejected_by_neutral_ml(self) -> None:
        strat = MLConfirmedStrategy(
            _AlwaysBuyStrategy(), _neutral_model(), _FEAT_NAMES,
            threshold=0.6,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        assert signal == Signal.HOLD

    def test_sell_confirmed_by_bearish_ml(self) -> None:
        strat = MLConfirmedStrategy(
            _AlwaysSellStrategy(), _bearish_model(), _FEAT_NAMES,
            threshold=0.6,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        assert signal == Signal.SELL

    def test_hold_passes_through(self) -> None:
        strat = MLConfirmedStrategy(
            _AlwaysHoldStrategy(), _bullish_model(), _FEAT_NAMES,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        assert signal == Signal.HOLD

    def test_warmup_returns_hold(self) -> None:
        strat = MLConfirmedStrategy(
            _AlwaysBuyStrategy(), _bullish_model(), _FEAT_NAMES,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 10)
        assert signal == Signal.HOLD

    def test_name(self) -> None:
        strat = MLConfirmedStrategy(
            _AlwaysBuyStrategy(), _bullish_model(), _FEAT_NAMES,
        )
        assert "ML-Confirmed" in strat.name

    def test_warmup_period(self) -> None:
        strat = MLConfirmedStrategy(
            _AlwaysBuyStrategy(), _bullish_model(), _FEAT_NAMES,
        )
        assert strat.warmup_period >= 201


# ------------------------------------------------------------------
# MLWeightedStrategy tests
# ------------------------------------------------------------------


class TestMLWeightedStrategy:
    def test_buy_with_soft_threshold(self) -> None:
        strat = MLWeightedStrategy(
            _AlwaysBuyStrategy(), _neutral_model(), _FEAT_NAMES,
            threshold=0.5,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        assert signal == Signal.BUY

    def test_buy_rejected_below_threshold(self) -> None:
        strat = MLWeightedStrategy(
            _AlwaysBuyStrategy(), _bearish_model(), _FEAT_NAMES,
            threshold=0.5,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        assert signal == Signal.HOLD


# ------------------------------------------------------------------
# MLOverrideStrategy tests
# ------------------------------------------------------------------


class TestMLOverrideStrategy:
    def test_ml_buys_when_strategy_holds(self) -> None:
        strat = MLOverrideStrategy(
            _AlwaysHoldStrategy(), _bullish_model(), _FEAT_NAMES,
            threshold=0.55,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        assert signal == Signal.BUY

    def test_ml_blocked_by_opposite_strategy(self) -> None:
        strat = MLOverrideStrategy(
            _AlwaysSellStrategy(), _bullish_model(), _FEAT_NAMES,
            threshold=0.55,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        # ML wants to buy, but strategy says SELL → blocked
        assert signal == Signal.HOLD

    def test_ml_sells_when_strategy_holds(self) -> None:
        strat = MLOverrideStrategy(
            _AlwaysHoldStrategy(), _bearish_model(), _FEAT_NAMES,
            threshold=0.55,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        assert signal == Signal.SELL

    def test_ml_sell_blocked_by_buy_strategy(self) -> None:
        strat = MLOverrideStrategy(
            _AlwaysBuyStrategy(), _bearish_model(), _FEAT_NAMES,
            threshold=0.55,
        )
        data = _make_data(250)
        signal = strat.evaluate(data, 210)
        # ML wants sell, strategy says BUY → blocked
        assert signal == Signal.HOLD


# ------------------------------------------------------------------
# HybridStage tests
# ------------------------------------------------------------------


def _make_mock_candidate() -> ModelCandidate:
    rng = np.random.RandomState(42)
    X = rng.randn(50, 63)
    y = rng.randint(0, 2, size=50)

    model = _bullish_model()
    training_result = MagicMock()
    training_result.model = model
    training_result.oos_accuracy = 0.55
    training_result.oos_roc_auc = 0.56
    training_result.optimal_threshold = 0.5
    training_result.feature_importances = {f"feat_{i}": 1.0 / 63 for i in range(63)}

    dataset = MLDataset(
        X=X, y=y,
        dates=tuple(f"2024-01-{i + 1:03d}" for i in range(50)),
        feature_names=_FEAT_NAMES,
        label_config=LabelConfig(forward_period=5, profit_threshold=0.0),
    )

    return ModelCandidate(
        model_id="test_model",
        training_result=training_result,
        label_config=LabelConfig(forward_period=5, profit_threshold=0.0),
        model_config=MLModelConfig(),
        dataset=dataset,
    )


class TestHybridStage:
    def test_creates_hybrids(self) -> None:
        candidate = _make_mock_candidate()
        data = _make_data(300)
        data_r = DataResult(
            symbol="SPY", data=data,
            date_range=(data[0].date, data[-1].date),
            bar_count=len(data),
        )
        training_r = TrainingStageResult(
            candidates=[candidate],
            best_by_auc=candidate,
        )
        convergence_r = ConvergenceResult(
            pairs=[
                ConvergencePair(
                    model_id="test_model",
                    strategy_name="sma",
                    agreement_rate=0.6,
                    agreement_win_rate=0.65,
                    disagreement_win_rate=0.45,
                    lift=0.20,
                    n_agreement_bars=100,
                    n_disagreement_bars=60,
                ),
            ],
            best_pair=None,
        )

        cfg = HybridConfig(modes=("confirmed", "weighted", "override"))
        stage = HybridStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(data_r, training_r, convergence_r)

        assert len(result.hybrid_strategies) == 3
        modes = {e.mode for e in result.hybrid_strategies}
        assert modes == {"confirmed", "weighted", "override"}

        for entry in result.hybrid_strategies:
            assert isinstance(entry.strategy, TradingStrategy)
            assert entry.model_id == "test_model"
            assert entry.base_strategy_name == "sma"

    def test_empty_convergence(self) -> None:
        candidate = _make_mock_candidate()
        data = _make_data(300)
        data_r = DataResult(
            symbol="SPY", data=data,
            date_range=(data[0].date, data[-1].date),
            bar_count=len(data),
        )
        training_r = TrainingStageResult(candidates=[candidate])
        convergence_r = ConvergenceResult(pairs=[])

        cfg = HybridConfig()
        stage = HybridStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(data_r, training_r, convergence_r)
        assert len(result.hybrid_strategies) == 0
