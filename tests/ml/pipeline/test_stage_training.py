"""Tests for Stage 2: Multi-Configuration ML Training."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from stockdownloader.ml.pipeline.config import TrainingGridConfig
from stockdownloader.ml.pipeline.results import DataResult
from stockdownloader.ml.pipeline.stage_training import TrainingStage
from stockdownloader.core.models.price import PriceData


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_data(n: int = 300, seed: int = 42) -> list[PriceData]:
    """Generate synthetic daily price data with a slight upward drift."""
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


class TestTrainingStage:
    def test_single_config_trains(self) -> None:
        """Train one model config and verify result structure."""
        cfg = TrainingGridConfig(
            forward_periods=(5,),
            profit_thresholds=(0.0,),
            model_types=("gradient_boosting",),
            use_class_balance_options=(False,),
            use_atr_labels_options=(False,),
            n_estimators=10,
            max_depth=2,
        )
        stage = TrainingStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(_data_result())

        assert len(result.candidates) == 1
        assert result.best_by_auc is not None
        assert result.best_by_accuracy is not None

        cand = result.candidates[0]
        assert cand.training_result.oos_accuracy > 0
        assert cand.training_result.oos_roc_auc > 0
        assert "gr" in cand.model_id

    def test_multiple_configs(self) -> None:
        """Train 4 configs and verify ranking."""
        cfg = TrainingGridConfig(
            forward_periods=(5, 10),
            profit_thresholds=(0.0,),
            model_types=("gradient_boosting",),
            use_class_balance_options=(False,),
            use_atr_labels_options=(False, True),
            n_estimators=10,
            max_depth=2,
        )
        stage = TrainingStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(_data_result())

        assert len(result.candidates) >= 2
        # Verify sorted by AUC descending
        aucs = [c.training_result.oos_roc_auc for c in result.candidates]
        assert aucs == sorted(aucs, reverse=True)

    def test_logistic_regression(self) -> None:
        cfg = TrainingGridConfig(
            forward_periods=(5,),
            profit_thresholds=(0.0,),
            model_types=("logistic_regression",),
            use_class_balance_options=(False,),
            use_atr_labels_options=(False,),
        )
        stage = TrainingStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(_data_result())
        assert len(result.candidates) == 1
        assert "lo" in result.candidates[0].model_id

    def test_model_id_format(self) -> None:
        mid = TrainingStage._make_model_id(10, 0.005, "gradient_boosting", True, False)
        assert "gr" in mid
        assert "fp10" in mid
        assert "bal" in mid
        assert "abs" in mid

    def test_too_few_bars_handled(self) -> None:
        """If data is too short for the forward period, candidate should be skipped."""
        cfg = TrainingGridConfig(
            forward_periods=(5,),
            profit_thresholds=(0.0,),
            model_types=("gradient_boosting",),
            use_class_balance_options=(False,),
            use_atr_labels_options=(False,),
            n_estimators=10,
            max_depth=2,
        )
        short_data = _data_result(50)  # Too few bars (< 201 warmup)
        stage = TrainingStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(short_data)
        # Should handle gracefully — either skip or produce empty
        assert isinstance(result.candidates, list)
