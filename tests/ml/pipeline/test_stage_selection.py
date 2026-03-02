"""Tests for Stage 6: Selection + PineScript Export."""

from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
from stockdownloader.ml.pipeline.config import SelectionConfig
from stockdownloader.ml.pipeline.results import (
    BacktestEntry,
    BacktestStageResult,
    ModelCandidate,
    TrainingStageResult,
)
from stockdownloader.ml.pipeline.stage_selection import SelectionStage
from stockdownloader.ml.trainer import MLModelConfig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mock_backtest_result(pnl: float = 5000, wr: float = 55.0) -> object:
    r = MagicMock()
    r.total_pnl = Decimal(str(pnl))
    r.win_rate = Decimal(str(wr))
    r.sharpe_ratio.return_value = Decimal("1.2")
    r.profit_factor = Decimal("1.5")
    r.max_drawdown = Decimal("8.0")
    r.total_trades = 30
    return r


def _make_backtest_entries() -> list[BacktestEntry]:
    return [
        BacktestEntry(
            strategy_name="Best Strategy",
            is_hybrid=True,
            mode="confirmed",
            model_id="test_model",
            result=_mock_backtest_result(10000, 60.0),
            composite_score=85.0,
        ),
        BacktestEntry(
            strategy_name="Second Strategy",
            is_hybrid=False,
            mode="baseline",
            model_id=None,
            result=_mock_backtest_result(5000, 55.0),
            composite_score=60.0,
        ),
        BacktestEntry(
            strategy_name="Third Strategy",
            is_hybrid=True,
            mode="override",
            model_id="test_model",
            result=_mock_backtest_result(2000, 50.0),
            composite_score=40.0,
        ),
    ]


def _make_training_result() -> TrainingStageResult:
    rng = np.random.RandomState(42)
    from stockdownloader.ml.feature_extractor import _ALL_FEATURE_NAMES

    n_feats = len(_ALL_FEATURE_NAMES)
    n = 200
    half = n // 2
    X = np.vstack([rng.randn(half, n_feats) - 0.5, rng.randn(half, n_feats) + 0.5])
    y = np.array([0] * half + [1] * half)
    dates = tuple(f"2024-01-{i + 1:03d}" for i in range(n))

    dataset = MLDataset(
        X=X, y=y, dates=dates,
        feature_names=_ALL_FEATURE_NAMES,
        label_config=LabelConfig(),
    )

    training_result = MagicMock()
    training_result.model = MagicMock()
    training_result.oos_accuracy = 0.55
    training_result.oos_roc_auc = 0.57
    training_result.feature_importances = {
        n: 1.0 / n_feats for n in _ALL_FEATURE_NAMES
    }

    candidate = ModelCandidate(
        model_id="test_model",
        training_result=training_result,
        label_config=LabelConfig(),
        model_config=MLModelConfig(),
        dataset=dataset,
    )

    return TrainingStageResult(
        candidates=[candidate],
        best_by_auc=candidate,
        best_by_accuracy=candidate,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSelectionStage:
    def test_selects_best(self) -> None:
        entries = _make_backtest_entries()
        bt_result = BacktestStageResult(entries=entries, ranked=entries)
        tr_result = _make_training_result()

        cfg = SelectionConfig(top_n=3, export_pine=False)
        stage = SelectionStage(cfg, print_fn=lambda *a, **k: None)
        best, pine, path = stage.run(bt_result, tr_result, "SPY")

        assert best is not None
        assert best.strategy_name == "Best Strategy"
        assert pine is None  # export_pine=False
        assert path is None

    def test_empty_results(self) -> None:
        bt_result = BacktestStageResult(entries=[], ranked=[])
        tr_result = TrainingStageResult(candidates=[])

        cfg = SelectionConfig()
        stage = SelectionStage(cfg, print_fn=lambda *a, **k: None)
        best, pine, path = stage.run(bt_result, tr_result, "SPY")
        assert best is None

    def test_pine_export(self, tmp_path: Path) -> None:
        entries = _make_backtest_entries()
        bt_result = BacktestStageResult(entries=entries, ranked=entries)
        tr_result = _make_training_result()

        cfg = SelectionConfig(
            top_n=3,
            export_pine=True,
            pine_depth=3,
            pine_top_features=5,
            output_dir=str(tmp_path),
        )
        stage = SelectionStage(cfg, print_fn=lambda *a, **k: None)
        best, pine, path = stage.run(bt_result, tr_result, "SPY")

        assert best is not None
        assert pine is not None
        assert "//@version=6" in pine
        assert "mlProb" in pine
        assert path is not None
        assert Path(path).exists()

    def test_find_candidate_by_model_id(self) -> None:
        tr = _make_training_result()
        entry = BacktestEntry(
            strategy_name="test",
            is_hybrid=True,
            mode="confirmed",
            model_id="test_model",
            result=_mock_backtest_result(),
            composite_score=50.0,
        )
        cand = SelectionStage._find_candidate(entry, tr)
        assert cand is not None
        assert cand.model_id == "test_model"

    def test_find_candidate_fallback(self) -> None:
        tr = _make_training_result()
        entry = BacktestEntry(
            strategy_name="test",
            is_hybrid=False,
            mode="baseline",
            model_id=None,
            result=_mock_backtest_result(),
            composite_score=50.0,
        )
        cand = SelectionStage._find_candidate(entry, tr)
        assert cand is not None  # Falls back to best_by_auc
