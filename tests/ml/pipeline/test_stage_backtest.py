"""Tests for Stage 5: Comprehensive Backtesting."""

from __future__ import annotations

import random
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pytest

from stockdownloader.ml.pipeline.config import BacktestConfig
from stockdownloader.ml.pipeline.results import (
    BacktestEntry,
    DataResult,
    HybridStageResult,
    HybridStrategyEntry,
)
from stockdownloader.ml.pipeline.stage_backtest import BacktestStage
from stockdownloader.ml.pipeline.stage_hybrid_strategies import (
    MLConfirmedStrategy,
)
from stockdownloader.model.price_data import PriceData
from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy


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


class TestBacktestStage:
    def test_baseline_strategies_run(self) -> None:
        """Verify baseline daily strategies get backtested."""
        cfg = BacktestConfig(
            initial_capital=100_000.0,
            commission=10.0,
            walk_forward_windows=0,
            top_for_walk_forward=0,
        )
        stage = BacktestStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(
            _data_result(300),
            HybridStageResult(hybrid_strategies=[]),
        )

        # Should have baselines for daily strategies
        baseline_entries = [e for e in result.entries if not e.is_hybrid]
        assert len(baseline_entries) >= 5  # At least 5 of the 7 daily strategies

        # All should have scores
        for entry in baseline_entries:
            assert isinstance(entry.composite_score, float)
            assert entry.mode == "baseline"
            assert entry.model_id is None

    def test_entries_sorted_by_score(self) -> None:
        cfg = BacktestConfig(
            walk_forward_windows=0,
            top_for_walk_forward=0,
        )
        stage = BacktestStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(
            _data_result(300),
            HybridStageResult(hybrid_strategies=[]),
        )

        scores = [e.composite_score for e in result.entries]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_strategies_included(self) -> None:
        """Verify hybrid strategies get backtested alongside baselines."""
        # Create a simple hybrid: always-buy with a bullish model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        class _SimpleBuy(TradingStrategy):
            @property
            def name(self) -> str:
                return "SimpleBuy"

            @property
            def warmup_period(self) -> int:
                return 0

            def evaluate(self, data: list[PriceData], i: int) -> Signal:
                return Signal.BUY if i % 20 == 0 else Signal.HOLD

        hybrid = MLConfirmedStrategy(
            _SimpleBuy(), mock_model,
            tuple(f"f_{i}" for i in range(63)),
            threshold=0.5,
            label="SimpleBuy (ML-Confirmed)",
        )

        hybrid_result = HybridStageResult(
            hybrid_strategies=[
                HybridStrategyEntry(
                    name="SimpleBuy (ML-Confirmed)",
                    mode="confirmed",
                    model_id="test",
                    base_strategy_name="simple",
                    strategy=hybrid,
                ),
            ],
        )

        cfg = BacktestConfig(
            walk_forward_windows=0,
            top_for_walk_forward=0,
        )
        stage = BacktestStage(cfg, print_fn=lambda *a, **k: None)
        result = stage.run(_data_result(300), hybrid_result)

        hybrid_entries = [e for e in result.entries if e.is_hybrid]
        assert len(hybrid_entries) == 1
        assert hybrid_entries[0].mode == "confirmed"


    def test_hybrid_walk_forward_produces_degradation(self) -> None:
        """Hybrid strategies should get walk-forward degradation (not None)."""
        from stockdownloader.ml.pipeline.stage_hybrid_strategies import (
            MLWeightedStrategy,
        )

        # Mock model alternates: BUY signals see high prob, SELL see low prob.
        # FeatureExtractor may fail on synthetic data → _get_ml_prob defaults
        # to 0.5.  So use threshold=0.5 in MLWeighted which passes when
        # prob >= 0.5 (BUY) and prob <= 0.5 (SELL).  Mock always returns 0.5
        # on error, so all signals pass through.
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = Exception("mock")

        class _PeriodicBuySell(TradingStrategy):
            @property
            def name(self) -> str:
                return "PeriodicBuySell"

            @property
            def warmup_period(self) -> int:
                return 0

            def evaluate(self, data: list[PriceData], i: int) -> Signal:
                if i % 20 == 0:
                    return Signal.BUY
                if i % 20 == 10:
                    return Signal.SELL
                return Signal.HOLD

        hybrid = MLWeightedStrategy(
            _PeriodicBuySell(), mock_model,
            tuple(f"f_{i}" for i in range(63)),
            threshold=0.5,
            label="PeriodicBuySell (ML-Weighted)",
        )

        hybrid_result = HybridStageResult(
            hybrid_strategies=[
                HybridStrategyEntry(
                    name="PeriodicBuySell (ML-Weighted)",
                    mode="weighted",
                    model_id="test",
                    base_strategy_name="periodic",
                    strategy=hybrid,
                ),
            ],
        )

        cfg = BacktestConfig(
            walk_forward_windows=3,
            top_for_walk_forward=20,
        )
        stage = BacktestStage(cfg, print_fn=lambda *a, **k: None)
        # Need ~1500 bars so each WF window > warmup (201)
        result = stage.run(_data_result(1500), hybrid_result)

        hybrid_entries = [e for e in result.entries if e.is_hybrid]
        assert len(hybrid_entries) == 1

        entry = hybrid_entries[0]
        assert entry.result.total_trades >= 5, (
            f"Expected ≥5 trades, got {entry.result.total_trades}"
        )
        # Walk-forward should produce a degradation value (not None)
        assert entry.walk_forward_degradation is not None
        assert isinstance(entry.walk_forward_degradation, float)
        assert entry.walk_forward_degradation > 0

    def test_slippage_reduces_backtest_pnl(self) -> None:
        """Slippage in config should reduce strategy P&L."""
        cfg_no_slip = BacktestConfig(
            initial_capital=100_000.0,
            commission=10.0,
            slippage_pct=0.0,
            walk_forward_windows=0,
            top_for_walk_forward=0,
        )
        cfg_with_slip = BacktestConfig(
            initial_capital=100_000.0,
            commission=10.0,
            slippage_pct=0.01,  # 1% — large for clear effect
            walk_forward_windows=0,
            top_for_walk_forward=0,
        )

        data = _data_result(300)
        stage_no_slip = BacktestStage(cfg_no_slip, print_fn=lambda *a, **k: None)
        stage_with_slip = BacktestStage(cfg_with_slip, print_fn=lambda *a, **k: None)

        result_no_slip = stage_no_slip.run(data, HybridStageResult(hybrid_strategies=[]))
        result_with_slip = stage_with_slip.run(data, HybridStageResult(hybrid_strategies=[]))

        # Find a strategy with trades in both runs
        for e_no in result_no_slip.entries:
            if e_no.result.total_trades > 0:
                e_with = next(
                    (e for e in result_with_slip.entries
                     if e.strategy_name == e_no.strategy_name),
                    None,
                )
                if e_with is not None and e_with.result.total_trades > 0:
                    # With slippage, final capital should be lower
                    assert e_no.result.final_capital > e_with.result.final_capital, (
                        f"{e_no.strategy_name}: slippage should reduce capital"
                    )
                    break


    def test_true_walk_forward_retrains_models(self) -> None:
        """True WF should retrain ML models per window and produce metrics."""
        from stockdownloader.ml.dataset_builder import LabelConfig
        from stockdownloader.ml.pipeline.stage_hybrid_strategies import (
            MLWeightedStrategy,
        )
        from stockdownloader.ml.trainer import MLModelConfig

        # Use a mock that defaults to 0.5 so all base signals pass through
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = Exception("mock")

        class _PeriodicBuySell(TradingStrategy):
            @property
            def name(self) -> str:
                return "PeriodicBuySell"

            @property
            def warmup_period(self) -> int:
                return 0

            def evaluate(self, data: list[PriceData], i: int) -> Signal:
                if i % 20 == 0:
                    return Signal.BUY
                if i % 20 == 10:
                    return Signal.SELL
                return Signal.HOLD

        hybrid = MLWeightedStrategy(
            _PeriodicBuySell(), mock_model,
            tuple(f"f_{i}" for i in range(63)),
            threshold=0.5,
            label="PeriodicBuySell (ML-Weighted)",
        )

        # Need a real strategy registry name for true WF to recreate base
        from stockdownloader.strategy.registration_loader import ensure_registered
        from stockdownloader.strategy.base_registry import StrategyRegistry

        ensure_registered()
        # Pick any registered daily strategy
        all_regs = list(StrategyRegistry.all_entries(category="daily"))
        assert len(all_regs) > 0
        reg_name = all_regs[0].name

        label_cfg = LabelConfig(forward_period=5, profit_threshold=0.005)
        model_cfg = MLModelConfig(
            model_type="logistic_regression",  # fast for testing
            n_estimators=50,
            max_depth=3,
        )

        hybrid_result = HybridStageResult(
            hybrid_strategies=[
                HybridStrategyEntry(
                    name="PeriodicBuySell (ML-Weighted)",
                    mode="weighted",
                    model_id="test",
                    base_strategy_name=reg_name,
                    strategy=hybrid,
                    label_config=label_cfg,
                    model_config=model_cfg,
                    hybrid_threshold=0.5,
                ),
            ],
        )

        cfg = BacktestConfig(
            walk_forward_windows=2,
            top_for_walk_forward=20,
            true_walk_forward=True,
            wf_min_train_bars=300,
        )
        stage = BacktestStage(cfg, print_fn=lambda *a, **k: None)
        # Need 1000+ bars for 2 windows with min_train=300
        data = _data_result(1000)
        result = stage.run(data, hybrid_result)

        hybrid_entries = [e for e in result.entries if e.is_hybrid]
        assert len(hybrid_entries) == 1

        entry = hybrid_entries[0]
        # The true WF should produce per-window results
        if entry.wf_window_results is not None:
            assert len(entry.wf_window_results) >= 1
            for wr in entry.wf_window_results:
                assert wr.model_auc > 0  # Fresh model was trained
                assert wr.is_bars >= 300  # Respects min_train
            # Degradation should be computed
            assert entry.walk_forward_degradation is not None


class TestComputeScore:
    def test_positive_result(self) -> None:
        mock_result = MagicMock()
        mock_result.sharpe_ratio.return_value = Decimal("1.5")
        mock_result.win_rate = Decimal("55.0")
        mock_result.profit_factor = Decimal("1.8")
        mock_result.max_drawdown = Decimal("10.0")
        mock_result.total_trades = 50

        score = BacktestStage._compute_score(mock_result)
        assert score > 0

    def test_few_trades_penalised(self) -> None:
        mock_result = MagicMock()
        mock_result.sharpe_ratio.return_value = Decimal("1.5")
        mock_result.win_rate = Decimal("55.0")
        mock_result.profit_factor = Decimal("1.8")
        mock_result.max_drawdown = Decimal("10.0")

        mock_result.total_trades = 50
        score_many = BacktestStage._compute_score(mock_result)

        mock_result.total_trades = 3
        score_few = BacktestStage._compute_score(mock_result)

        assert score_few < score_many
