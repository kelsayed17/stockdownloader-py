"""Tests for DailyStrategyOptimizer.

Uses a small synthetic dataset to verify the optimizer mechanics
run correctly.  Not testing for optimal parameters — just that the
optimizer executes, produces valid results, and doesn't crash.
"""

import io
from decimal import Decimal
from pathlib import Path

import pytest

from stockdownloader.strategy.registration_loader import ensure_registered
ensure_registered()

from stockdownloader.backtest.daily_strategy_optimizer import (
    DailyStrategyOptimizer,
    _ADAPTER_PARAM_SPACE,
)
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.optimizer_scoring import score as _score
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.strategy.base_registry import StrategyRegistry

# Real data file
_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "SPY" / "5m_bars.csv"


@pytest.fixture(scope="module")
def sample_data():
    """Load a small slice of real data for testing (first 2000 bars ≈ 25 days)."""
    if not _DATA_FILE.exists():
        pytest.skip("data/SPY/5m_bars.csv not found")
    data = IntradayCsvLoader.load_from_file(_DATA_FILE)
    if len(data) < 2000:
        pytest.skip("Not enough data for optimizer tests")
    return data[:2000]


class TestDailyStrategyOptimizerInit:

    def test_init_with_valid_strategy(self, sample_data):
        opt = DailyStrategyOptimizer("rsi", sample_data, verbose=False)
        assert opt._entry.name == "rsi"
        assert opt._run_count == 0
        assert opt._best_result is None

    def test_init_rejects_intraday(self, sample_data):
        with pytest.raises(ValueError, match="expected 'daily'"):
            DailyStrategyOptimizer("vwap-pullback", sample_data, verbose=False)

    def test_init_rejects_unknown(self, sample_data):
        with pytest.raises(ValueError, match="Unknown strategy"):
            DailyStrategyOptimizer("nonexistent", sample_data, verbose=False)

    def test_trading_days_computed(self, sample_data):
        opt = DailyStrategyOptimizer("rsi", sample_data, verbose=False)
        assert opt._trading_days > 0

    def test_default_adapter_kwargs(self, sample_data):
        opt = DailyStrategyOptimizer("rsi", sample_data, verbose=False)
        assert opt._best_adapter_kwargs["sl_atr_mult"] == Decimal("1.5")
        assert opt._best_adapter_kwargs["rr"] == Decimal("1.5")
        assert opt._best_adapter_kwargs["sl_cap"] == Decimal("2.00")

    def test_custom_adapter_kwargs(self, sample_data):
        custom = {"sl_atr_mult": Decimal("1.0"), "rr": Decimal("2.0"), "sl_cap": Decimal("1.50")}
        opt = DailyStrategyOptimizer("rsi", sample_data, adapter_kwargs=custom, verbose=False)
        assert opt._best_adapter_kwargs == custom


class TestBuildAndRun:

    def test_valid_combo_returns_result(self, sample_data):
        opt = DailyStrategyOptimizer("rsi", sample_data, verbose=False)
        result = opt._build_and_run(
            {"period": 14, "oversold": 30.0, "overbought": 70.0},
            {"sl_atr_mult": Decimal("1.5"), "rr": Decimal("1.5"), "sl_cap": Decimal("2.00")},
        )
        assert isinstance(result, BacktestResult)

    def test_invalid_combo_returns_none(self, sample_data):
        opt = DailyStrategyOptimizer("sma", sample_data, verbose=False)
        # SMA with short > long should raise ValueError in constructor
        result = opt._build_and_run(
            {"short_period": 200, "long_period": 5},
            {"sl_atr_mult": Decimal("1.5"), "rr": Decimal("1.5"), "sl_cap": Decimal("2.00")},
        )
        assert result is None

    def test_all_daily_strategies_baseline(self, sample_data):
        """Verify baseline run works for all registered daily strategies."""
        for entry in StrategyRegistry.all_entries(category="daily"):
            opt = DailyStrategyOptimizer(entry.name, sample_data, verbose=False)
            result = opt._build_and_run(
                opt._best_strategy_kwargs,
                opt._best_adapter_kwargs,
            )
            assert result is not None, f"{entry.name} baseline returned None"
            assert isinstance(result, BacktestResult)


class TestOptimize:

    def test_optimize_returns_tuple(self, sample_data):
        """Smoke test: full optimize on small dataset."""
        opt = DailyStrategyOptimizer("rsi", sample_data, verbose=False)
        kwargs, result = opt.optimize()
        assert isinstance(kwargs, dict)
        assert isinstance(result, BacktestResult)
        assert opt._run_count > 5  # should have tested multiple configs

    def test_optimize_result_has_strategy_name(self, sample_data):
        opt = DailyStrategyOptimizer("rsi", sample_data, verbose=False)
        _, result = opt.optimize()
        assert "RSI" in result.strategy_name

    def test_optimize_sma(self, sample_data):
        """Smoke test for SMA strategy optimization."""
        opt = DailyStrategyOptimizer("sma", sample_data, verbose=False)
        kwargs, result = opt.optimize()
        assert isinstance(result, BacktestResult)
        assert "SMA" in result.strategy_name

    def test_log_file_receives_output(self, sample_data):
        log_buf = io.StringIO()
        opt = DailyStrategyOptimizer("rsi", sample_data, verbose=True, log_file=log_buf)
        opt.optimize()
        log_content = log_buf.getvalue()
        assert "DAILY STRATEGY OPTIMIZER" in log_content
        assert "OPTIMIZATION COMPLETE" in log_content

    def test_log_file_not_used_when_none(self, sample_data):
        """Verify no crash when log_file is None."""
        opt = DailyStrategyOptimizer("rsi", sample_data, verbose=False, log_file=None)
        opt.optimize()
        # No exception = pass


class TestAdapterParamSpace:

    def test_adapter_space_has_expected_keys(self):
        assert "sl_atr_mult" in _ADAPTER_PARAM_SPACE
        assert "rr" in _ADAPTER_PARAM_SPACE
        assert "sl_cap" in _ADAPTER_PARAM_SPACE
        assert "allow_shorts" in _ADAPTER_PARAM_SPACE

    def test_adapter_space_values_are_lists(self):
        for key, values in _ADAPTER_PARAM_SPACE.items():
            assert isinstance(values, list), f"{key} should be a list"
            assert len(values) >= 2, f"{key} should have at least 2 values"
