"""Tests for StrategyOptimizer.

Uses a small synthetic dataset to verify the optimizer mechanics
run correctly. Not testing for optimal parameters — just that the
optimizer executes, produces valid results, and doesn't crash.
"""

import io
from decimal import Decimal
from pathlib import Path

import pytest

from stockdownloader.backtest.strategy_optimizer import (
    StrategyOptimizer,
    _run_backtest,
)
from stockdownloader.backtest.optimizer_scoring import score as _score, MIN_TRADES as _MIN_TRADES
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategy
from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategyConfig

# Real data file
_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "spy" / "5m_bars.csv"


@pytest.fixture(scope="module")
def sample_data():
    """Load a small slice of real data for testing (first 2000 bars ~ 25 days)."""
    if not _DATA_FILE.exists():
        pytest.skip("data/spy/5m_bars.csv not found")
    data = IntradayCsvLoader.load_from_file(_DATA_FILE)
    if len(data) < 2000:
        pytest.skip("Not enough data for optimizer tests")
    return data[:2000]


def _make_result_with_trades(
    name: str,
    pnl: Decimal,
    num_trades: int,
    win_pct: float = 0.5,
) -> BacktestResult:
    """Build a BacktestResult with the given number of closed trades."""
    from stockdownloader.model import Trade, Direction

    capital = Decimal("100000")
    r = BacktestResult(name, capital)
    r.final_capital = capital + pnl
    r.equity_curve = [
        capital + pnl * Decimal(str(i)) / Decimal(str(max(num_trades, 1)))
        for i in range(max(num_trades, 1) + 1)
    ]

    wins = int(num_trades * win_pct)
    for i in range(num_trades):
        t = Trade(Direction.LONG, Decimal("500"), Decimal("500"), 100)
        if i < wins:
            t.close("2025-01-01", Decimal("510"))
        else:
            t.close("2025-01-01", Decimal("495"))
        r.add_trade(t)

    return r


class TestScoreFunction:

    def test_score_returns_float(self):
        result = BacktestResult("test", Decimal("100000"))
        result.final_capital = Decimal("105000")
        result.equity_curve = [Decimal("100000"), Decimal("105000")]
        s = _score(result)
        assert isinstance(s, float)

    def test_higher_sharpe_gives_higher_score(self):
        from stockdownloader.model import Trade, Direction

        # Result with moderate performance
        r1 = BacktestResult("a", Decimal("100000"))
        r1.final_capital = Decimal("101000")
        r1.equity_curve = [Decimal("100000")] * 50 + [Decimal("101000")] * 50
        for _ in range(25):
            t = Trade(Direction.LONG, Decimal("500"), Decimal("500"), 100)
            t.close("2025-01-01", Decimal("502"))
            r1.add_trade(t)

        # Result with better performance
        r2 = BacktestResult("b", Decimal("100000"))
        r2.final_capital = Decimal("110000")
        r2.equity_curve = [Decimal(str(100000 + i * 100)) for i in range(100)]
        for _ in range(25):
            t = Trade(Direction.LONG, Decimal("500"), Decimal("500"), 100)
            t.close("2025-01-01", Decimal("510"))
            r2.add_trade(t)

        assert _score(r2) > _score(r1)

    def test_few_trades_penalized(self):
        r1 = BacktestResult("a", Decimal("100000"))
        r1.final_capital = Decimal("101000")
        r1.equity_curve = [Decimal("100000"), Decimal("101000")]
        # No trades = penalty (20 * 1 = 20 pts)
        assert _score(r1) < -10

    def test_min_trades_threshold(self):
        """Results below _MIN_TRADES get penalized."""
        assert _MIN_TRADES == 20

        r_few = _make_result_with_trades("few", Decimal("5000"), 10, win_pct=0.6)
        r_many = _make_result_with_trades("many", Decimal("5000"), 30, win_pct=0.6)

        score_few = _score(r_few)
        score_many = _score(r_many)

        # Same P&L/WR, but 10 < 20 trades -> penalized
        assert score_many > score_few
        # Penalty: (20-10) * 1 = 10 pts
        assert score_many - score_few > 0

    def test_100_trades_beats_50_trades(self):
        """More trades (above threshold) gives bonus, not just avoids penalty."""
        r1 = _make_result_with_trades("50t", Decimal("5000"), 50, win_pct=0.6)
        r2 = _make_result_with_trades("100t", Decimal("5000"), 100, win_pct=0.6)
        assert _score(r2) > _score(r1)

    def test_trading_days_bonus(self):
        """Passing trading_days gives a bonus for good trade frequency."""
        r = _make_result_with_trades("test", Decimal("5000"), 100, win_pct=0.6)
        score_no_days = _score(r, trading_days=0)
        score_with_days = _score(r, trading_days=200)  # 100/200 = 0.5 tpd = ideal
        assert score_with_days > score_no_days


class TestRunBacktest:

    def test_run_returns_result(self, sample_data):
        strategy = PullbackStrategy()
        result = _run_backtest(
            strategy, sample_data,
            Decimal("100000"), Decimal("0.01"),
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "VWAP Pullback"

    def test_run_with_custom_config(self, sample_data):
        config = PullbackStrategyConfig(rr=Decimal("2.0"), sl_atr=Decimal("1.0"))
        strategy = PullbackStrategy(config=config)
        result = _run_backtest(
            strategy, sample_data,
            Decimal("100000"), Decimal("0.01"),
        )
        assert isinstance(result, BacktestResult)


class TestOptimizer:

    def test_optimizer_init(self, sample_data):
        opt = StrategyOptimizer(sample_data, verbose=False)
        assert opt._run_count == 0
        assert opt._best_result is None
        assert opt._trading_days > 0

    def test_full_optimize_returns_list(self, sample_data):
        """Smoke test: run full optimize on tiny dataset."""
        opt = StrategyOptimizer(sample_data, verbose=False)
        results = opt.optimize()
        assert isinstance(results, list)
        assert len(results) > 0
        # Each entry is (name, result, score)
        for name, result, score in results:
            assert isinstance(name, str)
            assert isinstance(result, BacktestResult)
            assert isinstance(score, float)
        assert opt._run_count > 5  # should have tested many configs

    def test_log_file_receives_output(self, sample_data):
        """Verify that log_file gets output when verbose=True."""
        log_buf = io.StringIO()
        opt = StrategyOptimizer(sample_data, verbose=True, log_file=log_buf)
        # Run a quick baseline for one strategy
        strategy = PullbackStrategy()
        _run_backtest(strategy, sample_data, Decimal("100000"), Decimal("0.01"))
        opt._print("log-test-output")
        log_content = log_buf.getvalue()
        assert "log-test-output" in log_content

    def test_log_file_not_used_when_none(self, sample_data):
        """Verify no crash when log_file is None."""
        opt = StrategyOptimizer(sample_data, verbose=True, log_file=None)
        opt._print("no-log-test")
        # No exception = pass
