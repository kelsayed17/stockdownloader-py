"""Tests for Monte Carlo robustness testing in tournament engine."""
from __future__ import annotations

import pytest

from stockdownloader.backtest.tournament_engine import (
    ComboKey,
    MonteCarloPercentiles,
    MonteCarloResult,
    _compute_percentiles,
    _equity_max_drawdown,
    run_monte_carlo,
)


class TestComputePercentiles:
    def test_ordered_list(self):
        values = list(range(100))
        p = _compute_percentiles(values)
        assert p.p5 <= p.p25 <= p.p50 <= p.p75 <= p.p95

    def test_single_value(self):
        p = _compute_percentiles([42.0])
        assert p.p5 == p.p50 == p.p95 == 42.0

    def test_empty_list(self):
        p = _compute_percentiles([])
        assert p.p50 == 0.0

    def test_two_values(self):
        p = _compute_percentiles([10.0, 20.0])
        assert p.p50 == pytest.approx(15.0)
        assert p.p5 <= p.p95


class TestEquityMaxDrawdown:
    def test_no_drawdown(self):
        equity = [100.0, 110.0, 120.0, 130.0]
        assert _equity_max_drawdown(equity) == 0.0

    def test_drawdown(self):
        equity = [100.0, 110.0, 95.0, 105.0]
        # Peak 110, trough 95 -> DD = 15/110 = 13.64%
        dd = _equity_max_drawdown(equity)
        assert dd == pytest.approx(13.636, abs=0.01)

    def test_single_point(self):
        assert _equity_max_drawdown([100.0]) == 0.0

    def test_empty(self):
        assert _equity_max_drawdown([]) == 0.0


class TestRunMonteCarlo:
    def test_all_winning_trades(self):
        """All trades profitable -> robust, no penalty."""
        key = ComboKey("rsi", "5m")
        pnls = [100.0, 200.0, 150.0, 300.0, 250.0,
                180.0, 120.0, 90.0, 160.0, 210.0]
        _, mc, elapsed, error = run_monte_carlo(key, pnls, 100000.0, 500)

        assert error is None
        assert mc is not None
        assert mc.is_robust is True
        assert mc.mc_penalty == 0.0
        assert mc.bootstrap_return.p5 > 0

    def test_all_losing_trades(self):
        """All trades negative -> not robust, has penalty."""
        key = ComboKey("macd", "5m")
        pnls = [-100.0, -200.0, -150.0, -300.0, -250.0,
                -180.0, -120.0, -90.0, -160.0, -210.0]
        _, mc, elapsed, error = run_monte_carlo(key, pnls, 100000.0, 500)

        assert error is None
        assert mc is not None
        assert mc.is_robust is False
        assert mc.mc_penalty > 0
        assert mc.bootstrap_return.p5 < 0

    def test_mixed_trades(self):
        """Realistic mix of winning/losing trades."""
        key = ComboKey("smc", "15m")
        pnls = [100.0, -80.0, 150.0, -60.0, 200.0,
                -90.0, 120.0, -70.0, 180.0, -50.0,
                130.0, -85.0, 140.0, -65.0, 160.0]
        _, mc, elapsed, error = run_monte_carlo(key, pnls, 100000.0, 500)

        assert error is None
        assert mc is not None
        assert mc.n_trades == 15
        assert mc.n_simulations == 500
        # All percentiles should be finite
        assert all(abs(v) < 1e10 for v in [
            mc.max_drawdown.p5, mc.max_drawdown.p95,
            mc.total_return.p5, mc.total_return.p95,
            mc.bootstrap_return.p5, mc.bootstrap_return.p95,
        ])

    def test_percentile_ordering(self):
        """All percentiles maintain p5 <= p25 <= p50 <= p75 <= p95."""
        key = ComboKey("rsi", "5m")
        pnls = [100.0, -80.0, 150.0, -60.0, 200.0,
                -90.0, 120.0, -70.0, 180.0, -50.0]
        _, mc, _, _ = run_monte_carlo(key, pnls, 100000.0, 500)

        assert mc is not None
        for pctl in [mc.max_drawdown, mc.total_return, mc.final_equity,
                     mc.bootstrap_return, mc.bootstrap_sharpe]:
            assert pctl.p5 <= pctl.p25 <= pctl.p50 <= pctl.p75 <= pctl.p95

    def test_reproducibility(self):
        """Same inputs produce identical results (seeded RNG)."""
        key = ComboKey("rsi", "5m")
        pnls = [100.0, -80.0, 150.0, -60.0, 200.0,
                -90.0, 120.0, -70.0, 180.0, -50.0]

        _, mc1, _, _ = run_monte_carlo(key, pnls, 100000.0, 500)
        _, mc2, _, _ = run_monte_carlo(key, pnls, 100000.0, 500)

        assert mc1 is not None and mc2 is not None
        assert mc1.total_return.p50 == mc2.total_return.p50
        assert mc1.bootstrap_return.p5 == mc2.bootstrap_return.p5
        assert mc1.max_drawdown.p95 == mc2.max_drawdown.p95

    def test_single_trade_returns_none(self):
        """Single trade is insufficient for MC — returns None."""
        key = ComboKey("rsi", "5m")
        _, mc, _, error = run_monte_carlo(key, [100.0], 100000.0, 500)
        assert error is None
        assert mc is None

    def test_empty_returns_none(self):
        """Empty trade list returns None."""
        key = ComboKey("rsi", "5m")
        _, mc, _, error = run_monte_carlo(key, [], 100000.0, 500)
        assert error is None
        assert mc is None

    def test_shuffle_preserves_total_pnl(self):
        """Shuffling doesn't change total P&L — all shuffles sum to same total."""
        key = ComboKey("rsi", "5m")
        pnls = [100.0, -80.0, 150.0, -60.0, 200.0]
        expected_total = sum(pnls)
        expected_return = expected_total / 100000.0 * 100.0

        _, mc, _, _ = run_monte_carlo(key, pnls, 100000.0, 500)
        assert mc is not None
        # All shuffled returns should be the same (same P&Ls, just reordered)
        assert mc.total_return.p5 == pytest.approx(expected_return)
        assert mc.total_return.p95 == pytest.approx(expected_return)

    def test_penalty_zero_when_robust(self):
        """No penalty when strategy is robust."""
        key = ComboKey("rsi", "5m")
        # Strongly profitable trades
        pnls = [500.0, 400.0, 300.0, 200.0, 100.0,
                -50.0, -40.0, -30.0, -20.0, -10.0]
        _, mc, _, _ = run_monte_carlo(key, pnls, 100000.0, 500)

        assert mc is not None
        assert mc.is_robust is True
        assert mc.mc_penalty == 0.0

    def test_penalty_increases_with_fragility(self):
        """More negative bootstrap p5 -> larger penalty."""
        key1 = ComboKey("s1", "5m")
        # Slightly negative
        pnls1 = [50.0, -60.0, 40.0, -55.0, 30.0,
                 -45.0, 35.0, -50.0, 25.0, -40.0]
        _, mc1, _, _ = run_monte_carlo(key1, pnls1, 100000.0, 500)

        key2 = ComboKey("s2", "5m")
        # Very negative
        pnls2 = [-200.0, -300.0, 50.0, -250.0, -180.0,
                 -220.0, 30.0, -280.0, -190.0, -260.0]
        _, mc2, _, _ = run_monte_carlo(key2, pnls2, 100000.0, 500)

        assert mc1 is not None and mc2 is not None
        assert mc2.mc_penalty > mc1.mc_penalty
