"""Tests for the weekly wheel backtest engine."""
from __future__ import annotations

import pytest

from stockdownloader.backtesting.engines.wheel import (
    WheelBacktestEngine,
    WheelState,
    WeekRecord,
)


def _make_week(
    week_num: int,
    spy_open: float,
    spy_close: float,
    put_premium: float = 2.0,
    call_premium: float = 2.0,
    put_strike: float = 580.0,
    call_strike: float = 620.0,
    ml_prob: float = 0.50,
) -> WeekRecord:
    return WeekRecord(
        week_num=week_num,
        expiration_date=f"2025-02-{7 + week_num * 7:02d}",
        entry_date=f"2025-02-{3 + week_num * 7:02d}",
        spy_price_at_entry=spy_open,
        spy_price_at_expiry=spy_close,
        put_strike=put_strike,
        call_strike=call_strike,
        put_premium=put_premium,
        call_premium=call_premium,
        ml_prob=ml_prob,
    )


class TestWheelStateTransitions:

    def test_initial_state_is_cash(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        assert engine.state == WheelState.CASH

    def test_first_week_sells_put(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        week = _make_week(0, spy_open=600.0, spy_close=605.0)
        engine.process_week(week)
        assert engine.state == WheelState.PUT_PHASE
        assert engine.total_premium_collected > 0

    def test_put_assignment_transitions_to_call_phase(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        week = _make_week(0, spy_open=600.0, spy_close=570.0, put_strike=580.0, put_premium=3.0)
        engine.process_week(week)
        assert engine.state in (WheelState.HOLDING, WheelState.CALL_PHASE)
        assert engine.shares_held == 100

    def test_call_exercise_transitions_to_put_phase(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        engine._state = WheelState.CALL_PHASE
        engine._shares = 100
        engine._share_cost_basis = 580.0
        week = _make_week(0, spy_open=600.0, spy_close=625.0, call_strike=620.0, call_premium=2.0)
        engine.process_week(week)
        assert engine.state == WheelState.PUT_PHASE
        assert engine.shares_held == 0

    def test_otm_call_keeps_premium(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        engine._state = WheelState.CALL_PHASE
        engine._shares = 100
        engine._share_cost_basis = 580.0
        week = _make_week(0, spy_open=600.0, spy_close=610.0, call_strike=620.0, call_premium=2.0)
        engine.process_week(week)
        assert engine.state == WheelState.CALL_PHASE
        assert engine.shares_held == 100

    def test_ml_filter_skips_put(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1, skip_put_thresh=0.35)
        week = _make_week(0, spy_open=600.0, spy_close=605.0, ml_prob=0.30)
        engine.process_week(week, use_ml_filter=True)
        assert engine.state == WheelState.CASH
        assert engine.total_premium_collected == 0

    def test_ml_filter_skips_call(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1, skip_call_thresh=0.65)
        engine._state = WheelState.CALL_PHASE
        engine._shares = 100
        engine._share_cost_basis = 580.0
        week = _make_week(0, spy_open=600.0, spy_close=605.0, ml_prob=0.70)
        engine.process_week(week, use_ml_filter=True)
        assert engine.shares_held == 100


class TestBuyWriteMode:

    def test_buy_write_buys_shares_immediately(self):
        """Buy-write mode buys shares on week 1."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, buy_write=True,
        )
        week = _make_week(0, spy_open=500.0, spy_close=505.0)
        engine.process_week(week)
        assert engine.shares_held == 100
        assert engine.state == WheelState.CALL_PHASE

    def test_buy_write_collects_premium(self):
        """Buy-write collects call premium each week."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, buy_write=True,
        )
        week = _make_week(0, spy_open=500.0, spy_close=505.0, call_premium=3.0)
        engine.process_week(week)
        assert engine.total_premium_collected == 300.0  # 3.0 * 100

    def test_buy_write_otm_keeps_shares_and_premium(self):
        """OTM call: keep shares + premium, stay in CALL_PHASE."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, buy_write=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=510.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        assert engine.shares_held == 100
        assert engine.state == WheelState.CALL_PHASE
        metrics = engine.compute_metrics()
        assert metrics["n_calls_exercised"] == 0
        assert metrics["n_rebuys"] == 0

    def test_buy_write_itm_rebuys_immediately(self):
        """ITM call: called away + immediately re-buy at expiry price."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, buy_write=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=530.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        # Still holding shares after re-buy
        assert engine.shares_held == 100
        assert engine.state == WheelState.CALL_PHASE
        metrics = engine.compute_metrics()
        assert metrics["n_calls_exercised"] == 1
        assert metrics["n_rebuys"] == 1

    def test_buy_write_itm_cash_impact(self):
        """ITM call: verify cash accounting for called-away + re-buy."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, buy_write=True,
        )
        # Buy at 500, call strike 520, expires at 530
        week = _make_week(
            0, spy_open=500.0, spy_close=530.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        # Cash: 100000 - 50000 (buy) + 200 (prem) + 52000 (called) - 53000 (rebuy) = 49200
        expected_cash = 100_000 - 500 * 100 + 2.0 * 100 + 520 * 100 - 530 * 100
        assert abs(engine._cash - expected_cash) < 0.01

    def test_buy_write_ml_skips_call_on_rally(self):
        """ML filter skips selling call when prob > threshold (rally expected)."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            skip_call_thresh=0.65, buy_write=True,
        )
        week = _make_week(0, spy_open=500.0, spy_close=510.0, ml_prob=0.70)
        engine.process_week(week, use_ml_filter=True)
        assert engine.total_premium_collected == 0
        assert engine.shares_held == 100
        metrics = engine.compute_metrics()
        assert metrics["n_calls_skipped"] == 1

    def test_buy_write_equity_tracks_stock_plus_premium(self):
        """Buy-write equity should reflect stock appreciation + premium income."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, buy_write=True,
        )
        weeks = [
            _make_week(0, spy_open=500.0, spy_close=510.0, call_strike=520.0, call_premium=2.0),
            _make_week(1, spy_open=510.0, spy_close=520.0, call_strike=530.0, call_premium=2.0),
            _make_week(2, spy_open=520.0, spy_close=525.0, call_strike=535.0, call_premium=2.0),
        ]
        for w in weeks:
            engine.process_week(w)

        metrics = engine.compute_metrics()
        # Should be profitable (stock went 500->525 = +5%, plus premium income)
        assert metrics["total_return_pct"] > 0
        assert metrics["total_premium_collected"] == 600.0  # 3 weeks * 2.0 * 100
        assert metrics["n_calls_sold"] == 3

    def test_buy_write_never_sells_puts(self):
        """Buy-write mode never enters put phase or sells puts."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, buy_write=True,
        )
        weeks = [
            _make_week(i, spy_open=500.0, spy_close=505.0, call_strike=520.0, call_premium=2.0)
            for i in range(5)
        ]
        for w in weeks:
            engine.process_week(w)
        metrics = engine.compute_metrics()
        assert metrics["n_puts_sold"] == 0
        assert metrics["n_assignments"] == 0

    def test_buy_write_returns_n_rebuys_in_metrics(self):
        """Metrics include n_rebuys key."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, buy_write=True,
        )
        week = _make_week(0, spy_open=500.0, spy_close=505.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert "n_rebuys" in metrics


class TestWheelMetrics:

    def test_returns_metrics_dict(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        weeks = [_make_week(i, spy_open=600.0, spy_close=605.0) for i in range(4)]
        for w in weeks:
            engine.process_week(w)
        metrics = engine.compute_metrics()
        assert "total_return_pct" in metrics
        assert "total_premium_collected" in metrics
        assert "n_assignments" in metrics
        assert "n_calls_exercised" in metrics
        assert "sharpe" in metrics
        assert "max_drawdown_pct" in metrics

    def test_equity_curve_tracks_weekly(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        weeks = [_make_week(i, spy_open=600.0, spy_close=605.0) for i in range(4)]
        for w in weeks:
            engine.process_week(w)
        assert len(engine.equity_curve) == 4


class TestTransactionCosts:

    def test_commission_deducted_from_premium(self):
        """Commission reduces net premium collected."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            commission_per_contract=0.65,
        )
        week = _make_week(0, spy_open=600.0, spy_close=605.0, put_premium=3.0)
        engine.process_week(week)
        # Gross premium: 3.0 * 100 = 300. Commission: 0.65 * 1 = 0.65
        # Net premium: 299.35
        assert abs(engine.total_premium_collected - 299.35) < 0.01

    def test_zero_commission_matches_original(self):
        """With commission=0, behavior is identical to original engine."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            commission_per_contract=0.0,
        )
        week = _make_week(0, spy_open=600.0, spy_close=605.0, put_premium=3.0)
        engine.process_week(week)
        assert engine.total_premium_collected == 300.0

    def test_total_commissions_tracked_in_metrics(self):
        """Metrics dict includes total_commissions key."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            commission_per_contract=0.65,
        )
        weeks = [_make_week(i, spy_open=600.0, spy_close=605.0) for i in range(3)]
        for w in weeks:
            engine.process_week(w)
        metrics = engine.compute_metrics()
        assert "total_commissions" in metrics
        # 3 puts sold * 1 contract * 0.65 = 1.95
        assert abs(metrics["total_commissions"] - 1.95) < 0.01
