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
