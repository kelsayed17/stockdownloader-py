"""Tests for the iron condor overlay backtest engine."""
from __future__ import annotations

import pytest

from stockdownloader.backtesting.engines.iron_condor import (
    ICWeekRecord,
    IronCondorEngine,
)


def _make_ic_week(
    week_num: int,
    spy_open: float,
    spy_close: float,
    short_put_strike: float = 490.0,
    long_put_strike: float = 475.0,
    short_call_strike: float = 510.0,
    long_call_strike: float = 525.0,
    net_credit: float = 3.50,
    ml_prob: float = 0.50,
) -> ICWeekRecord:
    return ICWeekRecord(
        week_num=week_num,
        expiration_date=f"2025-02-{7 + week_num * 7:02d}",
        spy_price_at_entry=spy_open,
        spy_price_at_expiry=spy_close,
        short_put_strike=short_put_strike,
        long_put_strike=long_put_strike,
        short_call_strike=short_call_strike,
        long_call_strike=long_call_strike,
        net_credit_per_contract=net_credit,
        ml_prob=ml_prob,
    )


class TestIronCondorEngine:

    def test_ic_net_credit_collected(self):
        """IC collects net credit when sold."""
        engine = IronCondorEngine(capital=10_000.0)
        # Spread width=15, credit=3.50 -> max_loss=1150 -> n=floor(10000/1150)=8
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["total_credit"] == 3.50 * 8 * 100  # 2800
        assert metrics["n_ics_sold"] == 8

    def test_ic_max_profit_in_range(self):
        """SPY between short strikes -> keep full credit."""
        engine = IronCondorEngine(capital=10_000.0)
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["total_loss"] == 0.0
        assert metrics["final_equity"] > metrics["initial_capital"]

    def test_ic_max_loss_put_side(self):
        """SPY below long put -> max loss on put spread."""
        engine = IronCondorEngine(capital=10_000.0)
        week = _make_ic_week(0, spy_open=500.0, spy_close=460.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Put spread max loss = (490-475)*100*8 = 12000
        assert metrics["total_loss"] == 12_000.0
        assert metrics["n_max_loss_events"] >= 1

    def test_ic_max_loss_call_side(self):
        """SPY above long call -> max loss on call spread."""
        engine = IronCondorEngine(capital=10_000.0)
        week = _make_ic_week(0, spy_open=500.0, spy_close=540.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["total_loss"] == 12_000.0
        assert metrics["n_max_loss_events"] >= 1

    def test_ic_partial_loss_put(self):
        """SPY between long and short put -> partial put loss."""
        engine = IronCondorEngine(capital=10_000.0)
        week = _make_ic_week(0, spy_open=500.0, spy_close=485.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Put loss = (490-485)*100*8 = 4000
        assert metrics["total_loss"] == 4_000.0
        assert metrics["n_max_loss_events"] == 0

    def test_ic_partial_loss_call(self):
        """SPY between short and long call -> partial call loss."""
        engine = IronCondorEngine(capital=10_000.0)
        week = _make_ic_week(0, spy_open=500.0, spy_close=520.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Call loss = (520-510)*100*8 = 8000
        assert metrics["total_loss"] == 8_000.0

    def test_ic_sizing_by_max_loss(self):
        """Position sized by max loss per IC."""
        engine = IronCondorEngine(capital=5_000.0)
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # max_loss_per = 15*100 - 3.5*100 = 1150. n=floor(5000/1150)=4
        assert metrics["n_ics_sold"] == 4

    def test_ic_four_leg_commission(self):
        """Commission charged for 4 legs per IC contract."""
        engine = IronCondorEngine(capital=10_000.0, commission_per_contract=0.65)
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # max_loss = 1150, commission_per_ic = 0.65*4 = 2.60
        # n = floor(10000 / (1150 + 2.60)) = floor(10000/1152.60) = 8
        assert metrics["n_ics_sold"] == 8
        # 8 ICs * 4 legs * 0.65 = 20.80
        assert abs(metrics["total_commissions"] - 20.80) < 0.01

    def test_ic_ml_filter_skip(self):
        """Skips IC when ML signals strong directional move."""
        engine = IronCondorEngine(
            capital=10_000.0,
            skip_put_thresh=0.35,
            skip_call_thresh=0.65,
        )
        week_bear = _make_ic_week(0, spy_open=500.0, spy_close=500.0, ml_prob=0.25)
        engine.process_week(week_bear, use_ml_filter=True)
        metrics = engine.compute_metrics()
        assert metrics["n_ics_skipped"] >= 1
        assert metrics["n_ics_sold"] == 0

    def test_ic_metrics_dict(self):
        """Returns expected keys in metrics dict."""
        engine = IronCondorEngine(capital=10_000.0)
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        expected_keys = {
            "initial_capital", "final_equity", "total_return_pct",
            "total_credit", "total_loss", "total_commissions",
            "n_ics_sold", "n_ics_skipped", "n_max_loss_events",
            "weeks", "sharpe", "max_drawdown_pct", "annualized_return_pct",
        }
        assert expected_keys.issubset(set(metrics.keys()))
