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
    hedge_put_strike: float = 0.0,
    hedge_put_premium: float = 0.0,
    iv_percentile: float = 0.50,
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
        hedge_put_strike=hedge_put_strike,
        hedge_put_premium=hedge_put_premium,
        iv_percentile=iv_percentile,
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


class TestCombinedMode:

    def test_combined_buys_shares_immediately(self):
        """Combined mode buys shares on week 1."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(0, spy_open=500.0, spy_close=505.0)
        engine.process_week(week)
        assert engine.shares_held >= 100

    def test_combined_sells_both_cc_and_csp(self):
        """Combined mode collects premium from both calls and puts."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_sold"] >= 1
        assert metrics["n_puts_sold"] >= 1
        # Premium from both sides: at least 200 + 200 = 400
        assert engine.total_premium_collected >= 400.0

    def test_combined_cc_exercise_rebuys(self):
        """CC exercise in combined mode: called away + immediate re-buy."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=530.0,
            put_strike=480.0, put_premium=1.0,
            call_strike=520.0, call_premium=3.0,
        )
        engine.process_week(week)
        assert engine.shares_held >= 100  # Still holding after re-buy
        metrics = engine.compute_metrics()
        assert metrics["n_calls_exercised"] >= 1
        assert metrics["n_rebuys"] >= 1

    def test_combined_csp_assignment_grows_position(self):
        """CSP assignment in combined mode adds shares."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        # SPY at 500 -> buy 100 shares ($50K). Cash left ~$50K.
        # Put strike 480 -> CSP collateral $48K. Cash supports 1 CSP.
        # SPY drops to 470 -> put ITM, assigned -> acquire 100 more shares.
        week = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        assert engine.shares_held == 200  # 100 initial + 100 from CSP assignment
        metrics = engine.compute_metrics()
        assert metrics["n_assignments"] >= 1

    def test_combined_both_itm_settles_correctly(self):
        """Both CC and CSP ITM in same week: exercises cancel out."""
        engine = WheelBacktestEngine(
            initial_capital=110_000.0, contracts=1, combined=True,
        )
        # SPY at 500, put strike 510 (ITM), call strike 490 (ITM)
        # Capital $110K -> buy 100 shares at 500 ($50K) -> cash $60K
        # CC exercise: +$49K (strike) - $50.5K (re-buy) + $1.2K (prem) -> cash ~$59.7K
        # CSP collateral at 510: $51K fits in $59.7K -> 1 CSP sold, assigned
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=510.0, put_premium=12.0,
            call_strike=490.0, call_premium=12.0,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_exercised"] >= 1
        assert metrics["n_assignments"] >= 1

    def test_combined_csp_capped_at_contracts(self):
        """CSP count doesn't exceed the contracts parameter."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["n_puts_sold"] == 1
        assert engine.shares_held == 200  # 100 initial + 100 from 1 CSP

    def test_combined_ml_skips_cc_and_csp(self):
        """ML filter skips CC when bullish and CSP when bearish."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            skip_call_thresh=0.65, skip_put_thresh=0.35,
            combined=True,
        )
        # prob=0.70 -> skip CC (rally), do sell CSP
        week_bullish = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.70,
        )
        engine.process_week(week_bullish, use_ml_filter=True)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_skipped"] >= 1
        assert metrics["n_puts_sold"] >= 1

    def test_combined_dynamic_cc_count_after_assignment(self):
        """After CSP assignment, more CCs are sold next week."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        # Week 0: Buy 100 shares, sell 1 CC + 1 CSP, CSP assigned -> 200 shares
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        # Week 1: Should sell 2 CCs (200 shares / 100)
        week1 = _make_week(
            1, spy_open=470.0, spy_close=475.0,
            put_strike=460.0, put_premium=2.0,
            call_strike=490.0, call_premium=2.0,
        )
        engine.process_week(week1)
        metrics = engine.compute_metrics()
        # Week 0: 1 CC + week 1: 2 CCs = 3 total calls sold
        assert metrics["n_calls_sold"] == 3


class TestCombinedEdgeCases:

    def test_combined_no_cash_for_csp(self):
        """After CSP assignment eats all cash, no CSPs sold next week."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        # Week 0: CSP assigned -> 200 shares, ~$0 cash
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week0)
        puts_after_w0 = engine.compute_metrics()["n_puts_sold"]

        # Week 1: No cash for CSP. Only sell CCs.
        week1 = _make_week(
            1, spy_open=470.0, spy_close=475.0,
            put_strike=460.0, put_premium=2.0,
            call_strike=490.0, call_premium=2.0,
        )
        engine.process_week(week1)
        puts_after_w1 = engine.compute_metrics()["n_puts_sold"]
        # No new puts sold (no cash for CSP collateral)
        assert puts_after_w1 == puts_after_w0

    def test_combined_max_position_after_multiple_assignments(self):
        """Multiple CSP assignments grow position correctly."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1, combined=True,
        )
        # Week 0: Buy 100 shares at 500 ($50K). Cash=$150K. Sell 1 CSP at 480. Assigned.
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        # Week 1: Now 200 shares. Cash ~$102K (150K - 48K + premiums).
        # Sell 1 CSP at 460. Assigned again.
        week1 = _make_week(
            1, spy_open=470.0, spy_close=450.0,
            put_strike=460.0, put_premium=4.0,
            call_strike=490.0, call_premium=1.0,
        )
        engine.process_week(week1)
        assert engine.shares_held == 300

    def test_combined_cc_exercise_frees_cash_for_csp(self):
        """CC exercise frees cash, enabling CSP sale next week."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        # Week 0: Buy 100 at 500, CSP assigned at 480 -> 200 shares, ~$0 cash
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        # Week 1: CC exercised (price rallies above call strike)
        # 2 CCs exercised -> sell 200 shares at 490, re-buy at 495
        week1 = _make_week(
            1, spy_open=475.0, spy_close=495.0,
            put_strike=470.0, put_premium=2.0,
            call_strike=490.0, call_premium=3.0,
        )
        engine.process_week(week1)
        # After re-buy, should still hold 200 shares
        metrics = engine.compute_metrics()
        assert metrics["n_calls_exercised"] >= 2

    def test_combined_zero_premium_handled(self):
        """Zero put premium is handled gracefully (still sell call)."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=0.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_sold"] >= 1

    def test_combined_insufficient_capital_for_initial_buy(self):
        """Capital too low to buy even 1 contract of shares."""
        engine = WheelBacktestEngine(
            initial_capital=1_000.0, contracts=1, combined=True,
        )
        # SPY at 500 -> 100 shares = $50K. Capital=$1K. Can't afford it.
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        # Cash goes negative (the engine doesn't enforce margin).
        # Should still process without crashing.
        metrics = engine.compute_metrics()
        assert metrics["weeks"] == 1


class TestCollarMode:

    def test_collar_buys_protective_put(self):
        """Collar deducts hedge put cost from cash."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["total_hedge_cost"] == 150.0

    def test_collar_put_payout_on_crash(self):
        """Protective put pays out when SPY drops below hedge strike."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # After combined mode: 100 initial + 100 CSP assignment = 200 shares
        # Collar runs on 200 shares -> n_hedge = 2
        # Payout = (490 - 470) * 2 * 100 = 4000
        assert metrics["total_hedge_payout"] == 4000.0

    def test_collar_otm_expires_worthless(self):
        """OTM hedge put: no payout, only cost deducted."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=510.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["total_hedge_cost"] == 150.0
        assert metrics["total_hedge_payout"] == 0.0

    def test_collar_reduces_net_premium(self):
        """Net premium with collar is lower due to hedge cost."""
        engine_no = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine_no.process_week(week)
        premium_without = engine_no.total_premium_collected

        engine_yes = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        engine_yes.process_week(week)
        premium_with = engine_yes.total_premium_collected

        # Collar doesn't change premium collected, but cash is lower
        assert premium_with == premium_without
        metrics = engine_yes.compute_metrics()
        assert metrics["total_hedge_cost"] > 0

    def test_collar_ml_filter_unaffected(self):
        """ML filter still skips CC/CSP normally with collar enabled."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            skip_call_thresh=0.65, combined=True, collar=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
            ml_prob=0.70,
        )
        engine.process_week(week, use_ml_filter=True)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_skipped"] >= 1
        assert metrics["total_hedge_cost"] > 0

    def test_collar_hedge_metrics_tracked(self):
        """Metrics dict includes hedge cost and payout keys."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        weeks = [
            _make_week(i, spy_open=500.0, spy_close=505.0,
                       hedge_put_strike=490.0, hedge_put_premium=1.0)
            for i in range(3)
        ]
        for w in weeks:
            engine.process_week(w)
        metrics = engine.compute_metrics()
        assert "total_hedge_cost" in metrics
        assert "total_hedge_payout" in metrics
        assert abs(metrics["total_hedge_cost"] - 300.0) < 0.01

    def test_collar_commission_on_hedge(self):
        """Commission is charged on hedge put contracts."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            commission_per_contract=0.65,
            combined=True, collar=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Commissions: 1 CC (0.65) + 1 CSP (0.65) + 1 hedge (0.65) = 1.95
        assert abs(metrics["total_commissions"] - 1.95) < 0.01

    def test_collar_with_csp_assignment_grows_hedge(self):
        """After CSP assignment, more shares means more hedge puts next week."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        week1 = _make_week(
            1, spy_open=470.0, spy_close=475.0,
            put_strike=460.0, put_premium=2.0,
            call_strike=490.0, call_premium=2.0,
            hedge_put_strike=465.0, hedge_put_premium=1.00,
        )
        engine.process_week(week1)
        metrics = engine.compute_metrics()
        # Week 0: 200 shares after CSP assignment -> 2 hedges * 1.50 * 100 = 300
        # Week 1: still 200 shares (no new assignment) -> 2 hedges * 1.00 * 100 = 200
        # Total = 500
        assert abs(metrics["total_hedge_cost"] - 500.0) < 0.01


class TestVolScaling:

    def test_vol_scaling_high_iv_increases_csp_count(self):
        """High IV (80th pctile) -> 2x multiplier -> 2 CSPs sold."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=4.0,
            call_strike=520.0, call_premium=3.0,
            iv_percentile=0.80,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # High IV: mult=2.0, eff=min(3, 1*2)=2. Should sell 2 CSPs.
        assert metrics["n_puts_sold"] == 2

    def test_vol_scaling_low_iv_floors_at_one(self):
        """Low IV (10th pctile) -> 0.5x multiplier -> floor at 1 contract."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=1.0,
            call_strike=520.0, call_premium=1.0,
            iv_percentile=0.10,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Low IV: mult=0.5, int(1*0.5)=0, floor=1. Sell 1 CSP.
        assert metrics["n_puts_sold"] == 1

    def test_vol_scaling_respects_max_contracts(self):
        """Effective contracts capped at max_contracts."""
        engine = WheelBacktestEngine(
            initial_capital=500_000.0, contracts=2,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
            iv_percentile=0.90,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # mult=2.0, 2*2=4, cap at 3. Sell 3 CSPs.
        assert metrics["n_puts_sold"] == 3

    def test_vol_scaling_disabled_original_behavior(self):
        """vol_scaling=False -> same behavior as before."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1,
            combined=True, vol_scaling=False,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            iv_percentile=0.90,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Without vol scaling, ignores iv_percentile. Sell 1 CSP.
        assert metrics["n_puts_sold"] == 1

    def test_vol_scaling_cc_cap(self):
        """Vol scaling caps CC count in combined mode."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        # Week 0: CSP assigned -> 200 shares
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
            iv_percentile=0.10,  # low IV -> eff=1
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        # Week 1: 200 shares but low IV -> only 1 CC sold (not 2)
        week1 = _make_week(
            1, spy_open=470.0, spy_close=475.0,
            put_strike=460.0, put_premium=2.0,
            call_strike=490.0, call_premium=2.0,
            iv_percentile=0.10,  # low IV -> eff=1
        )
        engine.process_week(week1)
        metrics = engine.compute_metrics()
        # Week 0: 1 CC (eff=1, shares=100). Week 1: 1 CC (eff=1, capped from 2).
        assert metrics["n_calls_sold"] == 2  # not 3

    def test_vol_scaling_metrics_tracked(self):
        """Metrics include avg and max contracts traded."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        weeks = [
            _make_week(0, spy_open=500.0, spy_close=505.0, iv_percentile=0.20),
            _make_week(1, spy_open=505.0, spy_close=510.0, iv_percentile=0.60),
            _make_week(2, spy_open=510.0, spy_close=515.0, iv_percentile=0.90),
        ]
        for w in weeks:
            engine.process_week(w)
        metrics = engine.compute_metrics()
        assert "avg_contracts_traded" in metrics
        assert "max_contracts_traded" in metrics
        # Week 0: 0.20 -> 0.5x -> max(1,int(0.5))=1. Week 1: 0.60 -> 1.5x -> int(1.5)=1. Week 2: 0.90 -> 2.0x -> 2.
        assert metrics["max_contracts_traded"] == 2.0


class TestCrashAvoidance:

    def test_crash_avoidance_starts_invested(self):
        """With neutral probs, crash avoidance stays invested normally."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week, use_ml_filter=True)
        assert engine.shares_held >= 100

    def test_crash_avoidance_exits_on_strong_bearish(self):
        """Liquidates shares when ML prob drops below exit threshold."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25, re_entry_thresh=0.50,
        )
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)
        assert engine.shares_held >= 100

        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=1.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=True)
        assert engine.shares_held == 0
        metrics = engine.compute_metrics()
        assert metrics["n_crash_exits"] == 1

    def test_crash_avoidance_re_enters_on_recovery(self):
        """Re-enters market when prob recovers above re-entry threshold."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25, re_entry_thresh=0.50,
        )
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)

        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=1.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=True)
        assert engine.shares_held == 0

        week2 = _make_week(
            2, spy_open=490.0, spy_close=485.0,
            put_strike=470.0, put_premium=3.0,
            call_strike=510.0, call_premium=1.0,
            ml_prob=0.40,
        )
        engine.process_week(week2, use_ml_filter=True)
        assert engine.shares_held == 0

        week3 = _make_week(
            3, spy_open=485.0, spy_close=495.0,
            put_strike=470.0, put_premium=2.0,
            call_strike=510.0, call_premium=2.0,
            ml_prob=0.55,
        )
        engine.process_week(week3, use_ml_filter=True)
        assert engine.shares_held >= 100
        metrics = engine.compute_metrics()
        assert metrics["n_crash_reentries"] == 1

    def test_crash_avoidance_skips_options_while_in_cash(self):
        """No options sold while in cash mode."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25,
        )
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)
        m0 = engine.compute_metrics()

        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=5.0,
            call_strike=520.0, call_premium=5.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=True)
        m1 = engine.compute_metrics()

        assert m1["total_premium_collected"] == m0["total_premium_collected"]

    def test_crash_avoidance_equity_preserved_in_cash(self):
        """Equity in cash mode is just cash (no share exposure)."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25,
        )
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)

        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=1.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=True)
        cash_after_exit = engine._cash

        week2 = _make_week(
            2, spy_open=490.0, spy_close=450.0,
            put_strike=470.0, put_premium=5.0,
            call_strike=510.0, call_premium=0.5,
            ml_prob=0.15,
        )
        engine.process_week(week2, use_ml_filter=True)
        assert engine._cash == cash_after_exit
        assert engine.equity_curve[-1] == cash_after_exit

    def test_crash_avoidance_disabled_by_default(self):
        """crash_avoidance=False preserves original behavior."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=False,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.20,
        )
        engine.process_week(week, use_ml_filter=True)
        assert engine.shares_held >= 100

    def test_crash_avoidance_requires_ml_filter(self):
        """Crash avoidance only triggers when use_ml_filter=True."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25,
        )
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)
        assert engine.shares_held >= 100

        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=1.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=False)
        assert engine.shares_held >= 100
