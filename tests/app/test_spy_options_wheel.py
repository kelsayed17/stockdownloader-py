"""Tests for the spy-options-wheel CLI pipeline."""
from __future__ import annotations

import pytest

from stockdownloader.app.spy_options_wheel import _build_parser
from stockdownloader.backtesting.engines.wheel import WeekRecord, WheelBacktestEngine


class TestWheelParser:

    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.delta == 0.30
        assert args.initial_capital == 100_000.0
        assert args.contracts == 1
        assert args.skip_put_thresh == 0.35
        assert args.skip_call_thresh == 0.65
        assert args.no_ml_filter is False

    def test_custom_delta(self):
        parser = _build_parser()
        args = parser.parse_args(["--delta", "0.20"])
        assert args.delta == 0.20

    def test_no_ml_filter_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--no-ml-filter"])
        assert args.no_ml_filter is True

    def test_custom_thresholds(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--skip-put-thresh", "0.30",
            "--skip-call-thresh", "0.70",
        ])
        assert args.skip_put_thresh == 0.30
        assert args.skip_call_thresh == 0.70

    def test_date_range(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--from-date", "2022-01-01",
            "--to-date", "2026-01-01",
        ])
        assert args.from_date == "2022-01-01"
        assert args.to_date == "2026-01-01"

    def test_contracts_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--contracts", "2"])
        assert args.contracts == 2

    def test_walk_forward_windows(self):
        parser = _build_parser()
        args = parser.parse_args(["--walk-forward-windows", "8"])
        assert args.walk_forward_windows == 8

    def test_buy_write_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.buy_write is False

    def test_buy_write_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--buy-write"])
        assert args.buy_write is True


class TestCombinedParser:

    def test_combined_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.combined is False

    def test_combined_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--combined"])
        assert args.combined is True

    def test_commission_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.commission == 0.65

    def test_commission_custom(self):
        parser = _build_parser()
        args = parser.parse_args(["--commission", "1.00"])
        assert args.commission == 1.00

    def test_iv_filter_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.iv_filter is False

    def test_iv_filter_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--iv-filter"])
        assert args.iv_filter is True

    def test_min_iv_percentile_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.min_iv_percentile == 0.30


class TestWheelIntegration:
    """Integration smoke test for the wheel pipeline."""

    def test_full_wheel_lifecycle(self):
        """Run a complete wheel lifecycle: CSP -> assigned -> CC -> called away."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0,
            contracts=1,
            skip_put_thresh=0.35,
            skip_call_thresh=0.65,
        )

        weeks = [
            # Week 0: Sell put at 580, SPY closes at 605 (OTM) -> keep premium
            WeekRecord(0, "2025-02-07", "2025-02-03", 600.0, 605.0, 580.0, 620.0, 3.0, 2.0, 0.50),
            # Week 1: Sell put at 580, SPY drops to 570 (ITM) -> ASSIGNED
            WeekRecord(1, "2025-02-14", "2025-02-10", 600.0, 570.0, 580.0, 620.0, 4.0, 2.0, 0.50),
            # Week 2: Now in CALL_PHASE, sell call at 590, SPY at 575 (OTM) -> keep premium
            WeekRecord(2, "2025-02-21", "2025-02-17", 575.0, 575.0, 570.0, 590.0, 3.0, 2.5, 0.50),
            # Week 3: Sell call at 590, SPY rallies to 600 (ITM) -> CALLED AWAY
            WeekRecord(3, "2025-02-28", "2025-02-24", 580.0, 600.0, 570.0, 590.0, 3.0, 2.0, 0.50),
            # Week 4: Back to PUT_PHASE, sell put, SPY at 605 (OTM)
            WeekRecord(4, "2025-03-07", "2025-03-03", 600.0, 605.0, 580.0, 620.0, 3.0, 2.0, 0.50),
        ]

        for w in weeks:
            engine.process_week(w, use_ml_filter=True)

        metrics = engine.compute_metrics()

        assert metrics["n_assignments"] == 1  # Week 1
        assert metrics["n_calls_exercised"] == 1  # Week 3
        assert metrics["total_premium_collected"] > 0
        assert metrics["weeks"] == 5
        assert len(engine.equity_curve) == 5
        assert metrics["final_equity"] > metrics["initial_capital"]

    def test_buy_write_lifecycle(self):
        """Buy-write: buy shares, sell calls, get called away, re-buy."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0,
            contracts=1,
            skip_call_thresh=0.65,
            buy_write=True,
        )

        weeks = [
            # Week 0: Buy at 500, sell call at 520, closes 510 (OTM) -> keep premium
            WeekRecord(0, "2025-02-07", "2025-02-03", 500.0, 510.0, 480.0, 520.0, 0.0, 2.0, 0.50),
            # Week 1: Sell call at 520, closes 525 (ITM) -> called away + re-buy at 525
            WeekRecord(1, "2025-02-14", "2025-02-10", 510.0, 525.0, 490.0, 520.0, 0.0, 2.5, 0.50),
            # Week 2: Sell call at 540, closes 530 (OTM) -> keep premium
            WeekRecord(2, "2025-02-21", "2025-02-17", 525.0, 530.0, 510.0, 540.0, 0.0, 2.0, 0.50),
            # Week 3: ML prob 0.70 -> skip selling call (rally expected)
            WeekRecord(3, "2025-02-28", "2025-02-24", 530.0, 545.0, 510.0, 540.0, 0.0, 2.0, 0.70),
        ]

        for w in weeks:
            engine.process_week(w, use_ml_filter=True)

        metrics = engine.compute_metrics()

        assert metrics["n_calls_exercised"] == 1  # Week 1
        assert metrics["n_rebuys"] == 1  # Week 1 re-buy
        assert metrics["n_calls_sold"] == 3  # Weeks 0, 1, 2 (week 3 skipped)
        assert metrics["n_calls_skipped"] == 1  # Week 3
        assert metrics["n_puts_sold"] == 0  # Buy-write never sells puts
        assert metrics["total_premium_collected"] > 0
        assert metrics["weeks"] == 4
        assert engine.shares_held == 100  # Always holding

    def test_combined_lifecycle(self):
        """Combined mode: buy shares + sell CC + sell CSP, with assignment."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0,
            contracts=1,
            skip_call_thresh=0.65,
            skip_put_thresh=0.35,
            commission_per_contract=0.65,
            combined=True,
        )

        weeks = [
            # Week 0: Buy 100 at 500. Sell 1 CC at 520 + 1 CSP at 480. All OTM.
            WeekRecord(0, "2025-02-07", "2025-02-03", 500.0, 505.0, 480.0, 520.0, 2.0, 2.0, 0.50),
            # Week 1: SPY drops to 470. CSP at 480 ITM -> assigned. Now 200 shares.
            WeekRecord(1, "2025-02-14", "2025-02-10", 505.0, 470.0, 480.0, 520.0, 4.0, 1.0, 0.50),
            # Week 2: SPY rallies to 495. 2 CCs sold (200 shares). CC at 490 ITM -> called away + rebuy.
            WeekRecord(2, "2025-02-21", "2025-02-17", 475.0, 495.0, 460.0, 490.0, 2.0, 3.0, 0.50),
            # Week 3: ML prob 0.30 -> skip CSP (crash danger). Still sell CC.
            WeekRecord(3, "2025-02-28", "2025-02-24", 490.0, 495.0, 475.0, 510.0, 3.0, 2.0, 0.30),
        ]

        for w in weeks:
            engine.process_week(w, use_ml_filter=True)

        metrics = engine.compute_metrics()
        assert metrics["weeks"] == 4
        assert metrics["n_assignments"] >= 1  # Week 1 CSP
        assert metrics["n_calls_exercised"] >= 1  # Week 2 CC
        assert metrics["total_premium_collected"] > 0
        assert metrics["total_commissions"] > 0
        assert engine.shares_held >= 100


class TestAdvancedParser:

    def test_collar_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.collar is False

    def test_collar_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--collar"])
        assert args.collar is True

    def test_hedge_delta_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.hedge_delta == 0.10

    def test_hedge_delta_custom(self):
        parser = _build_parser()
        args = parser.parse_args(["--hedge-delta", "0.15"])
        assert args.hedge_delta == 0.15

    def test_vol_scaling_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.vol_scaling is False

    def test_vol_scaling_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--vol-scaling"])
        assert args.vol_scaling is True

    def test_max_contracts_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.max_contracts == 3

    def test_iron_condor_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.iron_condor is False

    def test_iron_condor_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--iron-condor"])
        assert args.iron_condor is True

    def test_ic_allocation_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.ic_allocation == 0.30

    def test_ic_allocation_custom(self):
        parser = _build_parser()
        args = parser.parse_args(["--ic-allocation", "0.20"])
        assert args.ic_allocation == 0.20
