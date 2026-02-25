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
