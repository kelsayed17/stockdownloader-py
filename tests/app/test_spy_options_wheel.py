"""Tests for the spy-options-wheel CLI pipeline."""
from __future__ import annotations

import pytest

from stockdownloader.app.spy_options_wheel import _build_parser


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
