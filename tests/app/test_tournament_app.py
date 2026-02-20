"""Tests for tournament_app — CLI smoke tests and arg parsing."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from stockdownloader.app.tournament.cli import (
    _parse_args,
    main,
)
from stockdownloader.app.tournament.helpers import (
    _box_title,
    _build_skip_set,
    _status_label,
)
from stockdownloader.backtest.tournament_engine import ComboKey, ComboResult


class TestBoxTitle:
    def test_returns_string(self):
        result = _box_title("TEST")
        assert "TEST" in result
        assert "\u2554" in result  # top-left corner
        assert "\u255d" in result  # bottom-right corner

    def test_custom_width(self):
        result = _box_title("HELLO", width=50)
        # Second line should be 50 chars between the vertical bars
        lines = result.split("\n")
        assert len(lines) == 3


class TestStatusLabel:
    def test_robust(self):
        assert _status_label(0.8) == "ROBUST"
        assert _status_label(1.0) == "ROBUST"
        assert _status_label(2.0) == "ROBUST"

    def test_acceptable(self):
        assert _status_label(0.5) == "ACCEPTABLE"
        assert _status_label(0.7) == "ACCEPTABLE"

    def test_overfit(self):
        assert _status_label(0.3) == "OVERFIT"
        assert _status_label(0.0) == "OVERFIT"
        assert _status_label(-1.0) == "OVERFIT"


class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.mode == "all"
        assert args.bracket_size == 16
        assert args.top_k == 5
        assert args.no_optimize is False
        assert args.no_walk_forward is False
        assert args.no_regime is False
        assert args.no_monte_carlo is False
        assert args.mc_top_n == 20
        assert args.mc_simulations == 1000
        assert args.timeframes is None
        assert args.category is None

    def test_mode(self):
        args = _parse_args(["--mode", "bracket"])
        assert args.mode == "bracket"

    def test_bracket_size(self):
        args = _parse_args(["--bracket-size", "8"])
        assert args.bracket_size == 8

    def test_no_optimize(self):
        args = _parse_args(["--no-optimize"])
        assert args.no_optimize is True

    def test_no_walk_forward(self):
        args = _parse_args(["--no-walk-forward"])
        assert args.no_walk_forward is True

    def test_both_skip_flags(self):
        args = _parse_args(["--no-optimize", "--no-walk-forward"])
        assert args.no_optimize is True
        assert args.no_walk_forward is True

    def test_no_regime(self):
        args = _parse_args(["--no-regime"])
        assert args.no_regime is True

    def test_no_monte_carlo(self):
        args = _parse_args(["--no-monte-carlo"])
        assert args.no_monte_carlo is True

    def test_mc_top_n(self):
        args = _parse_args(["--mc-top-n", "10"])
        assert args.mc_top_n == 10

    def test_mc_simulations(self):
        args = _parse_args(["--mc-simulations", "500"])
        assert args.mc_simulations == 500

    def test_timeframes(self):
        args = _parse_args(["--timeframes", "5m,15m,1h"])
        assert args.timeframes == "5m,15m,1h"

    def test_category(self):
        args = _parse_args(["--category", "intraday"])
        assert args.category == "intraday"

    def test_extra_csv(self):
        args = _parse_args(["--extra-csv", "15m:data/spy_15m.csv"])
        assert len(args.extra_csv) == 1
        assert args.extra_csv[0] == "15m:data/spy_15m.csv"

    def test_multiple_extra_csv(self):
        args = _parse_args([
            "--extra-csv", "15m:data/spy_15m.csv",
            "--extra-csv", "1h:data/spy_1h.csv",
        ])
        assert len(args.extra_csv) == 2

    def test_top_k(self):
        args = _parse_args(["--top-k", "3"])
        assert args.top_k == 3


class TestBuildSkipSet:
    def _make_combo(self, name: str, tf: str, trades: int) -> ComboResult:
        mock_result = MagicMock()
        mock_result.total_trades = trades
        mock_result.total_pnl = Decimal("1000")
        return ComboResult(
            key=ComboKey(name, tf),
            baseline=mock_result,
        )

    def test_skips_higher_tfs_after_zero(self):
        results = [
            self._make_combo("rsi", "5m", 50),
            self._make_combo("rsi", "15m", 20),
            self._make_combo("rsi", "30m", 0),  # zero trades
            self._make_combo("rsi", "1h", 0),
            self._make_combo("rsi", "4h", 0),
            self._make_combo("rsi", "1d", 0),
        ]
        skip = _build_skip_set(results)
        # 30m has 0 trades -> skip 30m, 1h, 4h, 1d
        assert ("rsi", "30m") in skip
        assert ("rsi", "1h") in skip
        assert ("rsi", "4h") in skip
        assert ("rsi", "1d") in skip
        # 5m and 15m should NOT be skipped
        assert ("rsi", "5m") not in skip
        assert ("rsi", "15m") not in skip

    def test_no_skips_when_all_have_trades(self):
        results = [
            self._make_combo("rsi", "5m", 50),
            self._make_combo("rsi", "15m", 20),
            self._make_combo("rsi", "30m", 10),
        ]
        skip = _build_skip_set(results)
        assert len(skip) == 0

    def test_empty_list(self):
        skip = _build_skip_set([])
        assert len(skip) == 0

    def test_different_strategies_independent(self):
        results = [
            self._make_combo("rsi", "5m", 50),
            self._make_combo("rsi", "15m", 0),  # skip from 15m
            self._make_combo("macd", "5m", 30),
            self._make_combo("macd", "15m", 10),  # macd still active at 15m
        ]
        skip = _build_skip_set(results)
        assert ("rsi", "15m") in skip
        assert ("rsi", "30m") in skip
        assert ("macd", "15m") not in skip


class TestMainCallable:
    def test_main_exists(self):
        """main() function is callable."""
        assert callable(main)
