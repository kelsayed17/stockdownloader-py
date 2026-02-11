"""Tests for ExitMechanismTradeResult and ExitMechanismSummary."""

from decimal import Decimal

import pytest

from stockdownloader.model.exit_mechanism_result import (
    ExitMechanismTradeResult,
    ExitMechanismSummary,
)
from stockdownloader.model.trade import Direction


def _make_result(pnl, capture=50, mechanism="TEST", trade_id=1, direction=Direction.LONG,
                 signal_type="PB"):
    return ExitMechanismTradeResult(
        mechanism_name=mechanism,
        trade_id=trade_id,
        direction=direction,
        signal_type=signal_type,
        entry_price=Decimal("680"),
        exit_price=Decimal("680") + Decimal(str(pnl)),
        exit_datetime="2025-12-01 11:00:00-05:00",
        pnl=Decimal(str(pnl)),
        r_multiple=Decimal(str(pnl)),
        holding_bars=10,
        peak_favorable=Decimal("2"),
        capture_pct=Decimal(str(capture)),
        exit_reason="trail_stop",
    )


class TestExitMechanismTradeResult:

    def test_create_result(self):
        r = _make_result(1.5)
        assert r.pnl == Decimal("1.5")
        assert r.mechanism_name == "TEST"
        assert r.capture_pct == Decimal("50")

    def test_frozen(self):
        r = _make_result(1.5)
        with pytest.raises(AttributeError):
            r.pnl = Decimal("999")


class TestExitMechanismSummary:

    def test_empty_summary(self):
        s = ExitMechanismSummary("TEST")
        assert s.total_trades == 0
        assert s.win_rate == Decimal("0")
        assert s.total_pnl == Decimal("0")

    def test_add_results(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.5))
        s.add_result(_make_result(-0.5))
        s.add_result(_make_result(2.0))
        assert s.total_trades == 3
        assert s.winning_trades == 2
        assert s.losing_trades == 1

    def test_win_rate(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.0))
        s.add_result(_make_result(-1.0))
        assert s.win_rate == Decimal("50.00")

    def test_total_pnl(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.5))
        s.add_result(_make_result(-0.5))
        assert s.total_pnl == Decimal("1.0")

    def test_profit_factor(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(3.0))
        s.add_result(_make_result(-1.0))
        assert s.profit_factor == Decimal("3.00")

    def test_profit_factor_no_losses(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.0))
        s.add_result(_make_result(2.0))
        assert s.profit_factor == Decimal("999.99")

    def test_avg_pnl(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.0))
        s.add_result(_make_result(3.0))
        assert s.avg_pnl == Decimal("2.0000")

    def test_median_pnl_odd(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.0))
        s.add_result(_make_result(3.0))
        s.add_result(_make_result(5.0))
        assert s.median_pnl == Decimal("3.0")

    def test_median_pnl_even(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.0))
        s.add_result(_make_result(3.0))
        assert s.median_pnl == Decimal("2.0000")

    def test_avg_capture_pct(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.0, capture=60))
        s.add_result(_make_result(2.0, capture=80))
        assert s.avg_capture_pct == Decimal("70.00")

    def test_sharpe_approx(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.0))
        s.add_result(_make_result(1.0))
        # Same P&L every trade -> std=0 -> sharpe=0
        assert s.sharpe_approx == Decimal("0")

    def test_sharpe_nonzero_std(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(2.0))
        s.add_result(_make_result(-1.0))
        s.add_result(_make_result(3.0))
        # Should produce a nonzero sharpe
        assert s.sharpe_approx != Decimal("0")

    def test_exit_reason_counts(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(1.0))
        s.add_result(_make_result(2.0))
        counts = s.exit_reason_counts()
        assert counts == {"trail_stop": 2}

    def test_max_loss_and_win(self):
        s = ExitMechanismSummary("TEST")
        s.add_result(_make_result(-2.0))
        s.add_result(_make_result(5.0))
        s.add_result(_make_result(-0.5))
        assert s.max_loss == Decimal("-2.0")
        assert s.max_win == Decimal("5.0")

    def test_validation_none_name(self):
        with pytest.raises(ValueError):
            ExitMechanismSummary(None)

    def test_validation_none_result(self):
        s = ExitMechanismSummary("TEST")
        with pytest.raises(ValueError):
            s.add_result(None)
