"""Tests for SMA Cross 20/21 tournament strategy."""

from __future__ import annotations

from decimal import Decimal

from stockdownloader.core.models.trade import IntradayAction
from tests.strategies.intraday.conftest import make_warmup_bars


class TestSMACross2021Config:
    """Test SMACross2021Config dataclass."""

    def test_default_values(self) -> None:
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Config
        c = SMACross2021Config()
        assert c.sma_short == 20
        assert c.sma_long == 21
        assert c.sl_mult == Decimal("1.5")
        assert c.rr_ratio == Decimal("1.5")
        assert c.sl_cap == Decimal("2.0")
        assert c.be_trigger == Decimal("0.5")
        assert c.max_day == 4
        assert c.spacing == 3
        assert c.circuit == 3
        assert c.day_loss == Decimal("3.0")
        assert c.allow_longs is True
        assert c.allow_shorts is False

    def test_inherits_infra_exit_config(self) -> None:
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Config
        from stockdownloader.strategies.intraday.base import InfraExitConfig
        assert issubclass(SMACross2021Config, InfraExitConfig)


class TestSMACross2021Strategy:
    """Test SMACross2021Strategy signal logic."""

    def test_strategy_name(self) -> None:
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
        s = SMACross2021Strategy()
        assert s.name == "SPY SMA 20/21"

    def test_warmup_period(self) -> None:
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
        s = SMACross2021Strategy()
        assert s.warmup_period == 78 * 15

    def test_hold_during_warmup(self) -> None:
        """Strategy should return HOLD during warmup period."""
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
        s = SMACross2021Strategy()
        bars = make_warmup_bars(100)
        s.on_session_start("2024-01-02")
        sig = s.evaluate(bars, 50)
        assert sig.action == IntradayAction.HOLD

    def test_long_only(self) -> None:
        """Strategy should be long-only (allow_shorts is False)."""
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
        s = SMACross2021Strategy()
        assert s._c.allow_longs is True
        assert s._c.allow_shorts is False

    def test_sl_tp_math(self) -> None:
        """SL/TP prices should match Pine formula -- long side only."""
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Config
        from stockdownloader.strategies.intraday.trade_mgmt import directional_sl_tp
        c = SMACross2021Config()
        entry = Decimal("450.00")
        sl_dist = Decimal("2.0")
        tp_dist = sl_dist * c.rr_ratio
        # Long side
        sl, tp = directional_sl_tp(True, entry, sl_dist, tp_dist)
        assert sl == Decimal("448.00")
        assert tp == Decimal("453.00")
