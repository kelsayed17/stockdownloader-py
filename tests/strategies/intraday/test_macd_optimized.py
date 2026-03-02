"""Tests for MACD Optimized 8/35/5 tournament strategy."""

from __future__ import annotations

from decimal import Decimal

from stockdownloader.core.models.trade import IntradayAction
from tests.strategies.intraday.conftest import make_warmup_bars


class TestMACDOptimizedConfig:
    """Test MACDOptimizedConfig dataclass."""

    def test_default_values(self) -> None:
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedConfig
        c = MACDOptimizedConfig()
        assert c.macd_fast == 8
        assert c.macd_slow == 35
        assert c.macd_signal == 5
        assert c.sl_mult == Decimal("1.5")
        assert c.rr_ratio == Decimal("1.5")
        assert c.sl_cap == Decimal("2.0")
        assert c.be_trigger == Decimal("0.5")
        assert c.max_day == 4
        assert c.spacing == 3
        assert c.circuit == 3
        assert c.day_loss == Decimal("3.0")
        assert c.allow_longs is True
        assert c.allow_shorts is True

    def test_inherits_infra_exit_config(self) -> None:
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedConfig
        from stockdownloader.strategies.intraday.base import InfraExitConfig
        assert issubclass(MACDOptimizedConfig, InfraExitConfig)


class TestMACDOptimizedStrategy:
    """Test MACDOptimizedStrategy signal logic."""

    def test_strategy_name(self) -> None:
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
        s = MACDOptimizedStrategy()
        assert s.name == "SPY MACD 8/35/5"

    def test_warmup_period(self) -> None:
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
        s = MACDOptimizedStrategy()
        assert s.warmup_period == 78 * 15

    def test_hold_during_warmup(self) -> None:
        """Strategy should return HOLD during warmup period."""
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
        s = MACDOptimizedStrategy()
        bars = make_warmup_bars(100)
        s.on_session_start("2024-01-02")
        sig = s.evaluate(bars, 50)
        assert sig.action == IntradayAction.HOLD

    def test_bidirectional(self) -> None:
        """Strategy should allow both longs and shorts."""
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
        s = MACDOptimizedStrategy()
        assert s._c.allow_longs is True
        assert s._c.allow_shorts is True

    def test_no_obv_filter(self) -> None:
        """Strategy should NOT have OBV state (pure MACD, unlike MACD+OBV)."""
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
        s = MACDOptimizedStrategy()
        assert not hasattr(s, '_obv_state')
