"""Tests for MACD Optimized 8/35/5 tournament strategy."""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import IntradayAction


def _bar(
    dt: str,
    o: float,
    h: float,
    l: float,
    c: float,
    v: int = 1000,
) -> IntradayPriceData:
    """Helper to create a 5-min bar."""
    return IntradayPriceData(
        date=dt,
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(l)),
        close=Decimal(str(c)),
        adj_close=Decimal(str(c)),
        volume=v,
    )


def _make_warmup_bars(n: int = 1200) -> list[IntradayPriceData]:
    """Generate n synthetic bars for warmup (78 bars/day)."""
    bars: list[IntradayPriceData] = []
    price = 450.0
    day_num = 0
    for i in range(n):
        bar_of_day = i % 78
        if bar_of_day == 0 and i > 0:
            day_num += 1
        date = f"2024-01-{2 + day_num:02d}"
        hour = 9 + (bar_of_day * 5 + 30) // 60
        minute = (bar_of_day * 5 + 30) % 60
        dt = f"{date} {hour:02d}:{minute:02d}:00-05:00"
        # Gentle uptrend with noise
        delta = 0.02 * (1 if i % 3 != 0 else -1)
        price += delta
        bars.append(_bar(dt, price - 0.05, price + 0.10, price - 0.10, price, 5000))
    return bars


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
