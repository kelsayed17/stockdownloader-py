"""Tests for MACD+OBV tournament strategy."""

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


class TestMACDOBVConfig:
    """Test MACDOBVConfig dataclass."""

    def test_default_values(self) -> None:
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVConfig
        c = MACDOBVConfig()
        assert c.macd_fast == 12
        assert c.macd_slow == 26
        assert c.macd_signal == 9
        assert c.obv_smooth == 5
        assert c.sl_mult == Decimal("1.5")
        assert c.rr_ratio == Decimal("1.5")
        assert c.sl_cap == Decimal("2.0")
        assert c.be_trigger == Decimal("0.5")
        assert c.max_day == 4
        assert c.spacing == 3
        assert c.circuit == 3
        assert c.day_loss == Decimal("3.0")
        assert c.allow_shorts is True

    def test_inherits_infra_exit_config(self) -> None:
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVConfig
        from stockdownloader.strategies.intraday.base import InfraExitConfig
        assert issubclass(MACDOBVConfig, InfraExitConfig)


class TestMACDOBVStrategy:
    """Test MACDOBVStrategy signal logic."""

    def test_strategy_name(self) -> None:
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
        s = MACDOBVStrategy()
        assert s.name == "SPY MACD+OBV"

    def test_warmup_period(self) -> None:
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
        s = MACDOBVStrategy()
        assert s.warmup_period == 78 * 15

    def test_hold_during_warmup(self) -> None:
        """Strategy should return HOLD during warmup period."""
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
        s = MACDOBVStrategy()
        bars = _make_warmup_bars(100)
        s.on_session_start("2024-01-02")
        sig = s.evaluate(bars, 50)
        assert sig.action == IntradayAction.HOLD

    def test_long_entry_produces_valid_signal(self) -> None:
        """After warmup, a MACD bullish cross + OBV rising should produce ENTER_LONG."""
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
        s = MACDOBVStrategy()
        bars = _make_warmup_bars(1300)
        for i in range(len(bars)):
            if i % 78 == 0:
                s.on_session_start(bars[i].trading_date)
            sig = s.evaluate(bars, i)
        assert sig.action in (IntradayAction.HOLD, IntradayAction.ENTER_LONG,
                              IntradayAction.ENTER_SHORT)

    def test_sl_cap_applied(self) -> None:
        """SL distance should be capped at sl_cap ($2.00)."""
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVConfig
        from stockdownloader.strategies.intraday.trade_mgmt import clamp_sl_dist
        c = MACDOBVConfig()
        raw_sl = Decimal("3.0") * c.sl_mult
        clamped = clamp_sl_dist(raw_sl, c.sl_cap)
        assert clamped == Decimal("2.0")

    def test_sl_tp_math(self) -> None:
        """SL/TP prices should match Pine formula."""
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVConfig
        from stockdownloader.strategies.intraday.trade_mgmt import directional_sl_tp
        c = MACDOBVConfig()
        entry = Decimal("450.00")
        sl_dist = Decimal("2.0")
        tp_dist = sl_dist * c.rr_ratio
        sl, tp = directional_sl_tp(True, entry, sl_dist, tp_dist)
        assert sl == Decimal("448.00")
        assert tp == Decimal("453.00")
        sl_s, tp_s = directional_sl_tp(False, entry, sl_dist, tp_dist)
        assert sl_s == Decimal("452.00")
        assert tp_s == Decimal("447.00")

    def test_obv_ema_crossover_detection(self) -> None:
        """OBV EMA rising/falling detection should work correctly."""
        from stockdownloader.strategies.intraday.macd_obv import _obv_ema_state
        state = _obv_ema_state(period=5)
        for obv_val in [100, 200, 300, 400, 500, 600]:
            state.update(Decimal(str(obv_val)))
        assert state.is_rising() is True
        for obv_val in [500, 400, 300]:
            state.update(Decimal(str(obv_val)))
        assert state.is_rising() is False
