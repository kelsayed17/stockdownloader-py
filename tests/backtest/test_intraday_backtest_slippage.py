"""Tests for IntradayBacktestEngine slippage model and input validation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import IntradayAction, IntradaySignal, HOLD
from stockdownloader.core.models.trade import Direction, TradeStatus
from stockdownloader.strategies.base import IntradayTradingStrategy

# All existing tests use slippage_pct=0 to verify core engine logic.
# Slippage-specific tests are in TestSlippage below.
_NO_SLIP = Decimal("0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _d(value) -> Decimal:
    """Shorthand for Decimal construction."""
    return Decimal(str(value))


def _make_bar(date: str, close, *, open_=None, high=None, low=None, volume: int = 1000) -> IntradayPriceData:
    """Build an IntradayPriceData bar with sensible defaults."""
    c = _d(close)
    o = _d(open_) if open_ is not None else c
    h = _d(high) if high is not None else c
    lo = _d(low) if low is not None else c
    return IntradayPriceData(date=date, open=o, high=h, low=lo, close=c, adj_close=c, volume=volume)


def _make_bars(prices: list, base_date: str = "2025-01-15") -> list[IntradayPriceData]:
    """Create a sequence of 5-minute bars from a list of close prices.

    Bars are timestamped starting at 09:30 and incrementing by 5 minutes.
    """
    bars: list[IntradayPriceData] = []
    hour, minute = 9, 30
    for price in prices:
        ts = f"{base_date} {hour:02d}:{minute:02d}:00-05:00"
        bars.append(_make_bar(ts, price))
        minute += 5
        if minute >= 60:
            minute -= 60
            hour += 1
    return bars


# ---------------------------------------------------------------------------
# Mock strategy
# ---------------------------------------------------------------------------

class _MockStrategy(IntradayTradingStrategy):
    """Strategy that returns configurable signals at specific bar indices.

    ``signal_map`` maps bar index -> IntradaySignal.  For indices not in the
    map, HOLD is returned.
    """

    def __init__(self, signal_map: dict[int, IntradaySignal], name: str = "MockStrategy") -> None:
        self._signal_map = signal_map
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, data: list[IntradayPriceData], current_index: int) -> IntradaySignal:
        return self._signal_map.get(current_index, HOLD)

    def on_session_start(self, trading_date: str) -> None:
        pass

    @property
    def warmup_period(self) -> int:
        return 0


def _long_entry_signal(risk_per_share="1.00", mode="test") -> IntradaySignal:
    return IntradaySignal(
        action=IntradayAction.ENTER_LONG,
        risk_per_share=_d(risk_per_share),
        mode=mode,
    )


def _short_entry_signal(risk_per_share="1.00", mode="test") -> IntradaySignal:
    return IntradaySignal(
        action=IntradayAction.ENTER_SHORT,
        risk_per_share=_d(risk_per_share),
        mode=mode,
    )


def _exit_signal() -> IntradaySignal:
    return IntradaySignal(action=IntradayAction.EXIT)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    """None strategy, empty data, and None initial_capital should raise ValueError."""

    def test_none_strategy_raises(self):
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        data = _make_bars([100, 100])
        with pytest.raises(ValueError, match="strategy"):
            engine.run(None, data)

    def test_empty_data_raises(self):
        strategy = _MockStrategy({})
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        with pytest.raises(ValueError, match="data"):
            engine.run(strategy, [])

    def test_none_data_raises(self):
        strategy = _MockStrategy({})
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        with pytest.raises((ValueError, TypeError)):
            engine.run(strategy, None)

    def test_none_initial_capital_raises(self):
        with pytest.raises((ValueError, TypeError)):
            IntradayBacktestEngine(initial_capital=None)


# ---------------------------------------------------------------------------
# Slippage model
# ---------------------------------------------------------------------------

class TestSlippage:
    """Verify the slippage model works against the trader on every fill."""

    def test_long_entry_fills_higher(self):
        """Long entry (buy) should fill at close * (1 + slippage_pct)."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])
        # 10 bps = 0.10% slippage
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_d("0.001"),
        )
        result = engine.run(strategy, data)

        trade = result.trades[0]
        # Entry: 100 * 1.001 = 100.10
        assert trade.entry_price == _d("100.1000")

    def test_long_exit_fills_lower(self):
        """Long exit (sell) should fill at close * (1 - slippage_pct)."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_d("0.001"),
        )
        result = engine.run(strategy, data)

        trade = result.trades[0]
        # Exit: 110 * 0.999 = 109.89
        assert trade.exit_price == _d("109.8900")

    def test_short_entry_fills_lower(self):
        """Short entry (sell) should fill at close * (1 - slippage_pct)."""
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 95, 90, 90])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_d("0.001"),
        )
        result = engine.run(strategy, data)

        trade = result.trades[0]
        # Short entry: 100 * 0.999 = 99.90
        assert trade.entry_price == _d("99.9000")

    def test_short_exit_fills_higher(self):
        """Short exit (buy) should fill at close * (1 + slippage_pct)."""
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 95, 90, 90])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_d("0.001"),
        )
        result = engine.run(strategy, data)

        trade = result.trades[0]
        # Short exit: 90 * 1.001 = 90.09
        assert trade.exit_price == _d("90.0900")

    def test_slippage_reduces_long_profit(self):
        """Slippage should reduce profit on long trades vs. no-slippage."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])

        no_slip_engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_NO_SLIP,
        )
        slip_engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_d("0.001"),
        )

        no_slip_result = no_slip_engine.run(strategy, data)
        slip_result = slip_engine.run(strategy, data)

        # With slippage, final capital should be less
        assert slip_result.final_capital < no_slip_result.final_capital

    def test_slippage_reduces_short_profit(self):
        """Slippage should reduce profit on short trades vs. no-slippage."""
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 95, 90, 90])

        no_slip_engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_NO_SLIP,
        )
        slip_engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_d("0.001"),
        )

        no_slip_result = no_slip_engine.run(strategy, data)
        slip_result = slip_engine.run(strategy, data)

        assert slip_result.final_capital < no_slip_result.final_capital

    def test_zero_slippage_matches_original_behavior(self):
        """With slippage_pct=0, results should exactly match no-slippage behavior."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 120, 120])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_NO_SLIP,
        )
        result = engine.run(strategy, data)

        assert result.trades[0].entry_price == _d(100)
        assert result.trades[0].exit_price == _d(120)
        assert result.trades[0].shares == 50
        assert result.final_capital == _d(11000)

    def test_default_slippage_is_2_bps(self):
        """Default slippage should be 2 bps (0.0002)."""
        engine = IntradayBacktestEngine(initial_capital=_d(10000))
        assert engine._slippage_pct == _d("0.0002")

    def test_slippage_exact_long_final_capital(self):
        """Verify exact P/L with known slippage on a long trade.

        Entry: close=100, slippage=0.1% -> fill=100.10, shares=50
            (risk_amount=100, risk_per_share=2.00 -> 50 shares,
             max_affordable=int(10000/100.10)=99 -> min(50,99)=50)
            cash = 10000 - 50*100.10 = 10000 - 5005 = 4995
        Exit: close=100, slippage=0.1% -> fill=99.90
            cash = 4995 + 50*99.90 = 4995 + 4995 = 9990
        final_capital = 9990
        """
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 100, 100, 100])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_d("0.001"),
        )
        result = engine.run(strategy, data)

        trade = result.trades[0]
        assert trade.entry_price == _d("100.1000")
        assert trade.exit_price == _d("99.9000")
        assert trade.shares == 50
        # Flat price + slippage = round-trip cost of 50*(100.10-99.90) = 50*0.20 = 10.00
        assert result.final_capital == _d("9990.0000")

    def test_force_close_with_slippage(self):
        """Force-close at end of data should also apply slippage."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            # No exit -> force-close
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_d("0.001"),
        )
        result = engine.run(strategy, data)

        trade = result.trades[0]
        # Force-close is a sell for long -> fill = 110 * 0.999 = 109.89
        assert trade.exit_price == _d("109.8900")
