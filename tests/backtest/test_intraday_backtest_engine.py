"""Tests for IntradayBacktestEngine."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import IntradayAction, IntradaySignal, HOLD
from stockdownloader.model.trade import Direction, TradeStatus
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy

# All existing tests use slippage_pct=0 to verify core engine logic.
# Slippage-specific tests are in TestSlippage at the bottom.
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
# 1. Long position lifecycle
# ---------------------------------------------------------------------------

class TestLongPositionLifecycle:
    """Enter long, price goes up, exit -- verify P/L is positive."""

    def test_long_profit(self):
        # Enter at bar 1 (close=100), exit at bar 3 (close=110)
        strategy = _MockStrategy({
            1: _long_entry_signal(),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.direction == Direction.LONG
        assert trade.status == TradeStatus.CLOSED
        assert trade.entry_price == _d(100)
        assert trade.exit_price == _d(110)
        assert trade.profit_loss > _d(0)

    def test_long_cash_tracking(self):
        """Cash should decrease on entry and increase on exit."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        # risk_per_trade=0.01 of 10000 = 100.  100 / 2.00 = 50 shares.
        # 50 shares * 100 = 5000 notional.
        data = _make_bars([100, 100, 105, 120, 120])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        trade = result.trades[0]
        assert trade.shares == 50
        # final_capital = 10000 - 5000 (buy) + 50*120 (sell) = 11000
        assert result.final_capital == _d(11000)

    def test_long_equity_curve_length(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert len(result.equity_curve) == len(data)


# ---------------------------------------------------------------------------
# 2. Short position lifecycle
# ---------------------------------------------------------------------------

class TestShortPositionLifecycle:
    """Enter short, price goes down, exit -- verify P/L is positive."""

    def test_short_profit(self):
        # Enter short at bar 1 (close=100), exit at bar 3 (close=90)
        strategy = _MockStrategy({
            1: _short_entry_signal(),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 95, 90, 90])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.direction == Direction.SHORT
        assert trade.status == TradeStatus.CLOSED
        assert trade.entry_price == _d(100)
        assert trade.exit_price == _d(90)
        assert trade.profit_loss > _d(0)

    def test_short_cash_and_margin_tracking(self):
        """Short entry: cash += notional, margin_hold = notional.
        Short exit: cash -= exit_notional, margin_hold = 0."""
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        # risk_per_trade=0.01 of 10000 = 100.  100/2.00 = 50 shares.
        # Entry at 100: notional = 5000. cash = 10000+5000 = 15000, margin_hold = 5000
        # Exit at 90:  cash = 15000 - 50*90 = 15000-4500 = 10500, margin_hold = 0
        # final_capital = 10500 - 0 = 10500
        data = _make_bars([100, 100, 95, 90, 90])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        trade = result.trades[0]
        assert trade.shares == 50
        assert result.final_capital == _d(10500)


# ---------------------------------------------------------------------------
# 3. Short position loss
# ---------------------------------------------------------------------------

class TestShortPositionLoss:
    """Enter short, price goes up, exit -- verify P/L is negative."""

    def test_short_loss(self):
        # Enter short at bar 1 (close=100), exit at bar 3 (close=110)
        strategy = _MockStrategy({
            1: _short_entry_signal(),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.direction == Direction.SHORT
        assert trade.profit_loss < _d(0)
        assert result.final_capital < _d(10000)

    def test_short_loss_final_capital(self):
        """Verify exact final capital with a short loss."""
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        # 50 shares shorted at 100, bought back at 110
        # cash after entry: 10000 + 5000 = 15000, margin = 5000
        # cash after exit: 15000 - 5500 = 9500, margin = 0
        # final = 9500
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.final_capital == _d(9500)


# ---------------------------------------------------------------------------
# 4. Commission deducted
# ---------------------------------------------------------------------------

class TestCommission:
    """Commission should be deducted on both entry and exit."""

    def test_commission_deducted_on_entry_and_exit(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 100, 100])

        engine_no_comm = IntradayBacktestEngine(
            initial_capital=_d(10000), commission=_d(0), slippage_pct=_NO_SLIP,
        )
        engine_with_comm = IntradayBacktestEngine(
            initial_capital=_d(10000), commission=_d(5), slippage_pct=_NO_SLIP,
        )

        result_no_comm = engine_no_comm.run(strategy, data)
        result_with_comm = engine_with_comm.run(strategy, data)

        # Commission of 5 on entry + 5 on exit = 10 total reduction
        expected_diff = _d(10)
        assert result_no_comm.final_capital - result_with_comm.final_capital == expected_diff

    def test_commission_exact_final_capital_long(self):
        """Flat price, long entry and exit -- only commissions cause P/L."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        # 50 shares at 100, same exit price. No gain, just commission.
        data = _make_bars([100, 100, 100, 100, 100])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), commission=_d(7), slippage_pct=_NO_SLIP,
        )
        result = engine.run(strategy, data)

        # final = 10000 - 7 (entry comm) - 7 (exit comm) = 9986
        assert result.final_capital == _d(9986)

    def test_commission_exact_final_capital_short(self):
        """Flat price, short entry and exit -- only commissions cause P/L."""
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 100, 100, 100])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), commission=_d(7), slippage_pct=_NO_SLIP,
        )
        result = engine.run(strategy, data)

        # short at 100 x 50: cash += 5000 => 15000, then -7 comm => 14993, margin=5000
        # exit at 100 x 50: cash -= 5000 => 9993, then -7 comm => 9986, margin=0
        # final = 9986
        assert result.final_capital == _d(9986)


# ---------------------------------------------------------------------------
# 5. Equity curve during long position
# ---------------------------------------------------------------------------

class TestEquityCurveLong:
    """Equity should equal cash + position_value at each bar."""

    def test_equity_equals_cash_plus_position_value(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        prices = [100, 100, 105, 110, 110]
        data = _make_bars(prices)
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        # 50 shares at 100.  cash after entry = 5000.
        # Bar 0: no position.  equity = 10000
        # Bar 1: entry.  cash = 5000, position = 50 * 100 = 5000.  equity = 10000
        # Bar 2: holding.  cash = 5000, position = 50 * 105 = 5250.  equity = 10250
        # Bar 3: exit.  cash = 5000 + 50*110 = 10500, no position.  equity = 10500
        # Bar 4: flat.  equity = 10500
        expected = [_d(10000), _d(10000), _d(10250), _d(10500), _d(10500)]
        assert result.equity_curve == expected

    def test_equity_curve_monotonic_on_rising_prices(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            4: _exit_signal(),
        })
        prices = [100, 100, 102, 104, 106, 106]
        data = _make_bars(prices)
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        # Equity should be non-decreasing while holding long on rising prices
        for i in range(1, len(result.equity_curve)):
            assert result.equity_curve[i] >= result.equity_curve[i - 1]


# ---------------------------------------------------------------------------
# 6. Equity curve during short position
# ---------------------------------------------------------------------------

class TestEquityCurveShort:
    """Equity should correctly reflect unrealized P/L for short positions."""

    def test_equity_with_short_unrealized_pl(self):
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        prices = [100, 100, 95, 90, 90]
        data = _make_bars(prices)
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        # Bar 0: no position, equity = initial capital
        assert result.equity_curve[0] == _d(10000)

        # Final capital is correct: 50 shares shorted at 100, closed at 90
        assert result.final_capital == _d(10500)

        # After exit (bars 3 and 4), equity matches final capital
        assert result.equity_curve[3] == _d(10500)
        assert result.equity_curve[4] == _d(10500)

    def test_short_equity_increases_as_price_drops(self):
        """When holding short, equity should increase as price drops."""
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            4: _exit_signal(),
        })
        prices = [100, 100, 98, 96, 94, 94]
        data = _make_bars(prices)
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        # While holding short (bars 1-3), equity should be non-decreasing
        # as the price drops
        for i in range(2, 4):
            assert result.equity_curve[i] >= result.equity_curve[i - 1], (
                f"Equity should increase as price drops: bar {i}"
            )


# ---------------------------------------------------------------------------
# 7. Force-close at end
# ---------------------------------------------------------------------------

class TestForceCloseAtEnd:
    """If a position is still open when data ends, it must be force-closed."""

    def test_long_force_closed(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            # No exit signal -- position stays open
        })
        data = _make_bars([100, 100, 105, 110, 115])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.status == TradeStatus.CLOSED
        assert trade.exit_price == _d(115)  # last bar's close

    def test_short_force_closed(self):
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
        })
        data = _make_bars([100, 100, 95, 90, 85])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.status == TradeStatus.CLOSED
        assert trade.exit_price == _d(85)

    def test_force_close_final_capital_long(self):
        """Final capital should reflect the force-close price."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
        })
        # 50 shares bought at 100, force-closed at 120
        data = _make_bars([100, 100, 110, 115, 120])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        # final = 10000 - 5000 + 50*120 = 10000 - 5000 + 6000 = 11000
        assert result.final_capital == _d(11000)

    def test_force_close_final_capital_short(self):
        """Final capital should reflect the force-close for short."""
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
        })
        # 50 shares shorted at 100, force-closed at 80
        data = _make_bars([100, 100, 90, 85, 80])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        # cash after entry: 15000, margin=5000
        # force-close: cash = 15000 - 50*80 = 15000 - 4000 = 11000, margin=0
        # final = 11000
        assert result.final_capital == _d(11000)

    def test_force_close_with_commission(self):
        """Force-close should also deduct commission."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
        })
        data = _make_bars([100, 100, 100, 100, 100])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), commission=_d(5), slippage_pct=_NO_SLIP,
        )
        result = engine.run(strategy, data)

        # 50 shares at 100. Buy: -5000 -5 comm. Force-close sell: +5000 -5 comm.
        # final = 10000 - 5 - 5 = 9990
        assert result.final_capital == _d(9990)


# ---------------------------------------------------------------------------
# 8. No double entry
# ---------------------------------------------------------------------------

class TestNoDoubleEntry:
    """If already in a position, additional entry signals should be ignored."""

    def test_second_long_entry_ignored(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            2: _long_entry_signal(risk_per_share="2.00"),  # should be ignored
            4: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 115, 115])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.entry_price == _d(100)

    def test_short_entry_while_long_ignored(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            2: _short_entry_signal(risk_per_share="2.00"),  # should be ignored
            4: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 115, 115])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.direction == Direction.LONG

    def test_long_entry_while_short_ignored(self):
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            2: _long_entry_signal(risk_per_share="2.00"),  # should be ignored
            4: _exit_signal(),
        })
        data = _make_bars([100, 100, 95, 90, 85, 85])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.direction == Direction.SHORT


# ---------------------------------------------------------------------------
# 9. Position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:
    """Shares should be computed correctly from risk_per_trade and risk_per_share."""

    def test_basic_position_sizing(self):
        # capital=10000, risk_per_trade=0.01, risk_amount=100
        # risk_per_share=2.00, shares=100/2=50
        # max_affordable=10000/100=100, min(50,100)=50
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 100, 100, 100])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.trades[0].shares == 50

    def test_position_sizing_capped_by_buying_power(self):
        """Shares should be capped by what the buying power can afford."""
        # capital=1000, risk_per_trade=0.01, risk_amount=10
        # risk_per_share=0.01, shares=10/0.01=1000
        # max_affordable=1000/100=10, min(1000,10)=10
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="0.01"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 100, 100, 100])
        engine = IntradayBacktestEngine(initial_capital=_d(1000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.trades[0].shares == 10

    def test_position_sizing_with_different_risk_per_trade(self):
        # capital=10000, risk_per_trade=0.02, risk_amount=200
        # risk_per_share=4.00, shares=200/4=50
        # max_affordable=10000/100=100, min(50,100)=50
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="4.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 100, 100, 100])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), risk_per_trade=_d("0.02"), slippage_pct=_NO_SLIP,
        )
        result = engine.run(strategy, data)

        assert result.trades[0].shares == 50

    def test_position_sizing_fractional_truncated(self):
        """Fractional shares should be truncated to int."""
        # capital=10000, risk_per_trade=0.01, risk_amount=100
        # risk_per_share=3.00, shares=int(100/3)=33
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="3.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 100, 100, 100])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.trades[0].shares == 33


# ---------------------------------------------------------------------------
# 10. Zero/negative risk_per_share
# ---------------------------------------------------------------------------

class TestZeroNegativeRiskPerShare:
    """When risk_per_share is zero or negative, no trade should be entered."""

    def test_zero_risk_per_share_no_trade(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="0"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 0
        assert result.final_capital == _d(10000)

    def test_negative_risk_per_share_no_trade(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="-1.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 0
        assert result.final_capital == _d(10000)

    def test_zero_risk_per_share_short_no_trade(self):
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="0"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 95, 90, 90])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 0
        assert result.final_capital == _d(10000)


# ---------------------------------------------------------------------------
# 11. Validation
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
# 12. Initial capital preservation
# ---------------------------------------------------------------------------

class TestInitialCapitalPreservation:
    """The initial_capital field on BacktestResult should remain unchanged."""

    def test_initial_capital_unchanged_after_long_trade(self):
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.initial_capital == _d(10000)
        assert result.final_capital != result.initial_capital  # trade moved capital

    def test_initial_capital_unchanged_after_short_trade(self):
        strategy = _MockStrategy({
            1: _short_entry_signal(risk_per_share="2.00"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 95, 90, 90])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.initial_capital == _d(10000)

    def test_initial_capital_unchanged_no_trades(self):
        strategy = _MockStrategy({})  # no signals at all
        data = _make_bars([100, 100, 100, 100, 100])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.initial_capital == _d(10000)
        assert result.final_capital == _d(10000)


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_exit_without_position_is_noop(self):
        """EXIT signal when no position is open should be ignored."""
        strategy = _MockStrategy({
            1: _exit_signal(),
            2: _exit_signal(),
        })
        data = _make_bars([100, 100, 100, 100])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 0
        assert result.final_capital == _d(10000)

    def test_hold_signals_do_nothing(self):
        """All HOLD signals should produce no trades."""
        strategy = _MockStrategy({})  # all HOLD
        data = _make_bars([100, 102, 104, 106])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 0
        assert result.final_capital == _d(10000)
        assert all(eq == _d(10000) for eq in result.equity_curve)

    def test_strategy_name_propagated(self):
        strategy = _MockStrategy({}, name="MyIntraday")
        data = _make_bars([100, 100])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.strategy_name == "MyIntraday"

    def test_start_and_end_dates_set(self):
        data = _make_bars([100, 100, 100])
        strategy = _MockStrategy({})
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.start_date == data[0].date
        assert result.end_date == data[-1].date

    def test_trade_mode_recorded(self):
        """The signal mode should be stored in result.trade_modes."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00", mode="breakout"),
            3: _exit_signal(),
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert hasattr(result, "trade_modes")
        assert result.trade_modes == ["breakout"]

    def test_multiple_trades_sequential(self):
        """Engine should support multiple sequential trades."""
        strategy = _MockStrategy({
            1: _long_entry_signal(risk_per_share="2.00"),
            2: _exit_signal(),
            4: _short_entry_signal(risk_per_share="2.00"),
            6: _exit_signal(),
        })
        data = _make_bars([100, 100, 110, 110, 110, 100, 90, 90])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 2
        assert result.trades[0].direction == Direction.LONG
        assert result.trades[1].direction == Direction.SHORT

    def test_single_bar_entry_and_force_close(self):
        """Entry on the last bar should be force-closed at the same bar."""
        strategy = _MockStrategy({
            0: _long_entry_signal(risk_per_share="2.00"),
        })
        data = _make_bars([100])
        engine = IntradayBacktestEngine(initial_capital=_d(10000), slippage_pct=_NO_SLIP)
        result = engine.run(strategy, data)

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.status == TradeStatus.CLOSED
        assert trade.entry_price == _d(100)
        assert trade.exit_price == _d(100)


# ---------------------------------------------------------------------------
# 13. Slippage model
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

        Entry: close=100, slippage=0.1% → fill=100.10, shares=50
            (risk_amount=100, risk_per_share=2.00 → 50 shares,
             max_affordable=int(10000/100.10)=99 → min(50,99)=50)
            cash = 10000 - 50*100.10 = 10000 - 5005 = 4995
        Exit: close=100, slippage=0.1% → fill=99.90
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
            # No exit → force-close
        })
        data = _make_bars([100, 100, 105, 110, 110])
        engine = IntradayBacktestEngine(
            initial_capital=_d(10000), slippage_pct=_d("0.001"),
        )
        result = engine.run(strategy, data)

        trade = result.trades[0]
        # Force-close is a sell for long → fill = 110 * 0.999 = 109.89
        assert trade.exit_price == _d("109.8900")
