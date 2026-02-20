"""Tests for BacktestEngine."""

from decimal import Decimal

import pytest

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.trade import TradeStatus
from stockdownloader.strategy.daily.sma_crossover_strategy import SMACrossoverStrategy
from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy


def _make_price_data(date, close):
    p = Decimal(str(close))
    return PriceData(date=date, open=p, high=p, low=p, close=p, adj_close=p, volume=1000)


def _generate_flat_price_data(count, price):
    return [_make_price_data(f"day{i}", price) for i in range(count)]


class _TestStrategy(TradingStrategy):
    """Simple test strategy for backtesting.

    Generates BUY at index warmup+1 and SELL at warmup+2.
    """

    def __init__(self, name, warmup):
        self._name = name
        self._warmup = warmup

    @property
    def name(self):
        return self._name

    def evaluate(self, data, current_index):
        if current_index == self._warmup + 1:
            return Signal.BUY
        if current_index == self._warmup + 2:
            return Signal.SELL
        return Signal.HOLD

    @property
    def warmup_period(self):
        return self._warmup


class _BuyOnlyStrategy(TradingStrategy):
    """Strategy that only buys at index 2 and never sells."""

    @property
    def name(self):
        return "Buy-Only"

    def evaluate(self, data, current_index):
        return Signal.BUY if current_index == 2 else Signal.HOLD

    @property
    def warmup_period(self):
        return 0


def test_run_with_null_strategy_throws():
    engine = BacktestEngine(Decimal("10000"), Decimal("0"))
    data = [_make_price_data("d1", 100)]
    with pytest.raises((ValueError, TypeError, AttributeError)):
        engine.run(None, data)


def test_run_with_null_data_throws():
    engine = BacktestEngine(Decimal("10000"), Decimal("0"))
    strategy = SMACrossoverStrategy(2, 5)
    with pytest.raises((ValueError, TypeError)):
        engine.run(strategy, None)


def test_run_with_empty_data_throws():
    engine = BacktestEngine(Decimal("10000"), Decimal("0"))
    strategy = SMACrossoverStrategy(2, 5)
    with pytest.raises(ValueError):
        engine.run(strategy, [])


def test_result_has_correct_strategy_name():
    engine = BacktestEngine(Decimal("10000"), Decimal("0"))
    strategy = SMACrossoverStrategy(2, 5)
    data = _generate_flat_price_data(10, 100)
    result = engine.run(strategy, data)
    assert result.strategy_name == "SMA Crossover (2/5)"


def test_no_trades_on_flat_prices():
    engine = BacktestEngine(Decimal("10000"), Decimal("0"))
    strategy = SMACrossoverStrategy(2, 5)
    data = _generate_flat_price_data(20, 100)
    result = engine.run(strategy, data)
    assert result.total_trades == 0
    assert Decimal("10000").compare(result.final_capital) == 0


def test_equity_curve_matches_data_size():
    engine = BacktestEngine(Decimal("10000"), Decimal("0"))
    strategy = SMACrossoverStrategy(2, 5)
    data = _generate_flat_price_data(20, 100)
    result = engine.run(strategy, data)
    assert len(result.equity_curve) == len(data)


def test_start_and_end_dates_are_set():
    engine = BacktestEngine(Decimal("10000"), Decimal("0"))
    strategy = SMACrossoverStrategy(2, 5)
    data = _generate_flat_price_data(10, 100)
    result = engine.run(strategy, data)
    assert result.start_date is not None
    assert result.end_date is not None
    assert result.start_date == data[0].date
    assert result.end_date == data[-1].date


def test_commission_reduces_profit():
    strategy = _TestStrategy("Buy-Sell", 2)
    engine_no_comm = BacktestEngine(Decimal("10000"), Decimal("0"))
    engine_with_comm = BacktestEngine(Decimal("10000"), Decimal("10"))

    data = [
        _make_price_data("d1", 50),
        _make_price_data("d2", 50),  # warmup
        _make_price_data("d3", 50),  # BUY
        _make_price_data("d4", 60),  # SELL
        _make_price_data("d5", 60),
    ]

    result_no_comm = engine_no_comm.run(strategy, data)
    result_with_comm = engine_with_comm.run(strategy, data)

    assert result_no_comm.final_capital > result_with_comm.final_capital, (
        "Commission should reduce final capital"
    )


def test_open_position_closed_at_end():
    strategy = _BuyOnlyStrategy()
    engine = BacktestEngine(Decimal("10000"), Decimal("0"))
    data = [
        _make_price_data("d1", 100),
        _make_price_data("d2", 100),
        _make_price_data("d3", 100),  # BUY here
        _make_price_data("d4", 110),
        _make_price_data("d5", 120),  # Force-closed here
    ]

    result = engine.run(strategy, data)
    assert result.total_trades == 1
    for t in result.trades:
        assert t.status == TradeStatus.CLOSED


def test_slippage_reduces_profit():
    """Slippage should worsen the fill price and reduce final capital."""
    strategy = _TestStrategy("Buy-Sell", 2)
    data = [
        _make_price_data("d1", 50),
        _make_price_data("d2", 50),
        _make_price_data("d3", 50),  # BUY
        _make_price_data("d4", 60),  # SELL
        _make_price_data("d5", 60),
    ]

    engine_no_slip = BacktestEngine(Decimal("10000"), Decimal("0"), slippage_pct=0)
    engine_with_slip = BacktestEngine(
        Decimal("10000"), Decimal("0"), slippage_pct=Decimal("0.01"),
    )

    result_no_slip = engine_no_slip.run(strategy, data)
    result_with_slip = engine_with_slip.run(strategy, data)

    assert result_no_slip.final_capital > result_with_slip.final_capital, (
        "Slippage should reduce final capital"
    )


def test_slippage_affects_entry_price():
    """Buy price should be higher than bar close when slippage is applied."""
    strategy = _TestStrategy("Buy-Sell", 0)  # warmup=0 → BUY at 1, SELL at 2
    data = [
        _make_price_data("d1", 100),
        _make_price_data("d2", 100),  # BUY at 100 * (1 + 0.01) = 101
        _make_price_data("d3", 110),  # SELL at 110 * (1 - 0.01) = 108.9
        _make_price_data("d4", 110),
    ]

    engine = BacktestEngine(
        Decimal("10000"), Decimal("0"), slippage_pct=Decimal("0.01"),
    )
    result = engine.run(strategy, data)

    assert result.total_trades == 1
    trade = result.trades[0]
    # Entry price should be 101 (100 * 1.01), not 100
    assert trade.entry_price == Decimal("101.00")
    # Exit price should be 108.9 (110 * 0.99), not 110
    assert trade.exit_price == Decimal("108.90")
