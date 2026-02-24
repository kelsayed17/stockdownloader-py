"""Tests for SMACrossoverStrategy."""

from decimal import Decimal

import pytest

from stockdownloader.core.models.price import PriceData
from stockdownloader.strategy.daily.simple_strategies import SMACrossoverStrategy
from stockdownloader.strategy.trading_strategy import Signal


def _make_price_data(date, close):
    p = Decimal(str(close))
    return PriceData(date=date, open=p, high=p, low=p, close=p, adj_close=p, volume=1000)


def _generate_price_data(count, start_price, increment):
    data = []
    for i in range(count):
        price = start_price + i * increment
        data.append(_make_price_data(f"2024-01-{i + 1:02d}", price))
    return data


def test_construction_with_valid_periods():
    strategy = SMACrossoverStrategy(10, 20)
    assert strategy.name == "SMA Crossover (10/20)"
    assert strategy.warmup_period == 20


def test_short_period_greater_than_long_throws():
    with pytest.raises(ValueError):
        SMACrossoverStrategy(20, 10)


def test_equal_periods_throws():
    with pytest.raises(ValueError):
        SMACrossoverStrategy(10, 10)


def test_zero_period_throws():
    with pytest.raises(ValueError):
        SMACrossoverStrategy(0, 10)


def test_negative_period_throws():
    with pytest.raises(ValueError):
        SMACrossoverStrategy(-5, 10)


def test_hold_during_warmup_period():
    strategy = SMACrossoverStrategy(2, 5)
    data = _generate_price_data(10, 100, 1)

    for i in range(5):
        assert strategy.evaluate(data, i) == Signal.HOLD


def test_buy_signal_on_golden_cross():
    strategy = SMACrossoverStrategy(3, 5)

    # Create data where short SMA crosses above long SMA
    data = []
    # Declining prices (short < long)
    for i in range(10):
        data.append(_make_price_data(f"day{i}", 100 - i))
    # Sharp upturn
    for i in range(10, 20):
        data.append(_make_price_data(f"day{i}", 90 + (i - 10) * 5))

    # Find the first BUY signal after warmup
    found_buy = False
    for i in range(5, len(data)):
        if strategy.evaluate(data, i) == Signal.BUY:
            found_buy = True
            break
    assert found_buy, "Should generate a BUY signal on golden cross"


def test_sell_signal_on_death_cross():
    strategy = SMACrossoverStrategy(3, 5)

    # Create data where short SMA crosses below long SMA
    data = []
    # Rising prices (short > long)
    for i in range(10):
        data.append(_make_price_data(f"day{i}", 100 + i * 3))
    # Sharp downturn
    for i in range(10, 20):
        data.append(_make_price_data(f"day{i}", 130 - (i - 10) * 5))

    found_sell = False
    for i in range(5, len(data)):
        if strategy.evaluate(data, i) == Signal.SELL:
            found_sell = True
            break
    assert found_sell, "Should generate a SELL signal on death cross"


def test_hold_when_no_signal_crossover():
    strategy = SMACrossoverStrategy(2, 3)
    # Flat prices -> no crossover
    data = _generate_price_data(10, 100, 0)

    for i in range(3, len(data)):
        assert strategy.evaluate(data, i) == Signal.HOLD
