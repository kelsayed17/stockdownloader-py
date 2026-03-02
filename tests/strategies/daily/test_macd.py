"""Tests for MACDStrategy."""

from decimal import Decimal

import pytest

from stockdownloader.core.models.price import PriceData
from stockdownloader.strategies.daily.simple import MACDStrategy
from stockdownloader.strategies.base import Signal


def _make_price_data(date, close):
    p = Decimal(str(max(1, close)))
    return PriceData(date=date, open=p, high=p, low=p, close=p, adj_close=p, volume=1000)


def _generate_price_data(count, start_price, increment):
    data = []
    for i in range(count):
        price = max(1, start_price + i * increment)
        data.append(_make_price_data(f"2024-01-{i + 1:02d}", price))
    return data


def test_construction_with_valid_params():
    strategy = MACDStrategy(12, 26, 9)
    assert strategy.name == "MACD (12/26/9)"
    assert strategy.warmup_period == 35


def test_fast_greater_than_slow_throws():
    with pytest.raises(ValueError):
        MACDStrategy(26, 12, 9)


def test_equal_fast_and_slow_throws():
    with pytest.raises(ValueError):
        MACDStrategy(12, 12, 9)


def test_zero_period_throws():
    with pytest.raises(ValueError):
        MACDStrategy(0, 26, 9)


def test_negative_period_throws():
    with pytest.raises(ValueError):
        MACDStrategy(-1, 26, 9)


def test_hold_during_warmup_period():
    strategy = MACDStrategy(12, 26, 9)
    data = _generate_price_data(50, 100, 0.5)

    for i in range(35):
        assert strategy.evaluate(data, i) == Signal.HOLD


def test_produces_signals_after_warmup():
    strategy = MACDStrategy(3, 7, 3)

    data = []
    # Declining phase
    for i in range(20):
        data.append(_make_price_data(f"day{i}", 100 - i * 2))
    # Sharp recovery
    for i in range(20, 40):
        data.append(_make_price_data(f"day{i}", 60 + (i - 20) * 4))

    found_signal = False
    for i in range(10, len(data)):
        signal = strategy.evaluate(data, i)
        if signal != Signal.HOLD:
            found_signal = True
            break
    assert found_signal, "Should produce BUY or SELL signal after warmup with trend reversal"


def test_hold_on_flat_prices():
    strategy = MACDStrategy(3, 7, 3)
    data = _generate_price_data(30, 100, 0)

    for i in range(10, len(data)):
        assert strategy.evaluate(data, i) == Signal.HOLD


def test_buy_signal_on_bullish_crossover():
    strategy = MACDStrategy(3, 7, 3)

    data = []
    # Decline
    for i in range(15):
        data.append(_make_price_data(f"day{i}", 100 - i * 3))
    # Strong reversal
    for i in range(15, 30):
        data.append(_make_price_data(f"day{i}", 55 + (i - 15) * 5))

    found_buy = False
    for i in range(10, len(data)):
        if strategy.evaluate(data, i) == Signal.BUY:
            found_buy = True
            break
    assert found_buy, "Should produce BUY on bullish MACD crossover"
