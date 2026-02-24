"""Tests for RSIStrategy."""

from decimal import Decimal

import pytest

from stockdownloader.core.models.price import PriceData
from stockdownloader.strategy.daily.simple_strategies import RSIStrategy
from stockdownloader.strategy.trading_strategy import Signal


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
    strategy = RSIStrategy(14, 30, 70)
    assert strategy.name == "RSI (14) [30/70]"
    assert strategy.warmup_period == 15


def test_zero_period_throws():
    with pytest.raises(ValueError):
        RSIStrategy(0, 30, 70)


def test_oversold_greater_than_overbought_throws():
    with pytest.raises(ValueError):
        RSIStrategy(14, 80, 70)


def test_equal_thresholds_throws():
    with pytest.raises(ValueError):
        RSIStrategy(14, 50, 50)


def test_negative_oversold_throws():
    with pytest.raises(ValueError):
        RSIStrategy(14, -10, 70)


def test_overbought_above_100_throws():
    with pytest.raises(ValueError):
        RSIStrategy(14, 30, 101)


def test_hold_during_warmup_period():
    strategy = RSIStrategy(14, 30, 70)
    data = _generate_price_data(20, 100, 1)

    for i in range(15):
        assert strategy.evaluate(data, i) == Signal.HOLD


def test_buy_signal_after_oversold_recovery():
    strategy = RSIStrategy(5, 30, 70)

    # Create sharp decline followed by recovery (RSI goes from oversold back up)
    data = []
    for i in range(8):
        data.append(_make_price_data(f"day{i}", 100 - i * 8))  # Sharp decline
    for i in range(8, 15):
        data.append(_make_price_data(f"day{i}", 36 + (i - 8) * 10))  # Recovery

    found_buy = False
    for i in range(6, len(data)):
        if strategy.evaluate(data, i) == Signal.BUY:
            found_buy = True
            break
    assert found_buy, "Should generate BUY when RSI crosses above oversold threshold"


def test_hold_on_flat_prices():
    strategy = RSIStrategy(5, 30, 70)
    data = _generate_price_data(20, 100, 0)

    for i in range(6, len(data)):
        assert strategy.evaluate(data, i) == Signal.HOLD


def test_rsi_max_when_all_gains():
    strategy = RSIStrategy(5, 30, 70)
    # Steady upward movement - RSI should be very high
    data = _generate_price_data(20, 50, 5)

    # After warmup, with all gains, RSI should be 100
    # So no BUY signal (above oversold), might generate SELL if crossing overbought
    any_buy = False
    for i in range(6, len(data)):
        if strategy.evaluate(data, i) == Signal.BUY:
            any_buy = True
    assert not any_buy, "No BUY signal when RSI is consistently high (all gains)"
