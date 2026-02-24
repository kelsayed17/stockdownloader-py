"""Tests for MomentumConfluenceStrategy."""

import random
from decimal import Decimal

from stockdownloader.core.models.price import PriceData
from stockdownloader.strategy.daily.momentum_confluence_strategy import MomentumConfluenceStrategy
from stockdownloader.strategy.trading_strategy import Signal

random.seed(42)


def _generate_test_data(days):
    data = []
    price = 100.0
    for i in range(days):
        price += (random.random() - 0.48) * 3
        price = max(50, price)
        data.append(
            PriceData(
                date="2020-01-01",
                open=Decimal(str(price - 1)),
                high=Decimal(str(price + 2)),
                low=Decimal(str(price - 2)),
                close=Decimal(str(price)),
                adj_close=Decimal(str(price)),
                volume=int(1_000_000 + random.random() * 5_000_000),
            )
        )
    return data


def test_constructor_default_values():
    strategy = MomentumConfluenceStrategy()
    assert "Momentum" in strategy.name
    assert "MACD" in strategy.name


def test_evaluate_hold_during_warmup():
    strategy = MomentumConfluenceStrategy()
    data = _generate_test_data(300)
    assert strategy.evaluate(data, 10) == Signal.HOLD


def test_evaluate_handles_enough_data():
    strategy = MomentumConfluenceStrategy()
    data = _generate_test_data(300)
    signal = strategy.evaluate(data, 250)
    assert signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


def test_warmup_period_covers_all_indicators():
    strategy = MomentumConfluenceStrategy()
    assert strategy.warmup_period >= 200, "Warmup should cover EMA(200) requirement"
