"""Integration test verifying that strategies produce correct signal sequences
when evaluated over real price data loaded from CSV.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.data.data_parsers import CsvPriceDataLoader
from stockdownloader.core.models.price import PriceData
from stockdownloader.strategy.daily.simple_strategies import MACDStrategy
from stockdownloader.strategy.daily.simple_strategies import RSIStrategy
from stockdownloader.strategy.daily.simple_strategies import SMACrossoverStrategy
from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy


@pytest.fixture(scope="module")
def price_data(test_data_path):
    data = CsvPriceDataLoader.load_from_file(str(test_data_path))
    assert len(data) > 0
    return data


def _collect_signals(strategy: TradingStrategy, data: list[PriceData]) -> list[Signal]:
    return [strategy.evaluate(data, i) for i in range(len(data))]


def test_sma_crossover_respects_warmup_period(price_data):
    strategy = SMACrossoverStrategy(20, 50)

    # During warmup period, all signals should be HOLD
    for i in range(strategy.warmup_period):
        assert strategy.evaluate(price_data, i) == Signal.HOLD, \
            f"Signal at index {i} should be HOLD during warmup"


def test_sma_crossover_generates_signals_after_warmup(price_data):
    strategy = SMACrossoverStrategy(20, 50)
    signals = _collect_signals(strategy, price_data)

    # Should have HOLD signals during warmup and possibly BUY/SELL after
    hold_count = sum(1 for s in signals if s == Signal.HOLD)
    assert hold_count >= strategy.warmup_period, \
        "Should have at least warmup period number of HOLD signals"

    # Total signals should match data size
    assert len(signals) == len(price_data)


def test_sma_crossover_buy_always_precedes_sell(price_data):
    strategy = SMACrossoverStrategy(20, 50)
    signals = _collect_signals(strategy, price_data)

    # Filter out HOLDs; the sequence of BUY/SELL should alternate
    action_signals = [s for s in signals if s != Signal.HOLD]

    if action_signals:
        for i in range(len(action_signals) - 1):
            assert action_signals[i] != action_signals[i + 1], \
                f"BUY and SELL signals should alternate at index {i}"


def test_rsi_respects_warmup_period(price_data):
    strategy = RSIStrategy(14, 30.0, 70.0)

    for i in range(strategy.warmup_period):
        assert strategy.evaluate(price_data, i) == Signal.HOLD, \
            f"RSI signal at index {i} should be HOLD during warmup"


def test_rsi_generates_valid_signals(price_data):
    strategy = RSIStrategy(14, 30.0, 70.0)
    signals = _collect_signals(strategy, price_data)

    assert len(signals) == len(price_data)

    # All signals must be one of BUY/SELL/HOLD
    for signal in signals:
        assert signal is not None
        assert signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


def test_rsi_narrow_thresholds_generate_more_signals(price_data):
    wide_thresholds = RSIStrategy(14, 30.0, 70.0)
    narrow_thresholds = RSIStrategy(14, 40.0, 60.0)

    wide_signals = sum(
        1 for s in _collect_signals(wide_thresholds, price_data) if s != Signal.HOLD
    )
    narrow_signals = sum(
        1 for s in _collect_signals(narrow_thresholds, price_data) if s != Signal.HOLD
    )

    # Narrower thresholds should generally produce more or equal signals
    assert narrow_signals >= wide_signals, \
        "Narrower RSI thresholds should produce more or equal trading signals"


def test_macd_respects_warmup_period(price_data):
    strategy = MACDStrategy(12, 26, 9)
    assert strategy.warmup_period == 35, "MACD warmup should be slow + signal"

    for i in range(strategy.warmup_period):
        assert strategy.evaluate(price_data, i) == Signal.HOLD, \
            f"MACD signal at index {i} should be HOLD during warmup"


def test_macd_generates_signals_after_warmup(price_data):
    strategy = MACDStrategy(12, 26, 9)
    signals = _collect_signals(strategy, price_data)

    assert len(signals) == len(price_data)

    # After warmup, there should be some non-HOLD signals over a full year of data
    action_signals = sum(1 for s in signals if s != Signal.HOLD)
    assert action_signals > 0, \
        "MACD should generate at least one trading signal over a year of data"


def test_macd_buy_and_sell_signals_alternate(price_data):
    strategy = MACDStrategy(12, 26, 9)
    signals = _collect_signals(strategy, price_data)

    action_signals = [s for s in signals if s != Signal.HOLD]

    for i in range(len(action_signals) - 1):
        assert action_signals[i] != action_signals[i + 1], \
            f"MACD BUY/SELL signals should alternate at index {i}"


def test_different_sma_periods_different_signals(price_data):
    short_periods = SMACrossoverStrategy(5, 20)
    long_periods = SMACrossoverStrategy(50, 200)

    short_signals = _collect_signals(short_periods, price_data)
    long_signals = _collect_signals(long_periods, price_data)

    # Short periods should generate more signals
    short_actions = sum(1 for s in short_signals if s != Signal.HOLD)
    long_actions = sum(1 for s in long_signals if s != Signal.HOLD)

    assert short_actions >= long_actions, \
        "Shorter SMA periods should generate more or equal trading signals"
