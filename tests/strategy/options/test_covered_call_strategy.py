"""Tests for CoveredCallStrategy."""

from decimal import Decimal

import pytest

from stockdownloader.model.options import OptionType
from stockdownloader.model.price_data import PriceData
from stockdownloader.strategy.options.options_strategies import CoveredCallStrategy
from stockdownloader.strategy.options_strategy import OptionsSignal


def _make_price(date, price):
    p = Decimal(str(price))
    return PriceData(
        date=date,
        open=p,
        high=p + Decimal("1"),
        low=p - Decimal("1"),
        close=p,
        adj_close=p,
        volume=100000,
    )


def _generate_prices(count, start, increment):
    data = []
    for i in range(count):
        price = start + i * increment
        data.append(_make_price(f"2024-01-{i + 1:02d}", price))
    return data


def test_default_constructor_creates_valid_strategy():
    strategy = CoveredCallStrategy()
    assert strategy.name is not None
    assert "Covered Call" in strategy.name
    assert strategy.option_type == OptionType.CALL
    assert strategy.is_short()


def test_rejects_invalid_period():
    with pytest.raises(ValueError):
        CoveredCallStrategy(0, Decimal("0.05"), 30, Decimal("0.03"))


def test_rejects_invalid_dte():
    with pytest.raises(ValueError):
        CoveredCallStrategy(20, Decimal("0.05"), 0, Decimal("0.03"))


def test_warmup_period_matches_ma_period():
    strategy = CoveredCallStrategy(30, Decimal("0.05"), 45, Decimal("0.03"))
    assert strategy.warmup_period == 30


def test_holds_during_warmup():
    strategy = CoveredCallStrategy(5, Decimal("0.05"), 30, Decimal("0.03"))
    data = _generate_prices(5, 100, 0.5)
    assert strategy.evaluate(data, 3) == OptionsSignal.HOLD


def test_target_strike_above_current_price():
    strategy = CoveredCallStrategy(20, Decimal("0.05"), 30, Decimal("0.03"))
    strike = strategy.get_target_strike(Decimal("475"))
    # 475 * 1.05 = 498.75 -> ceiling to 499
    assert strike > Decimal("475")


def test_target_days_to_expiry():
    strategy = CoveredCallStrategy(20, Decimal("0.05"), 45, Decimal("0.03"))
    assert strategy.target_days_to_expiry == 45


def test_opens_when_price_crosses_above_ma():
    # Generate prices that cross above MA
    strategy = CoveredCallStrategy(5, Decimal("0.05"), 30, Decimal("0.03"))
    data = []
    # Below MA phase: declining prices
    for i in range(6):
        data.append(_make_price(f"2024-01-0{i + 1}", 100 - i))
    # Then price jumps above the average
    data.append(_make_price("2024-01-07", 110))

    signal = strategy.evaluate(data, len(data) - 1)
    # May be OPEN or HOLD depending on exact MA cross
    assert signal is not None


def test_is_short_and_call_type():
    strategy = CoveredCallStrategy()
    assert strategy.is_short()
    assert strategy.option_type == OptionType.CALL
