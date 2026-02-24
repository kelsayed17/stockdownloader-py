"""Tests for ProtectivePutStrategy."""

from decimal import Decimal

import pytest

from stockdownloader.core.models.options import OptionType
from stockdownloader.core.models.price import PriceData
from stockdownloader.strategy.options.options_strategies import OptionsSignal
from stockdownloader.strategy.options.options_strategies import ProtectivePutStrategy


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
    strategy = ProtectivePutStrategy()
    assert strategy.name is not None
    assert "Protective Put" in strategy.name
    assert strategy.option_type == OptionType.PUT
    assert not strategy.is_short()


def test_rejects_invalid_period():
    with pytest.raises(ValueError):
        ProtectivePutStrategy(0, Decimal("0.05"), 30, 5)


def test_rejects_invalid_dte():
    with pytest.raises(ValueError):
        ProtectivePutStrategy(20, Decimal("0.05"), 0, 5)


def test_rejects_invalid_momentum_lookback():
    with pytest.raises(ValueError):
        ProtectivePutStrategy(20, Decimal("0.05"), 30, 0)


def test_warmup_period_is_max_of_ma_and_momentum():
    strategy = ProtectivePutStrategy(20, Decimal("0.05"), 30, 10)
    assert strategy.warmup_period == 20

    strategy2 = ProtectivePutStrategy(5, Decimal("0.05"), 30, 10)
    assert strategy2.warmup_period == 10


def test_holds_during_warmup():
    strategy = ProtectivePutStrategy(5, Decimal("0.05"), 30, 3)
    data = _generate_prices(5, 100, -0.5)
    assert strategy.evaluate(data, 3) == OptionsSignal.HOLD


def test_target_strike_below_current_price():
    strategy = ProtectivePutStrategy(20, Decimal("0.05"), 45, 5)
    strike = strategy.get_target_strike(Decimal("475"))
    # 475 * 0.95 = 451.25 -> floor to 451
    assert strike < Decimal("475")


def test_target_days_to_expiry():
    strategy = ProtectivePutStrategy(20, Decimal("0.05"), 60, 5)
    assert strategy.target_days_to_expiry == 60


def test_is_long_put_type():
    strategy = ProtectivePutStrategy()
    assert not strategy.is_short()
    assert strategy.option_type == OptionType.PUT


def test_signal_on_decline():
    strategy = ProtectivePutStrategy(5, Decimal("0.05"), 30, 3)
    data = []
    # Start above MA
    for i in range(6):
        data.append(_make_price(f"2024-01-{i + 1:02d}", 100 + i))
    # Then drop below MA
    data.append(_make_price("2024-01-07", 90))

    signal = strategy.evaluate(data, len(data) - 1)
    assert signal is not None


def test_name_contains_parameters():
    strategy = ProtectivePutStrategy(20, Decimal("0.05"), 45, 5)
    assert "MA20" in strategy.name
    assert "45DTE" in strategy.name
