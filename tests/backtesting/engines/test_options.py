"""Tests for OptionsBacktestEngine."""

import math
from decimal import Decimal

import pytest

from stockdownloader.backtesting.engines.options import OptionsBacktestEngine
from stockdownloader.core.models.price import PriceData
from stockdownloader.strategies.options.strategies import CoveredCallStrategy, ProtectivePutStrategy

CAPITAL = Decimal("100000")
NO_COMMISSION = Decimal("0")


def _make_price(date, price):
    p = Decimal(str(price))
    high = p + Decimal("1.50")
    low = p - Decimal("1.50")
    return PriceData(
        date=date, open=p, high=high, low=low, close=p, adj_close=p, volume=500000
    )


def _generate_uptrend(days):
    data = []
    price = 100.0
    for i in range(days):
        price += 0.5 + math.sin(i * 0.5) * 0.3
        data.append(_make_price(f"2024-01-{(i % 28) + 1:02d}", price))
    return data


def _generate_downtrend(days):
    data = []
    price = 100.0
    for i in range(days):
        price -= 0.3 + math.sin(i * 0.5) * 0.2
        price = max(price, 50)
        data.append(_make_price(f"2024-01-{(i % 28) + 1:02d}", price))
    return data


def _generate_flat_with_cross(days):
    data = []
    for i in range(days):
        price = 100 + math.sin(i * 0.3) * 5
        data.append(_make_price(f"2024-01-{(i % 28) + 1:02d}", price))
    return data


def test_rejects_null_strategy():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        engine.run(None, [])


def test_rejects_null_data():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    with pytest.raises((ValueError, TypeError)):
        engine.run(CoveredCallStrategy(), None)


def test_rejects_empty_data():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    with pytest.raises(ValueError):
        engine.run(CoveredCallStrategy(), [])


def test_runs_with_uptrend():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    strategy = CoveredCallStrategy(5, Decimal("0.05"), 10, Decimal("0.03"))
    data = _generate_uptrend(50)

    result = engine.run(strategy, data)

    assert result is not None
    assert result.strategy_name == strategy.name
    assert result.initial_capital == CAPITAL
    assert result.equity_curve is not None
    assert len(result.equity_curve) == len(data)


def test_runs_with_downtrend():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    strategy = ProtectivePutStrategy(5, Decimal("0.05"), 10, 3)
    data = _generate_downtrend(50)

    result = engine.run(strategy, data)

    assert result is not None
    assert result.start_date == data[0].date
    assert result.end_date == data[-1].date


def test_covered_call_in_flat_market():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    strategy = CoveredCallStrategy(5, Decimal("0.05"), 10, Decimal("0.03"))
    data = _generate_flat_with_cross(60)

    result = engine.run(strategy, data)
    assert result is not None
    assert result.final_capital is not None


def test_commission_is_deducted():
    commission = Decimal("5.00")
    engine = OptionsBacktestEngine(CAPITAL, commission)
    strategy = CoveredCallStrategy(5, Decimal("0.05"), 10, Decimal("0.03"))
    data = _generate_flat_with_cross(60)

    result_with_comm = engine.run(strategy, data)

    engine_no_comm = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    result_no_comm = engine_no_comm.run(strategy, data)

    assert result_with_comm.final_capital <= result_no_comm.final_capital, (
        "Commission should reduce final capital"
    )


def test_equity_curve_size_matches_data_size():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    strategy = CoveredCallStrategy(5, Decimal("0.05"), 10, Decimal("0.03"))
    data = _generate_uptrend(30)

    result = engine.run(strategy, data)
    assert len(result.equity_curve) == len(data)


def test_volume_is_captured_in_trades():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    strategy = CoveredCallStrategy(5, Decimal("0.05"), 10, Decimal("0.03"))
    data = _generate_flat_with_cross(60)

    result = engine.run(strategy, data)

    if result.trades:
        assert result.total_volume_traded > 0, "Volume should be tracked for all trades"
        for trade in result.trades:
            assert trade.entry_volume > 0, "Each trade should capture entry volume"


def test_handles_expiration_correctly():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    strategy = CoveredCallStrategy(3, Decimal("0.05"), 5, Decimal("0.10"))
    data = _generate_flat_with_cross(40)

    result = engine.run(strategy, data)
    assert result.final_capital is not None


def test_protective_put_in_downtrend():
    engine = OptionsBacktestEngine(CAPITAL, NO_COMMISSION)
    strategy = ProtectivePutStrategy(5, Decimal("0.05"), 10, 3)
    data = _generate_downtrend(50)

    result = engine.run(strategy, data)
    assert result is not None
    assert result.final_capital is not None


def test_custom_risk_free_rate():
    engine1 = OptionsBacktestEngine(CAPITAL, NO_COMMISSION, Decimal("0.01"))
    engine2 = OptionsBacktestEngine(CAPITAL, NO_COMMISSION, Decimal("0.10"))
    strategy = CoveredCallStrategy(5, Decimal("0.05"), 10, Decimal("0.03"))
    data = _generate_uptrend(30)

    assert engine1.run(strategy, data) is not None
    assert engine2.run(strategy, data) is not None
