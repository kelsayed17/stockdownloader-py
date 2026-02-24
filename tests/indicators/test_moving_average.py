"""Tests for moving_average_calculator utility functions."""

from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.core.models.price import PriceData
from stockdownloader.indicators import sma, ema


def _make_price_data(date, close):
    p = Decimal(str(close))
    return PriceData(date=date, open=p, high=p, low=p, close=p, adj_close=p, volume=1000)


def _generate_price_data(count, start, increment):
    data = []
    for i in range(count):
        data.append(_make_price_data(f"d{i}", start + i * increment))
    return data


def test_sma_of_uniform_prices_equals_price():
    data = _generate_price_data(10, 50.0, 0)
    result = sma(data, 9, 5)
    assert Decimal("50").compare(
        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_sma_of_linearly_increasing_prices():
    # Prices: 1, 2, 3, 4, 5
    data = _generate_price_data(5, 1.0, 1.0)
    result = sma(data, 4, 5)
    # Average of 1,2,3,4,5 = 3
    assert Decimal("3").compare(
        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_sma_with_smaller_window():
    # Prices: 10, 20, 30, 40, 50
    data = _generate_price_data(5, 10.0, 10.0)
    result = sma(data, 4, 3)
    # Average of last 3: 30, 40, 50 = 40
    assert Decimal("40").compare(
        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_ema_converges_to_price_on_flat():
    data = _generate_price_data(20, 100.0, 0)
    result = ema(data, 19, 5)
    assert Decimal("100").compare(
        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_ema_reacts_to_recent_prices_more_than_sma():
    data = []
    # Flat at 100 for 15 days then jump to 200
    for i in range(15):
        data.append(_make_price_data(f"d{i}", 100))
    for i in range(15, 20):
        data.append(_make_price_data(f"d{i}", 200))

    sma_val = sma(data, 19, 10)
    ema_val = ema(data, 19, 10)

    # EMA should be closer to 200 than SMA (more reactive)
    diff200_ema = abs(Decimal("200") - ema_val)
    diff200_sma = abs(Decimal("200") - sma_val)
    assert diff200_ema < diff200_sma, "EMA should be closer to recent prices than SMA"


def test_sma_period_of_1_equals_last_price():
    data = _generate_price_data(5, 10.0, 10.0)
    result = sma(data, 4, 1)
    assert Decimal("50").compare(
        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0
