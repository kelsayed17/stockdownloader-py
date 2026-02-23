"""Tests for black_scholes_calculator utility functions."""

import math
from decimal import Decimal

import pytest

from stockdownloader.model.options import OptionType
from stockdownloader.util.options.black_scholes import (
    price as bs_price,
    delta,
    theta,
    estimate_volatility,
    intrinsic_value,
)

SPOT = Decimal("100")
STRIKE_ATM = Decimal("100")
STRIKE_OTM_CALL = Decimal("110")
STRIKE_ITM_CALL = Decimal("90")
TIME_30D = Decimal("0.0822")  # ~30/365
TIME_1Y = Decimal("1")
RATE = Decimal("0.05")
VOL = Decimal("0.20")


def test_call_price_is_positive():
    price = bs_price(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, VOL)
    assert float(price) > 0, "ATM call should have positive price"


def test_put_price_is_positive():
    price = bs_price(OptionType.PUT, SPOT, STRIKE_ATM, TIME_30D, RATE, VOL)
    assert float(price) > 0, "ATM put should have positive price"


def test_call_price_increases_with_longer_expiry():
    short30d = bs_price(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, VOL)
    long1y = bs_price(OptionType.CALL, SPOT, STRIKE_ATM, TIME_1Y, RATE, VOL)
    assert long1y > short30d, "Longer expiry should increase call price"


def test_otm_call_cheaper_than_atm_call():
    atm = bs_price(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, VOL)
    otm = bs_price(OptionType.CALL, SPOT, STRIKE_OTM_CALL, TIME_30D, RATE, VOL)
    assert atm > otm, "OTM call should be cheaper than ATM"


def test_itm_call_more_expensive_than_atm_call():
    atm = bs_price(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, VOL)
    itm = bs_price(OptionType.CALL, SPOT, STRIKE_ITM_CALL, TIME_30D, RATE, VOL)
    assert itm > atm, "ITM call should be more expensive than ATM"


def test_put_call_parity():
    # Put-call parity: C - P = S - K*exp(-rT)
    call_price = bs_price(OptionType.CALL, SPOT, STRIKE_ATM, TIME_1Y, RATE, VOL)
    put_price = bs_price(OptionType.PUT, SPOT, STRIKE_ATM, TIME_1Y, RATE, VOL)

    S = float(SPOT)
    K = float(STRIKE_ATM)
    r = float(RATE)
    T = float(TIME_1Y)

    lhs = float(call_price) - float(put_price)
    rhs = S - K * math.exp(-r * T)

    assert abs(lhs - rhs) < 0.01, "Put-call parity should hold"


def test_expired_option_returns_intrinsic_value():
    itm_call = bs_price(OptionType.CALL, SPOT, STRIKE_ITM_CALL, Decimal("0"), RATE, VOL)
    # Intrinsic: 100 - 90 = 10
    assert Decimal("10.0000").compare(itm_call) == 0

    otm_call = bs_price(OptionType.CALL, SPOT, STRIKE_OTM_CALL, Decimal("0"), RATE, VOL)
    assert Decimal("0").compare(otm_call.normalize()) == 0


def test_zero_vol_returns_intrinsic():
    price = bs_price(OptionType.CALL, SPOT, STRIKE_ITM_CALL, TIME_30D, RATE, Decimal("0"))
    assert Decimal("10.0000").compare(price) == 0


def test_call_delta_between_zero_and_one():
    d = delta(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, VOL)
    assert 0 < float(d) < 1, f"Call delta should be between 0 and 1, got {d}"


def test_put_delta_between_neg_one_and_zero():
    d = delta(OptionType.PUT, SPOT, STRIKE_ATM, TIME_30D, RATE, VOL)
    assert -1 < float(d) < 0, f"Put delta should be between -1 and 0, got {d}"


def test_atm_call_delta_around_point_five():
    d = delta(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, VOL)
    assert 0.4 < float(d) < 0.7, f"ATM call delta should be around 0.5, got {d}"


def test_theta_is_negative_for_long_option():
    t = theta(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, VOL)
    assert float(t) < 0, "Theta should be negative for long call"


def test_estimate_volatility_from_prices():
    prices = []
    # Simulate slight uptrend with some noise
    for i in range(21):
        prices.append(Decimal(str(100 + i * 0.5 + math.sin(i) * 2)))

    vol = estimate_volatility(prices, 20)
    assert float(vol) > 0, "Volatility should be positive"
    assert float(vol) < 2.0, "Volatility should be reasonable"


def test_estimate_volatility_defaults_for_insufficient_data():
    vol = estimate_volatility(None, 20)
    assert Decimal("0.20").compare(vol.quantize(Decimal("0.01"))) == 0


def test_intrinsic_value_calculation():
    assert Decimal("10.0000").compare(
        intrinsic_value(OptionType.CALL, SPOT, STRIKE_ITM_CALL)
    ) == 0
    assert Decimal("0").compare(
        intrinsic_value(OptionType.CALL, SPOT, STRIKE_OTM_CALL).normalize()
    ) == 0
    assert Decimal("10.0000").compare(
        intrinsic_value(OptionType.PUT, SPOT, STRIKE_OTM_CALL)
    ) == 0
    assert Decimal("0").compare(
        intrinsic_value(OptionType.PUT, SPOT, STRIKE_ITM_CALL).normalize()
    ) == 0


def test_higher_vol_increases_price():
    low_vol = bs_price(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, Decimal("0.10"))
    high_vol = bs_price(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, Decimal("0.40"))
    assert high_vol > low_vol, "Higher vol should increase option price"
