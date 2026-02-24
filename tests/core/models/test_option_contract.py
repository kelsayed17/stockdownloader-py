"""Tests for OptionContract model."""

from decimal import Decimal

import pytest

from stockdownloader.core.models.options import OptionContract, OptionType


def _sample_call():
    return OptionContract(
        contract_symbol="SPY240119C00480000",
        type=OptionType.CALL,
        strike=Decimal("480.00"),
        expiration_date="2024-01-19",
        last_price=Decimal("5.50"),
        bid=Decimal("5.40"),
        ask=Decimal("5.60"),
        volume=1500,
        open_interest=25000,
        implied_volatility=Decimal("0.18"),
        delta=Decimal("0.45"),
        gamma=Decimal("0.02"),
        theta=Decimal("-0.15"),
        vega=Decimal("0.25"),
        in_the_money=False,
    )


def _sample_put():
    return OptionContract(
        contract_symbol="SPY240119P00460000",
        type=OptionType.PUT,
        strike=Decimal("460.00"),
        expiration_date="2024-01-19",
        last_price=Decimal("3.20"),
        bid=Decimal("3.10"),
        ask=Decimal("3.30"),
        volume=800,
        open_interest=18000,
        implied_volatility=Decimal("0.20"),
        delta=Decimal("-0.30"),
        gamma=Decimal("0.01"),
        theta=Decimal("-0.12"),
        vega=Decimal("0.20"),
        in_the_money=False,
    )


def test_constructor_validates_null_fields():
    with pytest.raises((ValueError, TypeError)):
        OptionContract(
            contract_symbol=None,
            type=OptionType.CALL,
            strike=Decimal("1"),
            expiration_date="2024-01-19",
            last_price=Decimal("1"),
            bid=Decimal("1"),
            ask=Decimal("1"),
            volume=0,
            open_interest=0,
            implied_volatility=Decimal("0"),
            delta=Decimal("0"),
            gamma=Decimal("0"),
            theta=Decimal("0"),
            vega=Decimal("0"),
            in_the_money=False,
        )


def test_constructor_rejects_negative_volume():
    with pytest.raises((ValueError, TypeError)):
        OptionContract(
            contract_symbol="SYM",
            type=OptionType.CALL,
            strike=Decimal("1"),
            expiration_date="2024-01-19",
            last_price=Decimal("1"),
            bid=Decimal("1"),
            ask=Decimal("1"),
            volume=-1,
            open_interest=0,
            implied_volatility=Decimal("0"),
            delta=Decimal("0"),
            gamma=Decimal("0"),
            theta=Decimal("0"),
            vega=Decimal("0"),
            in_the_money=False,
        )


def test_constructor_rejects_negative_open_interest():
    with pytest.raises((ValueError, TypeError)):
        OptionContract(
            contract_symbol="SYM",
            type=OptionType.CALL,
            strike=Decimal("1"),
            expiration_date="2024-01-19",
            last_price=Decimal("1"),
            bid=Decimal("1"),
            ask=Decimal("1"),
            volume=0,
            open_interest=-1,
            implied_volatility=Decimal("0"),
            delta=Decimal("0"),
            gamma=Decimal("0"),
            theta=Decimal("0"),
            vega=Decimal("0"),
            in_the_money=False,
        )


def test_mid_price_calculation():
    call = _sample_call()
    # (5.40 + 5.60) / 2 = 5.50
    assert call.mid_price() == Decimal("5.5000") or call.mid_price() == Decimal("5.50")


def test_spread_calculation():
    call = _sample_call()
    # 5.60 - 5.40 = 0.20
    assert call.spread() == Decimal("0.20")


def test_notional_value_calculation():
    call = _sample_call()
    # 5.50 * 100 = 550
    assert Decimal("550.00").compare(call.notional_value()) == 0


def test_record_accessors():
    call = _sample_call()
    assert call.contract_symbol == "SPY240119C00480000"
    assert call.type == OptionType.CALL
    assert call.strike == Decimal("480.00")
    assert call.expiration_date == "2024-01-19"
    assert call.volume == 1500
    assert call.open_interest == 25000
    assert not call.in_the_money


def test_put_contract_fields():
    put = _sample_put()
    assert put.type == OptionType.PUT
    assert put.strike == Decimal("460.00")
    assert put.volume == 800
    assert put.open_interest == 18000


def test_to_string_contains_key_fields():
    call = _sample_call()
    s = str(call)
    assert "CALL" in s
    assert "480" in s
    assert "vol:1500" in s
    assert "OI:25000" in s
