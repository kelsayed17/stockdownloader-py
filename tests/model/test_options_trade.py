"""Tests for OptionsTrade model."""

from decimal import Decimal

import pytest

from stockdownloader.model.options import (
    OptionType,
    OptionsTrade,
    OptionsDirection,
    OptionsTradeStatus,
)


def _make_long_call():
    return OptionsTrade(
        option_type=OptionType.CALL,
        direction=OptionsDirection.BUY,
        strike=Decimal("480"),
        expiration_date="2024-01-19",
        entry_date="2024-01-01",
        entry_premium=Decimal("5.00"),
        contracts=2,
        entry_volume=1000,
    )


def test_constructor_validates_null_fields():
    with pytest.raises(ValueError):
        OptionsTrade(
            option_type=None,
            direction=OptionsDirection.BUY,
            strike=Decimal("1"),
            expiration_date="2024-01-19",
            entry_date="2024-01-01",
            entry_premium=Decimal("1"),
            contracts=1,
            entry_volume=100,
        )


def test_constructor_rejects_non_positive_contracts():
    with pytest.raises(ValueError):
        OptionsTrade(
            option_type=OptionType.CALL,
            direction=OptionsDirection.BUY,
            strike=Decimal("1"),
            expiration_date="2024-01-19",
            entry_date="2024-01-01",
            entry_premium=Decimal("1"),
            contracts=0,
            entry_volume=100,
        )


def test_new_trade_is_open():
    trade = _make_long_call()
    assert trade.status == OptionsTradeStatus.OPEN
    assert trade.profit_loss == Decimal("0")


def test_close_long_call_with_profit():
    trade = _make_long_call()
    # Bought at 5.00, close at 8.00 -> profit = (8-5) * 2 * 100 = 600
    trade.close("2024-01-15", Decimal("8.00"))

    assert trade.status == OptionsTradeStatus.CLOSED
    assert Decimal("600.00").compare(trade.profit_loss.quantize(Decimal("0.01"))) == 0
    assert trade.is_win()


def test_close_long_call_with_loss():
    trade = _make_long_call()
    # Bought at 5.00, close at 2.00 -> loss = (2-5) * 2 * 100 = -600
    trade.close("2024-01-15", Decimal("2.00"))

    assert Decimal("-600.00").compare(trade.profit_loss.quantize(Decimal("0.01"))) == 0
    assert not trade.is_win()


def test_close_short_put_with_profit():
    trade = OptionsTrade(
        option_type=OptionType.PUT,
        direction=OptionsDirection.SELL,
        strike=Decimal("460"),
        expiration_date="2024-01-19",
        entry_date="2024-01-01",
        entry_premium=Decimal("4.00"),
        contracts=1,
        entry_volume=500,
    )
    # Sold at 4.00, close at 1.00 -> profit = (4-1) * 1 * 100 = 300
    trade.close("2024-01-15", Decimal("1.00"))

    assert Decimal("300.00").compare(trade.profit_loss.quantize(Decimal("0.01"))) == 0
    assert trade.is_win()


def test_expire_worthless():
    trade = OptionsTrade(
        option_type=OptionType.CALL,
        direction=OptionsDirection.SELL,
        strike=Decimal("500"),
        expiration_date="2024-01-19",
        entry_date="2024-01-01",
        entry_premium=Decimal("3.00"),
        contracts=1,
        entry_volume=200,
    )
    # Sold call expires worthless -> full premium profit = 3 * 1 * 100 = 300
    trade.expire("2024-01-19", Decimal("0"))

    assert trade.status == OptionsTradeStatus.EXPIRED
    assert Decimal("300.00").compare(trade.profit_loss.quantize(Decimal("0.01"))) == 0


def test_cannot_close_already_closed_trade():
    trade = _make_long_call()
    trade.close("2024-01-15", Decimal("6.00"))
    with pytest.raises(RuntimeError):
        trade.close("2024-01-16", Decimal("7.00"))


def test_cannot_expire_already_expired_trade():
    trade = _make_long_call()
    trade.expire("2024-01-19", Decimal("0"))
    with pytest.raises(RuntimeError):
        trade.expire("2024-01-19", Decimal("0"))


def test_total_entry_cost():
    trade = _make_long_call()
    # 5.00 * 2 * 100 = 1000
    assert Decimal("1000").compare(trade.total_entry_cost()) == 0


def test_return_percentage():
    trade = _make_long_call()
    trade.close("2024-01-15", Decimal("7.50"))
    # P/L = (7.50-5.00)*2*100 = 500, Cost = 5*2*100 = 1000, Return = 50%
    assert trade.return_pct > Decimal("49")
    assert trade.return_pct < Decimal("51")


def test_volume_is_captured():
    trade = _make_long_call()
    assert trade.entry_volume == 1000


def test_to_string_contains_key_fields():
    trade = _make_long_call()
    s = str(trade)
    assert "BUY" in s
    assert "CALL" in s
    assert "480.00" in s
    assert "vol:1000" in s
