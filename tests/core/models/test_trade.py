"""Tests for Trade model."""

from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.core.models.trade import Trade, Direction, TradeStatus


def test_create_long_trade():
    trade = Trade(Direction.LONG, "2024-01-01", Decimal("100"), 10)
    assert trade.direction == Direction.LONG
    assert trade.status == TradeStatus.OPEN
    assert trade.entry_date == "2024-01-01"
    assert trade.entry_price == Decimal("100")
    assert trade.shares == 10
    assert trade.profit_loss == Decimal("0")


def test_close_long_trade_with_profit():
    trade = Trade(Direction.LONG, "2024-01-01", Decimal("100"), 10)
    trade.close("2024-02-01", Decimal("120"))

    assert trade.status == TradeStatus.CLOSED
    assert trade.exit_date == "2024-02-01"
    assert trade.exit_price == Decimal("120")
    assert Decimal("200").compare(trade.profit_loss) == 0
    assert trade.is_win()


def test_close_long_trade_with_loss():
    trade = Trade(Direction.LONG, "2024-01-01", Decimal("100"), 10)
    trade.close("2024-02-01", Decimal("80"))

    assert trade.status == TradeStatus.CLOSED
    assert Decimal("-200").compare(trade.profit_loss) == 0
    assert not trade.is_win()


def test_close_short_trade_with_profit():
    trade = Trade(Direction.SHORT, "2024-01-01", Decimal("100"), 10)
    trade.close("2024-02-01", Decimal("80"))

    assert Decimal("200").compare(trade.profit_loss) == 0
    assert trade.is_win()


def test_close_short_trade_with_loss():
    trade = Trade(Direction.SHORT, "2024-01-01", Decimal("100"), 10)
    trade.close("2024-02-01", Decimal("120"))

    assert Decimal("-200").compare(trade.profit_loss) == 0
    assert not trade.is_win()


def test_closing_already_closed_trade_throws():
    trade = Trade(Direction.LONG, "2024-01-01", Decimal("100"), 10)
    trade.close("2024-02-01", Decimal("110"))

    with pytest.raises(RuntimeError):
        trade.close("2024-03-01", Decimal("120"))


def test_zero_shares_throws():
    with pytest.raises(ValueError):
        Trade(Direction.LONG, "2024-01-01", Decimal("100"), 0)


def test_negative_shares_throws():
    with pytest.raises(ValueError):
        Trade(Direction.LONG, "2024-01-01", Decimal("100"), -5)


def test_null_direction_throws():
    with pytest.raises(ValueError):
        Trade(None, "2024-01-01", Decimal("100"), 10)


def test_return_percentage_is_calculated():
    trade = Trade(Direction.LONG, "2024-01-01", Decimal("100"), 10)
    trade.close("2024-02-01", Decimal("110"))

    # 10% return
    assert trade.return_pct > Decimal("0")
    assert Decimal("10").compare(
        trade.return_pct.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_to_string_formats_correctly():
    trade = Trade(Direction.LONG, "2024-01-01", Decimal("100"), 10)
    trade.close("2024-02-01", Decimal("110"))

    s = str(trade)
    assert "LONG" in s
    assert "CLOSED" in s
    assert "2024-01-01" in s
    assert "2024-02-01" in s
