"""Tests for BacktestResult."""

from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.backtesting.results.result import BacktestResult
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.models.trade import Trade, Direction


def test_total_return_calculation():
    result = BacktestResult("Test", Decimal("10000"))
    result.final_capital = Decimal("12000")

    assert Decimal("20").compare(
        result.total_return.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_total_profit_loss():
    result = BacktestResult("Test", Decimal("10000"))
    result.final_capital = Decimal("12000")

    assert Decimal("2000").compare(result.total_pnl) == 0


def test_zero_trades_return_zero_win_rate():
    result = BacktestResult("Test", Decimal("10000"))
    assert result.win_rate == Decimal("0")


def test_win_rate_calculation():
    result = BacktestResult("Test", Decimal("10000"))

    win_trade = Trade(Direction.LONG, "d1", Decimal("100"), 10)
    win_trade.close("d2", Decimal("120"))

    loss_trade = Trade(Direction.LONG, "d3", Decimal("100"), 10)
    loss_trade.close("d4", Decimal("80"))

    result.add_trade(win_trade)
    result.add_trade(loss_trade)

    assert Decimal("50").compare(
        result.win_rate.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_winning_and_losing_trade_count():
    result = BacktestResult("Test", Decimal("10000"))

    win1 = Trade(Direction.LONG, "d1", Decimal("100"), 10)
    win1.close("d2", Decimal("120"))
    win2 = Trade(Direction.LONG, "d3", Decimal("100"), 10)
    win2.close("d4", Decimal("115"))
    loss = Trade(Direction.LONG, "d5", Decimal("100"), 10)
    loss.close("d6", Decimal("90"))

    result.add_trade(win1)
    result.add_trade(win2)
    result.add_trade(loss)

    assert result.total_trades == 3
    assert result.winning_trades == 2
    assert result.losing_trades == 1


def test_average_win_and_loss():
    result = BacktestResult("Test", Decimal("10000"))

    win = Trade(Direction.LONG, "d1", Decimal("100"), 10)
    win.close("d2", Decimal("120"))  # P/L = 200
    loss = Trade(Direction.LONG, "d3", Decimal("100"), 10)
    loss.close("d4", Decimal("80"))  # P/L = -200

    result.add_trade(win)
    result.add_trade(loss)

    assert Decimal("200").compare(result.average_win) == 0
    assert Decimal("-200").compare(result.average_loss) == 0


def test_profit_factor_with_no_losses():
    result = BacktestResult("Test", Decimal("10000"))

    win = Trade(Direction.LONG, "d1", Decimal("100"), 10)
    win.close("d2", Decimal("120"))
    result.add_trade(win)

    assert Decimal("999.99").compare(result.profit_factor) == 0


def test_profit_factor_calculation():
    result = BacktestResult("Test", Decimal("10000"))

    win = Trade(Direction.LONG, "d1", Decimal("100"), 10)
    win.close("d2", Decimal("150"))  # P/L = 500
    loss = Trade(Direction.LONG, "d3", Decimal("100"), 10)
    loss.close("d4", Decimal("50"))  # P/L = -500

    result.add_trade(win)
    result.add_trade(loss)

    assert Decimal("1").compare(result.profit_factor) == 0


def test_max_drawdown_on_empty_curve():
    result = BacktestResult("Test", Decimal("10000"))
    assert result.max_drawdown == Decimal("0")


def test_max_drawdown_calculation():
    result = BacktestResult("Test", Decimal("10000"))
    result.equity_curve = [
        Decimal("10000"),
        Decimal("11000"),
        Decimal("9000"),  # 18.18% drawdown from 11000
        Decimal("10500"),
    ]

    max_dd = result.max_drawdown
    assert max_dd > Decimal("0")
    assert max_dd < Decimal("20")


def test_sharpe_ratio_with_insufficient_data():
    result = BacktestResult("Test", Decimal("10000"))
    assert result.sharpe_ratio(252) == Decimal("0")


def test_sharpe_ratio_calculation():
    result = BacktestResult("Test", Decimal("10000"))
    result.equity_curve = [
        Decimal("10000"),
        Decimal("10100"),
        Decimal("10200"),
        Decimal("10300"),
        Decimal("10400"),
    ]

    sharpe = result.sharpe_ratio(252)
    assert sharpe > Decimal("0")


def test_buy_and_hold_return():
    result = BacktestResult("Test", Decimal("10000"))

    p = Decimal("100")
    data = [
        PriceData("d1", p, p, p, Decimal("100"), p, 1000),
        PriceData("d2", p, p, p, Decimal("150"), p, 1000),
    ]

    assert Decimal("50").compare(
        result.buy_and_hold_return(data).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_buy_and_hold_return_empty_data():
    result = BacktestResult("Test", Decimal("10000"))
    assert result.buy_and_hold_return([]) == Decimal("0")
