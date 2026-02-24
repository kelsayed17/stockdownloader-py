"""Tests for OptionsBacktestResult."""

from decimal import Decimal

import pytest

from stockdownloader.backtest.backtest_result import OptionsBacktestResult
from stockdownloader.core.models.options import OptionType, OptionsTrade, OptionsDirection


def _make_trade(entry_premium, exit_premium):
    trade = OptionsTrade(
        option_type=OptionType.CALL,
        direction=OptionsDirection.BUY,
        strike=Decimal("100"),
        expiration_date="2024-02-16",
        entry_date="2024-01-15",
        entry_premium=entry_premium,
        contracts=1,
        entry_volume=1000,
    )
    trade.close("2024-02-10", exit_premium)
    return trade


def test_initial_state():
    result = OptionsBacktestResult("TestStrategy", Decimal("100000"))
    assert result.strategy_name == "TestStrategy"
    assert result.initial_capital == Decimal("100000")
    assert result.final_capital == Decimal("100000")
    assert result.total_trades == 0
    assert result.total_volume_traded == 0


def test_total_return_calculation():
    result = OptionsBacktestResult("Test", Decimal("100000"))
    result.final_capital = Decimal("110000")
    # (110000 - 100000) / 100000 * 100 = 10%
    assert Decimal("10.000000").compare(result.total_return) == 0


def test_total_profit_loss():
    result = OptionsBacktestResult("Test", Decimal("100000"))
    result.final_capital = Decimal("95000")
    assert Decimal("-5000").compare(result.total_pnl) == 0


def test_win_rate_calculation():
    result = OptionsBacktestResult("Test", Decimal("100000"))

    win1 = _make_trade(Decimal("5.00"), Decimal("8.00"))
    win2 = _make_trade(Decimal("3.00"), Decimal("5.00"))
    lose = _make_trade(Decimal("5.00"), Decimal("2.00"))

    result.add_trade(win1)
    result.add_trade(win2)
    result.add_trade(lose)

    assert result.total_trades == 3
    assert result.winning_trades == 2
    assert result.losing_trades == 1
    # Win rate: 2/3 * 100 = 66.67%
    assert float(result.win_rate) > 66
    assert float(result.win_rate) < 67


def test_profit_factor():
    result = OptionsBacktestResult("Test", Decimal("100000"))

    win = _make_trade(Decimal("5.00"), Decimal("8.00"))
    lose = _make_trade(Decimal("5.00"), Decimal("3.00"))
    result.add_trade(win)
    result.add_trade(lose)

    pf = result.profit_factor
    assert float(pf) > 0


def test_profit_factor_no_losses():
    result = OptionsBacktestResult("Test", Decimal("100000"))
    result.add_trade(_make_trade(Decimal("5.00"), Decimal("8.00")))

    assert Decimal("999.99").compare(result.profit_factor) == 0


def test_max_drawdown():
    result = OptionsBacktestResult("Test", Decimal("100000"))
    result.equity_curve = [
        Decimal("100000"),
        Decimal("105000"),
        Decimal("95000"),  # drawdown from 105000
        Decimal("98000"),
        Decimal("110000"),
    ]

    dd = result.max_drawdown
    # Peak 105000, trough 95000 -> (105000-95000)/105000 = 9.52%
    assert float(dd) > 9
    assert float(dd) < 10


def test_sharpe_ratio():
    result = OptionsBacktestResult("Test", Decimal("100000"))
    result.equity_curve = [
        Decimal("100000"),
        Decimal("100500"),
        Decimal("101000"),
        Decimal("101500"),
        Decimal("102000"),
    ]

    sharpe = result.sharpe_ratio(252)
    assert float(sharpe) > 0, "Sharpe should be positive for consistent gains"


def test_sharpe_ratio_insufficient_data():
    result = OptionsBacktestResult("Test", Decimal("100000"))
    result.equity_curve = [Decimal("100000")]
    assert result.sharpe_ratio(252) == Decimal("0")


def test_average_premium_collected():
    result = OptionsBacktestResult("Test", Decimal("100000"))
    # Entry premium 5.00, 1 contract * 100 = $500
    result.add_trade(_make_trade(Decimal("5.00"), Decimal("3.00")))
    result.add_trade(_make_trade(Decimal("3.00"), Decimal("1.00")))

    avg = result.average_premium_collected
    # (500 + 300) / 2 = 400
    assert Decimal("400.00").compare(avg) == 0


def test_volume_tracking():
    result = OptionsBacktestResult("Test", Decimal("100000"))

    trade1 = OptionsTrade(
        option_type=OptionType.CALL,
        direction=OptionsDirection.BUY,
        strike=Decimal("100"),
        expiration_date="2024-02-16",
        entry_date="2024-01-15",
        entry_premium=Decimal("5.00"),
        contracts=1,
        entry_volume=5000,
    )
    trade1.close("2024-02-10", Decimal("7.00"))

    trade2 = OptionsTrade(
        option_type=OptionType.CALL,
        direction=OptionsDirection.BUY,
        strike=Decimal("100"),
        expiration_date="2024-03-15",
        entry_date="2024-02-15",
        entry_premium=Decimal("4.00"),
        contracts=1,
        entry_volume=3000,
    )
    trade2.close("2024-03-10", Decimal("6.00"))

    result.add_trade(trade1)
    result.add_trade(trade2)

    assert result.total_volume_traded == 8000


def test_closed_trades_excludes_open_trades():
    result = OptionsBacktestResult("Test", Decimal("100000"))

    closed_trade = _make_trade(Decimal("5.00"), Decimal("8.00"))
    open_trade = OptionsTrade(
        option_type=OptionType.CALL,
        direction=OptionsDirection.BUY,
        strike=Decimal("100"),
        expiration_date="2024-02-16",
        entry_date="2024-01-15",
        entry_premium=Decimal("5.00"),
        contracts=1,
        entry_volume=1000,
    )

    result.add_trade(closed_trade)
    result.add_trade(open_trade)

    assert len(result.closed_trades) == 1
    assert len(result.trades) == 2
