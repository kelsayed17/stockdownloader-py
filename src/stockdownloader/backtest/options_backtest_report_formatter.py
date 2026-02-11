"""Formats and prints options backtest reports to the console.

Includes options-specific metrics like premium collected, volume traded, and
theta decay.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.backtest.options_backtest_result import OptionsBacktestResult

_TRADING_DAYS_PER_YEAR = 252


def _scale2(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def print_report(result: OptionsBacktestResult) -> None:
    """Print a formatted options backtest report to the console."""
    sep = "=" * 80
    thin = "-" * 80

    print()
    print(sep)
    print(f"  OPTIONS BACKTEST REPORT: {result.strategy_name}")
    print(sep)
    print()
    print(f"  Period:              {result.start_date} to {result.end_date}")
    print(f"  Initial Capital:     ${_scale2(result.initial_capital)}")
    print(f"  Final Capital:       ${_scale2(result.final_capital)}")
    print()

    print(thin)
    print("  PERFORMANCE METRICS")
    print(thin)
    print(f"  Total Return:        {_scale2(result.total_return)}%")
    print(f"  Total P/L:           ${_scale2(result.total_pnl)}")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio(_TRADING_DAYS_PER_YEAR)}")
    print(f"  Max Drawdown:        {_scale2(result.max_drawdown)}%")
    print(f"  Profit Factor:       {result.profit_factor}")
    print()

    print(thin)
    print("  TRADE STATISTICS")
    print(thin)
    print(f"  Total Trades:        {result.total_trades}")
    print(f"  Winning Trades:      {result.winning_trades}")
    print(f"  Losing Trades:       {result.losing_trades}")
    print(f"  Win Rate:            {_scale2(result.win_rate)}%")
    print(f"  Average Win:         ${result.average_win}")
    print(f"  Average Loss:        ${result.average_loss}")
    print()

    print(thin)
    print("  OPTIONS-SPECIFIC METRICS")
    print(thin)
    print(f"  Avg Premium/Trade:   ${result.average_premium_collected}")
    print(f"  Total Volume:        {result.total_volume_traded:,d} contracts")
    print()

    closed = result.get_closed_trades()
    if closed:
        print(thin)
        print("  TRADE LOG")
        print(thin)
        for count, t in enumerate(closed, start=1):
            print(f"  #{count:<4d} {t}")

    print()
    print(sep)


def print_comparison(results: list[OptionsBacktestResult]) -> None:
    """Print a side-by-side comparison of multiple options strategy results."""
    sep = "=" * 100
    thin = "-" * 100

    print()
    print(sep)
    print("  OPTIONS STRATEGY COMPARISON")
    print(sep)
    print()

    print(
        f"  {'Strategy':<40s} {'Return':>10s} {'Sharpe':>10s} "
        f"{'MaxDD':>10s} {'Trades':>10s} {'WinRate':>10s} {'Volume':>10s}"
    )
    print(thin)

    for r in results:
        print(
            f"  {r.strategy_name:<40s} "
            f"{str(_scale2(r.total_return)) + '%':>10s} "
            f"{str(r.sharpe_ratio(_TRADING_DAYS_PER_YEAR)):>10s} "
            f"{str(_scale2(r.max_drawdown)) + '%':>10s} "
            f"{r.total_trades:>10d} "
            f"{str(_scale2(r.win_rate)) + '%':>10s} "
            f"{r.total_volume_traded:>10,d}"
        )

    print(thin)

    best: OptionsBacktestResult | None = None
    for r in results:
        if best is None or r.total_return > best.total_return:
            best = r

    if best is not None:
        print()
        print(f"  Best performing strategy: {best.strategy_name}")
        print(
            f"  Return: {_scale2(best.total_return)}% | "
            f"Sharpe: {best.sharpe_ratio(_TRADING_DAYS_PER_YEAR)} | "
            f"Max Drawdown: {_scale2(best.max_drawdown)}%"
        )

    print()
    print(sep)
    print()
    print("  DISCLAIMER: This is for educational purposes only.")
    print("  Options trading involves significant risk. Past performance")
    print("  does not guarantee future results.")
    print()
