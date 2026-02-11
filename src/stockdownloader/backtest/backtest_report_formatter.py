"""Formats and prints backtest reports to the console.

Separated from BacktestResult to keep result data and presentation concerns
distinct.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.model import PriceData

from stockdownloader.backtest.backtest_result import BacktestResult

_TRADING_DAYS_PER_YEAR = 252


def _scale2(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def print_report(result: BacktestResult, data: list[PriceData]) -> None:
    """Print a formatted backtest report to the console."""
    separator = "=" * 70
    thin_sep = "-" * 70

    print()
    print(separator)
    print(f"  BACKTEST REPORT: {result.strategy_name}")
    print(separator)
    print()
    print(f"  Period:              {result.start_date} to {result.end_date}")
    print(f"  Initial Capital:     ${_scale2(result.initial_capital)}")
    print(f"  Final Capital:       ${_scale2(result.final_capital)}")
    print()

    print(thin_sep)
    print("  PERFORMANCE METRICS")
    print(thin_sep)
    print(f"  Total Return:        {_scale2(result.total_return)}%")
    print(f"  Buy & Hold Return:   {_scale2(result.buy_and_hold_return(data))}%")
    print(f"  Total P/L:           ${_scale2(result.total_pnl)}")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio(_TRADING_DAYS_PER_YEAR)}")
    print(f"  Max Drawdown:        {_scale2(result.max_drawdown)}%")
    print(f"  Profit Factor:       {result.profit_factor}")
    print()

    print(thin_sep)
    print("  TRADE STATISTICS")
    print(thin_sep)
    print(f"  Total Trades:        {result.total_trades}")
    print(f"  Winning Trades:      {result.winning_trades}")
    print(f"  Losing Trades:       {result.losing_trades}")
    print(f"  Win Rate:            {_scale2(result.win_rate)}%")
    print(f"  Average Win:         ${result.average_win}")
    print(f"  Average Loss:        ${result.average_loss}")
    print()

    closed = result.get_closed_trades()
    if closed:
        print(thin_sep)
        print("  TRADE LOG")
        print(thin_sep)
        for count, t in enumerate(closed, start=1):
            print(f"  #{count:<4d} {t}")

    print()
    print(separator)


def print_comparison(results: list[BacktestResult], data: list[PriceData]) -> None:
    """Print a side-by-side comparison of multiple strategy results."""
    separator = "=" * 90
    thin_sep = "-" * 90

    print()
    print(separator)
    print("  STRATEGY COMPARISON SUMMARY")
    print(separator)
    print()

    buy_and_hold = Decimal("0")
    if data:
        first = data[0].close
        last = data[-1].close
        buy_and_hold = _scale2(
            ((last - first) / first).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )
            * Decimal("100")
        )

    print(
        f"  {'Strategy':<35s} {'Return':>10s} {'Sharpe':>10s} "
        f"{'MaxDD':>10s} {'Trades':>10s} {'Win Rate':>10s}"
    )
    print(thin_sep)
    print(
        f"  {'Buy & Hold (Benchmark)':<35s} {str(buy_and_hold) + '%':>10s} "
        f"{'N/A':>10s} {'N/A':>10s} {'1':>10s} {'N/A':>10s}"
    )

    for r in results:
        print(
            f"  {r.strategy_name:<35s} "
            f"{str(_scale2(r.total_return)) + '%':>10s} "
            f"{str(r.sharpe_ratio(_TRADING_DAYS_PER_YEAR)):>10s} "
            f"{str(_scale2(r.max_drawdown)) + '%':>10s} "
            f"{r.total_trades:>10d} "
            f"{str(_scale2(r.win_rate)) + '%':>10s}"
        )

    print(thin_sep)

    best: BacktestResult | None = None
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

        diff = _scale2(best.total_return - buy_and_hold)
        if best.total_return > buy_and_hold:
            print(f"  >> Outperformed Buy & Hold by {diff} percentage points")
        else:
            print(f"  >> Underperformed Buy & Hold by {abs(diff)} percentage points")

    print()
    print(separator)
    print()
    print("  DISCLAIMER: This is for educational purposes only.")
    print("  Past performance does not guarantee future results.")
    print("  Always do your own research before trading.")
    print()
