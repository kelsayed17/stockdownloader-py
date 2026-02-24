"""Unified report formatting for all backtest types.

Combines shared utility functions, daily equity report formatting,
options report formatting, and intraday report formatting into a
single module.

Public API
----------
Shared utilities:
    scale2, print_header, print_capital, print_initial_final_capital,
    print_performance_metrics, print_trade_statistics, print_trade_log,
    print_comparison_table, find_best, print_best_summary,
    print_disclaimer, print_footer

Daily equity reports:
    print_daily_report(result, data)
    print_daily_comparison(results, data)

Options reports:
    print_options_report(result)
    print_options_comparison(results)

Intraday reports:
    print_intraday_report(result, data)
    print_intraday_comparison(results, data)
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.backtesting.results.result import BaseBacktestResult
    from stockdownloader.core.models import PriceData

from stockdownloader.backtesting.results.result import BacktestResult, OptionsBacktestResult
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.backtesting.results.helpers import scale2

_TRADING_DAYS_PER_YEAR = 252
_BARS_PER_DAY = 78  # 5-minute bars in a 6.5-hour session


# ======================================================================
# Shared utility functions (from base_report_formatter)
# ======================================================================


# ------------------------------------------------------------------
# Section helpers
# ------------------------------------------------------------------


def print_header(
    title: str,
    strategy_name: str,
    width: int = 70,
) -> None:
    """Print the report header block."""
    sep = "=" * width
    print()
    print(sep)
    print(f"  {title}: {strategy_name}")
    print(sep)
    print()


def print_capital(result: BaseBacktestResult) -> None:
    """Print period and capital lines."""
    print(f"  Period:              {result.start_date} to {result.end_date}")


def print_initial_final_capital(result: BaseBacktestResult) -> None:
    """Print initial and final capital lines."""
    print(f"  Initial Capital:     ${scale2(result.initial_capital)}")
    print(f"  Final Capital:       ${scale2(result.final_capital)}")
    print()


def print_performance_metrics(
    result: BaseBacktestResult,
    sharpe: Decimal | None = None,
    *,
    include_buy_hold: bool = False,
    buy_hold_return: Decimal | None = None,
    thin_sep: str = "-" * 70,
) -> None:
    """Print the PERFORMANCE METRICS section.

    Parameters
    ----------
    result:
        The backtest result to report on.
    sharpe:
        Pre-computed Sharpe ratio.  If *None*, uses the default
        ``result.sharpe_ratio(_TRADING_DAYS_PER_YEAR)``.
    include_buy_hold:
        If *True*, print a "Buy & Hold Return" line.
    buy_hold_return:
        The buy-and-hold return percentage (required when *include_buy_hold*
        is *True*).
    thin_sep:
        The thin separator string.
    """
    if sharpe is None:
        sharpe = result.sharpe_ratio(_TRADING_DAYS_PER_YEAR)

    print(thin_sep)
    print("  PERFORMANCE METRICS")
    print(thin_sep)
    print(f"  Total Return:        {scale2(result.total_return)}%")
    if include_buy_hold and buy_hold_return is not None:
        print(f"  Buy & Hold Return:   {scale2(buy_hold_return)}%")
    print(f"  Total P/L:           ${scale2(result.total_pnl)}")
    print(f"  Sharpe Ratio:        {sharpe}")
    print(f"  Max Drawdown:        {scale2(result.max_drawdown)}%")
    print(f"  Profit Factor:       {result.profit_factor}")
    print()


def print_trade_statistics(
    result: BaseBacktestResult,
    *,
    extra_lines: list[str] | None = None,
    thin_sep: str = "-" * 70,
) -> None:
    """Print the TRADE STATISTICS section.

    Parameters
    ----------
    extra_lines:
        Additional lines to append after the standard metrics
        (e.g. "Trades/Day" for intraday reports).
    """
    print(thin_sep)
    print("  TRADE STATISTICS")
    print(thin_sep)
    print(f"  Total Trades:        {result.total_trades}")
    print(f"  Winning Trades:      {result.winning_trades}")
    print(f"  Losing Trades:       {result.losing_trades}")
    print(f"  Win Rate:            {scale2(result.win_rate)}%")
    print(f"  Average Win:         ${result.average_win}")
    print(f"  Average Loss:        ${result.average_loss}")
    if extra_lines:
        for line in extra_lines:
            print(line)
    print()


def print_trade_log(
    result: BaseBacktestResult,
    *,
    label: str = "TRADE LOG",
    limit: int | None = None,
    thin_sep: str = "-" * 70,
) -> None:
    """Print the TRADE LOG section.

    Parameters
    ----------
    label:
        Section header text.
    limit:
        If set, only show the last *limit* trades.
    """
    closed = result.closed_trades
    if not closed:
        return

    print(thin_sep)
    print(f"  {label}")
    print(thin_sep)

    if limit is not None and len(closed) > limit:
        display = closed[-limit:]
        start_num = len(closed) - limit + 1
    else:
        display = closed
        start_num = 1

    for count, t in enumerate(display, start=start_num):
        print(f"  #{count:<4d} {t}")


def print_comparison_table(
    results: list[BaseBacktestResult],
    *,
    sharpe_fn: ...,
    extra_columns: list[tuple[str, ...]] | None = None,
    name_width: int = 35,
    thin_sep: str = "-" * 90,
) -> None:
    """Print the strategy comparison table rows.

    Parameters
    ----------
    sharpe_fn:
        Callable ``(result) -> Decimal`` to compute the Sharpe ratio for
        each result (allows callers to use different annualisation).
    extra_columns:
        Optional list of ``(formatted_value,)`` tuples per result, appended
        after the standard columns.
    """
    for i, r in enumerate(results):
        sharpe = sharpe_fn(r)
        line = (
            f"  {r.strategy_name:<{name_width}s} "
            f"{str(scale2(r.total_return)) + '%':>10s} "
            f"{str(sharpe):>10s} "
            f"{str(scale2(r.max_drawdown)) + '%':>10s} "
            f"{r.total_trades:>10d} "
            f"{str(scale2(r.win_rate)) + '%':>10s}"
        )
        if extra_columns and i < len(extra_columns):
            for col in extra_columns[i]:
                line += f" {col}"
        print(line)


def find_best(results: list[BaseBacktestResult]) -> BaseBacktestResult | None:
    """Return the result with the highest total return, or *None*."""
    best: BaseBacktestResult | None = None
    for r in results:
        if best is None or r.total_return > best.total_return:
            best = r
    return best


def print_best_summary(
    best: BaseBacktestResult,
    sharpe: Decimal,
    *,
    buy_and_hold: Decimal | None = None,
) -> None:
    """Print the 'Best performing strategy' block."""
    print()
    print(f"  Best performing strategy: {best.strategy_name}")
    print(
        f"  Return: {scale2(best.total_return)}% | "
        f"Sharpe: {sharpe} | "
        f"Max Drawdown: {scale2(best.max_drawdown)}%"
    )

    if buy_and_hold is not None:
        diff = scale2(best.total_return - buy_and_hold)
        if best.total_return > buy_and_hold:
            print(f"  >> Outperformed Buy & Hold by {diff} percentage points")
        else:
            print(f"  >> Underperformed Buy & Hold by {abs(diff)} percentage points")


def print_disclaimer(*lines: str) -> None:
    """Print the disclaimer block."""
    print()
    for line in lines:
        print(f"  {line}")
    print()


def print_footer(width: int = 70) -> None:
    """Print the closing separator."""
    print()
    print("=" * width)


# ======================================================================
# Daily equity report functions (from backtest_report_formatter)
# ======================================================================


def print_daily_report(result: BacktestResult, data: list[PriceData]) -> None:
    """Print a formatted backtest report to the console."""
    thin_sep = "-" * 70

    print_header("BACKTEST REPORT", result.strategy_name, width=70)
    print_capital(result)
    print_initial_final_capital(result)

    print_performance_metrics(
        result,
        include_buy_hold=True,
        buy_hold_return=result.buy_and_hold_return(data),
        thin_sep=thin_sep,
    )

    print_trade_statistics(result, thin_sep=thin_sep)
    print_trade_log(result, thin_sep=thin_sep)

    print()
    print("=" * 70)


def print_daily_comparison(results: list[BacktestResult], data: list[PriceData]) -> None:
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
        buy_and_hold = scale2(
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

    print_comparison_table(
        results,
        sharpe_fn=lambda r: r.sharpe_ratio(_TRADING_DAYS_PER_YEAR),
        thin_sep=thin_sep,
    )

    print(thin_sep)

    best = find_best(results)
    if best is not None:
        print_best_summary(
            best,
            sharpe=best.sharpe_ratio(_TRADING_DAYS_PER_YEAR),
            buy_and_hold=buy_and_hold,
        )

    print()
    print(separator)
    print_disclaimer(
        "DISCLAIMER: This is for educational purposes only.",
        "Past performance does not guarantee future results.",
        "Always do your own research before trading.",
    )


# ======================================================================
# Options report functions (from options_backtest_report_formatter)
# ======================================================================


def print_options_report(result: OptionsBacktestResult) -> None:
    """Print a formatted options backtest report to the console."""
    thin = "-" * 80

    print_header("OPTIONS BACKTEST REPORT", result.strategy_name, width=80)
    print_capital(result)
    print_initial_final_capital(result)

    print_performance_metrics(result, thin_sep=thin)
    print_trade_statistics(result, thin_sep=thin)

    # Options-specific section
    print(thin)
    print("  OPTIONS-SPECIFIC METRICS")
    print(thin)
    print(f"  Avg Premium/Trade:   ${result.average_premium_collected}")
    print(f"  Total Volume:        {result.total_volume_traded:,d} contracts")
    print()

    print_trade_log(result, thin_sep=thin)

    print()
    print("=" * 80)


def print_options_comparison(results: list[OptionsBacktestResult]) -> None:
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

    # Build extra columns for volume
    extra_columns = [
        (f"{r.total_volume_traded:>10,d}",) for r in results
    ]

    print_comparison_table(
        results,
        sharpe_fn=lambda r: r.sharpe_ratio(_TRADING_DAYS_PER_YEAR),
        extra_columns=extra_columns,
        name_width=40,
        thin_sep=thin,
    )

    print(thin)

    best = find_best(results)
    if best is not None:
        print_best_summary(
            best,
            sharpe=best.sharpe_ratio(_TRADING_DAYS_PER_YEAR),
        )

    print()
    print(sep)
    print_disclaimer(
        "DISCLAIMER: This is for educational purposes only.",
        "Options trading involves significant risk. Past performance",
        "does not guarantee future results.",
    )


# ======================================================================
# Intraday report functions (from intraday_backtest_report_formatter)
# ======================================================================


def print_intraday_report(
    result: BacktestResult,
    data: list[IntradayPriceData],
) -> None:
    """Print a formatted intraday backtest report."""
    thin_sep = "-" * 70

    print_header("INTRADAY BACKTEST REPORT", result.strategy_name, width=70)
    print(f"  Period:              {result.start_date} to {result.end_date}")
    print(f"  Total Bars:          {len(data):,}")

    # Count trading days
    trading_days = len({
        bar.date[:10] for bar in data
    })
    print(f"  Trading Days:        {trading_days}")
    print_initial_final_capital(result)

    sharpe = result.sharpe_ratio(trading_days_per_year=252 * _BARS_PER_DAY)
    print_performance_metrics(result, sharpe=sharpe, thin_sep=thin_sep)

    # Build extra trade stats lines
    extra_lines: list[str] = []
    if trading_days > 0:
        trades_per_day = Decimal(str(result.total_trades)) / Decimal(
            str(trading_days)
        )
        extra_lines.append(
            f"  Trades/Day:          {trades_per_day.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
        )

    print_trade_statistics(result, extra_lines=extra_lines, thin_sep=thin_sep)

    # Mode distribution
    trade_modes: list[str] = result.trade_modes
    if trade_modes:
        print(thin_sep)
        print("  ENTRY MODE DISTRIBUTION")
        print(thin_sep)
        mode_counts = Counter(trade_modes)
        for mode, count in mode_counts.most_common():
            pct = Decimal(str(count)) / Decimal(str(len(trade_modes))) * 100
            print(
                f"  {mode:<20s} {count:>5d}  ({scale2(pct)}%)"
            )
        print()

    # Trade log (last 50 for intraday)
    print_trade_log(
        result,
        label="TRADE LOG (last 50)",
        limit=50,
        thin_sep=thin_sep,
    )

    print()
    print("=" * 70)


def print_intraday_comparison(
    results: list[BacktestResult],
    data: list[IntradayPriceData],
) -> None:
    """Print a side-by-side comparison of multiple intraday strategy results."""
    separator = "=" * 90
    thin_sep = "-" * 90

    print()
    print(separator)
    print("  INTRADAY STRATEGY COMPARISON SUMMARY")
    print(separator)
    print()

    print(
        f"  {'Strategy':<35s} {'Return':>10s} {'Sharpe':>10s} "
        f"{'MaxDD':>10s} {'Trades':>10s} {'Win Rate':>10s}"
    )
    print(thin_sep)

    sharpe_fn = lambda r: r.sharpe_ratio(trading_days_per_year=252 * _BARS_PER_DAY)

    print_comparison_table(
        results,
        sharpe_fn=sharpe_fn,
        thin_sep=thin_sep,
    )

    print(thin_sep)

    best = find_best(results)
    if best is not None:
        print_best_summary(
            best,
            sharpe=sharpe_fn(best),
        )

    print()
    print(separator)
    print_disclaimer(
        "DISCLAIMER: This is for educational purposes only.",
        "Past performance does not guarantee future results.",
    )
