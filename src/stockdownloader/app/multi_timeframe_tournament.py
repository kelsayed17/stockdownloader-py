"""Multi-timeframe strategy tournament on SPY.

Runs all strategies across 6 timeframes (5m, 15m, 30m, 1h, 4h, 1d)
with both long and short trades enabled, then ranks by win rate and P/L.

Usage::

    python -m stockdownloader.app.multi_timeframe_tournament

Output is written to both stdout and ``output/tournament_results.log``
so you can monitor progress with ``tail -f output/tournament_results.log``.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from operator import attrgetter
from pathlib import Path
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.data.intraday_csv_loader import IntradayCsvLoader
from stockdownloader.model import Direction
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.strategy.daily.bollinger_band_rsi_strategy import (
    BollingerBandRSIStrategy,
)
from stockdownloader.strategy.daily.breakout_strategy import BreakoutStrategy
from stockdownloader.strategy.daily.macd_strategy import MACDStrategy
from stockdownloader.strategy.daily.momentum_confluence_strategy import (
    MomentumConfluenceStrategy,
)
from stockdownloader.strategy.daily.multi_indicator_strategy import (
    MultiIndicatorStrategy,
)
from stockdownloader.strategy.daily.rsi_strategy import RSIStrategy
from stockdownloader.strategy.daily.sma_crossover_strategy import SMACrossoverStrategy
from stockdownloader.strategy.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
from stockdownloader.strategy.intraday.or_reversal_strategy import ORReversalStrategy
from stockdownloader.strategy.intraday.pattern_scalp_strategy import PatternScalpStrategy
from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategy
from stockdownloader.strategy.intraday.reversal_strategy import ReversalStrategy
from stockdownloader.util.big_decimal_math import quantize_decimal as _q
from stockdownloader.util.file_helper import TeeWriter
from stockdownloader.util.timeframe_aggregator import Timeframe, TimeframeAggregator

logger = logging.getLogger(__name__)

_INITIAL_CAPITAL = Decimal("100000")
_RISK_PER_TRADE = Decimal("0.01")
_DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "spy_5m_bars.csv"
_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

# Timeframes to test (ordered smallest to largest)
_TIMEFRAMES = [
    (Timeframe.M5, "5m"),
    (Timeframe.M15, "15m"),
    (Timeframe.M30, "30m"),
    (Timeframe.H1, "1h"),
    (Timeframe.H4, "4h"),
    (Timeframe.DAILY, "1d"),
]


@dataclass(slots=True)
class TournamentEntry:
    """One strategy × timeframe result."""

    timeframe: str
    strategy_name: str
    result: BacktestResult
    longs: int
    shorts: int


# Module-level output target (set in main)
_out: TeeWriter | None = None


def _print(msg: str = "") -> None:
    """Print to both stdout and log file."""
    if _out is not None:
        _out.write(msg + "\n")
    else:
        print(msg, flush=True)


def _build_daily_strategies():
    """Build all daily strategies to test."""
    return [
        SMACrossoverStrategy(short_period=9, long_period=21),
        SMACrossoverStrategy(short_period=50, long_period=200),
        RSIStrategy(period=14, oversold=30.0, overbought=70.0),
        MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
        BollingerBandRSIStrategy(),
        MomentumConfluenceStrategy(),
        BreakoutStrategy(),
        MultiIndicatorStrategy(),
    ]


def _count_directions(result: BacktestResult) -> tuple[int, int]:
    """Count long and short trades in a BacktestResult."""
    longs = 0
    shorts = 0
    for trade in result.closed_trades:
        if trade.direction == Direction.LONG:
            longs += 1
        else:
            shorts += 1
    return longs, shorts


def _run_single(
    strategy_name: str,
    adapted_strategy,
    data: list[IntradayPriceData],
    engine: IntradayBacktestEngine,
) -> BacktestResult | None:
    """Run a single backtest, return result or None on failure."""
    try:
        result = engine.run(adapted_strategy, data)
        return result
    except Exception as exc:
        logger.warning(
            "Strategy %s failed: %s", strategy_name, exc, exc_info=True,
        )
        _print(f"    ERROR: {strategy_name} — {exc}")
        return None


def _run_timeframe(
    tf_label: str,
    data: list[IntradayPriceData],
    include_vwap: bool = False,
) -> list[TournamentEntry]:
    """Run all strategies on a single timeframe and return results."""
    entries: list[TournamentEntry] = []
    engine = IntradayBacktestEngine(_INITIAL_CAPITAL, _RISK_PER_TRADE)

    # Daily strategies via adapter (allow_shorts=True for both long and short)
    daily_strategies = _build_daily_strategies()

    for daily_strat in daily_strategies:
        name = daily_strat.name
        _print(f"  Running: {name}...")

        adapter = DailyToIntradayAdapter(daily_strat, allow_shorts=True)
        result = _run_single(name, adapter, data, engine)

        if result is not None:
            longs, shorts = _count_directions(result)
            entries.append(TournamentEntry(
                timeframe=tf_label,
                strategy_name=name,
                result=result,
                longs=longs,
                shorts=shorts,
            ))

    # VWAP standalone strategies (5m only — designed for 5-minute bars)
    if include_vwap:
        for strat in [
            PullbackStrategy(),
            ReversalStrategy(),
            ORBreakoutStrategy(),
            ORReversalStrategy(),
            PatternScalpStrategy(),
        ]:
            strat_name = strat.name
            _print(f"  Running: {strat_name}...")
            result = _run_single(strat_name, strat, data, engine)

            if result is not None:
                longs, shorts = _count_directions(result)
                entries.append(TournamentEntry(
                    timeframe=tf_label,
                    strategy_name=strat_name,
                    result=result,
                    longs=longs,
                    shorts=shorts,
                ))

    return entries


def _print_timeframe_table(
    tf_label: str,
    bar_count: int,
    entries: list[TournamentEntry],
) -> None:
    """Print a formatted table for one timeframe."""
    _print()
    _print("\u2554" + "\u2550" * 100 + "\u2557")
    title = f"TIMEFRAME: {tf_label} ({bar_count:,} bars)"
    _print("\u2551" + title.center(100) + "\u2551")
    _print("\u255a" + "\u2550" * 100 + "\u255d")
    _print()

    if not entries:
        _print("  No results for this timeframe.")
        _print()
        return

    # Sort by P/L descending
    entries_sorted = sorted(entries, key=attrgetter("result.total_pnl"), reverse=True)

    # Compute Sharpe with appropriate annualization
    # For intraday: 252 trading days * bars_per_day
    trading_days = len({bar_date for e in entries[:1] for bar_date in
                        [e.result.start_date[:10] if e.result.start_date else ""]}) or 1

    header = (
        f"  {'Rank':<5s} {'Strategy':<38s} {'P/L':>12s}  {'WR':>6s}  "
        f"{'Trades':>6s}  {'Sharpe':>7s}  {'PF':>6s}  {'DD':>6s}  "
        f"{'Longs':>5s}  {'Shorts':>6s}"
    )
    _print(header)
    _print("  " + "\u2500" * 98)

    for rank, entry in enumerate(entries_sorted, 1):
        r = entry.result
        pnl = r.total_pnl
        sign = "+" if pnl >= 0 else ""
        sharpe = r.sharpe_ratio(trading_days_per_year=252)

        _print(
            f"  {rank:<5d} {entry.strategy_name:<38s} "
            f"{sign}${_q(pnl):>10,.2f}  "
            f"{_q(r.win_rate):>5.1f}%  "
            f"{r.total_trades:>6d}  "
            f"{_q(sharpe):>7.2f}  "
            f"{_q(r.profit_factor):>6.2f}  "
            f"{_q(r.max_drawdown):>5.2f}%  "
            f"{entry.longs:>5d}  "
            f"{entry.shorts:>6d}"
        )

    _print()

    # Best by win rate
    best_wr = max(entries_sorted, key=attrgetter("result.win_rate"))
    _print(
        f"  Best Win Rate:  {best_wr.strategy_name} "
        f"({_q(best_wr.result.win_rate):.1f}%, "
        f"{best_wr.result.total_trades} trades)"
    )
    best_pnl = entries_sorted[0]
    _print(
        f"  Best P/L:       {best_pnl.strategy_name} "
        f"({'+'if best_pnl.result.total_pnl >= 0 else ''}${_q(best_pnl.result.total_pnl):,.2f})"
    )
    _print()


def _print_grand_comparison(all_entries: list[TournamentEntry]) -> None:
    """Print cross-timeframe comparison ranked by P/L."""
    _print()
    _print("\u2554" + "\u2550" * 110 + "\u2557")
    title = "GRAND COMPARISON \u2014 ALL STRATEGIES \u00d7 ALL TIMEFRAMES (ranked by P/L)"
    _print("\u2551" + title.center(110) + "\u2551")
    _print("\u255a" + "\u2550" * 110 + "\u255d")
    _print()

    sorted_all = sorted(all_entries, key=attrgetter("result.total_pnl"), reverse=True)

    header = (
        f"  {'Rank':<5s} {'TF':<5s} {'Strategy':<38s} {'P/L':>12s}  "
        f"{'WR':>6s}  {'Trades':>6s}  {'Sharpe':>7s}  {'PF':>6s}  "
        f"{'Longs':>5s}  {'Shorts':>6s}"
    )
    _print(header)
    _print("  " + "\u2500" * 108)

    for rank, entry in enumerate(sorted_all, 1):
        r = entry.result
        pnl = r.total_pnl
        sign = "+" if pnl >= 0 else ""
        sharpe = r.sharpe_ratio(trading_days_per_year=252)

        _print(
            f"  {rank:<5d} {entry.timeframe:<5s} {entry.strategy_name:<38s} "
            f"{sign}${_q(pnl):>10,.2f}  "
            f"{_q(r.win_rate):>5.1f}%  "
            f"{r.total_trades:>6d}  "
            f"{_q(sharpe):>7.2f}  "
            f"{_q(r.profit_factor):>6.2f}  "
            f"{entry.longs:>5d}  "
            f"{entry.shorts:>6d}"
        )

    _print()


def _print_best_per_timeframe(all_entries: list[TournamentEntry]) -> None:
    """Print the best strategy for each timeframe."""
    _print()
    _print("\u2554" + "\u2550" * 90 + "\u2557")
    title = "BEST STRATEGY PER TIMEFRAME"
    _print("\u2551" + title.center(90) + "\u2551")
    _print("\u255a" + "\u2550" * 90 + "\u255d")
    _print()

    header = (
        f"  {'TF':<5s} {'Best by P/L':<38s} {'P/L':>12s}  "
        f"{'Best by WR':<30s} {'WR':>6s}"
    )
    _print(header)
    _print("  " + "\u2500" * 88)

    tf_order = ["5m", "15m", "30m", "1h", "4h", "1d"]
    for tf in tf_order:
        tf_entries = [e for e in all_entries if e.timeframe == tf]
        if not tf_entries:
            continue

        best_pnl = max(tf_entries, key=attrgetter("result.total_pnl"))
        best_wr = max(tf_entries, key=attrgetter("result.win_rate"))

        pnl = best_pnl.result.total_pnl
        sign = "+" if pnl >= 0 else ""

        _print(
            f"  {tf:<5s} {best_pnl.strategy_name:<38s} "
            f"{sign}${_q(pnl):>10,.2f}  "
            f"{best_wr.strategy_name:<30s} "
            f"{_q(best_wr.result.win_rate):>5.1f}%"
        )

    _print()

    # Overall best
    best_overall = max(all_entries, key=attrgetter("result.total_pnl"))
    pnl = best_overall.result.total_pnl
    sign = "+" if pnl >= 0 else ""
    _print(
        f"  OVERALL BEST:  {best_overall.strategy_name} on {best_overall.timeframe} "
        f"({sign}${_q(pnl):,.2f}, "
        f"{_q(best_overall.result.win_rate):.1f}% WR, "
        f"{best_overall.result.total_trades} trades)"
    )
    _print()


def _print_win_rate_ranking(all_entries: list[TournamentEntry]) -> None:
    """Print cross-timeframe comparison ranked by win rate."""
    _print()
    _print("\u2554" + "\u2550" * 110 + "\u2557")
    title = "WIN RATE RANKING \u2014 TOP 15"
    _print("\u2551" + title.center(110) + "\u2551")
    _print("\u255a" + "\u2550" * 110 + "\u255d")
    _print()

    # Only include entries with >=5 trades for meaningful WR
    qualifying = [e for e in all_entries if e.result.total_trades >= 5]
    sorted_wr = sorted(
        qualifying, key=attrgetter("result.win_rate"), reverse=True,
    )[:15]

    header = (
        f"  {'Rank':<5s} {'TF':<5s} {'Strategy':<38s} {'WR':>6s}  "
        f"{'Trades':>6s}  {'P/L':>12s}  {'PF':>6s}  "
        f"{'Longs':>5s}  {'Shorts':>6s}"
    )
    _print(header)
    _print("  " + "\u2500" * 108)

    for rank, entry in enumerate(sorted_wr, 1):
        r = entry.result
        pnl = r.total_pnl
        sign = "+" if pnl >= 0 else ""

        _print(
            f"  {rank:<5d} {entry.timeframe:<5s} {entry.strategy_name:<38s} "
            f"{_q(r.win_rate):>5.1f}%  "
            f"{r.total_trades:>6d}  "
            f"{sign}${_q(pnl):>10,.2f}  "
            f"{_q(r.profit_factor):>6.2f}  "
            f"{entry.longs:>5d}  "
            f"{entry.shorts:>6d}"
        )

    _print()


def main() -> None:
    """Run the multi-timeframe strategy tournament."""
    global _out

    # Set up output directory and log file
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _OUTPUT_DIR / "tournament_results.log"
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    _out = TeeWriter(log_file)

    start_time = time.time()

    _print("\u2554" + "\u2550" * 100 + "\u2557")
    title = "MULTI-TIMEFRAME STRATEGY TOURNAMENT \u2014 SPY"
    _print("\u2551" + title.center(100) + "\u2551")
    _print("\u255a" + "\u2550" * 100 + "\u255d")
    _print()

    # Load data
    _print(f"Loading SPY 5m data from {_DATA_FILE}...")
    raw_data = IntradayCsvLoader.load_from_file(_DATA_FILE)
    if not raw_data:
        _print(f"ERROR: Could not load data from {_DATA_FILE}")
        log_file.close()
        return

    _print(f"Loaded {len(raw_data):,} 5-minute bars")
    _print(f"Date range: {raw_data[0].date} to {raw_data[-1].date}")
    trading_days = len({bar.trading_date for bar in raw_data})
    _print(f"Trading days: {trading_days}")
    _print(f"Capital: ${_INITIAL_CAPITAL:,.2f}")
    _print(f"Risk per trade: {_RISK_PER_TRADE * 100}%")
    _print(f"Strategies: 8 daily (long+short) + VWAP v11.2 (5m only)")
    _print(f"Timeframes: 5m, 15m, 30m, 1h, 4h, 1d")
    _print()

    # Aggregate timeframes
    _print("Aggregating timeframes...")
    agg = TimeframeAggregator(raw_data)

    all_entries: list[TournamentEntry] = []

    for tf_enum, tf_label in _TIMEFRAMES:
        tf_data = agg.as_intraday_price_data(tf_enum)
        bar_count = len(tf_data)
        tf_days = len({bar.trading_date for bar in tf_data})

        _print()
        _print("=" * 102)
        _print(f"  TIMEFRAME: {tf_label} ({bar_count:,} bars, {tf_days} trading days)")
        _print("=" * 102)
        _print()

        include_vwap = tf_enum == Timeframe.M5
        entries = _run_timeframe(tf_label, tf_data, include_vwap=include_vwap)

        _print_timeframe_table(tf_label, bar_count, entries)
        all_entries.extend(entries)

    # Grand comparisons
    if all_entries:
        _print_grand_comparison(all_entries)
        _print_win_rate_ranking(all_entries)
        _print_best_per_timeframe(all_entries)

    elapsed = time.time() - start_time

    _print()
    _print("=" * 102)
    _print(f"  Tournament completed in {elapsed:.1f}s")
    _print(f"  Total strategy × timeframe combinations: {len(all_entries)}")
    _print(f"  Results logged to: {log_path}")
    _print("=" * 102)
    _print()
    _print("  DISCLAIMER: This is for educational purposes only.")
    _print("  Not financial advice. Past performance does not guarantee future results.")
    _print()

    log_file.close()


if __name__ == "__main__":
    main()
