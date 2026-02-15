"""Multi-timeframe parameter-optimized strategy tournament on SPY.

For each strategy × timeframe combination, dynamically tunes ALL indicator
inputs and adapter parameters (stop-loss, take-profit, risk/reward, shorts)
using greedy sequential search, then ranks the optimized results.

Usage::

    python -m stockdownloader.app.multi_timeframe_optimizer

Output is written to both stdout and ``output/optimizer_tournament.log``
so you can monitor progress with ``tail -f output/optimizer_tournament.log``.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from operator import attrgetter
from pathlib import Path
from typing import Any

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.daily_strategy_optimizer import DailyStrategyOptimizer
from stockdownloader.backtest.optimizer_scoring import score as _score
from stockdownloader.backtest.strategy_optimizer import StrategyOptimizer
from stockdownloader.data.intraday_csv_loader import IntradayCsvLoader
from stockdownloader.model import Direction
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.strategy.registry import StrategyRegistry
from stockdownloader.util.big_decimal_math import quantize_decimal as _q
from stockdownloader.util.file_helper import TeeWriter
from stockdownloader.util.timeframe_aggregator import Timeframe, TimeframeAggregator

logger = logging.getLogger(__name__)

_INITIAL_CAPITAL = Decimal("100000")
_RISK_PER_TRADE = Decimal("0.01")
_DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "spy_5m_bars.csv"
_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

# Timeframes to test
_TIMEFRAMES = [
    (Timeframe.M5, "5m"),
    (Timeframe.M15, "15m"),
    (Timeframe.M30, "30m"),
    (Timeframe.H1, "1h"),
    (Timeframe.H4, "4h"),
    (Timeframe.DAILY, "1d"),
]

# Daily strategy registry names
_DAILY_STRATEGIES = [
    "sma",
    "rsi",
    "macd",
    "bollinger",
    "momentum",
    "breakout",
    "multi",
]


@dataclass(slots=True)
class OptimizedEntry:
    """One optimized strategy × timeframe result."""

    timeframe: str
    strategy_name: str
    display_name: str
    result: BacktestResult
    best_kwargs: dict[str, Any]
    longs: int
    shorts: int
    configs_tested: int
    opt_time: float


_out: TeeWriter | None = None


def _print(msg: str = "") -> None:
    if _out is not None:
        _out.write(msg + "\n")
    else:
        print(msg, flush=True)


def _count_directions(result: BacktestResult) -> tuple[int, int]:
    longs = shorts = 0
    for trade in result.closed_trades:
        if trade.direction == Direction.LONG:
            longs += 1
        else:
            shorts += 1
    return longs, shorts


def _optimize_daily_strategy(
    strategy_name: str,
    data: list[IntradayPriceData],
    log_file: TextIO,
) -> OptimizedEntry | None:
    """Run DailyStrategyOptimizer for a single strategy, return entry."""
    entry = StrategyRegistry.get(strategy_name)

    try:
        opt = DailyStrategyOptimizer(
            strategy_name,
            data,
            initial_capital=_INITIAL_CAPITAL,
            risk_per_trade=_RISK_PER_TRADE,
            allow_shorts=True,
            verbose=True,
            log_file=log_file,
        )
        best_kwargs, best_result = opt.optimize()

        longs, shorts = _count_directions(best_result)

        return OptimizedEntry(
            timeframe="",  # filled by caller
            strategy_name=strategy_name,
            display_name=entry.display_name,
            result=best_result,
            best_kwargs=best_kwargs,
            longs=longs,
            shorts=shorts,
            configs_tested=opt._run_count,
            opt_time=0,  # filled by caller
        )

    except Exception as exc:
        logger.warning(
            "Optimizer failed for %s: %s", strategy_name, exc, exc_info=True,
        )
        _print(f"  ERROR optimizing {strategy_name}: {exc}")
        return None


def _optimize_vwap(
    data: list[IntradayPriceData],
    log_file: TextIO,
) -> OptimizedEntry | None:
    """Run StrategyOptimizer for VWAP v11.2, return entry."""
    try:
        opt = StrategyOptimizer(
            data,
            initial_capital=_INITIAL_CAPITAL,
            risk_per_trade=_RISK_PER_TRADE,
            verbose=True,
            log_file=log_file,
        )
        _best_config, best_result = opt.optimize()

        longs, shorts = _count_directions(best_result)

        return OptimizedEntry(
            timeframe="5m",
            strategy_name="vwap",
            display_name="VWAP v11.2 (optimized)",
            result=best_result,
            best_kwargs={},  # VWAP config is a dataclass, not kwargs
            longs=longs,
            shorts=shorts,
            configs_tested=opt._run_count,
            opt_time=0,
        )

    except Exception as exc:
        logger.warning("VWAP optimizer failed: %s", exc, exc_info=True)
        _print(f"  ERROR optimizing VWAP: {exc}")
        return None


def _print_timeframe_table(
    tf_label: str,
    bar_count: int,
    entries: list[OptimizedEntry],
) -> None:
    """Print optimized results table for one timeframe."""
    _print()
    _print("\u2554" + "\u2550" * 110 + "\u2557")
    title = f"OPTIMIZED RESULTS: {tf_label} ({bar_count:,} bars)"
    _print("\u2551" + title.center(110) + "\u2551")
    _print("\u255a" + "\u2550" * 110 + "\u255d")
    _print()

    if not entries:
        _print("  No results for this timeframe.")
        return

    sorted_entries = sorted(entries, key=attrgetter("result.total_pnl"), reverse=True)

    header = (
        f"  {'Rank':<5s} {'Strategy':<32s} {'P/L':>12s}  {'WR':>6s}  "
        f"{'Trades':>6s}  {'Sharpe':>7s}  {'PF':>6s}  {'DD':>6s}  "
        f"{'Longs':>5s}  {'Shorts':>6s}  {'Cfgs':>5s}"
    )
    _print(header)
    _print("  " + "\u2500" * 108)

    for rank, entry in enumerate(sorted_entries, 1):
        r = entry.result
        pnl = r.total_pnl
        sign = "+" if pnl >= 0 else ""
        sharpe = r.sharpe_ratio(trading_days_per_year=252)

        _print(
            f"  {rank:<5d} {entry.display_name:<32s} "
            f"{sign}${_q(pnl):>10,.2f}  "
            f"{_q(r.win_rate):>5.1f}%  "
            f"{r.total_trades:>6d}  "
            f"{_q(sharpe):>7.2f}  "
            f"{_q(r.profit_factor):>6.2f}  "
            f"{_q(r.max_drawdown):>5.2f}%  "
            f"{entry.longs:>5d}  "
            f"{entry.shorts:>6d}  "
            f"{entry.configs_tested:>5d}"
        )

    _print()

    # Print best params for each strategy
    _print("  OPTIMIZED PARAMETERS:")
    _print("  " + "\u2500" * 108)
    for entry in sorted_entries:
        if not entry.best_kwargs:
            continue
        params_str = ", ".join(
            f"{k}={v}" for k, v in sorted(entry.best_kwargs.items())
        )
        _print(f"  {entry.display_name}:")
        _print(f"    {params_str}")
    _print()


def _print_grand_comparison(all_entries: list[OptimizedEntry]) -> None:
    """Print cross-timeframe comparison."""
    _print()
    _print("\u2554" + "\u2550" * 115 + "\u2557")
    title = "GRAND COMPARISON \u2014 ALL OPTIMIZED STRATEGIES \u00d7 ALL TIMEFRAMES"
    _print("\u2551" + title.center(115) + "\u2551")
    _print("\u255a" + "\u2550" * 115 + "\u255d")
    _print()

    sorted_all = sorted(all_entries, key=attrgetter("result.total_pnl"), reverse=True)

    header = (
        f"  {'Rank':<5s} {'TF':<5s} {'Strategy':<32s} {'P/L':>12s}  "
        f"{'WR':>6s}  {'Trades':>6s}  {'Sharpe':>7s}  {'PF':>6s}  "
        f"{'Longs':>5s}  {'Shorts':>6s}  {'Cfgs':>5s}"
    )
    _print(header)
    _print("  " + "\u2500" * 113)

    for rank, entry in enumerate(sorted_all, 1):
        r = entry.result
        pnl = r.total_pnl
        sign = "+" if pnl >= 0 else ""
        sharpe = r.sharpe_ratio(trading_days_per_year=252)

        _print(
            f"  {rank:<5d} {entry.timeframe:<5s} {entry.display_name:<32s} "
            f"{sign}${_q(pnl):>10,.2f}  "
            f"{_q(r.win_rate):>5.1f}%  "
            f"{r.total_trades:>6d}  "
            f"{_q(sharpe):>7.2f}  "
            f"{_q(r.profit_factor):>6.2f}  "
            f"{entry.longs:>5d}  "
            f"{entry.shorts:>6d}  "
            f"{entry.configs_tested:>5d}"
        )

    _print()


def _print_best_per_timeframe(all_entries: list[OptimizedEntry]) -> None:
    """Print best strategy per timeframe + overall best."""
    _print()
    _print("\u2554" + "\u2550" * 100 + "\u2557")
    title = "BEST OPTIMIZED STRATEGY PER TIMEFRAME"
    _print("\u2551" + title.center(100) + "\u2551")
    _print("\u255a" + "\u2550" * 100 + "\u255d")
    _print()

    header = (
        f"  {'TF':<5s} {'Best by P/L':<32s} {'P/L':>12s}  "
        f"{'Best by WR':<28s} {'WR':>6s}  {'Trades':>6s}"
    )
    _print(header)
    _print("  " + "\u2500" * 98)

    for tf_label in ["5m", "15m", "30m", "1h", "4h", "1d"]:
        tf_entries = [e for e in all_entries if e.timeframe == tf_label]
        if not tf_entries:
            continue

        best_pnl = max(tf_entries, key=attrgetter("result.total_pnl"))
        best_wr = max(tf_entries, key=attrgetter("result.win_rate"))

        pnl = best_pnl.result.total_pnl
        sign = "+" if pnl >= 0 else ""

        _print(
            f"  {tf_label:<5s} {best_pnl.display_name:<32s} "
            f"{sign}${_q(pnl):>10,.2f}  "
            f"{best_wr.display_name:<28s} "
            f"{_q(best_wr.result.win_rate):>5.1f}%  "
            f"{best_wr.result.total_trades:>6d}"
        )

    _print()

    # Overall best
    best_overall = max(all_entries, key=attrgetter("result.total_pnl"))
    pnl = best_overall.result.total_pnl
    sign = "+" if pnl >= 0 else ""
    _print(
        f"  \u2605 OVERALL BEST: {best_overall.display_name} on {best_overall.timeframe} "
        f"({sign}${_q(pnl):,.2f}, "
        f"{_q(best_overall.result.win_rate):.1f}% WR, "
        f"{best_overall.result.total_trades} trades, "
        f"{best_overall.longs}L/{best_overall.shorts}S)"
    )

    # Print its optimal params
    if best_overall.best_kwargs:
        params_str = ", ".join(
            f"{k}={v}" for k, v in sorted(best_overall.best_kwargs.items())
        )
        _print(f"    Params: {params_str}")

    _print()


def main() -> None:
    """Run the multi-timeframe parameter-optimized tournament."""
    global _out

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _OUTPUT_DIR / "optimizer_tournament.log"
    log_file = open(log_path, "w", encoding="utf-8")
    _out = TeeWriter(log_file)

    total_start = time.time()

    _print("\u2554" + "\u2550" * 110 + "\u2557")
    title = "MULTI-TIMEFRAME PARAMETER-OPTIMIZED TOURNAMENT \u2014 SPY"
    _print("\u2551" + title.center(110) + "\u2551")
    _print("\u255a" + "\u2550" * 110 + "\u255d")
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
    _print()

    # Ensure strategies are registered
    _ensure_registrations()

    daily_names = [
        name for name in _DAILY_STRATEGIES
        if name in StrategyRegistry.list_names(category="daily")
    ]
    _print(f"Daily strategies to optimize: {len(daily_names)}")
    for name in daily_names:
        entry = StrategyRegistry.get(name)
        n_params = sum(len(v) for v in entry.param_space.values())
        _print(f"  - {entry.display_name} ({n_params} param values to test)")
    _print(f"Timeframes: {', '.join(tf[1] for tf in _TIMEFRAMES)}")
    _print(f"Plus VWAP v11.2 optimizer on 5m (~200 configs)")
    _print()

    # Aggregate timeframes
    _print("Aggregating timeframes...")
    agg = TimeframeAggregator(raw_data)

    all_entries: list[OptimizedEntry] = []
    total_configs = 0

    for tf_enum, tf_label in _TIMEFRAMES:
        tf_data = agg.as_intraday_price_data(tf_enum)
        bar_count = len(tf_data)
        tf_days = len({bar.trading_date for bar in tf_data})

        _print()
        _print("\u2554" + "\u2550" * 110 + "\u2557")
        tf_title = f"OPTIMIZING TIMEFRAME: {tf_label} ({bar_count:,} bars, {tf_days} trading days)"
        _print("\u2551" + tf_title.center(110) + "\u2551")
        _print("\u255a" + "\u2550" * 110 + "\u255d")
        _print()

        tf_entries: list[OptimizedEntry] = []

        # Optimize each daily strategy
        for strat_name in daily_names:
            entry = StrategyRegistry.get(strat_name)
            _print()
            _print(f"  >>> Optimizing: {entry.display_name} on {tf_label}...")
            _print()

            strat_start = time.time()
            opt_entry = _optimize_daily_strategy(strat_name, tf_data, log_file)
            strat_elapsed = time.time() - strat_start

            if opt_entry is not None:
                opt_entry.timeframe = tf_label
                opt_entry.opt_time = strat_elapsed
                tf_entries.append(opt_entry)
                total_configs += opt_entry.configs_tested

                pnl = opt_entry.result.total_pnl
                sign = "+" if pnl >= 0 else ""
                _print(
                    f"  <<< {entry.display_name} on {tf_label}: "
                    f"{sign}${_q(pnl):,.2f}  "
                    f"WR: {_q(opt_entry.result.win_rate):.1f}%  "
                    f"Trades: {opt_entry.result.total_trades}  "
                    f"({opt_entry.configs_tested} configs in {strat_elapsed:.1f}s)"
                )

        # VWAP (5m only)
        if tf_enum == Timeframe.M5:
            _print()
            _print(f"  >>> Optimizing: VWAP v11.2 on {tf_label}...")
            _print()

            vwap_start = time.time()
            vwap_entry = _optimize_vwap(tf_data, log_file)
            vwap_elapsed = time.time() - vwap_start

            if vwap_entry is not None:
                vwap_entry.opt_time = vwap_elapsed
                tf_entries.append(vwap_entry)
                total_configs += vwap_entry.configs_tested

                pnl = vwap_entry.result.total_pnl
                sign = "+" if pnl >= 0 else ""
                _print(
                    f"  <<< VWAP v11.2 on {tf_label}: "
                    f"{sign}${_q(pnl):,.2f}  "
                    f"WR: {_q(vwap_entry.result.win_rate):.1f}%  "
                    f"Trades: {vwap_entry.result.total_trades}  "
                    f"({vwap_entry.configs_tested} configs in {vwap_elapsed:.1f}s)"
                )

        _print_timeframe_table(tf_label, bar_count, tf_entries)
        all_entries.extend(tf_entries)

    # Grand comparisons
    if all_entries:
        _print_grand_comparison(all_entries)
        _print_best_per_timeframe(all_entries)

    total_elapsed = time.time() - total_start

    _print()
    _print("=" * 112)
    _print(f"  Tournament completed in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} minutes)")
    _print(f"  Total configurations tested: {total_configs:,}")
    _print(f"  Strategy × timeframe combinations: {len(all_entries)}")
    _print(f"  Results logged to: {log_path}")
    _print("=" * 112)
    _print()
    _print("  DISCLAIMER: This is for educational purposes only.")
    _print("  Not financial advice. Past performance does not guarantee future results.")
    _print()

    log_file.close()


def _ensure_registrations() -> None:
    """Ensure all strategy registrations are loaded."""
    from stockdownloader.strategy.registrations import ensure_registered
    ensure_registered()


if __name__ == "__main__":
    main()
