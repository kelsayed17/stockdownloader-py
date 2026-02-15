"""CLI for automated VWAP strategy parameter optimization.

Loads intraday bar data, runs a multi-phase parameter search, and
reports the best configuration along with before/after comparison.

Usage::

    strategy-optimize                                  # data/spy_5m_bars.csv
    strategy-optimize --csv data/spy_5m_bars.csv       # explicit CSV
    strategy-optimize --csv data/aapl_5m_bars.csv      # different symbol
    strategy-optimize --log opt.log                    # tailable log file
    strategy-optimize --all-strategies                 # compare all 8 strategies

Tip: for real-time output when running in the background::

    PYTHONUNBUFFERED=1 strategy-optimize --log opt.log &
    tail -f opt.log
"""
from __future__ import annotations

import argparse
import logging
from decimal import Decimal

from stockdownloader.backtest.strategy_optimizer import StrategyOptimizer
from stockdownloader.data.intraday_csv_loader import IntradayCsvLoader


def main() -> None:
    """Entry point for the strategy optimizer CLI."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Automated VWAP strategy parameter optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  strategy-optimize                            # default SPY CSV\n"
            "  strategy-optimize --csv data/spy_5m_bars.csv\n"
            "  strategy-optimize --capital 50000\n"
            "  strategy-optimize --log opt.log              # tail -f opt.log\n"
            "  strategy-optimize --all-strategies           # compare all strategies\n"
        ),
    )
    parser.add_argument(
        "--csv",
        dest="csv_file",
        default="data/spy_5m_bars.csv",
        help="Intraday CSV file path (default: data/spy_5m_bars.csv)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=0.01,
        help="Risk per trade as decimal (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--log",
        dest="log_file",
        default=None,
        help="Write output to a log file (use with 'tail -f' for real-time)",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        dest="all_strategies",
        help="After VWAP optimization, run all 8 strategies and print comparison",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Optimize a specific daily strategy by name (e.g., rsi, macd, sma).",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        dest="list_strategies",
        help="List all available strategies and exit.",
    )
    args = parser.parse_args()

    if args.list_strategies:
        from stockdownloader.strategy.registrations import ensure_registered
        from stockdownloader.strategy.registry import StrategyRegistry
        ensure_registered()
        print("Available strategies for optimization:")
        print()
        print("  Intraday (native VWAP optimizer):")
        for entry in StrategyRegistry.all_entries(category="intraday"):
            print(f"    {entry.name:<15s}  {entry.display_name}")
        print()
        print("  Daily (via DailyStrategyOptimizer):")
        for entry in StrategyRegistry.all_entries(category="daily"):
            params = ", ".join(entry.param_space.keys()) if entry.param_space else "(no tunable params)"
            print(f"    {entry.name:<15s}  {entry.display_name:<30s}  params: {params}")
        return

    log_fh = None
    if args.log_file:
        log_fh = open(args.log_file, "w", encoding="utf-8")

    try:
        print("Loading data...", flush=True)
        data = IntradayCsvLoader.load_from_file(args.csv_file)
        if not data:
            print(f"ERROR: No data loaded from {args.csv_file}", flush=True)
            return

        trading_days = len({bar.date[:10] for bar in data})
        print(f"Loaded {len(data):,} bars across {trading_days} trading days", flush=True)
        print(f"Date range: {data[0].date[:10]} to {data[-1].date[:10]}", flush=True)
        print(flush=True)

        capital = Decimal(str(args.capital))
        risk = Decimal(str(args.risk))

        if args.strategy:
            from stockdownloader.strategy.registrations import ensure_registered
            from stockdownloader.strategy.registry import StrategyRegistry
            ensure_registered()
            entry = StrategyRegistry.get(args.strategy)
            if entry.category == "intraday" and entry.name == "vwap":
                # Use the existing VWAP optimizer
                optimizer = StrategyOptimizer(
                    data,
                    initial_capital=capital,
                    risk_per_trade=risk,
                    log_file=log_fh,
                )
                optimizer.optimize()
            elif entry.category == "daily":
                from stockdownloader.backtest.daily_strategy_optimizer import DailyStrategyOptimizer
                opt = DailyStrategyOptimizer(
                    strategy_name=args.strategy,
                    data=data,
                    initial_capital=capital,
                    risk_per_trade=risk,
                    log_file=log_fh,
                )
                opt.optimize()
            else:
                print(f"ERROR: Strategy '{args.strategy}' (category '{entry.category}') "
                      "is not optimizable.", flush=True)
        elif args.all_strategies:
            optimizer = StrategyOptimizer(
                data,
                initial_capital=capital,
                risk_per_trade=risk,
                log_file=log_fh,
            )
            optimizer.optimize_all_strategies()
        else:
            optimizer = StrategyOptimizer(
                data,
                initial_capital=capital,
                risk_per_trade=risk,
                log_file=log_fh,
            )
            optimizer.optimize()
    finally:
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    main()
