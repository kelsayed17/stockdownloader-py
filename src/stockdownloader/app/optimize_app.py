"""Strategy optimization and walk-forward validation CLIs.

Combines the parameter optimizer and walk-forward validator into a
single module.

Optimization
------------
Loads intraday bar data, runs a multi-phase parameter search, and
reports the best configuration along with before/after comparison.

Usage::

    strategy-optimize                                  # data/SPY/5m_bars.csv
    strategy-optimize --csv data/SPY/5m_bars.csv       # explicit CSV
    strategy-optimize --csv data/AAPL/5m_bars.csv      # different symbol
    strategy-optimize --log opt.log                    # tailable log file
    strategy-optimize --all-strategies                 # compare all 8 strategies

Tip: for real-time output when running in the background::

    PYTHONUNBUFFERED=1 strategy-optimize --log opt.log &
    tail -f opt.log

Walk-forward validation
-----------------------
Loads intraday bar data and runs rolling walk-forward validation
across all native intraday strategies to detect overfitting.

Usage::

    walk-forward                                   # data/SPY/5m_bars.csv
    walk-forward --csv data/SPY/5m_bars.csv        # explicit CSV
    walk-forward --windows 5                       # number of WF windows
    walk-forward --is-ratio 0.7                    # IS/OOS split ratio
"""
from __future__ import annotations

import argparse
import logging
import time
from decimal import Decimal

from stockdownloader.app.app_helpers import add_intraday_csv_arg, add_log_arg
from stockdownloader.app.app_helpers import status_label as _status_label
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.strategy_optimizer import StrategyOptimizer
from stockdownloader.backtest.walk_forward import WalkForwardValidator
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy
from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategy
from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy
from stockdownloader.strategies.intraday.pullback import PullbackStrategy
from stockdownloader.strategies.intraday.reversal import ReversalStrategy
from stockdownloader.core.config import INITIAL_CAPITAL, RISK_PER_TRADE

logger = logging.getLogger(__name__)


# ===================================================================
# Strategy parameter optimization
# ===================================================================


def main_optimize() -> None:
    """Entry point for the strategy optimizer CLI."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Automated VWAP strategy parameter optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  strategy-optimize                            # default SPY CSV\n"
            "  strategy-optimize --csv data/SPY/5m_bars.csv\n"
            "  strategy-optimize --capital 50000\n"
            "  strategy-optimize --log opt.log              # tail -f opt.log\n"
            "  strategy-optimize --all-strategies           # compare all strategies\n"
        ),
    )
    add_intraday_csv_arg(parser)
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
    add_log_arg(parser, help="Write output to a log file (use with 'tail -f' for real-time)")
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
        from stockdownloader.strategies.loader import ensure_registered
        from stockdownloader.strategies.registry import StrategyRegistry
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
        logger.info("Loading data...")
        data = IntradayCsvLoader.load_from_file(args.csv_file)
        if not data:
            logger.error("No data loaded from %s", args.csv_file)
            return

        trading_days = len({bar.date[:10] for bar in data})
        logger.info("Loaded %s bars across %d trading days", f"{len(data):,}", trading_days)
        logger.info("Date range: %s to %s", data[0].date[:10], data[-1].date[:10])

        capital = Decimal(str(args.capital))
        risk = Decimal(str(args.risk))

        if args.strategy:
            from stockdownloader.strategies.loader import ensure_registered
            from stockdownloader.strategies.registry import StrategyRegistry
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


# ===================================================================
# Walk-forward validation
# ===================================================================


def main_walk_forward() -> None:
    """Entry point for the walk-forward validation CLI."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Walk-forward validation for intraday strategies",
    )
    add_intraday_csv_arg(parser)
    parser.add_argument(
        "--windows", type=int, default=5,
        help="Number of walk-forward windows (default: 5)",
    )
    parser.add_argument(
        "--is-ratio", type=float, default=0.7, dest="is_ratio",
        help="In-sample fraction (default: 0.7 = 70/30 split)",
    )
    args = parser.parse_args()

    # -- Load data --
    print("Loading data...", flush=True)
    data = IntradayCsvLoader.load_from_file(args.csv_file)
    if not data:
        print(f"ERROR: No data loaded from {args.csv_file}")
        return

    trading_days = len({bar.date[:10] for bar in data})
    print(f"Loaded {len(data):,} bars across {trading_days} trading days")
    print(f"Date range: {data[0].date[:10]} to {data[-1].date[:10]}")
    print()

    # -- Build validator and engine --
    validator = WalkForwardValidator(
        data, n_windows=args.windows, is_ratio=args.is_ratio,
    )
    engine = IntradayBacktestEngine(
        INITIAL_CAPITAL, RISK_PER_TRADE,
        vol_scale=False, dd_throttle=True,
    )

    print(
        f"Walk-forward config: {args.windows} windows, "
        f"{args.is_ratio:.0%} IS / {1 - args.is_ratio:.0%} OOS",
    )
    print()

    # -- Run validation --
    strategies = [
        ("VWAP Pullback", PullbackStrategy),
        ("VWAP Reversal", ReversalStrategy),
        ("OR Breakout", ORBreakoutStrategy),
        ("OR Reversal", ORReversalStrategy),
        ("Pattern Scalp", PatternScalpStrategy),
    ]

    print("=" * 78)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 78)
    print()

    results = []
    t0 = time.time()

    for name, cls in strategies:
        print(f"  Validating: {name}...", flush=True)
        result = validator.validate(
            strategy_factory=lambda c=cls: c(),
            engine=engine,
            strategy_name=name,
        )
        results.append(result)

        # Per-window detail
        for i, (is_r, oos_r) in enumerate(
            zip(result.in_sample_results, result.out_of_sample_results, strict=True),
        ):
            is_trades = is_r.total_trades
            oos_trades = oos_r.total_trades
            is_ret = float(is_r.total_pnl) / float(INITIAL_CAPITAL) * 100
            oos_ret = float(oos_r.total_pnl) / float(INITIAL_CAPITAL) * 100
            print(
                f"    Window {i + 1}: IS {is_ret:>+6.2f}% ({is_trades:>2d} trades)  "
                f"OOS {oos_ret:>+6.2f}% ({oos_trades:>2d} trades)",
            )

        _print_status(result)
        print()

    elapsed = time.time() - t0

    # -- Summary table --
    print("-" * 78)
    print(
        f"  {'Strategy':<25s} {'IS Score':>10s}  {'OOS Score':>10s}  "
        f"{'Degrad.':>8s}  {'Status':>12s}",
    )
    print("-" * 78)

    for r in results:
        status = _status_label(r.degradation_ratio)
        print(
            f"  {r.strategy_name:<25s} {r.in_sample_score:>10.2f}  "
            f"{r.out_of_sample_score:>10.2f}  "
            f"{r.degradation_ratio:>8.2f}  {status:>12s}",
        )

    print("-" * 78)
    print()
    print(f"  Time elapsed: {elapsed:.1f}s")
    print()
    print("  Legend: Degradation = OOS / IS score ratio")
    print("    >= 0.8 = ROBUST   0.5-0.8 = ACCEPTABLE   < 0.5 = OVERFIT")
    print()


def _print_status(result) -> None:  # noqa: ANN001
    """Print summary line for a single strategy."""
    status = _status_label(result.degradation_ratio)
    overfit_flag = " *** OVERFIT ***" if result.degradation_ratio < 0.5 else ""
    print(
        f"    => IS: {result.in_sample_score:>7.2f}  "
        f"OOS: {result.out_of_sample_score:>7.2f}  "
        f"Degradation: {result.degradation_ratio:>5.2f}  "
        f"[{status}]{overfit_flag}",
    )


if __name__ == "__main__":
    main_optimize()
