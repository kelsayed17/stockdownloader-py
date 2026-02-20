"""CLI for walk-forward validation of intraday strategies.

Loads intraday bar data and runs rolling walk-forward validation
across all native intraday strategies to detect overfitting.

Usage::

    walk-forward                                   # data/spy/5m_bars.csv
    walk-forward --csv data/spy/5m_bars.csv        # explicit CSV
    walk-forward --windows 5                       # number of WF windows
    walk-forward --is-ratio 0.7                    # IS/OOS split ratio
"""
from __future__ import annotations

import argparse
import logging
import time
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.walk_forward import WalkForwardValidator
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
from stockdownloader.strategy.intraday.or_reversal_strategy import ORReversalStrategy
from stockdownloader.strategy.intraday.pattern_scalp_strategy import PatternScalpStrategy
from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategy
from stockdownloader.strategy.intraday.reversal_strategy import ReversalStrategy
from stockdownloader.util.constants import DEFAULT_DATA_FILE, INITIAL_CAPITAL, RISK_PER_TRADE

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the walk-forward validation CLI."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Walk-forward validation for intraday strategies",
    )
    parser.add_argument(
        "--csv", dest="csv_file", default=str(DEFAULT_DATA_FILE),
        help="Intraday CSV file path (default: data/spy/5m_bars.csv)",
    )
    parser.add_argument(
        "--windows", type=int, default=5,
        help="Number of walk-forward windows (default: 5)",
    )
    parser.add_argument(
        "--is-ratio", type=float, default=0.7, dest="is_ratio",
        help="In-sample fraction (default: 0.7 = 70/30 split)",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    data = IntradayCsvLoader.load_from_file(args.csv_file)
    if not data:
        print(f"ERROR: No data loaded from {args.csv_file}")
        return

    trading_days = len({bar.date[:10] for bar in data})
    print(f"Loaded {len(data):,} bars across {trading_days} trading days")
    print(f"Date range: {data[0].date[:10]} to {data[-1].date[:10]}")
    print()

    # ── Build validator and engine ────────────────────────────────────
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

    # ── Run validation ────────────────────────────────────────────────
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

    # ── Summary table ─────────────────────────────────────────────────
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


def _status_label(degradation: float) -> str:
    """Map degradation ratio to a human-readable label."""
    if degradation >= 0.8:
        return "ROBUST"
    if degradation >= 0.5:
        return "ACCEPTABLE"
    return "OVERFIT"


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
    main()
