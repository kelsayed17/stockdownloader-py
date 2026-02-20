"""Tournament application entry points.

Combines the exit mechanism tournament and signal-stack combinatorial
tournament into a single module.

Exit tournament
---------------
Compares multiple exit mechanisms against historical trades using 5-minute
intraday bar data.

Usage::

    python -m stockdownloader.app.tournament_apps exit \\
        --bars data/spy/5m_bars.csv \\
        --trades trades.csv

    python -m stockdownloader.app.tournament_apps exit \\
        --bars data/spy/5m_bars.csv \\
        --trades trades.csv \\
        --stop-distance 0.96

Signal-stack tournament
-----------------------
Exhaustively tests all valid combinations of atomic signal generators
across timeframes, weights, aggregation modes, and thresholds.

Usage::

    python -m stockdownloader.app.tournament_apps signal-stack

Output is written to both stdout and ``output/signal_stack_tournament.log``
for monitoring with ``tail -f output/signal_stack_tournament.log``.

The tournament has three modes controlled by the ``--mode`` argument:

* ``quick`` -- max 3 generators, M5 only, reduced grid (~4,000 configs)
* ``standard`` -- max 4 generators, M5 + M15, balanced grid (~60,000 configs)
* ``full`` -- max 5 generators, all timeframes, full grid (~270,000 configs)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path

from stockdownloader.backtest.exit_tournament_engine import ExitTournamentEngine
from stockdownloader.backtest import exit_tournament_report_formatter
from stockdownloader.backtest.combinatorial_tester import (
    CombinatorialConfig,
    CombinatorialTester,
)
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.data.tradingview_trade_loader import TradingViewTradeLoader
from stockdownloader.strategy.base_registry import SignalGeneratorRegistry
from stockdownloader.strategy.exit_mechanisms import (
    AtrTrailExit,
    HybridExit,
    TimeDecayExit,
    TrailingStopExit,
    VwapBandExit,
    VwapCrossExit,
)
from stockdownloader.strategy.signals.stacked_signal_engine import AggregationMode
from stockdownloader.util.io_helpers import TeeWriter
from stockdownloader.util.timeframe_aggregator import Timeframe

from stockdownloader.util.config_loader import (
    DEFAULT_DATA_FILE as _DATA_FILE,
    DEFAULT_OUTPUT_DIR as _OUTPUT_DIR,
    INITIAL_CAPITAL as _INITIAL_CAPITAL,
    RISK_PER_TRADE as _RISK_PER_TRADE,
)

logger = logging.getLogger(__name__)

_LOG_FILE = _OUTPUT_DIR / "signal_stack_tournament.log"


# ===================================================================
# Exit tournament
# ===================================================================


def main_exit_tournament() -> None:
    """Entry point for the exit tournament application."""
    parser = argparse.ArgumentParser(
        description="Exit Mechanism Tournament",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  exit_tournament_app --bars data/spy/5m_bars.csv --trades trades.csv\n"
            "  exit_tournament_app --bars data/spy/5m_bars.csv --trades trades.csv "
            "--stop-distance 0.96\n"
        ),
    )
    parser.add_argument(
        "--bars", required=True, help="Path to 5-minute OHLCV CSV file"
    )
    parser.add_argument(
        "--trades", required=True, help="Path to TradingView trade export CSV"
    )
    parser.add_argument(
        "--stop-distance",
        type=float,
        default=None,
        help="Default stop distance (1R) in dollars if not inferred from signals",
    )
    parser.add_argument(
        "--tz-offset",
        type=int,
        default=0,
        help="Hours to add to TradingView timestamps (e.g. 3 for PST->EST)",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Label for the report header",
    )
    args = parser.parse_args()

    print("========================================")
    print("  Exit Mechanism Tournament")
    print("========================================")
    print()

    # -- Load intraday bars --

    print(f"Loading 5-minute bars from: {args.bars}")
    bars = IntradayCsvLoader.load_from_file(args.bars)
    if not bars:
        print("ERROR: No bar data loaded.")
        sys.exit(1)

    # Determine date range from bar data
    dates = sorted({b.date[:10] for b in bars})
    print(f"  {len(bars)} bars, {len(dates)} sessions")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print()

    # -- Load trades --

    print(f"Loading trades from: {args.trades}")
    default_sd = (
        Decimal(str(args.stop_distance))
        if args.stop_distance is not None
        else None
    )
    trades = TradingViewTradeLoader.load_from_file(
        args.trades,
        default_stop_distance=default_sd,
        timezone_offset_hours=args.tz_offset,
    )
    if not trades:
        print("ERROR: No trades loaded.")
        sys.exit(1)

    print(f"  {len(trades)} trades loaded")

    # Filter trades to those within the bar data window
    bar_date_set = set(dates)
    trades = [t for t in trades if t.entry_trading_date in bar_date_set]
    print(f"  {len(trades)} trades within bar data window")
    print()

    if not trades:
        print("ERROR: No trades overlap with bar data date range.")
        sys.exit(1)

    # -- Create exit mechanisms --

    mechanisms = [
        TrailingStopExit(),                                             # CURRENT_TRAIL
        VwapCrossExit(activation_r=Decimal("0.5"), name="VWAP_CROSS"),  # VWAP_CROSS
        VwapCrossExit(activation_r=Decimal("1.0"), name="VWAP_CROSS_LATE"),
        AtrTrailExit(),                                                 # ATR_TRAIL
        HybridExit(activation_r=Decimal("0.5"), name="HYBRID"),         # HYBRID
        HybridExit(activation_r=Decimal("1.0"), name="HYBRID_LATE"),
        VwapBandExit(),                                                 # VWAP_BAND
        TimeDecayExit(),                                                # TIME_DECAY
    ]

    # -- Run tournament --

    print("Running tournament...")
    engine = ExitTournamentEngine()
    result = engine.run(trades, bars, mechanisms)

    # -- Print report --

    exit_tournament_report_formatter.print_report(result, label=args.label)
    exit_tournament_report_formatter.print_comparison(result)


# ===================================================================
# Signal-stack combinatorial tournament
# ===================================================================


def _build_config(mode: str) -> CombinatorialConfig:
    """Build a :class:`CombinatorialConfig` for the given mode."""
    if mode == "quick":
        # Focus on the 8 best-performing generator types (from optimizer
        # findings: RSI, BB, SMA, MACD dominated).  Keeps combos to ~100
        # and total configs to ~800 -- finishes in ~20 minutes.
        return CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=3,
            min_category_diversity=2,
            timeframes=[Timeframe.M5],
            weights=[1.0],
            modes=[AggregationMode.WEIGHTED_AVERAGE, AggregationMode.UNANIMOUS],
            buy_thresholds=[0.2, 0.3],
            sell_thresholds=[0.2, 0.3],
            require_fire=True,
            allow_shorts=True,
            initial_capital=_INITIAL_CAPITAL,
            risk_per_trade=_RISK_PER_TRADE,
            generator_names=[
                "rsi", "macd", "sma_cross", "bb_touch",
                "stochastic", "ema_trend", "obv", "volume_surge",
            ],
        )

    if mode == "standard":
        # All 17 generators, 2-3 combos, M5 only, 2 modes, paired thresholds.
        # ~735 combos x 8 settings = ~5,880 configs (~2-3 hours).
        return CombinatorialConfig(
            min_combo_size=2,
            max_combo_size=3,
            min_category_diversity=2,
            timeframes=[Timeframe.M5],
            weights=[1.0],
            modes=[
                AggregationMode.WEIGHTED_AVERAGE,
                AggregationMode.UNANIMOUS,
            ],
            buy_thresholds=[0.2, 0.3],
            sell_thresholds=[0.2, 0.3],
            require_fire=True,
            allow_shorts=True,
            initial_capital=_INITIAL_CAPITAL,
            risk_per_trade=_RISK_PER_TRADE,
        )

    # full mode -- practical sweep: all combos up to size 4, 3 key
    # timeframes, 3 modes, and paired thresholds to keep configs ~50K
    return CombinatorialConfig(
        min_combo_size=2,
        max_combo_size=4,
        min_category_diversity=2,
        timeframes=[
            Timeframe.M5,
            Timeframe.H1,
            Timeframe.DAILY,
        ],
        weights=[1.0],
        modes=[
            AggregationMode.WEIGHTED_AVERAGE,
            AggregationMode.UNANIMOUS,
            AggregationMode.MAJORITY_VOTE,
        ],
        buy_thresholds=[0.2, 0.3, 0.5],
        sell_thresholds=[0.3],
        require_fire=True,
        allow_shorts=True,
        initial_capital=_INITIAL_CAPITAL,
        risk_per_trade=_RISK_PER_TRADE,
    )


def main_signal_stack(
    mode: str = "quick",
    log_path: Path | None = None,
    workers: int = 0,
    data_file: Path | None = None,
) -> None:
    """Run the signal-stack combinatorial tournament.

    Parameters
    ----------
    mode:
        Tournament mode (``quick``, ``standard``, ``full``).
    log_path:
        Custom log file path.
    workers:
        Number of parallel worker processes.  ``0`` (default) auto-detects
        CPU count.  ``1`` runs single-threaded.
    data_file:
        Custom data file path.  If ``None``, uses the default SPY CSV.
    """
    logging.basicConfig(level=logging.WARNING)

    # Ensure output directory exists
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = data_file or _DATA_FILE

    actual_log = log_path or _LOG_FILE
    with open(actual_log, "w", encoding="utf-8") as log_fh:
        tee = TeeWriter(log_fh)

        tee.write("=" * 80 + "\n")
        tee.write("SIGNAL STACK COMBINATORIAL TOURNAMENT\n")
        tee.write(f"Mode: {mode}\n")
        tee.write(f"Workers: {workers if workers else 'auto'}\n")
        tee.write("=" * 80 + "\n\n")

        # Load data
        tee.write(f"Loading data from {csv_path}...\n")
        data = IntradayCsvLoader.load_from_file(csv_path)
        trading_days = len({d.date[:10] for d in data})
        tee.write(
            f"Loaded {len(data)} bars across {trading_days} trading days\n\n"
        )

        # Import generators to trigger registration
        import stockdownloader.strategy.signals.generators  # noqa: F401

        # Show registered generators
        entries = {e.name: e for e in SignalGeneratorRegistry.all_entries()}
        tee.write(f"Registered {len(entries)} signal generators:\n")
        by_cat: dict[str, list[str]] = {}
        for e in entries.values():
            by_cat.setdefault(e.category, []).append(e.display_name)
        for cat in sorted(by_cat):
            tee.write(f"  {cat}: {', '.join(sorted(by_cat[cat]))}\n")
        tee.write("\n")

        # Build config and run
        config = _build_config(mode)
        tee.write(
            f"Config: combo size {config.min_combo_size}-{config.max_combo_size}, "
            f"timeframes={[tf.label for tf in config.timeframes]}, "
            f"modes={[m.value for m in config.modes]}\n\n"
        )

        start = time.time()
        tester = CombinatorialTester(config, data, log_fh)
        results = tester.run(workers=workers)
        elapsed = time.time() - start

        tee.write(f"\nTournament completed in {elapsed:.1f}s\n")
        tee.write(f"Total results: {len(results)}\n\n")

        # Print results
        top_table = CombinatorialTester.format_results(results, top_n=50)
        tee.write(top_table + "\n")

        # Frequency analysis
        freq = CombinatorialTester.frequency_analysis(results, top_n=50)
        tee.write(freq + "\n")

        # Category pair analysis
        cat_pairs = CombinatorialTester.category_pair_analysis(results, entries)
        tee.write(cat_pairs + "\n")

        tee.write(f"\nResults written to {actual_log}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tournament applications (exit tournament, signal-stack tournament)",
    )
    sub = parser.add_subparsers(dest="command")

    # -- exit sub-command --
    sub.add_parser("exit", help="Run exit mechanism tournament")

    # -- signal-stack sub-command --
    ss = sub.add_parser("signal-stack", help="Run signal-stack combinatorial tournament")
    ss.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="quick",
        help="Tournament mode (quick/standard/full)",
    )
    ss.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Custom log file path (default: output/signal_stack_tournament.log)",
    )
    ss.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes. "
            "0 = auto-detect CPU count (default), 1 = single-threaded."
        ),
    )
    ss.add_argument(
        "--file",
        type=str,
        default=None,
        help="Custom data file path (default: data/spy/5m_bars.csv)",
    )

    args = parser.parse_args()
    if args.command == "exit":
        main_exit_tournament()
    elif args.command == "signal-stack":
        log_p = Path(args.log_file) if args.log_file else None
        data_f = Path(args.file) if args.file else None
        main_signal_stack(args.mode, log_p, workers=args.workers, data_file=data_f)
    else:
        parser.print_help()
