"""Signal-stack combinatorial tournament on SPY.

Exhaustively tests all valid combinations of atomic signal generators
across timeframes, weights, aggregation modes, and thresholds.

Usage::

    python -m stockdownloader.app.signal_stack_tournament

Output is written to both stdout and ``output/signal_stack_tournament.log``
for monitoring with ``tail -f output/signal_stack_tournament.log``.

The tournament has three modes controlled by the ``--mode`` argument:

* ``quick`` — max 3 generators, M5 only, reduced grid (~4,000 configs)
* ``standard`` — max 4 generators, M5 + M15, balanced grid (~60,000 configs)
* ``full`` — max 5 generators, all timeframes, full grid (~270,000 configs)
"""
from __future__ import annotations

import argparse
import logging
import time
from decimal import Decimal
from pathlib import Path

from stockdownloader.backtest.combinatorial_tester import (
    CombinatorialConfig,
    CombinatorialTester,
)
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.strategy.base_registry import SignalGeneratorRegistry
from stockdownloader.strategy.signals.stacked_signal_engine import AggregationMode
from stockdownloader.util.file_helper import TeeWriter
from stockdownloader.util.timeframe_aggregator import Timeframe

from stockdownloader.util.constants import (
    DEFAULT_DATA_FILE as _DATA_FILE,
    DEFAULT_OUTPUT_DIR as _OUTPUT_DIR,
    INITIAL_CAPITAL as _INITIAL_CAPITAL,
    RISK_PER_TRADE as _RISK_PER_TRADE,
)

logger = logging.getLogger(__name__)

_LOG_FILE = _OUTPUT_DIR / "signal_stack_tournament.log"


def _build_config(mode: str) -> CombinatorialConfig:
    """Build a :class:`CombinatorialConfig` for the given mode."""
    if mode == "quick":
        # Focus on the 8 best-performing generator types (from optimizer
        # findings: RSI, BB, SMA, MACD dominated).  Keeps combos to ~100
        # and total configs to ~800 — finishes in ~20 minutes.
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
        # All 17 generators, 2–3 combos, M5 only, 2 modes, paired thresholds.
        # ~735 combos × 8 settings = ~5,880 configs (~2–3 hours).
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

    # full mode — practical sweep: all combos up to size 4, 3 key
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


def main(
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
            f"Config: combo size {config.min_combo_size}–{config.max_combo_size}, "
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
        description="Signal-stack combinatorial tournament",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="quick",
        help="Tournament mode (quick/standard/full)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Custom log file path (default: output/signal_stack_tournament.log)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes. "
            "0 = auto-detect CPU count (default), 1 = single-threaded."
        ),
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Custom data file path (default: data/spy/5m_bars.csv)",
    )
    args = parser.parse_args()
    log_path = Path(args.log_file) if args.log_file else None
    data_file = Path(args.file) if args.file else None
    main(args.mode, log_path, workers=args.workers, data_file=data_file)
