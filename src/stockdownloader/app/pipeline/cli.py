"""Pipeline CLI — argument parsing and entry point."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import TextIO

from stockdownloader.app.app_helpers import add_intraday_csv_arg, add_log_arg
from stockdownloader.app.pipeline.helpers import (
    load_daily_data,
    load_intraday_data,
    make_print_fn,
)
from stockdownloader.app.pipeline.models import SlotResult
from stockdownloader.app.pipeline.report import print_holistic_report
from stockdownloader.app.pipeline.stages import (
    _run_baseline,
    _run_optimize,
    _run_rebacktest,
    _run_walkforward,
)
from stockdownloader.strategy.registration_loader import ensure_registered
from stockdownloader.strategy.base_registry import StrategyRegistry
from stockdownloader.util.io import TeeWriter

logger = logging.getLogger(__name__)

_CATEGORIES = ("intraday", "daily", "options")
_STAGES = ("backtest", "optimize", "validate", "all")
_OPTIMIZE_MODES = ("wf", "full", "skip")


def main() -> None:
    """Entry point for the unified strategy pipeline."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Unified strategy pipeline — backtest, optimize, validate",
    )
    add_intraday_csv_arg(parser)
    parser.add_argument(
        "--stage", choices=_STAGES, default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--category", choices=_CATEGORIES, default=None,
        help="Filter to a single strategy category",
    )
    parser.add_argument(
        "--strategy", default=None,
        help="Run a single strategy by registry name (e.g., vwap-orr)",
    )
    parser.add_argument(
        "--optimize-mode", choices=_OPTIMIZE_MODES, default="wf",
        help="Optimizer mode: wf (walk-forward, default), full (full-data), skip",
    )
    add_log_arg(parser, help="Save output to log file")
    args = parser.parse_args()

    # Set up output
    tee: TeeWriter | None = None
    log_fh: TextIO | None = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")
        tee = TeeWriter(log_fh)
    out = make_print_fn(tee)

    try:
        _run_pipeline(args, out)
    finally:
        if log_fh:
            log_fh.close()


def _run_pipeline(args, out) -> None:
    """Execute the pipeline stages."""
    ensure_registered()
    t_start = time.time()

    out("=" * 78)
    out("  UNIFIED STRATEGY PIPELINE")
    out("=" * 78)

    # ── Load data ─────────────────────────────────────────────────────
    intraday_data = load_intraday_data(args.csv_file, out)
    if not intraday_data:
        return

    # Load daily data for options (aggregated from 5-min bars)
    daily_data = []
    categories = [args.category] if args.category else list(_CATEGORIES)
    if "options" in categories:
        try:
            daily_data = load_daily_data(intraday_data, out)
        except Exception as e:
            out(f"  WARNING: Could not aggregate daily data for options: {e}")
            out("  Options strategies will be skipped.")

    # ── Build slots ───────────────────────────────────────────────────
    slots: list[SlotResult] = []

    if args.strategy:
        # Single strategy mode
        entry = StrategyRegistry.get(args.strategy)
        slots.append(SlotResult(
            name=entry.name,
            display_name=entry.display_name,
            category=entry.category,
        ))
    else:
        for cat in categories:
            for entry in StrategyRegistry.all_entries(category=cat):
                slots.append(SlotResult(
                    name=entry.name,
                    display_name=entry.display_name,
                    category=entry.category,
                ))

    out(f"\n  Strategies: {len(slots)} "
        f"({', '.join(f'{c}: {sum(1 for s in slots if s.category == c)}' for c in _CATEGORIES if any(s.category == c for s in slots))})")

    stage = args.stage

    # ── Stage 1: Baseline ─────────────────────────────────────────────
    if stage in ("backtest", "all"):
        _run_baseline(slots, intraday_data, daily_data, out)

    # ── Stage 2: Optimize ─────────────────────────────────────────────
    optimize_mode = getattr(args, "optimize_mode", "wf")
    if stage in ("optimize", "all") and optimize_mode != "skip":
        if stage == "optimize" and not any(s.baseline for s in slots):
            # Need baseline first for optimize-only mode
            _run_baseline(slots, intraday_data, daily_data, out)
        _run_optimize(slots, intraday_data, optimize_mode, out)
    elif optimize_mode == "skip" and stage in ("optimize", "all"):
        out()
        out("=" * 78)
        out("  STAGE 2: OPTIMIZE (skipped — --optimize-mode skip)")
        out("=" * 78)

    # ── Stage 3: Re-backtest ──────────────────────────────────────────
    if stage == "all":
        _run_rebacktest(slots, intraday_data, out)

    # ── Stage 4: Walk-Forward ─────────────────────────────────────────
    if stage in ("validate", "all"):
        if stage == "validate" and not any(s.baseline for s in slots):
            _run_baseline(slots, intraday_data, daily_data, out)
        _run_walkforward(slots, intraday_data, out)

    # ── Holistic Report ───────────────────────────────────────────────
    elapsed = time.time() - t_start
    print_holistic_report(slots, intraday_data, elapsed, out)


if __name__ == "__main__":
    main()
