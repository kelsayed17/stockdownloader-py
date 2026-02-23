"""Tournament CLI — argument parsing and entry points."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from stockdownloader.app.app_helpers import (
    DEFAULT_DATA_FILE as _DATA_FILE,
    DEFAULT_OUTPUT_DIR as _OUTPUT_DIR,
)
from stockdownloader.app.tournament.helpers import _box_title, _load_data
from stockdownloader.app.tournament.stages import (
    _run_optimization,
    _run_round_robin,
    _run_walkforward,
)
from stockdownloader.app.tournament.stages_advanced import (
    _run_bracket,
    _run_monte_carlo_stage,
    _run_portfolio,
    _run_regime_analysis,
)
from stockdownloader.backtest.tournament_engine import (
    TournamentResult,
    apply_cross_timeframe_bonus,
)
from stockdownloader.strategy.registration_loader import ensure_registered
from stockdownloader.util.io import TeeWriter

_MODES = ("all", "roundrobin", "bracket", "portfolio")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-timeframe strategy tournament",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=_DATA_FILE,
        help="Path to base 5-minute bar CSV file",
    )
    parser.add_argument(
        "--extra-csv",
        action="append",
        default=[],
        help="Extra CSV for a timeframe: TF:path (e.g., 15m:data/spy_15m.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=_MODES,
        default="all",
        help="Tournament mode (default: all)",
    )
    parser.add_argument(
        "--bracket-size",
        type=int,
        default=16,
        help="Number of combos in elimination bracket (default: 16)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max strategies in portfolio (default: 5)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip walk-forward optimization (faster, uses baseline params)",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Skip walk-forward validation",
    )
    parser.add_argument(
        "--no-regime",
        action="store_true",
        help="Skip regime-aware analysis",
    )
    parser.add_argument(
        "--no-monte-carlo",
        action="store_true",
        help="Skip Monte Carlo robustness testing",
    )
    parser.add_argument(
        "--mc-top-n",
        type=int,
        default=20,
        help="Number of top strategies for Monte Carlo testing (default: 20)",
    )
    parser.add_argument(
        "--mc-simulations",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations (default: 1000)",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default=None,
        help="Comma-separated timeframes to test (e.g., 5m,15m,1h)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter strategies by category (intraday or daily)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file path (default: output/tournament_results.log)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the multi-timeframe strategy tournament."""
    args = _parse_args(argv)
    ensure_registered()

    # Set up output
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = args.log or (_OUTPUT_DIR / "tournament_results.log")
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    tee = TeeWriter(log_file)

    def out(msg: str = "") -> None:
        tee.write(msg + "\n")

    start_time = time.time()

    out(_box_title("MULTI-TIMEFRAME STRATEGY TOURNAMENT"))
    out()

    # Parse extra CSVs
    extra_csvs: dict[str, Path] = {}
    for spec in args.extra_csv:
        if ":" in spec:
            tf, path = spec.split(":", 1)
            extra_csvs[tf] = Path(path)

    # Parse timeframe filter
    tf_filter = None
    if args.timeframes:
        tf_filter = [t.strip() for t in args.timeframes.split(",")]

    # Load data
    tf_data = _load_data(args.csv_file, extra_csvs, tf_filter, out)
    if not tf_data:
        log_file.close()
        return

    mode = args.mode
    tournament = TournamentResult()

    # Stage 1: Round-Robin (always runs)
    if mode in ("all", "roundrobin", "bracket", "portfolio"):
        tournament.combos = _run_round_robin(tf_data, args.category, out)

    # Stage 1.5: Walk-Forward Optimization (optional)
    if not args.no_optimize and mode in ("all",):
        tournament.combos = _run_optimization(tournament.combos, tf_data, out)

    # Stage 2: Walk-Forward Validation (optional)
    if not args.no_walk_forward and mode in ("all",):
        tournament.combos = _run_walkforward(tournament.combos, tf_data, out)

    # Stage 2.5: Regime Analysis (optional)
    if not args.no_regime and mode in ("all",):
        tournament.combos = _run_regime_analysis(tournament.combos, tf_data, out)

    # Stage 2.75: Monte Carlo Robustness (optional)
    if not args.no_monte_carlo and mode in ("all",):
        tournament.combos = _run_monte_carlo_stage(
            tournament.combos, tf_data, args.mc_top_n, args.mc_simulations, out,
        )

    # Cross-timeframe consistency bonus (before bracket)
    if mode in ("all", "bracket", "portfolio") and len(tf_data) > 1:
        apply_cross_timeframe_bonus(tournament.combos)
        tournament.combos.sort(key=lambda c: c.tournament_score, reverse=True)

    # Stage 3: Elimination Bracket
    if mode in ("all", "bracket"):
        matches, champion = _run_bracket(
            tournament.combos, args.bracket_size, out,
        )
        tournament.matches = matches
        tournament.bracket_champion = champion

    # Stage 4: Portfolio Analysis
    if mode in ("all", "portfolio"):
        portfolio, corr = _run_portfolio(
            tournament.combos, args.top_k, out,
        )
        tournament.portfolio = portfolio
        tournament.correlation_matrix = corr

    # Final summary
    elapsed = time.time() - start_time
    out()
    out("=" * 100)
    out(f"  Tournament completed in {elapsed:.1f}s")
    out(f"  Total combinations tested: {len(tournament.combos)}")

    valid = [c for c in tournament.combos if c.best_result is not None]
    profitable = [c for c in valid if c.best_result.total_pnl > 0]
    out(f"  Profitable: {len(profitable)}/{len(valid)}")

    opt_count = sum(1 for c in tournament.combos if c.optimized is not None)
    if opt_count:
        out(f"  Optimized: {opt_count} combos improved via WF optimization")

    if tournament.bracket_champion:
        out(f"  Bracket Champion: {tournament.bracket_champion.label}")

    if tournament.portfolio:
        out(f"  Portfolio: {len(tournament.portfolio)} strategies")

    regime_count = sum(1 for c in tournament.combos if c.regime_analysis is not None)
    if regime_count:
        versatile = sum(
            1 for c in tournament.combos
            if c.regime_analysis and c.regime_analysis.regime_coverage >= 3
        )
        narrow = regime_count - versatile
        out(f"  Regime: {versatile} VERSATILE, {narrow} NARROW")

    mc_count = sum(1 for c in tournament.combos if c.monte_carlo is not None)
    if mc_count:
        mc_robust = sum(
            1 for c in tournament.combos
            if c.monte_carlo and c.monte_carlo.is_robust
        )
        mc_fragile = mc_count - mc_robust
        out(f"  Monte Carlo: {mc_robust} ROBUST, {mc_fragile} FRAGILE")

    wf_count = sum(1 for c in tournament.combos if c.wf_result is not None)
    if wf_count:
        robust = sum(
            1 for c in tournament.combos
            if c.wf_result and c.wf_result.degradation_ratio >= 0.8
        )
        acceptable = sum(
            1 for c in tournament.combos
            if c.wf_result and 0.5 <= c.wf_result.degradation_ratio < 0.8
        )
        overfit = sum(
            1 for c in tournament.combos
            if c.wf_result and c.wf_result.degradation_ratio < 0.5
        )
        out(f"  Walk-forward: {robust} ROBUST, {acceptable} ACCEPTABLE, {overfit} OVERFIT")

    out(f"  Results logged to: {log_path}")
    out("=" * 100)
    out()

    log_file.close()


# ======================================================================
# Backward-compatible entry points for consolidated CLI commands
# ======================================================================


def main_walkforward(argv: list[str] | None = None) -> None:
    """``grand-tournament`` -- walk-forward validated tournament.

    Equivalent to: ``tournament --no-optimize --no-regime --no-monte-carlo``
    """
    # Build default args that mimic the old grand_tournament behaviour:
    # round-robin + walk-forward validation only, no optimization stages.
    extra = [
        "--no-optimize",
        "--no-regime",
        "--no-monte-carlo",
    ]
    combined = extra + (argv or sys.argv[1:])
    main(combined)


def main_baseline(argv: list[str] | None = None) -> None:
    """``multi-timeframe-tournament`` -- baseline ranking (no optimisation).

    Equivalent to: ``tournament --mode roundrobin --no-optimize --no-walk-forward
    --no-regime --no-monte-carlo``
    """
    extra = [
        "--mode", "roundrobin",
        "--no-optimize",
        "--no-walk-forward",
        "--no-regime",
        "--no-monte-carlo",
    ]
    combined = extra + (argv or sys.argv[1:])
    main(combined)


def main_greedy(argv: list[str] | None = None) -> None:
    """``multi-timeframe-optimizer`` -- optimisation + ranking.

    Equivalent to: ``tournament --no-walk-forward --no-regime --no-monte-carlo``
    (runs round-robin + walk-forward optimization, skips validation stages)
    """
    extra = [
        "--no-walk-forward",
        "--no-regime",
        "--no-monte-carlo",
    ]
    combined = extra + (argv or sys.argv[1:])
    main(combined)


if __name__ == "__main__":
    main()
