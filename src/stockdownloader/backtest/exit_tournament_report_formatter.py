"""Formats and prints exit tournament reports to the console.

Follows the same module-level function pattern as
:mod:`report_formatter`.
"""
from __future__ import annotations

from operator import attrgetter
from stockdownloader.backtest.report_helpers import scale2 as _s2, scale3 as _s3

from stockdownloader.backtest.exit_tournament_result import ExitTournamentResult
from stockdownloader.core.models.exit_result import ExitMechanismSummary


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def print_report(result: ExitTournamentResult, label: str = "") -> None:
    """Print a comprehensive exit tournament report."""
    sep = "=" * 70
    thin = "-" * 95

    print()
    print(sep)
    print(f"  EXIT MECHANISM TOURNAMENT {label}")
    print(sep)
    print()
    print(f"  Trades simulated: {result.trades_simulated}")
    print(f"  Trades skipped:   {result.trades_skipped}")
    print()

    # ── Aggregate summary table ──────────────────────────────────────

    print(thin)
    print("  AGGREGATE RESULTS")
    print(thin)
    print()

    header = (
        f"  {'Mechanism':<16} {'Total P&L':>10} {'WR%':>6} {'PF':>6} "
        f"{'Avg P&L':>8} {'Med P&L':>8} {'Avg Cap%':>9} {'Med Cap%':>9} "
        f"{'Avg Bars':>9}"
    )
    print(header)
    print("  " + "-" * 93)

    summaries = _sorted_summaries(result)
    for s in summaries:
        print(
            f"  {s.mechanism_name:<16} "
            f"{_s2(s.total_pnl):>10} "
            f"{_s2(s.win_rate):>5}% "
            f"{_s2(s.profit_factor):>6} "
            f"{_s3(s.avg_pnl):>8} "
            f"{_s3(s.median_pnl):>8} "
            f"{_s2(s.avg_capture_pct):>8}% "
            f"{_s2(s.median_capture_pct):>8}% "
            f"{_s2(s.avg_holding_bars):>8}"
        )
    print()

    # ── Exit reason breakdown ────────────────────────────────────────

    print(thin)
    print("  EXIT REASON BREAKDOWN")
    print(thin)
    for s in summaries:
        counts = s.exit_reason_counts()
        parts = [f"{reason}: {cnt}" for reason, cnt in counts.items()]
        print(f"  {s.mechanism_name:<16} {', '.join(parts)}")
    print()

    # ── By direction ─────────────────────────────────────────────────

    print(thin)
    print("  RESULTS BY DIRECTION")
    print(thin)
    for direction_val in ("LONG", "SHORT"):
        print(f"\n  {direction_val}:")
        print(f"  {'Mechanism':<16} {'Total P&L':>10} {'WR%':>6} {'Avg Cap%':>9}")
        for s in summaries:
            breakdown = result.get_breakdown_by_direction(s.mechanism_name)
            if direction_val in breakdown:
                ds = breakdown[direction_val]
                print(
                    f"  {s.mechanism_name:<16} "
                    f"{_s2(ds.total_pnl):>10} "
                    f"{_s2(ds.win_rate):>5}% "
                    f"{_s2(ds.avg_capture_pct):>8}%"
                )
    print()

    # ── By signal type ───────────────────────────────────────────────

    print(thin)
    print("  RESULTS BY SIGNAL TYPE")
    print(thin)
    # Collect all signal types
    signal_types: set[str] = set()
    for s in summaries:
        breakdown = result.get_breakdown_by_signal_type(s.mechanism_name)
        signal_types.update(breakdown.keys())

    for stype in sorted(signal_types):
        # Check if any mechanism has >= 3 trades for this signal type
        any_data = False
        for s in summaries:
            bd = result.get_breakdown_by_signal_type(s.mechanism_name)
            if stype in bd and bd[stype].total_trades >= 3:
                any_data = True
                break
        if not any_data:
            continue

        print(f"\n  {stype}:")
        print(f"  {'Mechanism':<16} {'Total P&L':>10} {'WR%':>6} {'Avg Cap%':>9}")
        for s in summaries:
            bd = result.get_breakdown_by_signal_type(s.mechanism_name)
            if stype in bd and bd[stype].total_trades >= 3:
                ss = bd[stype]
                print(
                    f"  {s.mechanism_name:<16} "
                    f"{_s2(ss.total_pnl):>10} "
                    f"{_s2(ss.win_rate):>5}% "
                    f"{_s2(ss.avg_capture_pct):>8}%"
                )
    print()

    # ── Head-to-head ─────────────────────────────────────────────────

    print(thin)
    print("  HEAD-TO-HEAD: Best mechanism per trade")
    print(thin)
    h2h = result.head_to_head
    total_trades = result.trades_simulated or 1
    for mech, wins in h2h.items():
        pct = wins / total_trades * 100
        print(f"  {mech:<16} wins {wins:>3} / {total_trades} ({pct:.1f}%)")
    print()

    # ── Risk metrics ─────────────────────────────────────────────────

    print(thin)
    print("  RISK METRICS")
    print(thin)
    print(
        f"  {'Mechanism':<16} {'Max Loss':>10} {'Max Win':>10} "
        f"{'P&L Std':>10} {'Sharpe*':>8}"
    )
    for s in summaries:
        print(
            f"  {s.mechanism_name:<16} "
            f"{_s3(s.max_loss):>10} "
            f"{_s3(s.max_win):>10} "
            f"{_s3(s.pnl_std):>10} "
            f"{_s3(s.sharpe_approx):>8}"
        )
    print()
    print("  * Sharpe approximation = mean(trade P&L) / std(trade P&L)")

    # ── Best mechanism ───────────────────────────────────────────────

    print()
    best = result.get_best_mechanism("total_pnl")
    if best:
        s = result.get_summary(best)
        if s is not None:
            print(f"  BEST BY TOTAL P&L: {best}")
            print(
                f"  P&L: ${_s2(s.total_pnl)} | "
                f"WR: {_s2(s.win_rate)}% | "
                f"PF: {_s2(s.profit_factor)} | "
                f"Sharpe: {_s3(s.sharpe_approx)}"
            )
    print()
    print(sep)
    print()


def print_comparison(result: ExitTournamentResult) -> None:
    """Print a concise mechanism comparison summary."""
    sep = "=" * 90
    thin = "-" * 90

    print()
    print(sep)
    print("  EXIT MECHANISM COMPARISON SUMMARY")
    print(sep)
    print()

    header = (
        f"  {'Mechanism':<18s} {'Total P&L':>10s} {'Win Rate':>10s} "
        f"{'PF':>8s} {'Sharpe':>8s} {'Capture%':>10s}"
    )
    print(header)
    print(thin)

    for s in _sorted_summaries(result):
        print(
            f"  {s.mechanism_name:<18s} "
            f"{'$' + str(_s2(s.total_pnl)):>10s} "
            f"{str(_s2(s.win_rate)) + '%':>10s} "
            f"{str(_s2(s.profit_factor)):>8s} "
            f"{str(_s3(s.sharpe_approx)):>8s} "
            f"{str(_s2(s.avg_capture_pct)) + '%':>10s}"
        )

    print(thin)
    print()
    print("  DISCLAIMER: Results are simulations, not live trading.")
    print("  Stop distance and fill dynamics are approximated.")
    print()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _sorted_summaries(result: ExitTournamentResult) -> list[ExitMechanismSummary]:
    """Return mechanism summaries sorted by total P&L descending."""
    summaries = list(result.mechanism_summaries.values())
    summaries.sort(key=attrgetter("total_pnl"), reverse=True)
    return summaries
