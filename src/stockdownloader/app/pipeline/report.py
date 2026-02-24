"""Holistic pipeline report formatter."""
from __future__ import annotations

from stockdownloader.app.helpers import status_label
from stockdownloader.app.pipeline.helpers import unique_days
from stockdownloader.app.pipeline.models import SlotResult
from stockdownloader.backtesting.results.result import BacktestResult
from stockdownloader.core.models.price import IntradayPriceData


def print_holistic_report(
    slots: list[SlotResult],
    intraday_data: list[IntradayPriceData],
    elapsed: float,
    out,
) -> None:
    """Print the unified cross-category ranking."""
    out()
    out("=" * 100)
    out("  UNIFIED PIPELINE — HOLISTIC RANKING")
    out("=" * 100)
    out()

    trading_days = unique_days(intraday_data)

    # Sort by ranking score descending
    ranked = sorted(
        [s for s in slots if s.best_result is not None],
        key=lambda s: s.ranking_score(trading_days),
        reverse=True,
    )

    out(
        f"  {'Rank':>4s}  {'Category':>8s}  {'Strategy':<28s}  "
        f"{'Return':>8s}  {'Trades':>6s}  {'WR%':>5s}  "
        f"{'Sharpe':>6s}  {'OOS':>8s}  {'Degrade':>7s}  {'Status':>10s}"
    )
    out("  " + "-" * 96)

    for i, slot in enumerate(ranked, 1):
        r = slot.best_result
        ret = f"{float(r.total_return):>+7.2f}%"
        trades = f"{r.total_trades:>5d}"
        wr = f"{float(r.win_rate):>5.1f}"

        # Sharpe (use intraday Sharpe for intraday/daily, standard for options)
        if isinstance(r, BacktestResult):
            sharpe = f"{float(r.sharpe_ratio(252 * 78)):>6.2f}"
        else:
            sharpe = f"{float(r.sharpe_ratio(252)):>6.2f}"

        # OOS score and degradation
        if slot.wf_result is not None:
            oos = f"{slot.wf_result.out_of_sample_score:>+7.2f}"
            deg = f"{slot.wf_result.degradation_ratio:>6.2f}"
            status = status_label(slot.wf_result.degradation_ratio)
        else:
            oos = "    n/a"
            deg = "    n/a"
            status = ""

        # Optimization marker
        opt_marker = "*" if slot.optimized is not None else " "

        out(
            f"  {i:>4d}{opt_marker} {slot.category:>8s}  {slot.display_name:<28s}  "
            f"{ret:>8s}  {trades:>6s}  {wr:>5s}  "
            f"{sharpe:>6s}  {oos:>8s}  {deg:>7s}  {status:>10s}"
        )

    out("  " + "-" * 96)
    out()

    # Summary statistics
    total = len(ranked)
    profitable = sum(1 for s in ranked if s.best_result and s.best_result.total_pnl > 0)
    optimized = sum(1 for s in slots if s.optimized is not None)
    wf_tested = [s for s in slots if s.wf_result is not None]
    robust = sum(1 for s in wf_tested if s.wf_result.degradation_ratio >= 0.8)
    acceptable = sum(1 for s in wf_tested if 0.5 <= s.wf_result.degradation_ratio < 0.8)
    overfit = sum(1 for s in wf_tested if s.wf_result.degradation_ratio < 0.5)

    out(f"  Strategies evaluated: {total}")
    out(f"  Profitable: {profitable}/{total}")
    out(f"  Optimized: {optimized} strategies improved by optimizer")
    if wf_tested:
        out(f"  Walk-forward: {robust} ROBUST, {acceptable} ACCEPTABLE, {overfit} OVERFIT")
    out(f"  Time elapsed: {elapsed:.1f}s")
    out()
    out("  * = using optimized parameters")
    out("  Legend: Degradation = OOS/IS score ratio (>= 0.8 ROBUST, 0.5-0.8 ACCEPTABLE, < 0.5 OVERFIT)")
    out()

    # Best strategy highlight
    if ranked:
        best = ranked[0]
        out("  " + "=" * 60)
        out(f"  BEST STRATEGY: {best.display_name} ({best.category})")
        r = best.best_result
        out(f"    Return: {float(r.total_return):+.2f}%  P/L: ${r.total_pnl:>,.2f}")
        out(f"    Trades: {r.total_trades}  Win Rate: {float(r.win_rate):.1f}%")
        if best.wf_result:
            out(f"    Walk-forward: {status_label(best.wf_result.degradation_ratio)} "
                f"(degradation {best.wf_result.degradation_ratio:.2f})")
        out("  " + "=" * 60)
    out()
