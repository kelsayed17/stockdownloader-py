#!/usr/bin/env python3
"""Comprehensive strategy analysis with full professional trading statistics.

Runs every registered strategy, computes all available metrics, performs
walk-forward validation for overfitting detection, and outputs a complete
trading report with actionable recommendations.

Usage::

    python scripts/full_strategy_analysis.py [--csv PATH] [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_scoring import score as score_v1, score_v2, MIN_TRADES
from stockdownloader.backtest.walk_forward import WalkForwardValidator
from stockdownloader.data.intraday_csv_loader import IntradayCsvLoader
from stockdownloader.strategy.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.registrations import ensure_registered
from stockdownloader.strategy.registry import StrategyRegistry

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

W = 120  # report width
THIN = "─" * W
THICK = "═" * W
SECTION = "━" * W


def _s(val: Decimal | float, decimals: int = 2) -> str:
    """Format a Decimal or float to a fixed-width string."""
    return f"{float(val):.{decimals}f}"


def _pnl(val: Decimal) -> str:
    v = float(val)
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:>12,.2f}"


def _pct(val: Decimal | float) -> str:
    return f"{float(val):>7.2f}%"


def _ratio(val: Decimal | float) -> str:
    return f"{float(val):>7.2f}"


def header(title: str) -> str:
    return f"\n{THICK}\n  {title}\n{THICK}"


def section(title: str) -> str:
    return f"\n{SECTION}\n  {title}\n{SECTION}"


# ---------------------------------------------------------------------------
# Core metrics extraction
# ---------------------------------------------------------------------------

def extract_metrics(result: BacktestResult, trading_days: int) -> dict[str, Any]:
    """Pull every metric from a BacktestResult into a flat dict."""
    trades = result.total_trades
    wins = result.winning_trades
    losses = result.losing_trades
    avg_w = float(result.average_win)
    avg_l = float(result.average_loss)
    wr = float(result.win_rate) / 100.0

    # Expectancy per trade
    if trades > 0 and avg_l != 0:
        expectancy = (wr * avg_w) + ((1 - wr) * avg_l)  # avg_l is already negative
    else:
        expectancy = 0.0

    # Risk-Reward Ratio (avg win / avg loss magnitude)
    rrr = abs(avg_w / avg_l) if avg_l != 0 else 0.0

    # Kelly Criterion (optimal fraction of capital to risk)
    if rrr > 0 and avg_l != 0:
        kelly = wr - ((1 - wr) / rrr)
    else:
        kelly = 0.0

    # Annualized return
    total_ret = float(result.total_return) / 100.0
    years = trading_days / 252.0 if trading_days > 0 else 1.0
    if total_ret > -1.0 and years > 0:
        try:
            ann_return = ((1 + total_ret) ** (1.0 / years) - 1.0) * 100.0
        except (OverflowError, ZeroDivisionError):
            ann_return = 0.0
    else:
        ann_return = -100.0

    return {
        "total_pnl": float(result.total_pnl),
        "total_return": float(result.total_return),
        "ann_return": ann_return,
        "total_trades": trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": float(result.win_rate),
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "profit_factor": float(result.profit_factor),
        "max_drawdown": float(result.max_drawdown),
        "sharpe": float(result.sharpe_ratio(trading_days_per_year=252 * 78)),
        "sortino": float(result.sortino_ratio(trading_days_per_year=252 * 78)),
        "calmar": float(result.calmar_ratio()),
        "max_consec_wins": result.max_consecutive_wins,
        "max_consec_losses": result.max_consecutive_losses,
        "avg_duration_bars": result.avg_trade_duration_bars,
        "expectancy": expectancy,
        "rrr": rrr,
        "kelly": kelly,
        "score_v1": score_v1(result, trading_days=trading_days),
        "score_v2": score_v2(result, trading_days=trading_days),
        "trades_per_day": trades / trading_days if trading_days > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def run_walk_forward(
    strategy_factory,
    engine: IntradayBacktestEngine,
    data: list,
    name: str,
    n_windows: int = 5,
) -> dict[str, Any] | None:
    """Run walk-forward validation and return summary dict."""
    try:
        validator = WalkForwardValidator(data, n_windows=n_windows, is_ratio=0.7)
        wf = validator.validate(strategy_factory, engine, strategy_name=name)
        return {
            "is_score": wf.in_sample_score,
            "oos_score": wf.out_of_sample_score,
            "degradation": wf.degradation_ratio,
            "n_windows": len(wf.windows),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_full_metrics_table(name: str, m: dict[str, Any]) -> None:
    """Print a complete metrics card for one strategy."""
    print(f"\n  ┌{'─' * 58}┐")
    print(f"  │  {name:<54s}  │")
    print(f"  ├{'─' * 58}┤")

    def row(label, val):
        print(f"  │  {label:<30s} {val:>24s}  │")

    row("Total P/L", f"${m['total_pnl']:>+13,.2f}")
    row("Total Return", f"{m['total_return']:>+.2f}%")
    row("Annualized Return", f"{m['ann_return']:>+.2f}%")
    print(f"  │{'─' * 58}│")
    row("Total Trades", f"{m['total_trades']:>d}")
    row("Winners / Losers", f"{m['winning_trades']} / {m['losing_trades']}")
    row("Win Rate", f"{m['win_rate']:.2f}%")
    row("Avg Win", f"${m['avg_win']:>+,.2f}")
    row("Avg Loss", f"${m['avg_loss']:>,.2f}")
    row("Risk-Reward Ratio (Avg W/L)", f"{m['rrr']:.2f}")
    row("Expectancy / Trade", f"${m['expectancy']:>+,.2f}")
    row("Kelly Criterion", f"{m['kelly']:.2f}%" if m['kelly'] != 0 else "N/A")
    print(f"  │{'─' * 58}│")
    row("Profit Factor", f"{m['profit_factor']:.2f}")
    row("Sharpe Ratio", f"{m['sharpe']:.2f}")
    row("Sortino Ratio", f"{m['sortino']:.2f}")
    row("Calmar Ratio", f"{m['calmar']:.2f}")
    row("Max Drawdown", f"{m['max_drawdown']:.2f}%")
    print(f"  │{'─' * 58}│")
    row("Max Consecutive Wins", f"{m['max_consec_wins']}")
    row("Max Consecutive Losses", f"{m['max_consec_losses']}")
    row("Avg Trade Duration (bars)", f"{m['avg_duration_bars']:.1f}")
    row("Trades per Day", f"{m['trades_per_day']:.3f}")
    print(f"  │{'─' * 58}│")
    row("Score v1 (Sharpe-based)", f"{m['score_v1']:.1f}")
    row("Score v2 (Institutional)", f"{m['score_v2']:.1f}")
    print(f"  └{'─' * 58}┘")


def print_comparison_table(all_results: list[tuple[str, dict[str, Any]]]) -> None:
    """Print a compact comparison table of all strategies."""
    # Sort by score_v2 descending
    ranked = sorted(all_results, key=lambda x: x[1]["score_v2"], reverse=True)

    print(f"\n  {'Rank':<5s} {'Strategy':<28s} {'P/L':>13s} {'Return':>8s} "
          f"{'WR':>7s} {'Trades':>7s} {'PF':>6s} {'Sharpe':>7s} "
          f"{'Sortino':>8s} {'MaxDD':>7s} {'Expect':>9s} {'RRR':>5s} "
          f"{'ConsL':>6s} {'Sv2':>7s}")
    print(f"  {'─' * 5} {'─' * 28} {'─' * 13} {'─' * 8} "
          f"{'─' * 7} {'─' * 7} {'─' * 6} {'─' * 7} "
          f"{'─' * 8} {'─' * 7} {'─' * 9} {'─' * 5} "
          f"{'─' * 6} {'─' * 7}")

    for rank, (name, m) in enumerate(ranked, 1):
        sign = "+" if m["total_pnl"] >= 0 else ""
        print(
            f"  {rank:<5d} {name:<28s} "
            f"{sign}${m['total_pnl']:>10,.2f} "
            f"{m['total_return']:>+7.2f}% "
            f"{m['win_rate']:>6.1f}% "
            f"{m['total_trades']:>7d} "
            f"{m['profit_factor']:>6.2f} "
            f"{m['sharpe']:>7.2f} "
            f"{m['sortino']:>8.2f} "
            f"{m['max_drawdown']:>6.2f}% "
            f"${m['expectancy']:>+7.2f} "
            f"{m['rrr']:>5.2f} "
            f"{m['max_consec_losses']:>6d} "
            f"{m['score_v2']:>7.1f}"
        )

    print()


def print_walk_forward_table(wf_results: list[tuple[str, dict[str, Any]]]) -> None:
    """Print walk-forward overfitting analysis."""
    print(f"\n  {'Strategy':<28s} {'IS Score':>10s} {'OOS Score':>10s} "
          f"{'Degradation':>12s} {'Assessment':<25s}")
    print(f"  {'─' * 28} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 25}")

    for name, wf in wf_results:
        if "error" in wf:
            print(f"  {name:<28s} {'ERROR':>10s} {'—':>10s} {'—':>12s} {wf['error'][:25]}")
            continue

        deg = wf["degradation"]
        if deg >= 0.8:
            assessment = "✓ Robust"
        elif deg >= 0.5:
            assessment = "⚠ Moderate overfitting"
        elif deg >= 0.0:
            assessment = "✗ Likely overfit"
        else:
            assessment = "✗ Severe overfitting"

        print(
            f"  {name:<28s} "
            f"{wf['is_score']:>10.1f} "
            f"{wf['oos_score']:>10.1f} "
            f"{deg:>12.3f} "
            f"{assessment:<25s}"
        )
    print()


def print_risk_analysis(all_results: list[tuple[str, dict[str, Any]]]) -> None:
    """Print risk metrics deep dive."""
    profitable = [(n, m) for n, m in all_results if m["total_pnl"] > 0 and m["total_trades"] >= 10]
    if not profitable:
        print("  No strategies with >= 10 trades and positive P/L for risk analysis.")
        return

    profitable.sort(key=lambda x: x[1]["sortino"], reverse=True)

    print(f"\n  {'Strategy':<28s} {'Sortino':>8s} {'Calmar':>8s} {'MaxDD':>7s} "
          f"{'ConsecL':>8s} {'Kelly':>7s} {'TPD':>6s} {'AvgDur':>7s}")
    print(f"  {'─' * 28} {'─' * 8} {'─' * 8} {'─' * 7} "
          f"{'─' * 8} {'─' * 7} {'─' * 6} {'─' * 7}")

    for name, m in profitable:
        kelly_str = f"{m['kelly']:.1f}%" if m['kelly'] > 0 else "N/A"
        print(
            f"  {name:<28s} "
            f"{m['sortino']:>8.2f} "
            f"{m['calmar']:>8.2f} "
            f"{m['max_drawdown']:>6.2f}% "
            f"{m['max_consec_losses']:>8d} "
            f"{kelly_str:>7s} "
            f"{m['trades_per_day']:>6.3f} "
            f"{m['avg_duration_bars']:>7.1f}"
        )
    print()


def print_edge_quality_analysis(all_results: list[tuple[str, dict[str, Any]]]) -> None:
    """Analyze quality of the trading edge for each strategy."""
    active = [(n, m) for n, m in all_results if m["total_trades"] >= 5]
    if not active:
        return

    active.sort(key=lambda x: x[1]["expectancy"], reverse=True)

    print(f"\n  {'Strategy':<28s} {'Trades':>7s} {'WR':>7s} {'AvgWin':>10s} "
          f"{'AvgLoss':>10s} {'RRR':>6s} {'Expect':>9s} {'Edge':>12s}")
    print(f"  {'─' * 28} {'─' * 7} {'─' * 7} {'─' * 10} "
          f"{'─' * 10} {'─' * 6} {'─' * 9} {'─' * 12}")

    for name, m in active:
        exp = m["expectancy"]
        rrr = m["rrr"]
        wr = m["win_rate"]

        # Classify the edge
        if m["total_trades"] < MIN_TRADES:
            edge = "Insufficient"
        elif exp > 50 and rrr > 1.0:
            edge = "Strong"
        elif exp > 20:
            edge = "Moderate"
        elif exp > 0:
            edge = "Marginal"
        else:
            edge = "No Edge"

        # Additional classification: high-WR/low-RRR vs low-WR/high-RRR
        if wr > 60 and rrr < 1.0:
            style = " (freq-based)"
        elif wr < 45 and rrr > 1.5:
            style = " (RRR-based)"
        else:
            style = ""

        print(
            f"  {name:<28s} "
            f"{m['total_trades']:>7d} "
            f"{wr:>6.1f}% "
            f"${m['avg_win']:>8,.2f} "
            f"${m['avg_loss']:>8,.2f} "
            f"{rrr:>6.2f} "
            f"${exp:>+7.2f} "
            f"{edge + style:>12s}"
        )
    print()


def print_stat_significance(all_results: list[tuple[str, dict[str, Any]]]) -> None:
    """Evaluate statistical significance of results."""
    print(f"\n  {'Strategy':<28s} {'Trades':>7s} {'WR':>7s} "
          f"{'WR StdErr':>10s} {'95% CI':>18s} {'Significant':>12s}")
    print(f"  {'─' * 28} {'─' * 7} {'─' * 7} "
          f"{'─' * 10} {'─' * 18} {'─' * 12}")

    for name, m in sorted(all_results, key=lambda x: x[1]["total_trades"], reverse=True):
        n = m["total_trades"]
        if n < 2:
            print(f"  {name:<28s} {n:>7d} {'—':>7s} {'—':>10s} {'—':>18s} {'Too few':>12s}")
            continue

        wr = m["win_rate"] / 100.0
        # Standard error of a proportion: sqrt(p*(1-p)/n)
        se = math.sqrt(wr * (1 - wr) / n) * 100  # as percentage
        ci_low = max(0, m["win_rate"] - 1.96 * se)
        ci_high = min(100, m["win_rate"] + 1.96 * se)

        # Is the WR significantly > 50% (for directional strategies)?
        z_stat = (wr - 0.5) / (math.sqrt(0.25 / n)) if n > 0 else 0
        sig = "Yes (p<.05)" if z_stat > 1.645 else "No"

        print(
            f"  {name:<28s} "
            f"{n:>7d} "
            f"{m['win_rate']:>6.1f}% "
            f"±{se:>7.2f}% "
            f"[{ci_low:>5.1f}% – {ci_high:>5.1f}%] "
            f"{sig:>12s}"
        )
    print()


def print_recommendations(all_results: list[tuple[str, dict[str, Any]]],
                          wf_results: list[tuple[str, dict[str, Any]]]) -> None:
    """Generate actionable trading recommendations."""
    wf_map = {n: w for n, w in wf_results}

    # Classify strategies
    ready = []       # Good edge, robust, enough trades
    promising = []   # Good results but questions (overfitting, few trades)
    needs_work = []  # Positive but marginal
    avoid = []       # Negative or no edge

    for name, m in all_results:
        wf = wf_map.get(name, {})
        deg = wf.get("degradation", None)
        n = m["total_trades"]
        exp = m["expectancy"]
        pnl = m["total_pnl"]

        if pnl <= 0 or exp <= 0:
            avoid.append((name, m, "Negative expectancy"))
        elif n < 30:
            promising.append((name, m, f"Only {n} trades — insufficient sample"))
        elif deg is not None and deg < 0.5:
            promising.append((name, m, f"Walk-forward degradation {deg:.2f} — possible overfit"))
        elif n < MIN_TRADES:
            promising.append((name, m, f"{n} trades (< {MIN_TRADES} threshold)"))
        elif exp > 20 and m["profit_factor"] > 1.1 and m["max_drawdown"] < 25:
            ready.append((name, m, ""))
        elif exp > 0:
            needs_work.append((name, m, "Marginal edge"))
        else:
            avoid.append((name, m, "No clear edge"))

    if ready:
        print("\n  ✅ READY FOR LIVE CONSIDERATION (strong edge, robust, sufficient trades):")
        print(THIN)
        for name, m, _ in sorted(ready, key=lambda x: x[1]["expectancy"], reverse=True):
            print(f"    {name:<28s} Expect: ${m['expectancy']:>+.2f}/trade  "
                  f"WR: {m['win_rate']:.1f}%  PF: {m['profit_factor']:.2f}  "
                  f"MaxDD: {m['max_drawdown']:.1f}%  Trades: {m['total_trades']}")

    if promising:
        print(f"\n  ⚠️  PROMISING BUT NEEDS VALIDATION:")
        print(THIN)
        for name, m, reason in promising:
            print(f"    {name:<28s} {reason}")
            print(f"      P/L: ${m['total_pnl']:>+,.2f}  WR: {m['win_rate']:.1f}%  "
                  f"Trades: {m['total_trades']}")

    if needs_work:
        print(f"\n  🔧 NEEDS IMPROVEMENT (marginal edge):")
        print(THIN)
        for name, m, reason in needs_work:
            print(f"    {name:<28s} Expect: ${m['expectancy']:>+.2f}/trade  "
                  f"WR: {m['win_rate']:.1f}%  PF: {m['profit_factor']:.2f}")

    if avoid:
        print(f"\n  ❌ AVOID (negative or no edge):")
        print(THIN)
        for name, m, reason in avoid:
            print(f"    {name:<28s} {reason} — P/L: ${m['total_pnl']:>+,.2f}")

    # Capital allocation suggestion
    print(f"\n  💰 CAPITAL ALLOCATION SUGGESTION (Kelly-based):")
    print(THIN)
    allocatable = [(n, m) for n, m, _ in ready if m["kelly"] > 0]
    if allocatable:
        total_kelly = sum(m["kelly"] for _, m in allocatable)
        for name, m in sorted(allocatable, key=lambda x: x[1]["kelly"], reverse=True):
            # Half-Kelly for safety
            half_k = m["kelly"] / 2.0
            weight = (m["kelly"] / total_kelly) * 100 if total_kelly > 0 else 0
            print(f"    {name:<28s} Kelly: {m['kelly']:.1f}%  "
                  f"Half-Kelly: {half_k:.1f}%  Portfolio weight: {weight:.0f}%")
    else:
        print("    No strategies qualify for allocation yet.")

    # Key issues to address
    print(f"\n  📋 KEY ISSUES TO ADDRESS:")
    print(THIN)

    intraday = [(n, m) for n, m in all_results
                if any(n == e.display_name
                       for e in StrategyRegistry.all_entries(category="intraday"))]
    low_trade_intraday = [(n, m) for n, m in intraday if m["total_trades"] < 10]
    if low_trade_intraday:
        names = ", ".join(n for n, _ in low_trade_intraday)
        print(f"    1. CRITICAL: Intraday strategies generating almost no trades: {names}")
        print(f"       → Entry filters too aggressive for this dataset/period")
        print(f"       → Consider loosening: adx_thresh, min_score, rvol thresholds")
        print(f"       → Or expand the data period for more observations")

    high_dd = [(n, m) for n, m in all_results if m["max_drawdown"] > 15 and m["total_pnl"] > 0]
    if high_dd:
        print(f"    2. HIGH DRAWDOWN: {', '.join(n for n, _ in high_dd)}")
        print(f"       → Consider tighter stops or reduced position sizing")

    low_rrr = [(n, m) for n, m in all_results
               if m["total_trades"] >= 30 and m["rrr"] < 0.8 and m["total_pnl"] > 0]
    if low_rrr:
        print(f"    3. LOW RISK-REWARD: {', '.join(n for n, _ in low_rrr)}")
        print(f"       → Average losses larger than average wins")
        print(f"       → Consider tighter stops or wider targets")

    streak = [(n, m) for n, m in all_results if m["max_consec_losses"] >= 8]
    if streak:
        streak_strs = [f"{n} ({m['max_consec_losses']}L)" for n, m in streak]
        print(f"    4. LOSING STREAKS: {', '.join(streak_strs)}")
        print(f"       → Psychological risk in live trading")
        print(f"       → Consider regime filters or daily loss limits")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full strategy analysis")
    parser.add_argument("--csv", default="data/spy_5m_bars.csv",
                        help="Path to 5-minute bar CSV")
    parser.add_argument("--quick", action="store_true",
                        help="Skip walk-forward (faster)")
    args = parser.parse_args()

    ensure_registered()
    t0 = time.time()

    # Load data
    print(header("COMPREHENSIVE STRATEGY ANALYSIS"))
    print(f"\n  Loading data from {args.csv}...")
    data = IntradayCsvLoader.load_from_file(args.csv)
    trading_days = len({bar.date[:10] for bar in data})
    sessions = len({bar.trading_date for bar in data})
    date_range = f"{data[0].date[:10]} to {data[-1].date[:10]}" if data else "N/A"

    print(f"  Bars: {len(data):,}  |  Trading Days: {trading_days}  |  Sessions: {sessions}")
    print(f"  Period: {date_range}")
    print(f"  Capital: $100,000  |  Risk/Trade: 1.00%")

    capital = Decimal("100000")
    risk = Decimal("0.01")
    engine = IntradayBacktestEngine(capital, risk)

    # ── Section 1: Run all strategies ──────────────────────────────────────
    print(header("SECTION 1: FULL METRICS — ALL STRATEGIES"))

    all_results: list[tuple[str, dict[str, Any]]] = []
    backtest_results: dict[str, BacktestResult] = {}

    # Intraday strategies
    print(section("INTRADAY STRATEGIES"))
    for entry in StrategyRegistry.all_entries(category="intraday"):
        name = entry.display_name
        print(f"\n  Running: {name}...", end="", flush=True)
        t1 = time.time()
        strategy = entry.factory(**entry.default_kwargs)
        result = engine.run(strategy, data)
        elapsed = time.time() - t1
        print(f" done ({elapsed:.1f}s)")

        m = extract_metrics(result, trading_days)
        all_results.append((name, m))
        backtest_results[name] = result
        print_full_metrics_table(name, m)

    # Daily strategies via adapter
    print(section("DAILY STRATEGIES (via DailyToIntradayAdapter)"))
    for entry in StrategyRegistry.all_entries(category="daily"):
        name = entry.display_name
        print(f"\n  Running: {name}...", end="", flush=True)
        t1 = time.time()
        daily = entry.factory(**entry.default_kwargs)
        adapter = DailyToIntradayAdapter(daily)
        result = engine.run(adapter, data)
        elapsed = time.time() - t1
        print(f" done ({elapsed:.1f}s)")

        m = extract_metrics(result, trading_days)
        all_results.append((name, m))
        backtest_results[name] = result
        print_full_metrics_table(name, m)

    # ── Section 2: Comparison table ────────────────────────────────────────
    print(header("SECTION 2: STRATEGY COMPARISON (ranked by Score v2)"))
    print_comparison_table(all_results)

    # ── Section 3: Edge Quality Analysis ───────────────────────────────────
    print(header("SECTION 3: EDGE QUALITY ANALYSIS"))
    print("""
  Edge quality measures whether a strategy has a genuine, repeatable
  trading advantage. Key metrics: Expectancy (expected $ per trade),
  Risk-Reward Ratio (avg win / avg loss), and Win Rate.

  A positive expectancy with sufficient trades is the minimum bar.
  Higher RRR means fewer wins needed to be profitable.
""")
    print_edge_quality_analysis(all_results)

    # ── Section 4: Risk Analysis ───────────────────────────────────────────
    print(header("SECTION 4: RISK ANALYSIS"))
    print("""
  Sortino: Like Sharpe but only penalizes downside volatility (higher = better)
  Calmar: Annualized return / max drawdown (higher = better)
  Kelly: Theoretical optimal bet sizing (use half-Kelly in practice)
  ConsecL: Max consecutive losing trades (psychological risk)
  TPD: Trades per day (capital efficiency)
""")
    print_risk_analysis(all_results)

    # ── Section 5: Statistical Significance ────────────────────────────────
    print(header("SECTION 5: STATISTICAL SIGNIFICANCE"))
    print("""
  Tests whether win rates are statistically different from random (50%).
  95% confidence intervals show the likely range of true win rate.
  Strategies with < 30 trades cannot be reliably evaluated.
""")
    print_stat_significance(all_results)

    # ── Section 6: Walk-Forward Validation ─────────────────────────────────
    wf_results: list[tuple[str, dict[str, Any]]] = []
    if not args.quick:
        print(header("SECTION 6: WALK-FORWARD VALIDATION (Overfitting Detection)"))
        print("""
  Splits data into 5 rolling windows (70% in-sample / 30% out-of-sample).
  Degradation ratio = OOS score / IS score.
    ≥ 0.80 = Robust (strategy generalizes well)
    0.50–0.80 = Moderate overfitting risk
    < 0.50 = Likely overfit to training data
""")

        # Only run WF for strategies with enough trades
        wf_engine = IntradayBacktestEngine(capital, risk)
        for entry in StrategyRegistry.all_entries(category="intraday"):
            name = entry.display_name
            print(f"  Walk-forward: {name}...", end="", flush=True)
            t1 = time.time()
            wf = run_walk_forward(
                lambda e=entry: e.factory(**e.default_kwargs),
                wf_engine, data, name, n_windows=5,
            )
            elapsed = time.time() - t1
            print(f" done ({elapsed:.1f}s)")
            wf_results.append((name, wf))

        for entry in StrategyRegistry.all_entries(category="daily"):
            name = entry.display_name
            print(f"  Walk-forward: {name}...", end="", flush=True)
            t1 = time.time()

            def make_adapter(e=entry):
                daily = e.factory(**e.default_kwargs)
                return DailyToIntradayAdapter(daily)

            wf = run_walk_forward(make_adapter, wf_engine, data, name, n_windows=5)
            elapsed = time.time() - t1
            print(f" done ({elapsed:.1f}s)")
            wf_results.append((name, wf))

        print_walk_forward_table(wf_results)
    else:
        print(f"\n  (Walk-forward skipped with --quick flag)")

    # ── Section 7: Recommendations ─────────────────────────────────────────
    print(header("SECTION 7: RECOMMENDATIONS & ACTION ITEMS"))
    print_recommendations(all_results, wf_results)

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    print(THICK)
    print(f"  Analysis complete. Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    print(f"  Strategies analyzed: {len(all_results)}")
    if wf_results:
        print(f"  Walk-forward validations: {len(wf_results)}")
    print(THICK)
    print()
    print("  DISCLAIMER: This analysis is for educational/research purposes only.")
    print("  Past performance does not guarantee future results.")
    print("  Always conduct independent research before trading.")
    print()


if __name__ == "__main__":
    main()
