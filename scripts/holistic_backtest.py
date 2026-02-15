#!/usr/bin/env python3
"""Holistic backtest analysis: runs all strategies, exit mechanisms,
and indicator synergy checks on SPY 5-minute data.

Produces a comprehensive report covering:
1. Baseline performance of all strategies (5 intraday + 7 daily adapted)
2. Per-strategy parameter sensitivity
3. Daily strategy optimization for top candidates
4. Indicator synergy analysis
5. Consolidated recommendations

Usage:
    python scripts/holistic_backtest.py
    python scripts/holistic_backtest.py --csv data/spy_5m_bars.csv
    python scripts/holistic_backtest.py --quick   # Skip optimization
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_scoring import score as compute_score
from stockdownloader.data.intraday_csv_loader import IntradayCsvLoader
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.strategy.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.registry import StrategyRegistry

from stockdownloader.strategy.registrations import ensure_registered
ensure_registered()

CAPITAL = Decimal("100000")
RISK = Decimal("0.01")
_ZERO = Decimal("0")
_HUNDRED = Decimal("100")
_BARS_PER_DAY = 78


def _s2(v: Decimal) -> Decimal:
    return v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# -- Data Loading ----------------------------------------------------------

def load_data(csv_path: str) -> list[IntradayPriceData]:
    """Load intraday data from CSV."""
    data = IntradayCsvLoader.load_from_file(csv_path)
    if not data:
        print(f"ERROR: No data loaded from {csv_path}")
        sys.exit(1)
    return data


# -- Backtest Runner -------------------------------------------------------

def run_backtest(
    strategy,
    data: list[IntradayPriceData],
) -> BacktestResult:
    engine = IntradayBacktestEngine(CAPITAL, RISK)
    return engine.run(strategy, data)


# -- Section 1: All-Strategy Baseline --------------------------------------

def run_baseline(data: list[IntradayPriceData]) -> list[tuple[str, BacktestResult, float]]:
    """Run all strategies with default params and return ranked results."""
    trading_days = len({bar.date[:10] for bar in data})
    results: list[tuple[str, BacktestResult, float]] = []

    # 1. Intraday strategies (5 standalone)
    for entry in StrategyRegistry.all_entries(category="intraday"):
        name = entry.display_name
        print(f"  Running: {name}...", end="", flush=True)
        t0 = time.time()
        strategy = StrategyRegistry.create(entry.name)
        result = run_backtest(strategy, data)
        elapsed = time.time() - t0
        s = compute_score(result, trading_days=trading_days)
        results.append((name, result, s))
        print(f" done ({elapsed:.1f}s)")

    # 2. All daily strategies via adapter
    for entry in StrategyRegistry.all_entries(category="daily"):
        name = entry.display_name
        print(f"  Running: {name}...", end="", flush=True)
        t0 = time.time()
        daily = StrategyRegistry.create(entry.name)
        adapter = DailyToIntradayAdapter(daily)
        result = run_backtest(adapter, data)
        elapsed = time.time() - t0
        s = compute_score(result, trading_days=trading_days)
        results.append((name, result, s))
        print(f" done ({elapsed:.1f}s)")

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def print_baseline_report(
    results: list[tuple[str, BacktestResult, float]],
    data: list[IntradayPriceData],
) -> None:
    """Print the all-strategy baseline comparison table."""
    trading_days = len({bar.date[:10] for bar in data})
    sep = "=" * 120
    thin = "-" * 120

    print()
    print(sep)
    print("  SECTION 1: ALL-STRATEGY BASELINE COMPARISON")
    print(sep)
    print()
    print(f"  Data: SPY 5-minute bars")
    print(f"  Bars: {len(data):,}  |  Trading Days: {trading_days}")
    print(f"  Period: {data[0].date[:10]} to {data[-1].date[:10]}")
    print(f"  Capital: ${CAPITAL:,.0f}  |  Risk/Trade: {RISK*100}%")
    print()

    header = (
        f"  {'Rank':<5s} {'Strategy':<30s} {'P/L':>12s}  {'Return':>8s}  "
        f"{'WR':>6s}  {'PF':>6s}  {'Sharpe':>7s}  {'MaxDD':>7s}  "
        f"{'Trades':>7s}  {'Avg Win':>9s}  {'Avg Loss':>9s}  {'Score':>7s}"
    )
    print(header)
    print(thin)

    for rank, (name, r, score) in enumerate(results, 1):
        pnl = r.total_pnl
        sign = "+" if pnl >= 0 else ""
        sharpe = r.sharpe_ratio(trading_days_per_year=252 * _BARS_PER_DAY)
        print(
            f"  {rank:<5d} {name:<30s} {sign}${pnl:>10,.2f}  "
            f"{_s2(r.total_return):>7}%  "
            f"{_s2(r.win_rate):>5}%  "
            f"{_s2(r.profit_factor):>6}  "
            f"{sharpe:>7}  "
            f"{_s2(r.max_drawdown):>6}%  "
            f"{r.total_trades:>7d}  "
            f"${r.average_win:>8}  "
            f"${r.average_loss:>8}  "
            f"{score:>7.1f}"
        )

    print(thin)
    print()


# -- Section 2: Indicator Synergy Analysis ---------------------------------

def analyze_indicator_synergy(
    results: list[tuple[str, BacktestResult, float]],
    data: list[IntradayPriceData],
) -> None:
    """Analyze which indicators work well together based on strategy performance."""
    sep = "=" * 120
    thin = "-" * 120

    print(sep)
    print("  SECTION 2: INDICATOR SYNERGY ANALYSIS")
    print(sep)
    print()

    # Map strategies to their indicator ingredients
    indicator_map = {
        "SMA Crossover": ["SMA(short)", "SMA(long)"],
        "RSI Strategy": ["RSI(14)"],
        "MACD Strategy": ["EMA(12)", "EMA(26)", "MACD Signal(9)"],
        "Bollinger Band + RSI": ["BB(20,2)", "RSI(14)", "ADX(14)"],
        "Breakout Strategy": ["BB(20,2)", "BB Squeeze", "Volume"],
        "Momentum Confluence": ["EMA(fast)", "EMA(slow)", "MACD", "ADX(14)", "EMA(200) Trend"],
        "Multi-Indicator Confluence": ["SMA(20)", "RSI(14)", "MACD", "BB(20,2)", "ADX(14)", "Volume"],
        "VWAP Pullback": ["VWAP", "VWAP Bands(s)", "ATR(14)", "ADX(14)", "EMA(9/21)"],
        "VWAP Reversal": ["VWAP", "VWAP Bands(s)", "ATR(14)", "ADX(14)"],
        "OR Breakout": ["Opening Range", "ATR(14)", "Relative Volume", "VWAP"],
        "OR Reversal": ["Opening Range", "ATR(14)", "Relative Volume", "VWAP", "ADX(14)"],
        "Pattern Scalp": ["Opening Range", "ATR(14)", "Engulfing Patterns"],
    }

    # Identify indicators used by top-performing strategies
    profitable = [(n, r, s) for n, r, s in results if r.total_pnl > 0]
    losing = [(n, r, s) for n, r, s in results if r.total_pnl <= 0]

    print("  PROFITABLE STRATEGIES AND THEIR INDICATORS:")
    print(thin)
    for name, r, s in profitable:
        indicators = indicator_map.get(name, ["Unknown"])
        print(f"    {name:<30s} P/L: +${r.total_pnl:>10,.2f}  Indicators: {', '.join(indicators)}")
    print()

    if losing:
        print("  UNPROFITABLE STRATEGIES AND THEIR INDICATORS:")
        print(thin)
        for name, r, s in losing:
            indicators = indicator_map.get(name, ["Unknown"])
            print(f"    {name:<30s} P/L: -${abs(r.total_pnl):>10,.2f}  Indicators: {', '.join(indicators)}")
        print()

    # Count indicator frequency in profitable strategies
    profitable_indicators: Counter = Counter()
    for name, _, _ in profitable:
        for ind in indicator_map.get(name, []):
            profitable_indicators[ind] += 1

    losing_indicators: Counter = Counter()
    for name, _, _ in losing:
        for ind in indicator_map.get(name, []):
            losing_indicators[ind] += 1

    all_indicators = set(profitable_indicators.keys()) | set(losing_indicators.keys())

    print("  INDICATOR EFFECTIVENESS RANKING:")
    print(thin)
    print(f"    {'Indicator':<25s} {'In Profitable':>14s}  {'In Losing':>10s}  {'Assessment':<30s}")
    print(f"    {chr(9472) * 25} {chr(9472) * 14}  {chr(9472) * 10}  {chr(9472) * 30}")

    scored_indicators = []
    for ind in all_indicators:
        p_count = profitable_indicators.get(ind, 0)
        l_count = losing_indicators.get(ind, 0)
        total = p_count + l_count
        if total > 0:
            effectiveness = p_count / total
        else:
            effectiveness = 0
        scored_indicators.append((ind, p_count, l_count, effectiveness))

    scored_indicators.sort(key=lambda x: (-x[3], -x[1]))

    for ind, p_count, l_count, eff in scored_indicators:
        if eff >= 0.8:
            assessment = "Strong positive signal"
        elif eff >= 0.5:
            assessment = "Moderately effective"
        elif eff > 0:
            assessment = "Context-dependent"
        else:
            assessment = "Weak / not standalone"
        print(f"    {ind:<25s} {p_count:>14d}  {l_count:>10d}  {assessment}")

    print()


# -- Section 3: Per-Strategy Parameter Sensitivity -------------------------

def run_intraday_sensitivity(data: list[IntradayPriceData]) -> None:
    """Test sensitivity of key parameters for each intraday strategy."""
    sep = "=" * 120
    thin = "-" * 120
    trading_days = len({bar.date[:10] for bar in data})

    print(sep)
    print("  SECTION 3: INTRADAY PARAMETER SENSITIVITY")
    print(sep)
    print()

    for entry in StrategyRegistry.all_entries(category="intraday"):
        if not entry.param_space:
            continue

        name = entry.display_name
        print(f"  --- {name} ---")

        # Baseline
        baseline_strategy = StrategyRegistry.create(entry.name)
        baseline_result = run_backtest(baseline_strategy, data)
        baseline_score = compute_score(baseline_result, trading_days=trading_days)

        print(f"  Baseline: P/L=${_s2(baseline_result.total_pnl):>10}  "
              f"WR={_s2(baseline_result.win_rate)}%  "
              f"Trades={baseline_result.total_trades}  "
              f"Score={baseline_score:.1f}")
        print()

        best_overrides: dict = {}

        for param, values in entry.param_space.items():
            # Get default value from the strategy's config
            default_strategy = entry.factory()
            config = default_strategy._infra._c
            current_val = getattr(config, param)
            print(f"  {param} (default={current_val}):")

            best_for_param = (current_val, baseline_score)

            for val in values:
                if val == current_val:
                    continue
                try:
                    config_cls = type(config)
                    trial_config = dataclasses.replace(config, **{param: val})
                    strategy = entry.factory(config=trial_config)
                except (TypeError, ValueError):
                    continue

                result = run_backtest(strategy, data)
                s = compute_score(result, trading_days=trading_days)
                pnl = result.total_pnl

                improved = " ***" if s > best_for_param[1] else ""
                sign = "+" if pnl >= 0 else ""
                print(
                    f"    {param}={str(val):<12s}  "
                    f"P/L: {sign}${pnl:>10,.2f}  "
                    f"WR: {_s2(result.win_rate):>5}%  "
                    f"Trades: {result.total_trades:>4d}  "
                    f"Score: {s:>7.1f}{improved}"
                )

                if s > best_for_param[1]:
                    best_for_param = (val, s)

            if best_for_param[0] != current_val:
                best_overrides[param] = best_for_param[0]
                print(f"    >>> Best: {param}={best_for_param[0]} (score {best_for_param[1]:.1f})")

            print()

        # Try combined best overrides
        if best_overrides:
            print(thin)
            print(f"  COMBINED BEST PARAMETERS for {name}:")
            print(thin)
            config = default_strategy._infra._c
            for param, val in best_overrides.items():
                print(f"    {param}: {getattr(config, param)} -> {val}")

            try:
                config_cls = type(config)
                combined_config = dataclasses.replace(config, **best_overrides)
                combined_strategy = entry.factory(config=combined_config)
                combined_result = run_backtest(combined_strategy, data)
                combined_score = compute_score(combined_result, trading_days=trading_days)

                print()
                print(f"  Combined Result:")
                print(f"    P/L:    ${_s2(combined_result.total_pnl):>10}  (baseline: ${_s2(baseline_result.total_pnl):>10})")
                print(f"    WR:     {_s2(combined_result.win_rate)}%  (baseline: {_s2(baseline_result.win_rate)}%)")
                print(f"    Trades: {combined_result.total_trades}  (baseline: {baseline_result.total_trades})")
                print(f"    Score:  {combined_score:.1f}  (baseline: {baseline_score:.1f})")

                delta_pnl = combined_result.total_pnl - baseline_result.total_pnl
                sign = "+" if delta_pnl >= 0 else ""
                print(f"    Delta:  {sign}${_s2(delta_pnl)}")
            except Exception as e:
                print(f"  ERROR combining parameters: {e}")

        print()


# -- Section 4: Daily Strategy Optimization --------------------------------

def optimize_daily_strategies(
    baseline_results: list[tuple[str, BacktestResult, float]],
    data: list[IntradayPriceData],
) -> list[tuple[str, BacktestResult, float, dict]]:
    """Optimize the top daily strategies that show promise."""
    sep = "=" * 120
    thin = "-" * 120
    trading_days = len({bar.date[:10] for bar in data})

    print(sep)
    print("  SECTION 4: DAILY STRATEGY OPTIMIZATION")
    print(sep)
    print()

    optimized_results: list[tuple[str, BacktestResult, float, dict]] = []

    for entry in StrategyRegistry.all_entries(category="daily"):
        name = entry.display_name
        print(f"  Optimizing: {name}...")

        baseline = next((r for n, r, _ in baseline_results if n == name), None)
        if baseline is None:
            continue

        best_kwargs = dict(entry.default_kwargs)
        best_adapter_kwargs = {"sl_atr_mult": Decimal("1.5"), "rr": Decimal("1.5"), "sl_cap": Decimal("2.00")}
        best_score = compute_score(baseline, trading_days=trading_days)
        best_result = baseline

        # Phase 1: Strategy params
        if entry.param_space:
            for param, values in entry.param_space.items():
                param_best = (best_kwargs.get(param), best_score)
                for val in values:
                    if val == best_kwargs.get(param):
                        continue
                    trial = {**best_kwargs, param: val}
                    try:
                        strat = entry.factory(**trial)
                        adapter = DailyToIntradayAdapter(strat, **best_adapter_kwargs)
                        result = run_backtest(adapter, data)
                        s = compute_score(result, trading_days=trading_days)
                        if s > param_best[1]:
                            param_best = (val, s)
                    except (ValueError, TypeError):
                        continue

                if param_best[0] != best_kwargs.get(param) and param_best[0] is not None:
                    best_kwargs[param] = param_best[0]
                    strat = entry.factory(**best_kwargs)
                    adapter = DailyToIntradayAdapter(strat, **best_adapter_kwargs)
                    result = run_backtest(adapter, data)
                    s = compute_score(result, trading_days=trading_days)
                    if s > best_score:
                        best_score = s
                        best_result = result

        # Phase 2: Adapter params
        adapter_space = {
            "sl_atr_mult": [Decimal("0.8"), Decimal("1.0"), Decimal("1.3"), Decimal("1.5"), Decimal("1.8")],
            "rr": [Decimal("1.0"), Decimal("1.2"), Decimal("1.5"), Decimal("2.0")],
            "sl_cap": [Decimal("1.00"), Decimal("1.50"), Decimal("2.00"), Decimal("2.50")],
        }
        for param, values in adapter_space.items():
            param_best = (best_adapter_kwargs.get(param), best_score)
            for val in values:
                if val == best_adapter_kwargs.get(param):
                    continue
                trial_adapter = {**best_adapter_kwargs, param: val}
                strat = entry.factory(**best_kwargs)
                adapter = DailyToIntradayAdapter(strat, **trial_adapter)
                result = run_backtest(adapter, data)
                s = compute_score(result, trading_days=trading_days)
                if s > param_best[1]:
                    param_best = (val, s)

            if param_best[0] != best_adapter_kwargs.get(param) and param_best[0] is not None:
                best_adapter_kwargs[param] = param_best[0]
                strat = entry.factory(**best_kwargs)
                adapter = DailyToIntradayAdapter(strat, **best_adapter_kwargs)
                result = run_backtest(adapter, data)
                s = compute_score(result, trading_days=trading_days)
                if s > best_score:
                    best_score = s
                    best_result = result

        combined = {**best_kwargs, **best_adapter_kwargs}
        optimized_results.append((name, best_result, best_score, combined))

        base_pnl = baseline.total_pnl
        opt_pnl = best_result.total_pnl
        delta = opt_pnl - base_pnl
        sign = "+" if delta >= 0 else ""
        print(
            f"    Baseline: P/L=${_s2(base_pnl):>10}  WR={_s2(baseline.win_rate)}%  Trades={baseline.total_trades}")
        print(
            f"    Optimized: P/L=${_s2(opt_pnl):>10}  WR={_s2(best_result.win_rate)}%  "
            f"Trades={best_result.total_trades}  ({sign}${_s2(delta)})")

        changed = {k: v for k, v in combined.items()
                   if str(v) != str({**entry.default_kwargs,
                                     "sl_atr_mult": Decimal("1.5"),
                                     "rr": Decimal("1.5"),
                                     "sl_cap": Decimal("2.00")}.get(k, ""))}
        if changed:
            print(f"    Changed: {changed}")
        print()

    optimized_results.sort(key=lambda x: x[2], reverse=True)

    print(thin)
    print("  OPTIMIZED DAILY STRATEGY RANKING:")
    print(thin)
    print(f"    {'Rank':<5s} {'Strategy':<30s} {'P/L':>12s}  {'WR':>6s}  {'Trades':>7s}  {'Score':>7s}")
    print(f"    {chr(9472) * 5} {chr(9472) * 30} {chr(9472) * 12}  {chr(9472) * 6}  {chr(9472) * 7}  {chr(9472) * 7}")

    for rank, (name, r, score, _) in enumerate(optimized_results, 1):
        pnl = r.total_pnl
        sign = "+" if pnl >= 0 else ""
        print(
            f"    {rank:<5d} {name:<30s} {sign}${pnl:>10,.2f}  "
            f"{_s2(r.win_rate):>5}%  {r.total_trades:>7d}  {score:>7.1f}"
        )
    print()

    return optimized_results


# -- Section 5: Consolidated Recommendations -------------------------------

def print_recommendations(
    baseline: list[tuple[str, BacktestResult, float]],
    data: list[IntradayPriceData],
) -> None:
    """Print actionable recommendations based on all analysis."""
    sep = "=" * 120
    thin = "-" * 120

    print(sep)
    print("  SECTION 5: CONSOLIDATED RECOMMENDATIONS")
    print(sep)
    print()

    trading_days = len({bar.date[:10] for bar in data})

    # Sort by different criteria
    by_pnl = sorted(baseline, key=lambda x: x[1].total_pnl, reverse=True)
    by_wr = sorted(baseline, key=lambda x: float(x[1].win_rate), reverse=True)
    by_sharpe = sorted(baseline, key=lambda x: float(x[1].sharpe_ratio(252 * _BARS_PER_DAY)), reverse=True)
    by_pf = sorted(baseline, key=lambda x: float(x[1].profit_factor), reverse=True)

    print("  BEST BY METRIC:")
    print(f"    Best P/L:          {by_pnl[0][0]:<30s} ${_s2(by_pnl[0][1].total_pnl):>10}")
    print(f"    Best Win Rate:     {by_wr[0][0]:<30s} {_s2(by_wr[0][1].win_rate)}%")
    print(f"    Best Sharpe:       {by_sharpe[0][0]:<30s} {by_sharpe[0][1].sharpe_ratio(252*_BARS_PER_DAY)}")
    print(f"    Best Profit Factor:{by_pf[0][0]:<30s} {by_pf[0][1].profit_factor}")
    print()

    print("  STRATEGY RECOMMENDATIONS:")
    print(thin)

    profitable_count = sum(1 for _, r, _ in baseline if r.total_pnl > 0)
    total = len(baseline)

    print(f"    {profitable_count}/{total} strategies profitable over {trading_days} trading days")
    print()

    for rank, (name, r, score) in enumerate(baseline[:3], 1):
        pnl = r.total_pnl
        if pnl > 0:
            print(f"    #{rank} {name}:")
            print(f"       Return: {_s2(r.total_return)}% | WR: {_s2(r.win_rate)}% | "
                  f"PF: {r.profit_factor} | MaxDD: {_s2(r.max_drawdown)}%")
            print()

    print("  RISK MANAGEMENT NOTES:")
    print(thin)
    worst_dd = max(baseline, key=lambda x: float(x[1].max_drawdown))
    best_dd = min(baseline, key=lambda x: float(x[1].max_drawdown))
    print(f"    Worst MaxDD: {worst_dd[0]} at {_s2(worst_dd[1].max_drawdown)}%")
    print(f"    Best MaxDD:  {best_dd[0]} at {_s2(best_dd[1].max_drawdown)}%")
    print(f"    Recommendation: Keep per-trade risk at 1% and daily loss limit at 3%")
    print()

    print(sep)
    print()
    print("  DISCLAIMER: This analysis is for educational/research purposes only.")
    print("  Past performance does not guarantee future results.")
    print("  Always conduct independent research before trading.")
    print()
    print(sep)


# -- Main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Holistic Strategy Backtest Analysis")
    parser.add_argument("--csv", default="data/spy_5m_bars.csv", help="Path to 5-minute CSV")
    parser.add_argument("--quick", action="store_true", help="Skip optimization phases")
    args = parser.parse_args()

    total_start = time.time()

    print()
    print("=" * 120)
    print("  HOLISTIC STRATEGY BACKTEST & ANALYSIS")
    print("=" * 120)
    print()

    # Load data
    print("Loading data...", end="", flush=True)
    t0 = time.time()
    data = load_data(args.csv)
    trading_days = len({bar.date[:10] for bar in data})
    print(f" {len(data):,} bars, {trading_days} sessions ({time.time()-t0:.1f}s)")
    print(f"  Date range: {data[0].date[:10]} to {data[-1].date[:10]}")
    print()

    # Section 1: Baseline
    print("-" * 120)
    print("  Running baseline backtests...")
    print("-" * 120)
    baseline = run_baseline(data)
    print_baseline_report(baseline, data)

    # Section 2: Indicator synergy
    analyze_indicator_synergy(baseline, data)

    if not args.quick:
        # Section 3: Intraday parameter sensitivity
        print("  Running intraday parameter sensitivity (this may take a few minutes)...")
        run_intraday_sensitivity(data)

        # Section 4: Daily strategy optimization
        print("  Running daily strategy optimization...")
        optimize_daily_strategies(baseline, data)

    # Section 5: Recommendations
    print_recommendations(baseline, data)

    total_elapsed = time.time() - total_start
    print(f"  Total analysis time: {total_elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
