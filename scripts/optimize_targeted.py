#!/usr/bin/env python3
"""Targeted optimization: per-strategy sensitivity + fast daily strategies.

Skips Momentum Confluence and Multi-Indicator (too slow at 390s and 1368s per run).
Focuses on strategies that run in < 10s per iteration.
"""
from __future__ import annotations

import dataclasses
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

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


def _s2(v: Decimal) -> Decimal:
    return v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def run_backtest(strategy, data):
    engine = IntradayBacktestEngine(CAPITAL, RISK)
    return engine.run(strategy, data)


def main():
    total_start = time.time()

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/spy_5m_bars.csv"
    print(f"\nLoading data from {csv_path}...", end="", flush=True)
    data = IntradayCsvLoader.load_from_file(csv_path)
    trading_days = len({bar.date[:10] for bar in data})
    print(f" {len(data):,} bars, {trading_days} sessions")

    sep = "=" * 110
    thin = "-" * 110

    # ==================================================================
    # PART 1: PER-STRATEGY PARAMETER SENSITIVITY
    # ==================================================================
    print(f"\n{sep}")
    print("  INTRADAY STRATEGY PARAMETER SENSITIVITY")
    print(sep)

    total_configs = 0

    for entry in StrategyRegistry.all_entries(category="intraday"):
        if not entry.param_space:
            continue

        name = entry.display_name
        print(f"\n  --- {name} ---")

        # Get baseline
        default_strategy = entry.factory()
        config = default_strategy._infra._c
        baseline_result = run_backtest(default_strategy, data)
        baseline_score = compute_score(baseline_result, trading_days=trading_days)

        print(f"\n  Baseline: P/L=${_s2(baseline_result.total_pnl)}  "
              f"WR={_s2(baseline_result.win_rate)}%  "
              f"PF={baseline_result.profit_factor}  "
              f"Sharpe={baseline_result.sharpe_ratio(252*78)}  "
              f"MaxDD={_s2(baseline_result.max_drawdown)}%  "
              f"Trades={baseline_result.total_trades}  "
              f"Score={baseline_score:.1f}\n")

        best_overrides: dict = {}

        for param, values in entry.param_space.items():
            current_val = getattr(config, param)
            best_for_param = (current_val, baseline_score)
            print(f"  Testing {param} (default={current_val}):")

            for val in values:
                if val == current_val:
                    continue
                total_configs += 1
                try:
                    trial_config = dataclasses.replace(config, **{param: val})
                    strategy = entry.factory(config=trial_config)
                except (TypeError, ValueError):
                    continue
                result = run_backtest(strategy, data)
                s = compute_score(result, trading_days=trading_days)
                pnl = result.total_pnl
                improved = " *" if s > best_for_param[1] else ""
                sign = "+" if pnl >= 0 else ""
                print(
                    f"    {param}={str(val):<12s} P/L:{sign}${pnl:>10,.2f} "
                    f"WR:{_s2(result.win_rate):>5}% PF:{result.profit_factor:>5} "
                    f"Trades:{result.total_trades:>4d} Score:{s:>7.1f}{improved}"
                )
                if s > best_for_param[1]:
                    best_for_param = (val, s)

            if best_for_param[0] != current_val:
                best_overrides[param] = best_for_param[0]
                print(f"    >>> Best: {param}={best_for_param[0]}")
            print()

        # Combined result
        if best_overrides:
            print(thin)
            print(f"  COMBINED OPTIMAL PARAMETERS for {name}:")
            print(thin)
            for p, v in sorted(best_overrides.items()):
                print(f"    {p}: {getattr(config, p)} -> {v}")

            try:
                combined = dataclasses.replace(config, **best_overrides)
                result = run_backtest(entry.factory(config=combined), data)
                s = compute_score(result, trading_days=trading_days)
                delta = result.total_pnl - baseline_result.total_pnl
                sign = "+" if delta >= 0 else ""
                print(f"\n  Combined:  P/L=${_s2(result.total_pnl)}  WR={_s2(result.win_rate)}%  "
                      f"PF={result.profit_factor}  Trades={result.total_trades}  Score={s:.1f}")
                print(f"  Baseline:  P/L=${_s2(baseline_result.total_pnl)}  WR={_s2(baseline_result.win_rate)}%  "
                      f"PF={baseline_result.profit_factor}  Trades={baseline_result.total_trades}  Score={baseline_score:.1f}")
                print(f"  Delta:     {sign}${_s2(delta)}")

                # Greedy sequential validation
                print(f"\n  Greedy sequential validation:")
                greedy_config = config
                greedy_score = baseline_score
                for p, v in best_overrides.items():
                    trial = dataclasses.replace(greedy_config, **{p: v})
                    result = run_backtest(entry.factory(config=trial), data)
                    s = compute_score(result, trading_days=trading_days)
                    if s > greedy_score:
                        greedy_config = trial
                        greedy_score = s
                        print(f"    Applied {p}={v} -> Score={s:.1f} (accepted)")
                    else:
                        print(f"    Tried   {p}={v} -> Score={s:.1f} (rejected, conflicts)")

                final_result = run_backtest(entry.factory(config=greedy_config), data)
                print(f"\n  Greedy Best: P/L=${_s2(final_result.total_pnl)}  WR={_s2(final_result.win_rate)}%  "
                      f"PF={final_result.profit_factor}  Trades={final_result.total_trades}  Score={greedy_score:.1f}")
            except (TypeError, ValueError) as exc:
                print(f"  ERROR combining parameters: {exc}")

    # ==================================================================
    # PART 2: FAST DAILY STRATEGY OPTIMIZATION
    # ==================================================================
    print(f"\n{sep}")
    print("  DAILY STRATEGY OPTIMIZATION (fast strategies only)")
    print(f"{sep}\n")

    # Only optimize fast strategies (skip momentum, multi)
    fast_strategies = ["sma", "rsi", "macd", "bollinger", "breakout"]

    for strat_name in fast_strategies:
        entry = StrategyRegistry.get(strat_name)
        name = entry.display_name
        print(f"  Optimizing: {name}")

        best_kwargs = dict(entry.default_kwargs)
        best_adapter = {"sl_atr_mult": Decimal("1.5"), "rr": Decimal("1.5"), "sl_cap": Decimal("2.00")}

        # Get baseline score
        strat = entry.factory(**best_kwargs)
        adapter = DailyToIntradayAdapter(strat, **best_adapter)
        base_result = run_backtest(adapter, data)
        best_score = compute_score(base_result, trading_days=trading_days)
        best_result = base_result

        # Phase 1: Strategy params
        if entry.param_space:
            for param, values in entry.param_space.items():
                param_best = (best_kwargs.get(param), best_score)
                for val in values:
                    if val == best_kwargs.get(param):
                        continue
                    trial = {**best_kwargs, param: val}
                    try:
                        s_obj = entry.factory(**trial)
                        a = DailyToIntradayAdapter(s_obj, **best_adapter)
                        r = run_backtest(a, data)
                        s = compute_score(r, trading_days=trading_days)
                        if s > param_best[1]:
                            param_best = (val, s)
                    except (ValueError, TypeError):
                        continue
                if param_best[0] != best_kwargs.get(param) and param_best[0] is not None:
                    best_kwargs[param] = param_best[0]

            # Re-evaluate with all strategy param changes
            strat = entry.factory(**best_kwargs)
            adapter = DailyToIntradayAdapter(strat, **best_adapter)
            r = run_backtest(adapter, data)
            s = compute_score(r, trading_days=trading_days)
            if s > best_score:
                best_score = s
                best_result = r

        # Phase 2: Adapter params
        adapter_space = {
            "sl_atr_mult": [Decimal("0.8"), Decimal("1.0"), Decimal("1.3"), Decimal("1.5"), Decimal("1.8"), Decimal("2.0")],
            "rr": [Decimal("1.0"), Decimal("1.2"), Decimal("1.5"), Decimal("2.0"), Decimal("2.5")],
            "sl_cap": [Decimal("0.75"), Decimal("1.00"), Decimal("1.50"), Decimal("2.00"), Decimal("2.50"), Decimal("3.00")],
        }
        for param, values in adapter_space.items():
            param_best = (best_adapter.get(param), best_score)
            for val in values:
                if val == best_adapter.get(param):
                    continue
                trial_a = {**best_adapter, param: val}
                strat = entry.factory(**best_kwargs)
                a = DailyToIntradayAdapter(strat, **trial_a)
                r = run_backtest(a, data)
                s = compute_score(r, trading_days=trading_days)
                if s > param_best[1]:
                    param_best = (val, s)
            if param_best[0] != best_adapter.get(param) and param_best[0] is not None:
                best_adapter[param] = param_best[0]

        # Final with all adapter changes
        strat = entry.factory(**best_kwargs)
        adapter = DailyToIntradayAdapter(strat, **best_adapter)
        r = run_backtest(adapter, data)
        s = compute_score(r, trading_days=trading_days)
        if s > best_score:
            best_score = s
            best_result = r

        # Report
        delta = best_result.total_pnl - base_result.total_pnl
        sign = "+" if delta >= 0 else ""
        print(f"    Baseline:  P/L=${_s2(base_result.total_pnl):>10}  WR={_s2(base_result.win_rate):>5}%  "
              f"PF={base_result.profit_factor}  Trades={base_result.total_trades}")
        print(f"    Optimized: P/L=${_s2(best_result.total_pnl):>10}  WR={_s2(best_result.win_rate):>5}%  "
              f"PF={best_result.profit_factor}  Trades={best_result.total_trades}  ({sign}${_s2(delta)})")

        all_defaults = {**entry.default_kwargs, "sl_atr_mult": Decimal("1.5"), "rr": Decimal("1.5"), "sl_cap": Decimal("2.00")}
        combined = {**best_kwargs, **best_adapter}
        changed = {k: v for k, v in combined.items() if str(v) != str(all_defaults.get(k, ""))}
        if changed:
            print(f"    Changed:   {changed}")
        print()

    elapsed = time.time() - total_start
    print(f"\n  Total optimization time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print()


if __name__ == "__main__":
    main()
