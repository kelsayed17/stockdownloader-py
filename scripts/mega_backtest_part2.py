#!/usr/bin/env python3
"""Part 2: Run the 189 failed configs with correct config= pattern.

Strategies that take config= (not **overrides):
  DmiVwapStrategy, ORBreakoutStrategy, ORReversalStrategy, ReversalStrategy,
  PatternScalpStrategy, PullbackStrategy, AVWAPPullbackStrategy, SMCStructureStrategy
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.math import ZERO

logging.disable(logging.CRITICAL)

DATA_DIR = ROOT / "data" / "SPY"
CSV_5M = DATA_DIR / "5m_bars.csv"
RESULTS_CSV = DATA_DIR / "backtest_results_part2.csv"

D = Decimal


def load_5m_bars() -> list[IntradayPriceData]:
    bars: list[IntradayPriceData] = []
    with open(CSV_5M) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(IntradayPriceData(
                date=row["Datetime"],
                open=D(row["Open"]),
                high=D(row["High"]),
                low=D(row["Low"]),
                close=D(row["Close"]),
                adj_close=D(row["Close"]),
                volume=int(row["Volume"]),
            ))
    return bars


def extract_result(result, name: str, config_label: str, timeframe: str) -> dict:
    try:
        sharpe = float(result.sharpe_ratio())
    except Exception:
        sharpe = 0.0
    try:
        sortino = float(result.sortino_ratio())
    except Exception:
        sortino = 0.0
    try:
        calmar = float(result.calmar_ratio())
    except Exception:
        calmar = 0.0
    return {
        "strategy": name,
        "config": config_label,
        "timeframe": timeframe,
        "total_return_pct": float(result.total_return),
        "total_pnl": float(result.total_pnl),
        "total_trades": result.total_trades,
        "win_rate_pct": float(result.win_rate),
        "profit_factor": float(result.profit_factor),
        "max_drawdown_pct": float(result.max_drawdown),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "avg_win": float(result.average_win),
        "avg_loss": float(result.average_loss),
    }


def get_configs_with_factory():
    """Return (name, config_label, factory_callable) tuples for config-based strategies."""
    from stockdownloader.strategies.intraday.dmi_vwap import DmiVwapStrategy, DmiVwapConfig
    from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy, ORBreakoutStrategyConfig
    from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategy, ORReversalStrategyConfig
    from stockdownloader.strategies.intraday.reversal import ReversalStrategy, ReversalStrategyConfig
    from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy, PatternScalpStrategyConfig
    from stockdownloader.strategies.intraday.pullback import PullbackStrategy, PullbackStrategyConfig
    from stockdownloader.strategies.intraday.avwap_pullback import AVWAPPullbackStrategy, AVWAPPullbackConfig
    from stockdownloader.strategies.intraday.smc_structure import SMCStructureStrategy, SMCStructureConfig

    configs = []

    # --- DMI+VWAP (11 configs) ---
    name = "DmiVwapStrategy"
    configs.append((name, "default",
                     lambda: DmiVwapStrategy()))
    configs.append((name, "adx-20",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(adx_threshold=D("20")))))
    configs.append((name, "adx-30",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(adx_threshold=D("30")))))
    configs.append((name, "adx-rising",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(require_adx_rising=True))))
    configs.append((name, "rr-1.5",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(rr=D("1.5")))))
    configs.append((name, "rr-2.5",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(rr=D("2.5")))))
    configs.append((name, "rr-3.0",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(rr=D("3.0")))))
    configs.append((name, "dmi-10",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(dmi_period=10))))
    configs.append((name, "dmi-20",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(dmi_period=20))))
    configs.append((name, "tight-sl",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(sl_atr_mult=D("1.0")))))
    configs.append((name, "wide-sl",
                     lambda: DmiVwapStrategy(config=DmiVwapConfig(sl_atr_mult=D("2.0")))))

    # --- OR Breakout (10 configs) ---
    name = "ORBreakoutStrategy"
    configs.append((name, "default",
                     lambda: ORBreakoutStrategy()))
    configs.append((name, "or-15",
                     lambda: ORBreakoutStrategy(config=ORBreakoutStrategyConfig(orb_window=15))))
    configs.append((name, "or-45",
                     lambda: ORBreakoutStrategy(config=ORBreakoutStrategyConfig(orb_window=45))))
    configs.append((name, "or-60",
                     lambda: ORBreakoutStrategy(config=ORBreakoutStrategyConfig(orb_window=60))))
    configs.append((name, "rvol-1.5",
                     lambda: ORBreakoutStrategy(config=ORBreakoutStrategyConfig(orb_rvol=D("1.5")))))
    configs.append((name, "rvol-3.0",
                     lambda: ORBreakoutStrategy(config=ORBreakoutStrategyConfig(orb_rvol=D("3.0")))))
    configs.append((name, "no-vwap-filter",
                     lambda: ORBreakoutStrategy(config=ORBreakoutStrategyConfig(orb_vwap_align=False))))
    configs.append((name, "no-adx-filter",
                     lambda: ORBreakoutStrategy(config=ORBreakoutStrategyConfig(orb_adx_filter=False))))
    configs.append((name, "no-gap-filter",
                     lambda: ORBreakoutStrategy(config=ORBreakoutStrategyConfig(orb_gap_filter=False))))
    configs.append((name, "all-filters-off",
                     lambda: ORBreakoutStrategy(config=ORBreakoutStrategyConfig(
                         orb_vwap_align=False, orb_adx_filter=False, orb_gap_filter=False))))

    # --- OR Reversal (8 configs) ---
    name = "ORReversalStrategy"
    configs.append((name, "default",
                     lambda: ORReversalStrategy()))
    configs.append((name, "prox-0.4",
                     lambda: ORReversalStrategy(config=ORReversalStrategyConfig(orr_prox=D("0.4")))))
    configs.append((name, "prox-0.8",
                     lambda: ORReversalStrategy(config=ORReversalStrategyConfig(orr_prox=D("0.8")))))
    configs.append((name, "or-30",
                     lambda: ORReversalStrategy(config=ORReversalStrategyConfig(orr_window=30))))
    configs.append((name, "no-break-req",
                     lambda: ORReversalStrategy(config=ORReversalStrategyConfig(orr_require_break=False))))
    configs.append((name, "no-filters",
                     lambda: ORReversalStrategy(config=ORReversalStrategyConfig(
                         orr_adx_filter=False, orr_gap_filter=False))))
    configs.append((name, "long-only",
                     lambda: ORReversalStrategy(config=ORReversalStrategyConfig(allow_shorts=False))))
    configs.append((name, "tight-sl",
                     lambda: ORReversalStrategy(config=ORReversalStrategyConfig(orr_sl_atr=D("0.3")))))

    # --- Reversal (8 configs) ---
    name = "ReversalStrategy"
    configs.append((name, "default",
                     lambda: ReversalStrategy()))
    configs.append((name, "band-2s",
                     lambda: ReversalStrategy(config=ReversalStrategyConfig(rev_band="2σ"))))
    configs.append((name, "with-shorts",
                     lambda: ReversalStrategy(config=ReversalStrategyConfig(rev_shorts=True))))
    configs.append((name, "tp-rr-2.0",
                     lambda: ReversalStrategy(config=ReversalStrategyConfig(rev_tp_mode="rr", rev_rr=D("2.0")))))
    configs.append((name, "no-sr-req",
                     lambda: ReversalStrategy(config=ReversalStrategyConfig(rev_require_sr=False))))
    configs.append((name, "low-score",
                     lambda: ReversalStrategy(config=ReversalStrategyConfig(min_score=3))))
    configs.append((name, "tight-sl",
                     lambda: ReversalStrategy(config=ReversalStrategyConfig(rev_sl_atr=D("0.7"), rev_sl_cap=D("1.0")))))
    configs.append((name, "wide-sl",
                     lambda: ReversalStrategy(config=ReversalStrategyConfig(rev_sl_atr=D("1.5"), rev_sl_cap=D("2.5")))))

    # --- Pattern Scalp (8 configs) ---
    name = "PatternScalpStrategy"
    configs.append((name, "default",
                     lambda: PatternScalpStrategy()))
    configs.append((name, "rvol-1.0",
                     lambda: PatternScalpStrategy(config=PatternScalpStrategyConfig(ps_rvol=D("1.0")))))
    configs.append((name, "rvol-2.0",
                     lambda: PatternScalpStrategy(config=PatternScalpStrategyConfig(ps_rvol=D("2.0")))))
    configs.append((name, "no-sma-filter",
                     lambda: PatternScalpStrategy(config=PatternScalpStrategyConfig(ps_sma_filter=False))))
    configs.append((name, "high-rr",
                     lambda: PatternScalpStrategy(config=PatternScalpStrategyConfig(ps_min_rr=D("2.0")))))
    configs.append((name, "tight-sl",
                     lambda: PatternScalpStrategy(config=PatternScalpStrategyConfig(ps_sl_atr=D("1.0"), ps_sl_cap=D("1.0")))))
    configs.append((name, "wide-sl",
                     lambda: PatternScalpStrategy(config=PatternScalpStrategyConfig(ps_sl_atr=D("1.8"), ps_sl_cap=D("2.5")))))
    configs.append((name, "no-htf",
                     lambda: PatternScalpStrategy(config=PatternScalpStrategyConfig(ps_htf_align=False))))

    # --- Pullback (10 configs) ---
    name = "PullbackStrategy"
    configs.append((name, "default",
                     lambda: PullbackStrategy()))
    configs.append((name, "rr-2.5",
                     lambda: PullbackStrategy(config=PullbackStrategyConfig(rr=D("2.5")))))
    configs.append((name, "rr-1.0",
                     lambda: PullbackStrategy(config=PullbackStrategyConfig(rr=D("1.0")))))
    configs.append((name, "no-htf",
                     lambda: PullbackStrategy(config=PullbackStrategyConfig(htf_align=False))))
    configs.append((name, "no-vwap-bias",
                     lambda: PullbackStrategy(config=PullbackStrategyConfig(pb_vwap_bias=False))))
    configs.append((name, "no-cvd-filter",
                     lambda: PullbackStrategy(config=PullbackStrategyConfig(cvd_long_filter=False))))
    configs.append((name, "no-lrs-filter",
                     lambda: PullbackStrategy(config=PullbackStrategyConfig(lrs_short_filter=False))))
    configs.append((name, "all-filters-off",
                     lambda: PullbackStrategy(config=PullbackStrategyConfig(
                         htf_align=False, pb_vwap_bias=False, cvd_long_filter=False, lrs_short_filter=False))))
    configs.append((name, "tight-sl",
                     lambda: PullbackStrategy(config=PullbackStrategyConfig(sl_atr=D("1.0"), sl_cap=D("1.5")))))
    configs.append((name, "wide-sl",
                     lambda: PullbackStrategy(config=PullbackStrategyConfig(sl_atr=D("1.8"), sl_cap=D("3.0")))))

    # --- AVWAP Pullback (8 configs) ---
    name = "AVWAPPullbackStrategy"
    configs.append((name, "default",
                     lambda: AVWAPPullbackStrategy()))
    configs.append((name, "rr-1.5",
                     lambda: AVWAPPullbackStrategy(config=AVWAPPullbackConfig(avwap_rr=D("1.5")))))
    configs.append((name, "rr-3.0",
                     lambda: AVWAPPullbackStrategy(config=AVWAPPullbackConfig(avwap_rr=D("3.0")))))
    configs.append((name, "no-vwap-agree",
                     lambda: AVWAPPullbackStrategy(config=AVWAPPullbackConfig(avwap_session_vwap_agree=False))))
    configs.append((name, "no-htf",
                     lambda: AVWAPPullbackStrategy(config=AVWAPPullbackConfig(avwap_htf_align=False))))
    configs.append((name, "with-shorts",
                     lambda: AVWAPPullbackStrategy(config=AVWAPPullbackConfig(avwap_shorts=True))))
    configs.append((name, "tight-sl",
                     lambda: AVWAPPullbackStrategy(config=AVWAPPullbackConfig(avwap_sl_atr=D("1.0"), avwap_sl_cap=D("1.5")))))
    configs.append((name, "wide-sl",
                     lambda: AVWAPPullbackStrategy(config=AVWAPPullbackConfig(avwap_sl_atr=D("1.8"), avwap_sl_cap=D("3.0")))))

    # --- SMC Structure (8 configs) ---
    name = "SMCStructureStrategy"
    configs.append((name, "default",
                     lambda: SMCStructureStrategy()))
    configs.append((name, "no-bos-req",
                     lambda: SMCStructureStrategy(config=SMCStructureConfig(smc_require_bos=False))))
    configs.append((name, "no-sweep",
                     lambda: SMCStructureStrategy(config=SMCStructureConfig(smc_sweep_entry=False))))
    configs.append((name, "vwap-agree",
                     lambda: SMCStructureStrategy(config=SMCStructureConfig(smc_vwap_agree=True))))
    configs.append((name, "rr-2.0",
                     lambda: SMCStructureStrategy(config=SMCStructureConfig(smc_rr=D("2.0")))))
    configs.append((name, "with-shorts",
                     lambda: SMCStructureStrategy(config=SMCStructureConfig(smc_shorts=True))))
    configs.append((name, "tight-sl",
                     lambda: SMCStructureStrategy(config=SMCStructureConfig(smc_sl_atr=D("1.5"), smc_sl_cap=D("1.0")))))
    configs.append((name, "low-score",
                     lambda: SMCStructureStrategy(config=SMCStructureConfig(min_score=2))))

    return configs


def main():
    t0 = time.time()
    print("=" * 80)
    print("MEGA BACKTEST PART 2: Config-based strategies")
    print("=" * 80)

    print("\nLoading 5-minute bars...")
    bars_5m = load_5m_bars()
    print(f"  {len(bars_5m)} bars loaded")

    all_configs = get_configs_with_factory()
    risk_levels = [
        ("risk-0.5%", D("0.005")),
        ("risk-1%", D("0.01")),
        ("risk-2%", D("0.02")),
    ]

    total = len(all_configs) * len(risk_levels)
    print(f"\nRunning {len(all_configs)} configs × {len(risk_levels)} risk levels = {total} combinations")

    INITIAL_CAPITAL = D("100000")
    all_results = []
    done = 0
    errors = 0

    for name, config_label, factory in all_configs:
        for risk_label, risk_pct in risk_levels:
            done += 1
            full_label = f"{config_label} | {risk_label}"
            try:
                strategy = factory()
                engine = IntradayBacktestEngine(
                    initial_capital=INITIAL_CAPITAL,
                    risk_per_trade=risk_pct,
                    slippage_pct=D("0.0002"),
                )
                result = engine.run(strategy, bars_5m)
                row = extract_result(result, name, full_label, "5m")
                all_results.append(row)
                if done % 15 == 0 or done == total:
                    elapsed = time.time() - t0
                    print(f"  [{done}/{total}] {elapsed:.0f}s | "
                          f"{name} {config_label} {risk_label}: "
                          f"{row['total_return_pct']:+.2f}% | "
                          f"{row['total_trades']} trades | "
                          f"WR {row['win_rate_pct']:.1f}%")
            except Exception as e:
                errors += 1
                print(f"  [{done}/{total}] ERROR {name} {config_label} {risk_label}: {e}")

    print(f"\nCompleted: {done - errors} succeeded, {errors} errors")

    # Save part 2 results
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Part 2 results saved to: {RESULTS_CSV}")

    # Print part 2 leaderboard
    sorted_results = sorted(all_results, key=lambda r: r["total_return_pct"], reverse=True)
    print(f"\n{'=' * 120}")
    print("PART 2 LEADERBOARD — TOP 20")
    print(f"{'=' * 120}")
    header = (f"{'Rank':>4} {'Strategy':<30} {'Config':<35} "
              f"{'Return%':>9} {'Trades':>6} {'WR%':>6} {'PF':>6} "
              f"{'Sharpe':>7} {'MaxDD%':>7}")
    print(header)
    print("-" * 120)
    for rank, row in enumerate(sorted_results[:20], 1):
        print(f"{rank:4d} {row['strategy']:<30} {row['config']:<35} "
              f"{row['total_return_pct']:>+9.2f} {row['total_trades']:>6} "
              f"{row['win_rate_pct']:>6.1f} {row['profit_factor']:>6.2f} "
              f"{row['sharpe']:>7.2f} {row['max_drawdown_pct']:>7.2f}")

    print(f"\n{'=' * 80}")
    print("BOTTOM 10")
    print(f"{'=' * 80}")
    worst = sorted(all_results, key=lambda r: r["total_return_pct"])[:10]
    for rank, row in enumerate(worst, 1):
        print(f"{rank:4d} {row['strategy']:<30} {row['config']:<35} "
              f"{row['total_return_pct']:>+9.2f} {row['total_trades']:>6} "
              f"{row['win_rate_pct']:>6.1f} {row['profit_factor']:>6.2f} "
              f"{row['sharpe']:>7.2f} {row['max_drawdown_pct']:>7.2f}")

    # Summary by family
    from collections import defaultdict
    by_family = defaultdict(list)
    for r in all_results:
        by_family[r["strategy"]].append(r)

    print(f"\n{'=' * 80}")
    print("SUMMARY BY FAMILY")
    print(f"{'=' * 80}")
    for family in sorted(by_family.keys()):
        best = max(by_family[family], key=lambda r: r["total_return_pct"])
        n = len(by_family[family])
        n_profit = sum(1 for r in by_family[family] if r["total_return_pct"] > 0)
        avg_ret = sum(r["total_return_pct"] for r in by_family[family]) / n
        print(f"\n  {family} ({n} configs, {n_profit} profitable, avg {avg_ret:+.2f}%)")
        print(f"    Best: {best['config']:<40} {best['total_return_pct']:>+8.2f}% | "
              f"WR {best['win_rate_pct']:.1f}% | PF {best['profit_factor']:.2f} | "
              f"Sharpe {best['sharpe']:.2f} | Trades {best['total_trades']}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
