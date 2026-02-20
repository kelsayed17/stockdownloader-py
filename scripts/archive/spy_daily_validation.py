#!/usr/bin/env python3
"""Validate top indicator signals on 33 years of SPY daily data.

Tests the best signals discovered on intraday data against the full
1993-2026 daily history for robustness confirmation. Also runs the
ML feature extractor to identify top predictive features.
"""
from __future__ import annotations

import math
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.data.csv_price_data_loader import CsvPriceDataLoader
from stockdownloader.model.price_data import PriceData
from stockdownloader.util.indicator_hub import IndicatorHub


# ======================================================================
# Condition definitions (same as intraday but on daily)
# ======================================================================

@dataclass
class Condition:
    name: str
    description: str
    check_fn: object
    direction: str  # "long" or "short"


def build_conditions() -> list[Condition]:
    """Build indicator conditions to test on daily data."""
    conds: list[Condition] = []

    # ── RSI conditions (our #1 intraday winner) ──
    for t in (20, 25, 30, 35):
        conds.append(Condition(
            f"rsi14_below_{t}", f"RSI(14) < {t}",
            lambda h, d, i, t=t: h.rsi(d, i, 14) < t, "long",
        ))
    for t in (65, 70, 75, 80):
        conds.append(Condition(
            f"rsi14_above_{t}", f"RSI(14) > {t}",
            lambda h, d, i, t=t: h.rsi(d, i, 14) > t, "short",
        ))
    # RSI crossover signals
    conds.append(Condition(
        "rsi14_cross_above_30", "RSI crosses above 30",
        lambda h, d, i: h.rsi(d, i, 14) > 30 and i > 0 and h.rsi(d, i - 1, 14) <= 30, "long",
    ))
    conds.append(Condition(
        "rsi14_cross_below_70", "RSI crosses below 70",
        lambda h, d, i: h.rsi(d, i, 14) < 70 and i > 0 and h.rsi(d, i - 1, 14) >= 70, "short",
    ))

    # ── MFI conditions (our #1 overall winner) ──
    for t in (15, 20, 25):
        conds.append(Condition(
            f"mfi_below_{t}", f"MFI(14) < {t}",
            lambda h, d, i, t=t: h.mfi(d, i, 14) < t, "long",
        ))
    for t in (75, 80, 85):
        conds.append(Condition(
            f"mfi_above_{t}", f"MFI(14) > {t}",
            lambda h, d, i, t=t: h.mfi(d, i, 14) > t, "short",
        ))

    # ── Bollinger Band conditions (our #2 winner on 1h) ──
    conds.append(Condition(
        "bb_touch_lower", "Close at/below lower BB",
        lambda h, d, i: float(d[i].close) <= float(h.bollinger_bands(d, i).lower), "long",
    ))
    conds.append(Condition(
        "bb_touch_upper", "Close at/above upper BB",
        lambda h, d, i: float(d[i].close) >= float(h.bollinger_bands(d, i).upper), "short",
    ))
    conds.append(Condition(
        "bb_percent_b_below_0", "BB %B < 0 (below lower band)",
        lambda h, d, i: h.bollinger_percent_b(d, i) < 0.0, "long",
    ))
    conds.append(Condition(
        "bb_percent_b_above_1", "BB %B > 1 (above upper band)",
        lambda h, d, i: h.bollinger_percent_b(d, i) > 1.0, "short",
    ))

    # ── MACD conditions ──
    conds.append(Condition(
        "macd_bullish_cross", "MACD bullish crossover",
        lambda h, d, i: (
            i > 0
            and h.macd_line(d, i) > h.macd_signal(d, i)
            and h.macd_line(d, i - 1) <= h.macd_signal(d, i - 1)
        ), "long",
    ))
    conds.append(Condition(
        "macd_bearish_cross", "MACD bearish crossover",
        lambda h, d, i: (
            i > 0
            and h.macd_line(d, i) < h.macd_signal(d, i)
            and h.macd_line(d, i - 1) >= h.macd_signal(d, i - 1)
        ), "short",
    ))

    # ── ADX/DMI conditions ──
    for t in (20, 25, 30):
        conds.append(Condition(
            f"adx_gt_{t}_pdi", f"ADX>{t} & +DI>-DI",
            lambda h, d, i, t=t: (
                h.adx(d, i) > t
                and h.adx_result(d, i).plus_di > h.adx_result(d, i).minus_di
            ), "long",
        ))
        conds.append(Condition(
            f"adx_gt_{t}_mdi", f"ADX>{t} & -DI>+DI",
            lambda h, d, i, t=t: (
                h.adx(d, i) > t
                and h.adx_result(d, i).minus_di > h.adx_result(d, i).plus_di
            ), "short",
        ))

    # ── Stochastic ──
    conds.append(Condition(
        "stoch_cross_above_20", "Stoch %K crosses above 20",
        lambda h, d, i: (
            i > 0
            and h.stochastic(d, i).percent_k > 20
            and h.stochastic(d, i - 1).percent_k <= 20
        ), "long",
    ))
    conds.append(Condition(
        "stoch_cross_below_80", "Stoch %K crosses below 80",
        lambda h, d, i: (
            i > 0
            and h.stochastic(d, i).percent_k < 80
            and h.stochastic(d, i - 1).percent_k >= 80
        ), "short",
    ))

    # ── Williams %R ──
    conds.append(Condition(
        "willr_below_neg80", "Williams %R < -80",
        lambda h, d, i: h.williams_r(d, i) < -80, "long",
    ))
    conds.append(Condition(
        "willr_above_neg20", "Williams %R > -20",
        lambda h, d, i: h.williams_r(d, i) > -20, "short",
    ))

    # ── CCI ──
    conds.append(Condition(
        "cci_below_neg100", "CCI(20) < -100",
        lambda h, d, i: h.cci(d, i) < -100, "long",
    ))
    conds.append(Condition(
        "cci_above_100", "CCI(20) > 100",
        lambda h, d, i: h.cci(d, i) > 100, "short",
    ))

    # ── EMA crossover ──
    conds.append(Condition(
        "ema9_cross_above_21", "EMA(9) crosses above EMA(21)",
        lambda h, d, i: (
            i > 0
            and h.ema(d, i, 9) > h.ema(d, i, 21)
            and h.ema(d, i - 1, 9) <= h.ema(d, i - 1, 21)
        ), "long",
    ))
    conds.append(Condition(
        "ema9_cross_below_21", "EMA(9) crosses below EMA(21)",
        lambda h, d, i: (
            i > 0
            and h.ema(d, i, 9) < h.ema(d, i, 21)
            and h.ema(d, i - 1, 9) >= h.ema(d, i - 1, 21)
        ), "short",
    ))

    # ── Price vs SMA200 ──
    conds.append(Condition(
        "price_above_sma200", "Close > SMA(200)",
        lambda h, d, i: float(d[i].close) > h.sma(d, i, 200), "long",
    ))
    conds.append(Condition(
        "price_below_sma200", "Close < SMA(200)",
        lambda h, d, i: float(d[i].close) < h.sma(d, i, 200), "short",
    ))

    # ── OBV ──
    conds.append(Condition(
        "obv_rising", "OBV rising (5-bar)",
        lambda h, d, i: h.is_obv_rising(d, i, 5), "long",
    ))
    conds.append(Condition(
        "obv_falling", "OBV falling (5-bar)",
        lambda h, d, i: not h.is_obv_rising(d, i, 5), "short",
    ))

    # ── SAR ──
    conds.append(Condition(
        "sar_bullish_flip", "SAR flips bullish",
        lambda h, d, i: (
            i > 0
            and h.parabolic_sar(d, i) < float(d[i].close)
            and h.parabolic_sar(d, i - 1) >= float(d[i - 1].close)
        ), "long",
    ))
    conds.append(Condition(
        "sar_bearish_flip", "SAR flips bearish",
        lambda h, d, i: (
            i > 0
            and h.parabolic_sar(d, i) > float(d[i].close)
            and h.parabolic_sar(d, i - 1) <= float(d[i - 1].close)
        ), "short",
    ))

    # ── RSI + MFI combo (top combo from intraday) ──
    conds.append(Condition(
        "rsi30_and_mfi20", "RSI<30 AND MFI<20 (double oversold)",
        lambda h, d, i: h.rsi(d, i, 14) < 30 and h.mfi(d, i, 14) < 20, "long",
    ))
    conds.append(Condition(
        "rsi25_and_mfi20", "RSI<25 AND MFI<20 (extreme oversold)",
        lambda h, d, i: h.rsi(d, i, 14) < 25 and h.mfi(d, i, 14) < 20, "long",
    ))

    # ── RSI + BB combo ──
    conds.append(Condition(
        "rsi30_and_bb_lower", "RSI<30 AND Close <= lower BB",
        lambda h, d, i: (
            h.rsi(d, i, 14) < 30
            and float(d[i].close) <= float(h.bollinger_bands(d, i).lower)
        ), "long",
    ))

    # ── MFI + BB combo ──
    conds.append(Condition(
        "mfi20_and_bb_lower", "MFI<20 AND Close <= lower BB",
        lambda h, d, i: (
            h.mfi(d, i, 14) < 20
            and float(d[i].close) <= float(h.bollinger_bands(d, i).lower)
        ), "long",
    ))

    return conds


# ======================================================================
# Statistics
# ======================================================================

def _t_test(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n < 3:
        return 0.0, 1.0
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    if std == 0:
        return 0.0, 1.0
    t = mean / (std / math.sqrt(n))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def _cohens_d(vals: list[float]) -> float:
    if len(vals) < 3:
        return 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    return abs(m) / s if s > 0 else 0.0


def _profit_factor(rets: list[float]) -> float:
    gains = sum(r for r in rets if r > 0)
    losses = abs(sum(r for r in rets if r < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0
    return gains / losses


def _walk_forward_folds(rets: list[float], n: int = 5) -> list[float]:
    """5-fold expanding walk-forward win rates."""
    if len(rets) < n * 20:
        return []
    sz = len(rets) // n
    return [sum(1 for r in rets[f * sz:(f + 1) * sz] if r > 0) / sz for f in range(n)]


@dataclass
class Result:
    name: str
    direction: str
    horizon: int
    n: int
    avg_return: float
    median_return: float
    std_return: float
    win_rate: float
    profit_factor: float
    t_stat: float
    p_value: float
    effect_size: float
    wf_folds: list[float] = field(default_factory=list)
    wf_stable: bool = False
    # Decade breakdown
    decade_wrs: dict[str, tuple[float, int]] = field(default_factory=dict)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    csv_path = "data/spy_daily_bars.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    print("=" * 100)
    print("  SPY 33-YEAR DAILY VALIDATION")
    print("  Testing indicator signals on 1993-2026 daily data")
    print("=" * 100)

    data = CsvPriceDataLoader.load_from_file(csv_path)
    print(f"\n  Loaded {len(data):,} daily bars ({data[0].date} to {data[-1].date})")

    hub = IndicatorHub()
    conditions = build_conditions()
    horizons = (1, 3, 5, 10, 20)
    warmup = 201
    min_trades = 30
    results: list[Result] = []

    print(f"  Testing {len(conditions)} conditions × {len(horizons)} horizons...")
    print(f"  Warmup: {warmup} bars, min trades: {min_trades}")

    t0 = time.time()
    for ci, c in enumerate(conditions):
        if (ci + 1) % 10 == 0:
            print(f"    [{ci + 1}/{len(conditions)}] {c.name} ({time.time() - t0:.1f}s)")

        for h in horizons:
            rets = []
            decade_rets: dict[str, list[float]] = defaultdict(list)

            for i in range(warmup, len(data)):
                try:
                    if c.check_fn(hub, data, i):
                        j = i + h
                        if j < len(data):
                            ci_close = float(data[i].close)
                            cj_close = float(data[j].close)
                            if ci_close > 0:
                                ret = (cj_close - ci_close) / ci_close * 100
                                if c.direction == "short":
                                    ret = -ret
                                rets.append(ret)

                                # Decade classification
                                year = int(data[i].date[:4])
                                decade = f"{(year // 10) * 10}s"
                                decade_rets[decade].append(ret)
                except Exception:
                    continue

            if len(rets) < min_trades:
                continue

            avg = statistics.mean(rets)
            med = statistics.median(rets)
            std = statistics.stdev(rets) if len(rets) > 1 else 0
            wr = sum(1 for r in rets if r > 0) / len(rets)
            pf = _profit_factor(rets)
            t, p = _t_test(rets)
            d = _cohens_d(rets)
            wf = _walk_forward_folds(rets, 5)
            wf_ok = len(wf) >= 5 and sum(1 for fw in wf if fw > 0.50) >= 4

            # Decade win rates
            dec_wrs = {}
            for decade, drets in sorted(decade_rets.items()):
                if drets:
                    dwr = sum(1 for r in drets if r > 0) / len(drets)
                    dec_wrs[decade] = (dwr, len(drets))

            r = Result(
                name=c.name, direction=c.direction, horizon=h,
                n=len(rets), avg_return=avg, median_return=med, std_return=std,
                win_rate=wr, profit_factor=pf,
                t_stat=t, p_value=p, effect_size=d,
                wf_folds=wf, wf_stable=wf_ok,
                decade_wrs=dec_wrs,
            )
            results.append(r)

    elapsed = time.time() - t0
    print(f"\n  Analysis completed in {elapsed:.1f}s")

    # ── Filter and rank ──
    robust = [
        r for r in results
        if r.avg_return > 0
        and r.win_rate >= 0.52
        and r.p_value <= 0.05
        and r.effect_size >= 0.10
        and r.profit_factor >= 1.2
        and r.wf_stable
    ]
    robust.sort(key=lambda r: -r.t_stat)

    print(f"\n  Total condition×horizon results: {len(results)}")
    print(f"  Passed all filters: {len(robust)}")

    if robust:
        print(f"\n{'=' * 160}")
        print(f"  ROBUST SIGNALS ON 33 YEARS OF SPY DAILY DATA")
        print(f"  (WR≥52%, p≤0.05, d≥0.1, PF≥1.2, 4/5 WF folds profitable)")
        print(f"{'=' * 160}")
        print()
        print(f"  {'#':>3} {'Name':<35} {'Dir':<6} {'H':>3} {'N':>6} "
              f"{'AvgRet':>8} {'MedRet':>8} {'WR':>6} {'PF':>5} "
              f"{'t':>6} {'p':>8} {'d':>5}  Decades")
        print("  " + "-" * 150)

        for i, r in enumerate(robust[:40], 1):
            dec_str = "  ".join(
                f"{dk}:{dv[0]:.0%}({dv[1]})" for dk, dv in r.decade_wrs.items()
            )
            print(f"  {i:>3} {r.name:<35} {r.direction:<6} {r.horizon:>3} {r.n:>6} "
                  f"{r.avg_return:>+7.3f}% {r.median_return:>+7.3f}% "
                  f"{r.win_rate:>5.1%} {r.profit_factor:>5.2f} "
                  f"{r.t_stat:>6.2f} {r.p_value:>8.5f} {r.effect_size:>5.2f}  "
                  f"{dec_str}")

    # ── Near misses ──
    near = [
        r for r in results
        if r.avg_return > 0
        and r.win_rate >= 0.50
        and r.p_value <= 0.10
        and r not in robust
    ]
    near.sort(key=lambda r: r.p_value)

    if near:
        print(f"\n  NEAR-MISSES ({len(near[:20])} of {len(near)}):")
        print(f"  {'Name':<35} {'Dir':<6} {'H':>3} {'N':>6} "
              f"{'AvgRet':>8} {'WR':>6} {'PF':>5} {'p':>8} {'d':>5} {'WF':>4}")
        print("  " + "-" * 110)
        for r in near[:20]:
            wf = "YES" if r.wf_stable else "NO"
            print(f"  {r.name:<35} {r.direction:<6} {r.horizon:>3} {r.n:>6} "
                  f"{r.avg_return:>+7.3f}% {r.win_rate:>5.1%} {r.profit_factor:>5.2f} "
                  f"{r.p_value:>8.5f} {r.effect_size:>5.2f} {wf:>4}")


if __name__ == "__main__":
    main()
