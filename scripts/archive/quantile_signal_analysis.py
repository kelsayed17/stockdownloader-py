#!/usr/bin/env python3
"""Deep quantitative quintile analysis of indicator-based entry signals on SPY 5m bars.

Splits data into 60% discovery / 40% validation with a 100-bar gap.
Computes 16 indicators per bar, bins into quintiles, measures forward
returns at multiple horizons, ranks by t-statistic, and validates
out-of-sample.  Also tests 2-indicator combinations.
"""
from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

# ── project imports ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.util.indicators.hub import IndicatorHub

# ── constants ─────────────────────────────────────────────────────
DATA_FILE = ROOT / "data" / "spy_5m_bars.csv"
WARMUP = 201          # bars needed before indicators are stable
GAP = 100             # bars between IS and OOS
HORIZONS = [1, 3, 5, 10, 20]
N_QUINTILES = 5
TOP_N = 10            # top single conditions to carry forward
TOP_COMBOS = 15       # top combos to display

# ── helpers ───────────────────────────────────────────────────────

def t_stat(values: list[float]) -> tuple[float, float]:
    """Return (t-statistic, two-sided p-value) for H0: mean == 0."""
    n = len(values)
    if n < 3:
        return 0.0, 1.0
    m = mean(values)
    s = stdev(values)
    if s == 0:
        return 0.0, 1.0
    t = m / (s / math.sqrt(n))
    # Approximate two-tailed p from t-distribution using normal for large n
    # (good enough for n > 30; most quintiles will have thousands of bars)
    p = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return round(t, 3), round(p, 6)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def quintile_label(q: int) -> str:
    return f"Q{q+1}"


def pct(d: Decimal, base: Decimal) -> float:
    """Percentage change (float)."""
    if base == 0:
        return 0.0
    return float((d - base) / base) * 100.0


# ── load data ─────────────────────────────────────────────────────
print("=" * 80)
print("DEEP QUANTILE SIGNAL ANALYSIS — SPY 5-minute bars")
print("=" * 80)

t0 = time.time()
bars = IntradayCsvLoader.load_from_file(str(DATA_FILE))
print(f"\nLoaded {len(bars):,} bars from {DATA_FILE.name}")
print(f"  Date range: {bars[0].date[:10]} .. {bars[-1].date[:10]}")

# ── split ─────────────────────────────────────────────────────────
n = len(bars)
split_idx = int(n * 0.60)
oos_start = split_idx + GAP

print(f"\n  Discovery (IS) : bars {WARMUP:,} – {split_idx-1:,}  ({split_idx - WARMUP:,} usable bars)")
print(f"  Gap            : bars {split_idx:,} – {oos_start-1:,}  ({GAP} bars)")
print(f"  Validation (OOS): bars {oos_start:,} – {n-1:,}  ({n - oos_start:,} bars)")

# ── compute indicators (IS) ──────────────────────────────────────
print("\nComputing indicators on discovery set …")
hub = IndicatorHub()

# We'll store rows as dicts:  { indicator_name: float_value }
# and forward returns separately.

def compute_row(data, idx, hub_obj):
    """Compute all indicator values for bar *idx*. Returns dict or None if any fail."""
    try:
        close = float(data[idx].close)
        rsi_val = float(hub_obj.rsi(data, idx, 14))
        mfi_val = float(hub_obj.mfi(data, idx, 14))
        stoch = hub_obj.stochastic(data, idx, 14, 3)
        stoch_k = float(stoch.percent_k)
        wil_r = float(hub_obj.williams_r(data, idx, 14))
        cci_val = float(hub_obj.cci(data, idx, 20))
        bb_pctb = float(hub_obj.bollinger_percent_b(data, idx, 20))
        bb = hub_obj.bollinger_bands(data, idx, 20, 2.0)
        bb_width = float(bb.width)
        macd_h = float(hub_obj.macd_histogram(data, idx, 12, 26, 9))
        adx_res = hub_obj.adx(data, idx, 14)
        adx_val = float(adx_res.adx)
        plus_di = float(adx_res.plus_di)
        minus_di = float(adx_res.minus_di)
        ema9 = float(hub_obj.ema(data, idx, 9))
        ema21 = float(hub_obj.ema(data, idx, 21))
        ema_spread = ((ema9 - ema21) / ema21 * 100.0) if ema21 != 0 else 0.0
        vwap_val = float(hub_obj.vwap(data, idx, 20))
        price_vs_vwap = ((close - vwap_val) / vwap_val * 100.0) if vwap_val != 0 else 0.0
        avg_vol = float(hub_obj.average_volume(data, idx, 20))
        cur_vol = float(data[idx].volume)
        vol_ratio = (cur_vol / avg_vol) if avg_vol > 0 else 1.0
        atr_val = float(hub_obj.atr(data, idx, 14))
        atr_pct = (atr_val / close * 100.0) if close > 0 else 0.0
        obv_rising = 1.0 if hub_obj.is_obv_rising(data, idx, 5) else 0.0
        di_spread = plus_di - minus_di

        return {
            "RSI_14": rsi_val,
            "MFI_14": mfi_val,
            "Stoch_%K": stoch_k,
            "Williams_%R": wil_r,
            "CCI_20": cci_val,
            "BB_%B": bb_pctb,
            "BB_Width": bb_width,
            "MACD_Hist": macd_h,
            "ADX_14": adx_val,
            "+DI_14": plus_di,
            "-DI_14": minus_di,
            "DI_Spread": di_spread,
            "EMA_9v21_Spread": ema_spread,
            "Price_vs_VWAP": price_vs_vwap,
            "Vol_Ratio": vol_ratio,
            "ATR_%_Price": atr_pct,
            "OBV_Rising": obv_rising,
        }
    except Exception:
        return None


def compute_forward_returns(data, idx, horizons):
    """Forward returns (%) at each horizon."""
    close_now = float(data[idx].close)
    if close_now == 0:
        return None
    fwd = {}
    for h in horizons:
        if idx + h < len(data):
            fwd[h] = (float(data[idx + h].close) - close_now) / close_now * 100.0
        else:
            fwd[h] = None
    return fwd


# -- build IS dataset --
is_rows = []  # list of (indicators_dict, fwd_returns_dict)
progress_every = 5000
for i in range(WARMUP, split_idx):
    row = compute_row(bars, i, hub)
    if row is None:
        continue
    fwd = compute_forward_returns(bars, i, HORIZONS)
    if fwd is None:
        continue
    is_rows.append((row, fwd))
    if (i - WARMUP) % progress_every == 0 and i > WARMUP:
        print(f"  ... {i - WARMUP:,} bars processed")

print(f"  IS dataset: {len(is_rows):,} rows  ({time.time()-t0:.1f}s elapsed)")

# ── quintile analysis ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("QUINTILE ANALYSIS — DISCOVERY SET (IN-SAMPLE)")
print("=" * 80)

indicator_names = list(is_rows[0][0].keys())

# For each indicator, collect (value, fwd_returns)
# Sort by value, split into quintiles

def quintile_analysis(rows, ind_name, horizons):
    """Perform quintile analysis for one indicator. Returns list of quintile dicts."""
    pairs = [(r[0][ind_name], r[1]) for r in rows if r[0][ind_name] is not None]
    pairs.sort(key=lambda x: x[0])
    n_pts = len(pairs)
    q_size = n_pts // N_QUINTILES

    results = []
    for q in range(N_QUINTILES):
        start = q * q_size
        end = (q + 1) * q_size if q < N_QUINTILES - 1 else n_pts
        subset = pairs[start:end]
        vals = [x[0] for x in subset]
        val_min = min(vals) if vals else 0
        val_max = max(vals) if vals else 0

        qr = {
            "quintile": q,
            "label": quintile_label(q),
            "count": len(subset),
            "val_range": (round(val_min, 4), round(val_max, 4)),
        }
        for h in horizons:
            rets = [x[1][h] for x in subset if x[1][h] is not None]
            if len(rets) < 3:
                qr[f"avg_{h}"] = 0.0
                qr[f"wr_{h}"] = 0.0
                qr[f"t_{h}"] = 0.0
                qr[f"p_{h}"] = 1.0
                continue
            avg_r = mean(rets)
            wr = sum(1 for r in rets if r > 0) / len(rets) * 100.0
            t, p = t_stat(rets)
            qr[f"avg_{h}"] = round(avg_r, 5)
            qr[f"wr_{h}"] = round(wr, 2)
            qr[f"t_{h}"] = t
            qr[f"p_{h}"] = p
        results.append(qr)
    return results


# Collect all quintile results and rank by best forward return
all_conditions = []  # (indicator, quintile, horizon, avg_ret, wr, t, p, val_range)

for ind in indicator_names:
    qr = quintile_analysis(is_rows, ind, HORIZONS)

    # Print per-indicator table for 5-bar horizon
    h = 5
    print(f"\n{'─' * 70}")
    print(f"  {ind}  (5-bar fwd return)")
    print(f"{'─' * 70}")
    print(f"  {'Quintile':8s} {'Range':>24s} {'N':>6s} {'AvgRet%':>9s} {'WinRate%':>9s} {'t-stat':>8s} {'p-val':>8s}")
    for q in qr:
        print(f"  {q['label']:8s} [{q['val_range'][0]:>10.4f}, {q['val_range'][1]:>10.4f}] "
              f"{q['count']:6d} {q[f'avg_{h}']:>9.5f} {q[f'wr_{h}']:>8.2f}% "
              f"{q[f't_{h}']:>8.3f} {q[f'p_{h}']:>8.6f}")

    # Collect all for ranking
    for q in qr:
        for hh in HORIZONS:
            all_conditions.append({
                "indicator": ind,
                "quintile": q["label"],
                "horizon": hh,
                "avg_ret": q[f"avg_{hh}"],
                "win_rate": q[f"wr_{hh}"],
                "t_stat": q[f"t_{hh}"],
                "p_val": q[f"p_{hh}"],
                "val_range": q["val_range"],
                "count": q["count"],
            })

# ── rank best single conditions ──────────────────────────────────
print("\n\n" + "=" * 80)
print("TOP SINGLE CONDITIONS — RANKED BY ABS(t-stat) WITH p < 0.05")
print("=" * 80)

# Filter significant
sig = [c for c in all_conditions if c["p_val"] < 0.05]
sig.sort(key=lambda c: abs(c["t_stat"]), reverse=True)

# Show top long (positive avg_ret) and top short (negative)
sig_long = [c for c in sig if c["avg_ret"] > 0]
sig_short = [c for c in sig if c["avg_ret"] < 0]

print(f"\n  {'#':>3s} {'Indicator':>20s} {'Q':>4s} {'H':>3s} {'AvgRet%':>9s} {'WR%':>7s} {'t':>8s} {'p':>10s} {'N':>6s} {'Range'}")
print("  " + "-" * 110)

for rank, c in enumerate(sig_long[:30], 1):
    print(f"  {rank:3d} {c['indicator']:>20s} {c['quintile']:>4s} {c['horizon']:>3d} "
          f"{c['avg_ret']:>9.5f} {c['win_rate']:>6.2f}% {c['t_stat']:>8.3f} {c['p_val']:>10.6f} "
          f"{c['count']:>6d} [{c['val_range'][0]:.4f},{c['val_range'][1]:.4f}]")

print(f"\n  {'#':>3s} {'Indicator':>20s} {'Q':>4s} {'H':>3s} {'AvgRet%':>9s} {'WR%':>7s} {'t':>8s} {'p':>10s} {'N':>6s} {'Range'}")
print("  " + "-" * 110)
print("  BEST SHORT SIGNALS (negative forward return = profitable short)")
for rank, c in enumerate(sig_short[:20], 1):
    print(f"  {rank:3d} {c['indicator']:>20s} {c['quintile']:>4s} {c['horizon']:>3d} "
          f"{c['avg_ret']:>9.5f} {c['win_rate']:>6.2f}% {c['t_stat']:>8.3f} {c['p_val']:>10.6f} "
          f"{c['count']:>6d} [{c['val_range'][0]:.4f},{c['val_range'][1]:.4f}]")

# ── select top-N for OOS validation ──────────────────────────────
# Pick unique indicator+quintile combos from top LONG conditions (horizon=5)
seen = set()
top_conditions = []
for c in sig_long:
    if c["horizon"] != 5:
        continue
    key = (c["indicator"], c["quintile"])
    if key not in seen:
        seen.add(key)
        top_conditions.append(c)
    if len(top_conditions) >= TOP_N:
        break

# If we don't have enough at horizon 5, add from other horizons
if len(top_conditions) < TOP_N:
    for c in sig_long:
        key = (c["indicator"], c["quintile"])
        if key not in seen:
            seen.add(key)
            top_conditions.append(c)
        if len(top_conditions) >= TOP_N:
            break

print(f"\n\nSelected {len(top_conditions)} conditions for OOS validation.")

# ── build OOS dataset ─────────────────────────────────────────────
print("\nComputing indicators on validation set …")
hub2 = IndicatorHub()  # fresh hub for OOS

oos_rows = []
for i in range(oos_start, n):
    row = compute_row(bars, i, hub2)
    if row is None:
        continue
    fwd = compute_forward_returns(bars, i, HORIZONS)
    if fwd is None:
        continue
    oos_rows.append((row, fwd))

print(f"  OOS dataset: {len(oos_rows):,} rows")

# ── need quintile boundaries from IS ──────────────────────────────
def get_quintile_boundaries(rows, ind_name):
    """Get boundary values separating quintiles."""
    vals = sorted([r[0][ind_name] for r in rows])
    n_pts = len(vals)
    q_size = n_pts // N_QUINTILES
    boundaries = []
    for q in range(1, N_QUINTILES):
        boundaries.append(vals[q * q_size])
    return boundaries


def assign_quintile(value, boundaries):
    """Assign a value to a quintile given the IS boundaries."""
    for i, b in enumerate(boundaries):
        if value < b:
            return i
    return N_QUINTILES - 1


# Precompute IS boundaries for each indicator
is_boundaries = {}
for ind in indicator_names:
    is_boundaries[ind] = get_quintile_boundaries(is_rows, ind)

# ── OOS validation of top single conditions ───────────────────────
print("\n" + "=" * 80)
print("OUT-OF-SAMPLE VALIDATION — TOP SINGLE CONDITIONS")
print("=" * 80)

h_oos = 5  # primary horizon
print(f"\n  {'#':>3s} {'Indicator':>20s} {'Q':>4s} {'IS_Avg%':>9s} {'IS_WR%':>7s} {'IS_t':>8s} "
      f"{'OOS_Avg%':>9s} {'OOS_WR%':>8s} {'OOS_t':>8s} {'OOS_p':>10s} {'OOS_N':>6s} {'Surv?':>6s}")
print("  " + "-" * 120)

surviving_conditions = []

for rank, cond in enumerate(top_conditions, 1):
    ind = cond["indicator"]
    q_target = int(cond["quintile"][1]) - 1  # Q1 -> 0, Q2 -> 1, etc.
    bounds = is_boundaries[ind]

    # Gather OOS bars in the same quintile
    oos_rets = []
    for r in oos_rows:
        q_assigned = assign_quintile(r[0][ind], bounds)
        if q_assigned == q_target and r[1][h_oos] is not None:
            oos_rets.append(r[1][h_oos])

    if len(oos_rets) < 10:
        oos_avg = 0.0
        oos_wr = 0.0
        oos_t_val = 0.0
        oos_p_val = 1.0
    else:
        oos_avg = mean(oos_rets)
        oos_wr = sum(1 for r in oos_rets if r > 0) / len(oos_rets) * 100.0
        oos_t_val, oos_p_val = t_stat(oos_rets)

    survives = "YES" if (oos_avg > 0 and oos_p_val < 0.10) else "no"
    if survives == "YES":
        surviving_conditions.append({
            "indicator": ind,
            "quintile": cond["quintile"],
            "q_idx": q_target,
            "is_avg": cond["avg_ret"],
            "is_wr": cond["win_rate"],
            "is_t": cond["t_stat"],
            "oos_avg": round(oos_avg, 5),
            "oos_wr": round(oos_wr, 2),
            "oos_t": oos_t_val,
            "oos_p": oos_p_val,
            "oos_n": len(oos_rets),
        })

    print(f"  {rank:3d} {ind:>20s} {cond['quintile']:>4s} {cond['avg_ret']:>9.5f} {cond['win_rate']:>6.2f}% "
          f"{cond['t_stat']:>8.3f} {oos_avg:>9.5f} {oos_wr:>7.2f}% {oos_t_val:>8.3f} "
          f"{oos_p_val:>10.6f} {len(oos_rets):>6d} {survives:>6s}")

print(f"\n  {len(surviving_conditions)} / {len(top_conditions)} conditions survived OOS validation.")

# ── 2-indicator combo analysis ────────────────────────────────────
print("\n\n" + "=" * 80)
print("2-INDICATOR COMBINATIONS — IS + OOS")
print("=" * 80)

# Use top significant conditions (both long and short friendly)
# Pick top-15 unique indicator+quintile combos (any horizon, highest t-stat)
combo_pool_seen = set()
combo_pool = []
for c in sig_long:
    key = (c["indicator"], c["quintile"])
    if key not in combo_pool_seen:
        combo_pool_seen.add(key)
        combo_pool.append({
            "indicator": c["indicator"],
            "quintile": c["quintile"],
            "q_idx": int(c["quintile"][1]) - 1,
        })
    if len(combo_pool) >= 15:
        break

print(f"\n  Testing {len(list(combinations(combo_pool, 2)))} 2-indicator combos from top-{len(combo_pool)} conditions ...")

combo_results = []

for cA, cB in combinations(combo_pool, 2):
    if cA["indicator"] == cB["indicator"]:
        continue  # skip same-indicator combos

    indA, qA = cA["indicator"], cA["q_idx"]
    indB, qB = cB["indicator"], cB["q_idx"]
    boundsA = is_boundaries[indA]
    boundsB = is_boundaries[indB]

    # IS
    is_rets = []
    for r in is_rows:
        qAssA = assign_quintile(r[0][indA], boundsA)
        qAssB = assign_quintile(r[0][indB], boundsB)
        if qAssA == qA and qAssB == qB and r[1][h_oos] is not None:
            is_rets.append(r[1][h_oos])

    if len(is_rets) < 20:
        continue

    is_avg = mean(is_rets)
    is_wr = sum(1 for r in is_rets if r > 0) / len(is_rets) * 100.0
    is_t, is_p = t_stat(is_rets)

    if is_p > 0.05 or is_avg <= 0:
        continue

    # OOS
    oos_rets = []
    for r in oos_rows:
        qAssA = assign_quintile(r[0][indA], boundsA)
        qAssB = assign_quintile(r[0][indB], boundsB)
        if qAssA == qA and qAssB == qB and r[1][h_oos] is not None:
            oos_rets.append(r[1][h_oos])

    if len(oos_rets) < 10:
        oos_avg, oos_wr, oos_t_val, oos_p_val = 0.0, 0.0, 0.0, 1.0
    else:
        oos_avg = mean(oos_rets)
        oos_wr = sum(1 for r in oos_rets if r > 0) / len(oos_rets) * 100.0
        oos_t_val, oos_p_val = t_stat(oos_rets)

    combo_results.append({
        "condA": f"{cA['indicator']} {cA['quintile']}",
        "condB": f"{cB['indicator']} {cB['quintile']}",
        "is_n": len(is_rets),
        "is_avg": round(is_avg, 5),
        "is_wr": round(is_wr, 2),
        "is_t": is_t,
        "is_p": is_p,
        "oos_n": len(oos_rets),
        "oos_avg": round(oos_avg, 5),
        "oos_wr": round(oos_wr, 2),
        "oos_t": oos_t_val,
        "oos_p": oos_p_val,
    })

# Sort by OOS t-stat
combo_results.sort(key=lambda c: c["oos_t"], reverse=True)

print(f"\n  {len(combo_results)} combos passed IS filter (p<0.05, positive return).")
print(f"\n  {'#':>3s} {'Condition A':>30s} {'Condition B':>30s} "
      f"{'IS_N':>6s} {'IS_Avg%':>9s} {'IS_WR%':>7s} {'IS_t':>7s} "
      f"{'OOS_N':>6s} {'OOS_Avg%':>9s} {'OOS_WR%':>8s} {'OOS_t':>7s} {'OOS_p':>10s}")
print("  " + "-" * 160)

for rank, c in enumerate(combo_results[:TOP_COMBOS], 1):
    surv = " ***" if c["oos_avg"] > 0 and c["oos_p"] < 0.10 else ""
    print(f"  {rank:3d} {c['condA']:>30s} {c['condB']:>30s} "
          f"{c['is_n']:>6d} {c['is_avg']:>9.5f} {c['is_wr']:>6.2f}% {c['is_t']:>7.3f} "
          f"{c['oos_n']:>6d} {c['oos_avg']:>9.5f} {c['oos_wr']:>7.2f}% {c['oos_t']:>7.3f} "
          f"{c['oos_p']:>10.6f}{surv}")

# ── also show best combos sorted by OOS avg return ───────────────
combo_by_ret = sorted(combo_results, key=lambda c: c["oos_avg"], reverse=True)
print(f"\n  TOP COMBOS by OOS Average Return:")
print(f"  {'#':>3s} {'Condition A':>30s} {'Condition B':>30s} "
      f"{'IS_N':>6s} {'IS_Avg%':>9s} {'OOS_N':>6s} {'OOS_Avg%':>9s} {'OOS_WR%':>8s} {'OOS_t':>7s} {'OOS_p':>10s}")
print("  " + "-" * 140)

for rank, c in enumerate(combo_by_ret[:TOP_COMBOS], 1):
    surv = " ***" if c["oos_avg"] > 0 and c["oos_p"] < 0.10 else ""
    print(f"  {rank:3d} {c['condA']:>30s} {c['condB']:>30s} "
          f"{c['is_n']:>6d} {c['is_avg']:>9.5f} {c['oos_n']:>6d} {c['oos_avg']:>9.5f} "
          f"{c['oos_wr']:>7.2f}% {c['oos_t']:>7.3f} {c['oos_p']:>10.6f}{surv}")

# ── multi-horizon summary of surviving singles ────────────────────
print("\n\n" + "=" * 80)
print("MULTI-HORIZON OOS SUMMARY OF SURVIVING SINGLE CONDITIONS")
print("=" * 80)

if surviving_conditions:
    for sc in surviving_conditions:
        ind = sc["indicator"]
        q_target = sc["q_idx"]
        bounds = is_boundaries[ind]
        print(f"\n  {ind} {sc['quintile']}:")
        print(f"  {'Horizon':>8s} {'OOS_Avg%':>10s} {'OOS_WR%':>9s} {'t-stat':>8s} {'p-val':>10s} {'N':>6s}")
        for hh in HORIZONS:
            oos_rets_h = []
            for r in oos_rows:
                q_assigned = assign_quintile(r[0][ind], bounds)
                if q_assigned == q_target and r[1][hh] is not None:
                    oos_rets_h.append(r[1][hh])
            if len(oos_rets_h) < 10:
                continue
            avg_h = mean(oos_rets_h)
            wr_h = sum(1 for r in oos_rets_h if r > 0) / len(oos_rets_h) * 100.0
            t_h, p_h = t_stat(oos_rets_h)
            sig_mark = " *" if p_h < 0.05 else ""
            print(f"  {hh:>8d} {avg_h:>10.5f} {wr_h:>8.2f}% {t_h:>8.3f} {p_h:>10.6f} {len(oos_rets_h):>6d}{sig_mark}")
else:
    print("\n  No single conditions survived OOS validation at p < 0.10.")

# ── final summary ─────────────────────────────────────────────────
elapsed = time.time() - t0
print("\n\n" + "=" * 80)
print(f"ANALYSIS COMPLETE — {elapsed:.1f}s total")
print("=" * 80)
print(f"  Bars analyzed     : {len(bars):,}")
print(f"  IS sample         : {len(is_rows):,} rows")
print(f"  OOS sample        : {len(oos_rows):,} rows")
print(f"  Indicators tested : {len(indicator_names)}")
print(f"  Total conditions  : {len(all_conditions):,}  (indicator x quintile x horizon)")
print(f"  Significant (p<.05): {len(sig):,}")
print(f"  OOS survivors     : {len(surviving_conditions)}")
print(f"  2-combo IS passes : {len(combo_results)}")

# Count OOS-validated combos
oos_combo_survivors = [c for c in combo_results if c["oos_avg"] > 0 and c["oos_p"] < 0.10]
print(f"  2-combo OOS surv. : {len(oos_combo_survivors)}")
print()
