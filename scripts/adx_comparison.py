#!/usr/bin/env python3
"""Compare Python ADX & VWAP band values against TradingView trade signals.

Loads TV trades CSV, extracts PB entry signals, matches to Python's 5m bars,
computes indicators via IndicatorHub, and prints side-by-side comparison
tables showing divergence magnitudes.

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python scripts/adx_comparison.py
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# ── Project imports ──────────────────────────────────────────────────────
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.indicators.hub import IndicatorHub
from stockdownloader.core.models.price import IntradayPriceData

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
TV_CSV = Path.home() / "Downloads" / "VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv"
BARS_CSV = BASE / "data" / "SPY" / "5m_bars.csv"

# ── TV timezone offset (TV chart is Pacific; ET = TV + 3h) ──────────────
TV_TO_ET_HOURS = 3


# =========================================================================
# Parse TV trades
# =========================================================================

def parse_tv_trades(csv_path: Path) -> list[dict]:
    """Parse TV CSV and extract PB entry trades with signal metadata."""
    trades: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            signal = row.get("Signal", "")
            if not signal.startswith("PB|"):
                continue
            if "Entry" not in row.get("Type", ""):
                continue

            # Parse signal fields
            fields: dict[str, str] = {}
            for part in signal.split("|"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    fields[k] = v

            # Extract ADX and VD (VWAP delta)
            adx_tv = fields.get("ADX")
            vd_tv = fields.get("VD")
            if adx_tv is None:
                continue

            # Parse TV datetime and convert to ET
            tv_dt_str = row["Date and time"].strip()
            tv_dt = datetime.strptime(tv_dt_str, "%Y-%m-%d %H:%M")
            et_dt = tv_dt + timedelta(hours=TV_TO_ET_HOURS)

            trades.append({
                "trade_num": row["Trade #"],
                "tv_time": tv_dt_str,
                "et_time": et_dt,
                "direction": "S" if "Entry short" in row["Type"] else "L",
                "price": float(row["Price USD"]),
                "adx_tv": float(adx_tv),
                "vd_tv": float(vd_tv) if vd_tv else None,
                "signal": signal,
            })

    return trades


# =========================================================================
# Build datetime index for 5m bars
# =========================================================================

def build_bar_index(
    data: list[IntradayPriceData],
) -> dict[str, int]:
    """Map 'YYYY-MM-DD HH:MM' -> bar index for fast lookup."""
    idx: dict[str, int] = {}
    for i, bar in enumerate(data):
        # bar.date is like '2025-02-20 09:30:00-05:00'
        key = bar.date[:16]  # 'YYYY-MM-DD HH:MM'
        idx[key] = i
    return idx


# =========================================================================
# Main comparison
# =========================================================================

def main() -> None:
    print("=" * 90)
    print("ADX & VWAP Comparison: Python vs TradingView")
    print("=" * 90)

    # ── Load data ────────────────────────────────────────────────────────
    print(f"\nLoading 5m bars from {BARS_CSV} ...")
    data = IntradayCsvLoader.load_from_file(str(BARS_CSV))
    if not data:
        print("ERROR: No data loaded")
        sys.exit(1)
    print(f"  Loaded {len(data):,} bars, "
          f"{data[0].date[:10]} to {data[-1].date[:10]}")

    bar_index = build_bar_index(data)
    print(f"  Built index with {len(bar_index):,} entries")

    print(f"\nLoading TV trades from {TV_CSV} ...")
    trades = parse_tv_trades(TV_CSV)
    print(f"  Found {len(trades)} PB entry trades")

    # ── Compute indicators ───────────────────────────────────────────────
    hub = IndicatorHub()

    adx_results: list[dict] = []
    vd_results: list[dict] = []
    matched = 0
    missed = 0

    for t in trades:
        et_dt: datetime = t["et_time"]

        # The TV signal fires at the close of the bar *before* the entry bar.
        # TV reports the entry bar time. The signal was computed on the
        # previous bar. Try matching the signal bar (entry - 5min) first,
        # then fall back to the entry bar itself.
        signal_dt = et_dt - timedelta(minutes=5)
        signal_key = signal_dt.strftime("%Y-%m-%d %H:%M")
        entry_key = et_dt.strftime("%Y-%m-%d %H:%M")

        idx = bar_index.get(signal_key)
        matched_key = signal_key
        if idx is None:
            idx = bar_index.get(entry_key)
            matched_key = entry_key
        if idx is None:
            missed += 1
            continue

        matched += 1

        # Compute ADX via streaming accumulator (same as PullbackStrategy)
        adx_result = hub.adx(data, idx, period=14)
        py_adx = float(adx_result.adx)
        tv_adx = t["adx_tv"]
        delta_adx = py_adx - tv_adx

        adx_results.append({
            "trade": t["trade_num"],
            "dir": t["direction"],
            "tv_time": t["tv_time"],
            "matched": matched_key,
            "bar_idx": idx,
            "tv_adx": tv_adx,
            "py_adx": py_adx,
            "delta": delta_adx,
            "pct": (delta_adx / tv_adx * 100) if tv_adx != 0 else 0,
        })

        # VWAP bands comparison
        if t["vd_tv"] is not None:
            vwap_bands = hub.session_vwap_bands(data, idx)
            py_std = float(vwap_bands.std_dev)
            py_vwap = float(vwap_bands.vwap)

            # Raw VWAP slope
            py_slope_raw = float(hub.vwap_slope(data, idx, lookback=5))

            # ATR for normalization
            py_atr = float(hub.atr(data, idx, period=14))

            # Candidate normalizations for VD:
            # 1) abs(slope) / ATR
            vd_by_atr = abs(py_slope_raw) / py_atr if py_atr > 0 else 0.0
            # 2) abs(slope) / std_dev
            vd_by_std = abs(py_slope_raw) / py_std if py_std > 0 else 0.0
            # 3) Raw abs(slope)
            vd_abs = abs(py_slope_raw)

            vd_results.append({
                "trade": t["trade_num"],
                "dir": t["direction"],
                "tv_time": t["tv_time"],
                "tv_vd": t["vd_tv"],
                "py_slope": py_slope_raw,
                "vd_by_atr": vd_by_atr,
                "vd_by_std": vd_by_std,
                "vd_abs": vd_abs,
                "py_vwap": py_vwap,
                "py_std": py_std,
                "py_atr": py_atr,
            })

    # ── Print ADX table ──────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"ADX COMPARISON (matched {matched}/{matched + missed} trades)")
    print(f"{'=' * 90}")
    print(f"{'Trade':>6} {'Dir':>3} {'TV Time':>18} {'Matched Bar':>18} "
          f"{'TV ADX':>8} {'Py ADX':>8} {'Delta':>8} {'Pct%':>7}")
    print("-" * 90)

    abs_deltas = []
    for r in adx_results:
        print(f"{r['trade']:>6} {r['dir']:>3} {r['tv_time']:>18} {r['matched']:>18} "
              f"{r['tv_adx']:>8.1f} {r['py_adx']:>8.1f} {r['delta']:>+8.1f} {r['pct']:>+7.1f}")
        abs_deltas.append(abs(r["delta"]))

    if abs_deltas:
        mean_delta = sum(abs_deltas) / len(abs_deltas)
        max_delta = max(abs_deltas)
        print("-" * 90)
        print(f"{'':>56} Mean |delta|: {mean_delta:>6.2f}")
        print(f"{'':>56}  Max |delta|: {max_delta:>6.2f}")
        print(f"{'':>56}    Std  dev: {(sum((d - mean_delta)**2 for d in abs_deltas) / len(abs_deltas))**0.5:>6.2f}")

    # ── Print VD / VWAP table ────────────────────────────────────────────
    if vd_results:
        print(f"\n{'=' * 110}")
        print(f"VWAP DELTA (VD) COMPARISON -- Testing normalizations")
        print(f"{'=' * 110}")
        print(f"{'Trade':>6} {'Dir':>3} {'TV Time':>18} "
              f"{'TV VD':>7} {'|slope|/ATR':>11} {'|slope|/std':>11} {'|slope|':>8} "
              f"{'Py ATR':>7} {'Py std':>8}")
        print("-" * 110)

        deltas_atr = []
        deltas_std = []
        for r in vd_results:
            d_atr = abs(r["vd_by_atr"] - r["tv_vd"])
            d_std = abs(r["vd_by_std"] - r["tv_vd"])
            deltas_atr.append(d_atr)
            deltas_std.append(d_std)
            print(f"{r['trade']:>6} {r['dir']:>3} {r['tv_time']:>18} "
                  f"{r['tv_vd']:>7.2f} {r['vd_by_atr']:>11.3f} {r['vd_by_std']:>11.3f} "
                  f"{r['vd_abs']:>8.3f} {r['py_atr']:>7.3f} {r['py_std']:>8.4f}")

        if deltas_atr:
            print("-" * 110)
            m_atr = sum(deltas_atr) / len(deltas_atr)
            m_std = sum(deltas_std) / len(deltas_std)
            print(f"  Mean |TV - |slope|/ATR| = {m_atr:.3f}   "
                  f"Mean |TV - |slope|/std| = {m_std:.3f}")
            print(f"   Max |TV - |slope|/ATR| = {max(deltas_atr):.3f}   "
                  f" Max |TV - |slope|/std| = {max(deltas_std):.3f}")
            # Correlation check
            corr_atr = sum(r["vd_by_atr"] * r["tv_vd"] for r in vd_results)
            corr_std = sum(r["vd_by_std"] * r["tv_vd"] for r in vd_results)
            print(f"  Note: |slope|/ATR has lower error when the normalization matches TV.")

    # ── Smoothing analysis ───────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("ADX SMOOTHING METHOD ANALYSIS")
    print(f"{'=' * 90}")
    print("""
Python StreamingADX implementation:
  - Uses Wilder's smoothing for +DM, -DM, TR:
      smooth_val = smooth_val - (smooth_val / period) + current_val
  - ADX = SMA of last 'period' DX values (simple average of DX buffer)
  - This differs from PineScript's ta.dmi() which uses Wilder's smoothing
    for the ADX line itself:
      ADX = prev_ADX * (period-1)/period + DX/period

PineScript ta.dmi() / ta.adx():
  - +DI/-DI: Wilder's smoothing (same as Python)
  - ADX line: Wilder's smoothing of DX (NOT simple average)
  - Formula: ADX[i] = (ADX[i-1] * (period-1) + DX[i]) / period

Key difference:
  Python uses SMA of DX for ADX, PineScript uses Wilder-smoothed DX.
  Wilder's smoothing gives more weight to recent values (like an EMA),
  while SMA weights all DX values equally.
""")

    # ── Directional analysis ─────────────────────────────────────────────
    if adx_results:
        positive = sum(1 for r in adx_results if r["delta"] > 0)
        negative = sum(1 for r in adx_results if r["delta"] < 0)
        zero = sum(1 for r in adx_results if r["delta"] == 0)
        print(f"Bias: Py > TV in {positive}/{len(adx_results)} cases, "
              f"Py < TV in {negative}/{len(adx_results)} cases")


if __name__ == "__main__":
    main()
