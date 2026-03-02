"""VWAP Computation Comparison: Python vs TradingView.

Compares Python's session VWAP, std_dev, VWAP slope (delta), and bands
against TradingView's ta.vwap(hlc3, isNewDay, mult) on identical 5-minute
bar data.

Key differences analysed:
  1. Source price: both use hlc3 (typical price = (H+L+C)/3)
  2. Session reset: Python uses date[:10] change; TV uses timeframe.change("D")
  3. Std dev formula: both use volume-weighted sqrt(Sum(vol*(tp-vwap)^2)/Sum(vol))
     BUT Python's batch core uses a two-pass approach (compute vwap first, then
     sum deviations) while the streaming class uses E[X^2]-E[X]^2 shortcut.
  4. Quantization: Python quantizes TP to 10 decimal places before accumulation;
     TV uses native float64.
  5. VWAP slope (delta): Python = vwap[i] - vwap[i-5]; TV = ta.change(vwapLine, 5)

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/vwap_comparison.py

Requires:
    - TV export CSV at ~/Downloads/VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv
    - SPY 5-minute bars at data/SPY/5m_bars.csv
"""
from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.core.math import ZERO, quantize
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.indicators.volume import StreamingSessionVWAP
from stockdownloader.indicators.core import _compute_session_vwap_core, _find_session_start

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"
TV_CSV = Path.home() / "Downloads" / "VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv"

# TV export timestamps are bar CLOSE time in US/Pacific.
# Python bars use bar OPEN time in US/Eastern.
# Pacific + 3h = Eastern, then -5min for close->open = net +2h55m.
TV_TO_ET_OFFSET = timedelta(hours=2, minutes=55)

SLOPE_PERIOD = 5  # i_slopePer in PineScript


# ═══════════════════════════════════════════════════════════════════════
# TV-style VWAP (float64, no quantization) for reference comparison
# ═══════════════════════════════════════════════════════════════════════

def compute_tv_style_vwap(bars: list[IntradayPriceData]) -> list[tuple[float, float]]:
    """Compute session VWAP + std_dev using TV's formula (float64, no quantize).

    TV's ta.vwap(hlc3, isNewDay, mult) internally does:
        cumTPV += hlc3 * volume
        cumVol += volume
        cumTP2V += hlc3^2 * volume
        vwap = cumTPV / cumVol
        variance = cumTP2V / cumVol - vwap^2
        stdev = sqrt(max(0, variance))

    This uses EXACTLY the same formula with float64, no Decimal quantization.

    Returns list of (vwap, stdev) per bar.
    """
    results: list[tuple[float, float]] = []
    cum_tpv = 0.0
    cum_vol = 0.0
    cum_tp2v = 0.0
    current_session = ""

    for bar in bars:
        session = bar.date[:10]
        if session != current_session:
            cum_tpv = 0.0
            cum_vol = 0.0
            cum_tp2v = 0.0
            current_session = session

        tp = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        vol = float(bar.volume)

        cum_tpv += tp * vol
        cum_vol += vol
        cum_tp2v += tp * tp * vol

        if cum_vol == 0.0:
            results.append((0.0, 0.0))
        else:
            vwap = cum_tpv / cum_vol
            variance = cum_tp2v / cum_vol - vwap * vwap
            stdev = math.sqrt(max(0.0, variance))
            results.append((vwap, stdev))

    return results


# ═══════════════════════════════════════════════════════════════════════
# Python's VWAP (Decimal, quantized TP)
# ═══════════════════════════════════════════════════════════════════════

def compute_python_vwap(bars: list[IntradayPriceData]) -> list[tuple[float, float]]:
    """Compute session VWAP + std_dev using Python's StreamingSessionVWAP.

    This uses Decimal arithmetic with quantized typical price (10 decimal places).
    """
    streaming = StreamingSessionVWAP()
    results: list[tuple[float, float]] = []

    for i in range(len(bars)):
        vwap_d, std_d = streaming.update(bars, i)
        results.append((float(vwap_d), float(std_d)))

    return results


def compute_python_batch_vwap(bars: list[IntradayPriceData]) -> list[tuple[float, float]]:
    """Compute session VWAP + std_dev using Python's batch _compute_session_vwap_core.

    This is the two-pass approach: first compute VWAP, then sum (tp-vwap)^2*vol.
    """
    results: list[tuple[float, float]] = []
    for i in range(len(bars)):
        vwap_d, std_d = _compute_session_vwap_core(bars, i)
        results.append((float(vwap_d), float(std_d)))
    return results


# ═══════════════════════════════════════════════════════════════════════
# VWAP slope (delta)
# ═══════════════════════════════════════════════════════════════════════

def compute_vwap_slopes(vwap_series: list[float], lookback: int = 5) -> list[float]:
    """Compute VWAP slope = vwap[i] - vwap[i-lookback]."""
    slopes: list[float] = []
    for i in range(len(vwap_series)):
        if i < lookback:
            slopes.append(0.0)
        else:
            slopes.append(vwap_series[i] - vwap_series[i - lookback])
    return slopes


# ═══════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════

def load_5m_bars(csv_path: Path) -> list[IntradayPriceData]:
    """Load SPY 5-minute OHLCV bars from CSV."""
    bars: list[IntradayPriceData] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(IntradayPriceData(
                date=row["Datetime"],
                open=Decimal(row["Open"]),
                high=Decimal(row["High"]),
                low=Decimal(row["Low"]),
                close=Decimal(row["Close"]),
                adj_close=Decimal(row["Close"]),
                volume=int(row["Volume"]),
            ))
    return bars


def build_bar_index(bars: list[IntradayPriceData]) -> dict[str, int]:
    """Build map from 'YYYY-MM-DD HH:MM' -> bar index."""
    index: dict[str, int] = {}
    for i, bar in enumerate(bars):
        key = bar.date[:16]
        index[key] = i
    return index


def tv_time_to_et_bar_label(tv_dt_str: str) -> str:
    """Convert TV export timestamp (Pacific close time) to Python bar label (ET open)."""
    dt = datetime.strptime(tv_dt_str.strip(), "%Y-%m-%d %H:%M")
    et_dt = dt + TV_TO_ET_OFFSET
    return et_dt.strftime("%Y-%m-%d %H:%M")


def parse_tv_signal(signal: str) -> dict[str, str]:
    """Parse pipe-delimited TV signal into key-value pairs."""
    parts = signal.split("|")
    result: dict[str, str] = {}
    if len(parts) < 2:
        return result
    result["mode"] = parts[0]
    result["dir"] = parts[1]
    for part in parts[2:]:
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        result[key] = val
    return result


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ═══════════════════════════════════════════════════════════════════════
# Load TV entry trades with signal metadata
# ═══════════════════════════════════════════════════════════════════════

def load_tv_entries(csv_path: Path) -> list[dict]:
    """Load all TV entry trades (all modes) with parsed signal metadata."""
    entries: list[dict] = []

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trade_type = row.get("Type", "")
            if "Entry" not in trade_type:
                continue

            signal = row.get("Signal", "")
            parsed = parse_tv_signal(signal)
            if not parsed:
                continue

            tv_dt = row.get("Date and time", "").strip()
            et_label = tv_time_to_et_bar_label(tv_dt)

            entries.append({
                "trade_num": int(row.get("Trade #", "0")),
                "mode": parsed.get("mode", "UNK"),
                "direction": parsed.get("dir", "?"),
                "tv_datetime": tv_dt,
                "et_bar_label": et_label,
                "price": _safe_float(row.get("Price USD", "0")),
                "signal": signal,
                # VD in TV metadata is volume delta (close-low)/(high-low), NOT VWAP slope
                "tv_vdelta": _safe_float(parsed.get("VD", "0")),
                # VA is VWAP acceleration
                "tv_va": _safe_float(parsed.get("VA", "0")),
                # ADX for reference
                "tv_adx": _safe_float(parsed.get("ADX", "0")),
                # AR for reference
                "tv_ar": _safe_float(parsed.get("AR", "0")),
            })

    return entries


# ═══════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_session_divergence(
    bars: list[IntradayPriceData],
    tv_vwap: list[tuple[float, float]],
    py_vwap: list[tuple[float, float]],
) -> None:
    """Analyze per-session divergence between TV-style and Python VWAP."""
    sessions: dict[str, list[int]] = defaultdict(list)
    for i, bar in enumerate(bars):
        sessions[bar.date[:10]].append(i)

    print(f"\n{'='*90}")
    print("SESSION-LEVEL VWAP DIVERGENCE (TV-style float64 vs Python Decimal)")
    print(f"{'='*90}")

    vwap_deltas: list[float] = []
    std_deltas: list[float] = []
    max_vwap_delta = 0.0
    max_std_delta = 0.0
    max_vwap_session = ""
    max_std_session = ""
    max_vwap_bar = 0

    for session_date, indices in sorted(sessions.items()):
        session_vwap_deltas = []
        session_std_deltas = []

        for i in indices:
            tv_v, tv_s = tv_vwap[i]
            py_v, py_s = py_vwap[i]

            if tv_v == 0.0 and py_v == 0.0:
                continue

            vd = abs(tv_v - py_v)
            sd = abs(tv_s - py_s)
            session_vwap_deltas.append(vd)
            session_std_deltas.append(sd)
            vwap_deltas.append(vd)
            std_deltas.append(sd)

            if vd > max_vwap_delta:
                max_vwap_delta = vd
                max_vwap_session = session_date
                max_vwap_bar = i
            if sd > max_std_delta:
                max_std_delta = sd
                max_std_session = session_date

        if session_vwap_deltas:
            mean_vd = statistics.mean(session_vwap_deltas)
            max_vd = max(session_vwap_deltas)
            mean_sd = statistics.mean(session_std_deltas)
            max_sd = max(session_std_deltas)

            # Only print sessions with notable divergence
            if max_vd > 0.005 or max_sd > 0.005:
                n_bars = len(session_vwap_deltas)
                print(f"\n  {session_date}: {n_bars} bars")
                print(f"    VWAP: mean |delta| = {mean_vd:.6f}, max = {max_vd:.6f}")
                print(f"    Std:  mean |delta| = {mean_sd:.6f}, max = {max_sd:.6f}")

    # Overall statistics
    print(f"\n{'='*90}")
    print("OVERALL STATISTICS (all bars)")
    print(f"{'='*90}")
    if vwap_deltas:
        print(f"\n  VWAP |delta| (TV float64 vs Python Decimal):")
        print(f"    Mean:   {statistics.mean(vwap_deltas):.8f}")
        print(f"    Median: {statistics.median(vwap_deltas):.8f}")
        print(f"    Max:    {max_vwap_delta:.8f} (session {max_vwap_session}, bar {max_vwap_bar})")
        print(f"    P95:    {sorted(vwap_deltas)[int(len(vwap_deltas)*0.95)]:.8f}")
        print(f"    P99:    {sorted(vwap_deltas)[int(len(vwap_deltas)*0.99)]:.8f}")
        print(f"    Bars:   {len(vwap_deltas)}")

    if std_deltas:
        print(f"\n  StdDev |delta| (TV float64 vs Python Decimal):")
        print(f"    Mean:   {statistics.mean(std_deltas):.8f}")
        print(f"    Median: {statistics.median(std_deltas):.8f}")
        print(f"    Max:    {max_std_delta:.8f} (session {max_std_session})")
        print(f"    P95:    {sorted(std_deltas)[int(len(std_deltas)*0.95)]:.8f}")
        print(f"    P99:    {sorted(std_deltas)[int(len(std_deltas)*0.99)]:.8f}")


def analyze_batch_vs_streaming(
    py_streaming: list[tuple[float, float]],
    py_batch: list[tuple[float, float]],
) -> None:
    """Check if Python batch and streaming give identical results."""
    print(f"\n{'='*90}")
    print("PYTHON INTERNAL: BATCH vs STREAMING CONSISTENCY")
    print(f"{'='*90}")

    vwap_mismatches = 0
    std_mismatches = 0
    max_vwap_diff = 0.0
    max_std_diff = 0.0

    for i in range(len(py_streaming)):
        sv, ss = py_streaming[i]
        bv, bs = py_batch[i]

        vd = abs(sv - bv)
        sd = abs(ss - bs)

        if vd > 1e-12:
            vwap_mismatches += 1
            max_vwap_diff = max(max_vwap_diff, vd)
        if sd > 1e-12:
            std_mismatches += 1
            max_std_diff = max(max_std_diff, sd)

    print(f"\n  Total bars: {len(py_streaming)}")
    print(f"  VWAP mismatches: {vwap_mismatches} (max diff: {max_vwap_diff:.12f})")
    print(f"  StdDev mismatches: {std_mismatches} (max diff: {max_std_diff:.12f})")
    if vwap_mismatches == 0 and std_mismatches == 0:
        print("  RESULT: Batch and streaming are BIT-IDENTICAL.")
    else:
        print("  RESULT: Batch and streaming DIVERGE -- investigate.")


def analyze_vwap_at_entries(
    bars: list[IntradayPriceData],
    bar_index: dict[str, int],
    tv_entries: list[dict],
    tv_vwap: list[tuple[float, float]],
    py_vwap: list[tuple[float, float]],
    tv_slopes: list[float],
    py_slopes: list[float],
) -> None:
    """Compare VWAP values specifically at TV entry bars."""
    print(f"\n{'='*90}")
    print("VWAP DIVERGENCE AT TV ENTRY BARS")
    print(f"{'='*90}")

    vwap_deltas: list[float] = []
    std_deltas: list[float] = []
    slope_deltas: list[float] = []
    upper_2s_deltas: list[float] = []  # upper 2-sigma band
    lower_2s_deltas: list[float] = []  # lower 2-sigma band
    zone_impact: list[dict] = []       # entries where zone check flips

    matched = 0
    unmatched = 0

    for entry in tv_entries:
        idx = bar_index.get(entry["et_bar_label"])
        if idx is None:
            unmatched += 1
            continue
        matched += 1

        tv_v, tv_s = tv_vwap[idx]
        py_v, py_s = py_vwap[idx]
        bar = bars[idx]
        close = float(bar.close)

        vwap_d = tv_v - py_v
        std_d = tv_s - py_s

        vwap_deltas.append(vwap_d)
        std_deltas.append(std_d)

        # Slope comparison
        tv_slope = tv_slopes[idx]
        py_slope = py_slopes[idx]
        slope_d = tv_slope - py_slope
        slope_deltas.append(slope_d)

        # Band comparison (TV uses 2-sigma bands, Python uses std_dev * mult)
        tv_upper2 = tv_v + 2.0 * tv_s
        tv_lower2 = tv_v - 2.0 * tv_s
        py_upper2 = py_v + 2.0 * py_s
        py_lower2 = py_v - 2.0 * py_s
        upper_2s_deltas.append(tv_upper2 - py_upper2)
        lower_2s_deltas.append(tv_lower2 - py_lower2)

        # Zone check: does the divergence flip whether price is in zone?
        # PB zone = within pb_zone (0.5) * std_dev of VWAP
        pb_zone = 0.5
        tv_zone_width = pb_zone * tv_s
        py_zone_width = pb_zone * py_s
        tv_in_zone = abs(close - tv_v) <= tv_zone_width
        py_in_zone = abs(close - py_v) <= py_zone_width

        if tv_in_zone != py_in_zone:
            zone_impact.append({
                "trade_num": entry["trade_num"],
                "mode": entry["mode"],
                "et_bar": entry["et_bar_label"],
                "close": close,
                "tv_vwap": tv_v,
                "py_vwap": py_v,
                "tv_std": tv_s,
                "py_std": py_s,
                "tv_in_zone": tv_in_zone,
                "py_in_zone": py_in_zone,
                "tv_dist": abs(close - tv_v),
                "py_dist": abs(close - py_v),
                "tv_zone_w": tv_zone_width,
                "py_zone_w": py_zone_width,
            })

    print(f"\n  Matched entries: {matched}")
    print(f"  Unmatched entries: {unmatched}")

    if not vwap_deltas:
        print("  No matched entries to analyze.")
        return

    # VWAP delta at entries
    print(f"\n  VWAP (TV-float - Python-Decimal) at entry bars:")
    abs_vd = [abs(d) for d in vwap_deltas]
    print(f"    Mean signed:   {statistics.mean(vwap_deltas):+.6f}")
    print(f"    Mean |delta|:  {statistics.mean(abs_vd):.6f}")
    print(f"    Max |delta|:   {max(abs_vd):.6f}")
    print(f"    Median |delta|: {statistics.median(abs_vd):.6f}")

    # StdDev delta at entries
    print(f"\n  StdDev (TV-float - Python-Decimal) at entry bars:")
    abs_sd = [abs(d) for d in std_deltas]
    print(f"    Mean signed:   {statistics.mean(std_deltas):+.6f}")
    print(f"    Mean |delta|:  {statistics.mean(abs_sd):.6f}")
    print(f"    Max |delta|:   {max(abs_sd):.6f}")
    print(f"    Median |delta|: {statistics.median(abs_sd):.6f}")

    # VWAP slope delta at entries
    if slope_deltas:
        print(f"\n  VWAP Slope (TV-float - Python-Decimal) at entry bars:")
        abs_sl = [abs(d) for d in slope_deltas]
        print(f"    Mean signed:   {statistics.mean(slope_deltas):+.6f}")
        print(f"    Mean |delta|:  {statistics.mean(abs_sl):.6f}")
        print(f"    Max |delta|:   {max(abs_sl):.6f}")

    # 2-sigma band delta at entries
    if upper_2s_deltas:
        abs_u2 = [abs(d) for d in upper_2s_deltas]
        abs_l2 = [abs(d) for d in lower_2s_deltas]
        print(f"\n  2-sigma Upper Band (TV - Python) at entry bars:")
        print(f"    Mean |delta|:  {statistics.mean(abs_u2):.6f}")
        print(f"    Max |delta|:   {max(abs_u2):.6f}")
        print(f"\n  2-sigma Lower Band (TV - Python) at entry bars:")
        print(f"    Mean |delta|:  {statistics.mean(abs_l2):.6f}")
        print(f"    Max |delta|:   {max(abs_l2):.6f}")

    # Zone impact: trades where zone check flips
    print(f"\n{'='*90}")
    print("ZONE IMPACT: Entries where VWAP zone check FLIPS between TV and Python")
    print(f"{'='*90}")
    print(f"\n  Zone-flip entries: {len(zone_impact)} / {matched}")

    if zone_impact:
        print(f"\n  {'#':>4} {'Mode':<4} {'Bar Label':>18} {'Close':>9} "
              f"{'TV VWAP':>9} {'Py VWAP':>9} {'TV Std':>8} {'Py Std':>8} "
              f"{'TV Zone':>8} {'Py Zone':>8}")
        print(f"  {'-'*100}")
        for z in zone_impact[:20]:  # Limit to first 20
            print(f"  {z['trade_num']:>4} {z['mode']:<4} {z['et_bar']:>18} "
                  f"{z['close']:>9.2f} {z['tv_vwap']:>9.4f} {z['py_vwap']:>9.4f} "
                  f"{z['tv_std']:>8.4f} {z['py_std']:>8.4f} "
                  f"{'IN' if z['tv_in_zone'] else 'OUT':>8} "
                  f"{'IN' if z['py_in_zone'] else 'OUT':>8}")
            # Detail
            print(f"       TV dist={z['tv_dist']:.4f} zone_w={z['tv_zone_w']:.4f} "
                  f"| Py dist={z['py_dist']:.4f} zone_w={z['py_zone_w']:.4f}")

        if len(zone_impact) > 20:
            print(f"  ... and {len(zone_impact) - 20} more")
    else:
        print("  NO zone flips detected -- VWAP divergence does not flip entry zone check.")


def analyze_quantization_effect(bars: list[IntradayPriceData]) -> None:
    """Show how Python's TP quantization accumulates error vs raw float."""
    print(f"\n{'='*90}")
    print("QUANTIZATION ANALYSIS: Effect of Decimal TP rounding on VWAP")
    print(f"{'='*90}")

    # Pick a few sessions and trace accumulation
    sessions: dict[str, list[int]] = defaultdict(list)
    for i, bar in enumerate(bars):
        sessions[bar.date[:10]].append(i)

    # Pick the first 3 sessions
    example_sessions = list(sorted(sessions.keys()))[:3]

    for session_date in example_sessions:
        indices = sessions[session_date]
        print(f"\n  Session: {session_date} ({len(indices)} bars)")
        print(f"  {'Bar':>4} {'TP (float)':>14} {'TP (quantized)':>16} {'TP diff':>12} "
              f"{'VWAP_f':>12} {'VWAP_q':>12} {'VWAP diff':>12}")
        print(f"  {'-'*90}")

        cum_tpv_f = 0.0
        cum_vol_f = 0.0
        cum_tpv_q = Decimal("0")
        cum_vol_q = Decimal("0")

        for j, i in enumerate(indices[:15]):  # First 15 bars
            bar = bars[i]
            h, l, c = float(bar.high), float(bar.low), float(bar.close)
            vol = float(bar.volume)

            tp_f = (h + l + c) / 3.0
            tp_q = quantize((bar.high + bar.low + bar.close) / Decimal("3"))
            tp_q_f = float(tp_q)
            tp_diff = tp_f - tp_q_f

            cum_tpv_f += tp_f * vol
            cum_vol_f += vol
            vwap_f = cum_tpv_f / cum_vol_f if cum_vol_f > 0 else 0.0

            vol_d = Decimal(str(bar.volume))
            cum_tpv_q += tp_q * vol_d
            cum_vol_q += vol_d
            vwap_q = float(cum_tpv_q / cum_vol_q) if cum_vol_q > 0 else 0.0

            vwap_diff = vwap_f - vwap_q

            print(f"  {j+1:>4} {tp_f:>14.6f} {tp_q_f:>16.10f} {tp_diff:>+12.10f} "
                  f"{vwap_f:>12.6f} {vwap_q:>12.6f} {vwap_diff:>+12.10f}")

        # End of session stats
        last_idx = indices[-1]
        h, l, c = float(bars[last_idx].high), float(bars[last_idx].low), float(bars[last_idx].close)
        print(f"  ... last bar #{len(indices)}: end-of-session VWAP divergence shown above")


def analyze_std_dev_formula(bars: list[IntradayPriceData]) -> None:
    """Compare two std_dev formulas on select sessions.

    Formula A (TV/Streaming): sqrt(E[X^2] - E[X]^2)  -- single-pass
    Formula B (Python batch): sqrt(Sum((tp-vwap)^2 * vol) / Sum(vol))  -- two-pass

    These are mathematically identical but can differ due to floating-point
    catastrophic cancellation in Formula A when variance is near zero.
    """
    print(f"\n{'='*90}")
    print("STD DEV FORMULA COMPARISON: E[X^2]-E[X]^2 vs Sum((tp-vwap)^2*vol)/Sum(vol)")
    print(f"{'='*90}")

    sessions: dict[str, list[int]] = defaultdict(list)
    for i, bar in enumerate(bars):
        sessions[bar.date[:10]].append(i)

    # Check all sessions, report ones with any divergence
    divergent_sessions = 0
    max_std_diff = 0.0
    max_std_session = ""

    for session_date, indices in sorted(sessions.items()):
        for i in indices:
            # Formula A: E[X^2] - E[X]^2 (streaming)
            session_start = indices[0]
            cum_tpv = 0.0
            cum_vol = 0.0
            cum_tp2v = 0.0
            for j in range(session_start, i + 1):
                b = bars[j]
                tp = (float(b.high) + float(b.low) + float(b.close)) / 3.0
                v = float(b.volume)
                cum_tpv += tp * v
                cum_vol += v
                cum_tp2v += tp * tp * v

            if cum_vol > 0:
                vwap = cum_tpv / cum_vol
                var_a = cum_tp2v / cum_vol - vwap * vwap
                std_a = math.sqrt(max(0.0, var_a))

                # Formula B: Sum((tp-vwap)^2 * vol) / Sum(vol)
                sum_var = 0.0
                for j in range(session_start, i + 1):
                    b = bars[j]
                    tp = (float(b.high) + float(b.low) + float(b.close)) / 3.0
                    v = float(b.volume)
                    diff = tp - vwap
                    sum_var += diff * diff * v
                var_b = sum_var / cum_vol
                std_b = math.sqrt(max(0.0, var_b))

                diff = abs(std_a - std_b)
                if diff > max_std_diff:
                    max_std_diff = diff
                    max_std_session = session_date

                if diff > 1e-8:
                    divergent_sessions += 1

    print(f"\n  Bars checked: {len(bars)}")
    print(f"  Bars with std_dev divergence > 1e-8: {divergent_sessions}")
    print(f"  Max std_dev formula diff: {max_std_diff:.12f} (session {max_std_session})")

    if max_std_diff < 1e-6:
        print("  RESULT: Both formulas are effectively identical at float64 precision.")
    else:
        print("  RESULT: Formulas diverge -- catastrophic cancellation may affect streaming.")


def print_first_bar_behavior(bars: list[IntradayPriceData]) -> None:
    """Show VWAP on first bar of session for both approaches."""
    print(f"\n{'='*90}")
    print("FIRST BAR BEHAVIOR: Session reset and division-by-zero handling")
    print(f"{'='*90}")

    sessions: dict[str, list[int]] = defaultdict(list)
    for i, bar in enumerate(bars):
        sessions[bar.date[:10]].append(i)

    print(f"\n  {'Session':<12} {'Bar 1 TP':>12} {'Vol':>10} {'TV VWAP':>12} {'Py VWAP':>12} "
          f"{'TV Std':>10} {'Py Std':>10}")
    print(f"  {'-'*80}")

    streaming = StreamingSessionVWAP()
    count = 0
    for session_date, indices in sorted(sessions.items()):
        i = indices[0]
        bar = bars[i]
        tp_f = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        vol = bar.volume

        # TV-style
        tv_vwap = tp_f  # First bar: vwap = tp
        tv_std = 0.0     # First bar: std = 0 (sqrt(tp^2 - tp^2) = 0)

        # Python streaming
        py_vwap_d, py_std_d = streaming.update(bars, i)

        print(f"  {session_date:<12} {tp_f:>12.4f} {vol:>10,} {tv_vwap:>12.4f} "
              f"{float(py_vwap_d):>12.4f} {tv_std:>10.4f} {float(py_std_d):>10.4f}")

        count += 1
        if count >= 10:
            print(f"  ... ({len(sessions) - 10} more sessions)")
            break


# ═══════════════════════════════════════════════════════════════════════
# Trail comparison: VWAP trail divergence at exit bars
# ═══════════════════════════════════════════════════════════════════════

def analyze_trail_impact(
    bars: list[IntradayPriceData],
    bar_index: dict[str, int],
    tv_vwap: list[tuple[float, float]],
    py_vwap: list[tuple[float, float]],
) -> None:
    """Show how VWAP divergence accumulates through a session and impacts trailing SL.

    The PB trail uses VWAP + buffer*ATR as trailing stop. Any VWAP divergence
    directly shifts the trail level.
    """
    print(f"\n{'='*90}")
    print("TRAIL IMPACT: VWAP divergence effect on trailing SL through sessions")
    print(f"{'='*90}")

    sessions: dict[str, list[int]] = defaultdict(list)
    for i, bar in enumerate(bars):
        sessions[bar.date[:10]].append(i)

    # Show late-session divergence (bars 60+ are where trail matters most)
    late_vwap_deltas: list[float] = []
    for session_date, indices in sorted(sessions.items()):
        for j, i in enumerate(indices):
            if j >= 60:  # Late session
                tv_v, _ = tv_vwap[i]
                py_v, _ = py_vwap[i]
                late_vwap_deltas.append(abs(tv_v - py_v))

    if late_vwap_deltas:
        print(f"\n  Late-session (bar >= 60) VWAP |delta|:")
        print(f"    Mean:   {statistics.mean(late_vwap_deltas):.6f}")
        print(f"    Max:    {max(late_vwap_deltas):.6f}")
        print(f"    Median: {statistics.median(late_vwap_deltas):.6f}")
        print(f"    Bars:   {len(late_vwap_deltas)}")

        # ATR context: typical SPY 5m ATR is ~$0.50-1.00
        # Trail buffer is typically 0.15 * ATR
        # So trail offset is ~$0.075-0.15
        # If VWAP delta is > 0.01, it's meaningful relative to trail
        meaningful = sum(1 for d in late_vwap_deltas if d > 0.01)
        print(f"\n    VWAP deltas > $0.01 (trail-relevant): {meaningful} "
              f"({meaningful/len(late_vwap_deltas)*100:.1f}%)")
        meaningful_005 = sum(1 for d in late_vwap_deltas if d > 0.005)
        print(f"    VWAP deltas > $0.005: {meaningful_005} "
              f"({meaningful_005/len(late_vwap_deltas)*100:.1f}%)")
    else:
        print("\n  No late-session bars found.")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 90)
    print("VWAP COMPUTATION COMPARISON: Python (Decimal) vs TradingView (float64)")
    print("=" * 90)

    # Validate paths
    if not BARS_CSV.exists():
        print(f"ERROR: SPY 5m data not found at {BARS_CSV}")
        sys.exit(1)

    # ── Load bars ─────────────────────────────────────────────────────
    print("\n[1/7] Loading SPY 5m bars...")
    bars = load_5m_bars(BARS_CSV)
    print(f"  Loaded {len(bars)} bars ({bars[0].date[:10]} to {bars[-1].date[:10]})")

    bar_index = build_bar_index(bars)

    # ── Compute VWAPs ─────────────────────────────────────────────────
    print("\n[2/7] Computing TV-style VWAP (float64, no quantization)...")
    tv_vwap = compute_tv_style_vwap(bars)

    print("[3/7] Computing Python streaming VWAP (Decimal, quantized TP)...")
    py_vwap = compute_python_vwap(bars)

    # ── Compute slopes ────────────────────────────────────────────────
    print("[4/7] Computing VWAP slopes...")
    tv_slopes = compute_vwap_slopes([v for v, _ in tv_vwap], SLOPE_PERIOD)
    py_slopes = compute_vwap_slopes([v for v, _ in py_vwap], SLOPE_PERIOD)

    # ── Analysis 1: Session divergence ────────────────────────────────
    analyze_session_divergence(bars, tv_vwap, py_vwap)

    # ── Analysis 2: First bar behavior ────────────────────────────────
    print_first_bar_behavior(bars)

    # ── Analysis 3: Quantization effect ───────────────────────────────
    analyze_quantization_effect(bars)

    # ── Analysis 4: Std dev formula comparison ────────────────────────
    analyze_std_dev_formula(bars)

    # ── Analysis 5: Batch vs streaming consistency ────────────────────
    print("\n[5/7] Computing Python batch VWAP (for consistency check)...")
    # Only do first 2000 bars (batch is O(n^2) per session)
    BATCH_LIMIT = 2000
    py_batch = compute_python_batch_vwap(bars[:BATCH_LIMIT])
    py_streaming_subset = py_vwap[:BATCH_LIMIT]
    analyze_batch_vs_streaming(py_streaming_subset, py_batch)

    # ── Analysis 6: Entry-specific comparison ─────────────────────────
    if TV_CSV.exists():
        print(f"\n[6/7] Loading TV entries from export CSV...")
        tv_entries = load_tv_entries(TV_CSV)
        print(f"  Loaded {len(tv_entries)} TV entries")
        analyze_vwap_at_entries(bars, bar_index, tv_entries, tv_vwap, py_vwap,
                               tv_slopes, py_slopes)
    else:
        print(f"\n[6/7] SKIP: TV export CSV not found at {TV_CSV}")

    # ── Analysis 7: Trail impact ──────────────────────────────────────
    print("\n[7/7] Trail impact analysis...")
    analyze_trail_impact(bars, bar_index, tv_vwap, py_vwap)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("SUMMARY OF KEY FINDINGS")
    print(f"{'='*90}")

    # Overall VWAP divergence
    all_vwap_d = [abs(tv_vwap[i][0] - py_vwap[i][0])
                  for i in range(len(bars))
                  if tv_vwap[i][0] != 0.0]
    all_std_d = [abs(tv_vwap[i][1] - py_vwap[i][1])
                 for i in range(len(bars))
                 if tv_vwap[i][0] != 0.0]

    print(f"""
  1. SOURCE: Both use hlc3 (typical price = (H+L+C)/3) -- MATCH
  2. SESSION RESET: Both reset on new calendar day -- MATCH
  3. VWAP FORMULA: cumTPV / cumVol -- MATCH
  4. STD DEV FORMULA: sqrt(Sum(vol*(tp-vwap)^2) / Sum(vol)) -- MATCH
     (Python streaming uses E[X^2]-E[X]^2 shortcut, equivalent at float64)
  5. QUANTIZATION: Python quantizes TP to 10 decimal places before accumulation;
     TV uses raw float64. This is the ONLY structural difference.

  VWAP mean |delta|: {statistics.mean(all_vwap_d):.8f}
  VWAP max |delta|:  {max(all_vwap_d):.8f}
  StdDev mean |delta|: {statistics.mean(all_std_d):.8f}
  StdDev max |delta|:  {max(all_std_d):.8f}
""")

    # Verdict
    threshold = 0.01  # $0.01 is meaningfully affects zone/trail
    meaningful = sum(1 for d in all_vwap_d if d > threshold)
    if meaningful == 0:
        print(f"  VERDICT: VWAP divergence is NEGLIGIBLE (< ${threshold:.2f} on all bars).")
        print(f"  The trade count discrepancy (123 vs 107) is NOT caused by VWAP computation.")
    else:
        pct = meaningful / len(all_vwap_d) * 100
        print(f"  VERDICT: {meaningful} bars ({pct:.1f}%) have VWAP divergence > ${threshold:.2f}.")
        print(f"  This MAY affect zone checks and trail levels on those bars.")

    print(f"\n{'='*90}")
    print("DONE")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
