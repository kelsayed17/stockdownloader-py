"""Trade-by-trade comparison: Python unified vs TradingView export.

Parses TV trades from CSV, runs the Python unified strategy, and matches
trades to identify divergence root causes.

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/tv_trade_compare.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

D = Decimal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import Direction
from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy, UnifiedVWAPConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"
TV_CSV = Path("/Users/kelsayed/Downloads/VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-10_e0116.csv")

INITIAL_CAPITAL = D("100000")
RISK_PER_TRADE = D("0.01")
SLIPPAGE_PCT = D("0.0002")

# TV times are Pacific → add 3h for Eastern
TV_TZ_OFFSET_HOURS = 3


# ===================================================================
# Data loading
# ===================================================================

def load_5m_bars(csv_path: Path) -> list[IntradayPriceData]:
    bars: list[IntradayPriceData] = []
    with open(csv_path) as f:
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


# ===================================================================
# TV trade parsing
# ===================================================================

def parse_tv_trades(csv_path: Path) -> list[dict]:
    """Parse TV export CSV into list of trade dicts."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows_by_trade: dict[str, list[dict]] = {}
        for row in reader:
            tn = row.get("Trade #", "").strip()
            if not tn:
                continue
            rows_by_trade.setdefault(tn, []).append(row)

    trades = []
    for tn, rows in sorted(rows_by_trade.items(), key=lambda x: int(x[0])):
        if len(rows) < 2:
            continue

        # Sort by datetime
        rows.sort(key=lambda r: r.get("Date and time", ""))

        # First row = entry (earlier datetime), second = exit
        # But TV exports exit row first in the file! After sorting, the earlier
        # datetime is the entry. Let's detect which is entry vs exit:
        entry_row = None
        exit_row = None
        for r in rows:
            type_col = r.get("Type", "").lower()
            if "entry" in type_col:
                entry_row = r
            elif "exit" in type_col:
                exit_row = r

        if not entry_row or not exit_row:
            continue

        # Parse direction
        signal = entry_row.get("Signal", "")
        if "|L|" in signal:
            direction = "LONG"
        elif "|S|" in signal:
            direction = "SHORT"
        else:
            type_col = entry_row.get("Type", "").lower()
            if "long" in type_col:
                direction = "LONG"
            else:
                direction = "SHORT"

        # Parse mode from signal (first token before |)
        mode = signal.split("|")[0].strip() if signal else "UNK"

        # Parse exit reason from exit row's signal
        exit_signal = exit_row.get("Signal", "").strip()

        # Parse entry/exit datetimes and convert PT → ET
        entry_dt_str = entry_row.get("Date and time", "").strip()
        exit_dt_str = exit_row.get("Date and time", "").strip()
        entry_dt_et = _convert_pt_to_et(entry_dt_str)
        exit_dt_et = _convert_pt_to_et(exit_dt_str)

        # Parse prices
        entry_price = _parse_dec(entry_row.get("Price USD", ""))
        exit_price = _parse_dec(exit_row.get("Price USD", ""))

        # Parse PnL
        pnl = _parse_dec(exit_row.get("Net P&L USD", ""))

        # Parse shares
        shares_str = entry_row.get("Position size (qty)", "0")
        try:
            shares = int(shares_str)
        except ValueError:
            shares = 0

        trades.append({
            "trade_num": int(tn),
            "direction": direction,
            "mode": mode,
            "exit_reason": exit_signal,
            "entry_dt_pt": entry_dt_str,
            "exit_dt_pt": exit_dt_str,
            "entry_dt_et": entry_dt_et,
            "exit_dt_et": exit_dt_et,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "shares": shares,
        })

    return trades


def _convert_pt_to_et(dt_str: str) -> str:
    """Convert Pacific Time datetime string to Eastern Time."""
    if not dt_str:
        return ""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        dt_et = dt + timedelta(hours=TV_TZ_OFFSET_HOURS)
        return dt_et.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return dt_str


def _parse_dec(s: str) -> Decimal:
    """Parse decimal from string, handling commas and currency symbols."""
    if not s:
        return D("0")
    cleaned = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return D(cleaned)
    except Exception:
        return D("0")


# ===================================================================
# Python strategy runner
# ===================================================================

def build_unified() -> UnifiedVWAPStrategy:
    config = UnifiedVWAPConfig(
        max_day=2, spacing=3, be_trigger=D("0.5"), trail_buf=D("0.15"),
        trail_keep_tp=True, close_eod=True, circuit=3, day_loss=D("3.0"),
        adx_thresh=D("21"), allow_longs=True, allow_shorts=True,
    )
    pb_overrides = dict(
        pb_zone=D("0.5"), pb_body=D("0.15"), rr=D("1.4"), sl_atr=D("1.3"),
        sl_cap=D("1.50"), trend_bars=3, htf_align=True, ar_filter=True,
        ar_thresh=D("0.9"), ar_cap=D("1.15"), va_filter=True, va_min=D("-0.1"),
        cvd_long_filter=True, lrs_short_filter=True, lrs_thresh=D("0.08"),
        max_vxc=6, w_vol=3, w_sr=2, w_rsi=1, w_time=0, w_pq=1, w_box=0,
        min_score=3, min_score_long=5, pq_max_cross=3, no_friday_short=True,
        no_monday_long=True, trail_vwap=True, pb_vwap_bias=False, pb_tp_mode="rr",
    )
    ps_overrides = dict(
        ps_atr_pct=D("30.0"), ps_window=12, ps_rvol=D("1.0"), ps_engulf=D("0.35"),
        ps_sl_mode="Day Extreme", ps_sl_atr=D("1.5"), ps_sl_cap=D("2.50"),
        ps_tp_pct=D("75.0"), ps_sma_filter=False, ps_htf_align=False,
        ps_time_gate=False, ps_min_rr=D("0.3"),
    )
    orb_overrides = dict(
        orb_window=20, orb_rvol=D("2.0"), orb_sl_mode="OR Opposite",
        orb_sl_atr=D("1.5"), orb_sl_cap=D("2.50"), orb_vwap_align=True,
        orb_body_min=D("0.2"), orb_entry_mode="aggressive", orb_trail_atr=D("1.5"),
        orb_htf_align=False, orb_gap_filter=False, orb_adx_filter=False,
    )
    rev_overrides = dict(
        rev_band="2\u03c3", rev_body=D("0.20"), rev_sl_atr=D("1.0"),
        rev_sl_cap=D("1.50"), rev_shorts=False, rev_min_rr=D("0.3"),
        rev_tp_mode="vwap", rev_vwap_flat_tol=D("0.05"), rev_require_sr=False,
        rev_hug_limit=20, rev_can_trade_bar=11, min_score=3,
    )
    return UnifiedVWAPStrategy(
        config=config, pb_overrides=pb_overrides, ps_overrides=ps_overrides,
        orb_overrides=orb_overrides, rev_overrides=rev_overrides,
    )


def run_python_backtest(bars: list[IntradayPriceData]) -> list[dict]:
    """Run unified strategy and return per-trade details."""
    strategy = build_unified()
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        slippage_pct=SLIPPAGE_PCT,
        fixed_capital=True,
        next_bar_fill=True,
        trigger_exit_fill=True,
    )
    result = engine.run(strategy, bars)

    py_trades = []
    for idx, trade in enumerate(result.trades):
        mode = result.trade_modes[idx] if idx < len(result.trade_modes) else "UNK"
        exit_reason = result.trade_exit_reasons[idx] if idx < len(result.trade_exit_reasons) else "UNK"

        py_trades.append({
            "idx": idx + 1,
            "direction": trade.direction.value,
            "mode": mode,
            "exit_reason": exit_reason,
            "entry_dt": trade.entry_date,
            "exit_dt": trade.exit_date or "",
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price or D("0"),
            "pnl": trade.profit_loss,
            "shares": trade.shares,
        })

    return py_trades


# ===================================================================
# Trade matching
# ===================================================================

def match_trades(
    tv_trades: list[dict],
    py_trades: list[dict],
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Match TV trades to Python trades by date+direction+mode proximity.

    Returns (matched_pairs, unmatched_tv, unmatched_py).
    """
    matched = []
    used_py = set()

    for tv in tv_trades:
        best_py_idx = None
        best_score = -1

        tv_entry_date = tv["entry_dt_et"][:10] if tv["entry_dt_et"] else ""
        tv_entry_time = tv["entry_dt_et"][11:16] if len(tv["entry_dt_et"]) >= 16 else ""

        for j, py in enumerate(py_trades):
            if j in used_py:
                continue

            py_entry_date = py["entry_dt"][:10] if py["entry_dt"] else ""
            py_entry_time = py["entry_dt"][11:16] if len(py["entry_dt"]) >= 16 else ""

            # Must match on date
            if tv_entry_date != py_entry_date:
                continue

            score = 0
            # Direction match
            if tv["direction"] == py["direction"]:
                score += 10
            else:
                continue  # direction mismatch = no match

            # Mode match
            if tv["mode"] == py["mode"]:
                score += 5

            # Time proximity (prefer closer entry times)
            if tv_entry_time and py_entry_time:
                try:
                    tv_t = datetime.strptime(tv_entry_time, "%H:%M")
                    py_t = datetime.strptime(py_entry_time, "%H:%M")
                    diff_min = abs((tv_t - py_t).total_seconds()) / 60
                    if diff_min <= 5:
                        score += 4
                    elif diff_min <= 15:
                        score += 2
                    elif diff_min <= 30:
                        score += 1
                except ValueError:
                    pass

            if score > best_score:
                best_score = score
                best_py_idx = j

        if best_py_idx is not None and best_score >= 10:
            matched.append((tv, py_trades[best_py_idx]))
            used_py.add(best_py_idx)

    unmatched_tv = [tv for i, tv in enumerate(tv_trades)
                    if not any(tv is m[0] for m in matched)]
    unmatched_py = [py for j, py in enumerate(py_trades)
                    if j not in used_py]

    return matched, unmatched_tv, unmatched_py


# ===================================================================
# Reporting
# ===================================================================

def print_header(title: str) -> None:
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")


def print_section(title: str) -> None:
    print(f"\n  --- {title} ---")


def report_summary(tv_trades, py_trades, matched, unmatched_tv, unmatched_py):
    """Print overall summary."""
    print_header("TRADE-BY-TRADE COMPARISON: Python vs TradingView")

    tv_pnl = sum(t["pnl"] for t in tv_trades)
    py_pnl = sum(t["pnl"] for t in py_trades)
    tv_wins = sum(1 for t in tv_trades if t["pnl"] > 0)
    py_wins = sum(1 for t in py_trades if t["pnl"] > 0)

    print(f"\n  {'Metric':<30} {'TradingView':>15} {'Python':>15} {'Delta':>15}")
    print(f"  {'-' * 75}")
    print(f"  {'Total Trades':<30} {len(tv_trades):>15} {len(py_trades):>15} {len(py_trades) - len(tv_trades):>+15}")
    print(f"  {'Total PnL ($)':<30} ${float(tv_pnl):>14,.2f} ${float(py_pnl):>14,.2f} ${float(py_pnl - tv_pnl):>+14,.2f}")
    tv_wr = tv_wins / len(tv_trades) * 100 if tv_trades else 0
    py_wr = py_wins / len(py_trades) * 100 if py_trades else 0
    print(f"  {'Win Rate (%)':<30} {tv_wr:>14.1f}% {py_wr:>14.1f}% {py_wr - tv_wr:>+14.1f}%")
    print(f"  {'Matched Pairs':<30} {len(matched):>15}")
    print(f"  {'Unmatched TV (missing in Py)':<30} {len(unmatched_tv):>15}")
    print(f"  {'Unmatched Py (extra in Py)':<30} {len(unmatched_py):>15}")


def report_mode_breakdown(tv_trades, py_trades):
    """Show mode breakdown for both TV and Python."""
    print_section("MODE BREAKDOWN")

    tv_modes = Counter(t["mode"] for t in tv_trades)
    py_modes = Counter(t["mode"] for t in py_trades)

    all_modes = sorted(set(list(tv_modes.keys()) + list(py_modes.keys())))

    print(f"\n  {'Mode':<8} {'TV Count':>10} {'TV PnL':>12} {'TV WR':>8} "
          f"{'Py Count':>10} {'Py PnL':>12} {'Py WR':>8}")
    print(f"  {'-' * 72}")

    for mode in all_modes:
        tv_m = [t for t in tv_trades if t["mode"] == mode]
        py_m = [t for t in py_trades if t["mode"] == mode]
        tv_pnl = sum(t["pnl"] for t in tv_m)
        py_pnl = sum(t["pnl"] for t in py_m)
        tv_wr = sum(1 for t in tv_m if t["pnl"] > 0) / len(tv_m) * 100 if tv_m else 0
        py_wr = sum(1 for t in py_m if t["pnl"] > 0) / len(py_m) * 100 if py_m else 0

        print(f"  {mode:<8} {len(tv_m):>10} ${float(tv_pnl):>11,.2f} {tv_wr:>7.1f}% "
              f"{len(py_m):>10} ${float(py_pnl):>11,.2f} {py_wr:>7.1f}%")


def report_exit_reason_breakdown(tv_trades, py_trades):
    """Show exit reason breakdown."""
    print_section("EXIT REASON BREAKDOWN")

    tv_exits = Counter(t["exit_reason"] for t in tv_trades)
    py_exits = Counter(t["exit_reason"] for t in py_trades)

    print(f"\n  TradingView exit reasons:")
    for reason, count in tv_exits.most_common():
        pnl = sum(t["pnl"] for t in tv_trades if t["exit_reason"] == reason)
        wr = sum(1 for t in tv_trades if t["exit_reason"] == reason and t["pnl"] > 0) / count * 100
        print(f"    {reason:<15} {count:>4} trades  PnL=${float(pnl):>+10,.2f}  WR={wr:.0f}%")

    print(f"\n  Python exit reasons:")
    for reason, count in py_exits.most_common():
        pnl = sum(t["pnl"] for t in py_trades if t["exit_reason"] == reason)
        wr = sum(1 for t in py_trades if t["exit_reason"] == reason and t["pnl"] > 0) / count * 100
        print(f"    {reason:<15} {count:>4} trades  PnL=${float(pnl):>+10,.2f}  WR={wr:.0f}%")


def report_matched_divergence(matched):
    """Analyze divergence within matched pairs."""
    print_section("MATCHED TRADE DIVERGENCE ANALYSIS")

    if not matched:
        print("  No matched trades to analyze.")
        return

    # Categorize divergence
    categories = defaultdict(list)

    for tv, py in matched:
        tv_pnl = float(tv["pnl"])
        py_pnl = float(py["pnl"])
        pnl_delta = py_pnl - tv_pnl

        # Entry price difference
        entry_diff = float(py["entry_price"] - tv["entry_price"])

        # Exit price difference
        exit_diff = float(py["exit_price"] - tv["exit_price"])

        # Categorize
        if abs(pnl_delta) < 5:
            categories["close_match"].append((tv, py, pnl_delta))
        elif tv_pnl > 0 and py_pnl <= 0:
            categories["tv_win_py_lose"].append((tv, py, pnl_delta))
        elif tv_pnl <= 0 and py_pnl > 0:
            categories["tv_lose_py_win"].append((tv, py, pnl_delta))
        elif abs(pnl_delta) > 50:
            categories["large_delta"].append((tv, py, pnl_delta))
        else:
            categories["small_delta"].append((tv, py, pnl_delta))

    print(f"\n  {'Category':<35} {'Count':>6} {'Sum Delta PnL':>15}")
    print(f"  {'-' * 58}")
    for cat, label in [
        ("close_match", "Close match (|delta| < $5)"),
        ("tv_win_py_lose", "TV WIN, Python LOSE (flip)"),
        ("tv_lose_py_win", "TV LOSE, Python WIN (flip)"),
        ("large_delta", "Large delta (|delta| > $50)"),
        ("small_delta", "Small delta ($5-$50)"),
    ]:
        items = categories.get(cat, [])
        delta_sum = sum(d for _, _, d in items)
        print(f"  {label:<35} {len(items):>6} ${delta_sum:>+14,.2f}")

    # Detail the worst flip cases
    flips = categories.get("tv_win_py_lose", [])
    if flips:
        print(f"\n  TOP TV-WIN/PY-LOSE FLIPS (most damaging):")
        print(f"  {'#':>4} {'Mode':>5} {'Dir':>6} {'TV Entry':>12} {'Py Entry':>12} "
              f"{'TV Exit':>10} {'Py Exit':>10} {'TV PnL':>10} {'Py PnL':>10} "
              f"{'TV ExitR':>12} {'Py ExitR':>12}")
        print(f"  {'-' * 115}")
        flips.sort(key=lambda x: x[2])  # worst first
        for tv, py, delta in flips[:20]:
            print(f"  {tv['trade_num']:>4} {tv['mode']:>5} {tv['direction']:>6} "
                  f"${float(tv['entry_price']):>11.2f} ${float(py['entry_price']):>11.2f} "
                  f"${float(tv['exit_price']):>9.2f} ${float(py['exit_price']):>9.2f} "
                  f"${float(tv['pnl']):>+9.2f} ${float(py['pnl']):>+9.2f} "
                  f"{tv['exit_reason']:>12} {py['exit_reason']:>12}")

    # Also show large deltas
    large = categories.get("large_delta", [])
    if large:
        print(f"\n  LARGE DELTA TRADES (|delta| > $50):")
        print(f"  {'#':>4} {'Mode':>5} {'Dir':>6} {'TV Entry':>12} {'Py Entry':>12} "
              f"{'TV PnL':>10} {'Py PnL':>10} {'Delta':>10} "
              f"{'TV ExitR':>12} {'Py ExitR':>12}")
        print(f"  {'-' * 105}")
        large.sort(key=lambda x: x[2])
        for tv, py, delta in large[:15]:
            print(f"  {tv['trade_num']:>4} {tv['mode']:>5} {tv['direction']:>6} "
                  f"${float(tv['entry_price']):>11.2f} ${float(py['entry_price']):>11.2f} "
                  f"${float(tv['pnl']):>+9.2f} ${float(py['pnl']):>+9.2f} "
                  f"${delta:>+9.2f} "
                  f"{tv['exit_reason']:>12} {py['exit_reason']:>12}")


def report_unmatched(unmatched_tv, unmatched_py):
    """Show unmatched trades."""
    if unmatched_tv:
        print_section(f"UNMATCHED TV TRADES ({len(unmatched_tv)} — missing from Python)")
        print(f"  {'#':>4} {'Mode':>5} {'Dir':>6} {'Entry (ET)':>18} {'Exit (ET)':>18} "
              f"{'EntryP':>10} {'ExitP':>10} {'PnL':>10} {'ExitR':>12}")
        print(f"  {'-' * 100}")
        for tv in unmatched_tv:
            print(f"  {tv['trade_num']:>4} {tv['mode']:>5} {tv['direction']:>6} "
                  f"{tv['entry_dt_et']:>18} {tv['exit_dt_et']:>18} "
                  f"${float(tv['entry_price']):>9.2f} ${float(tv['exit_price']):>9.2f} "
                  f"${float(tv['pnl']):>+9.2f} {tv['exit_reason']:>12}")
        missing_pnl = sum(t["pnl"] for t in unmatched_tv)
        print(f"  {'TOTAL':>4} {'':>5} {'':>6} {'':>18} {'':>18} "
              f"{'':>10} {'':>10} ${float(missing_pnl):>+9.2f}")

    if unmatched_py:
        print_section(f"UNMATCHED PYTHON TRADES ({len(unmatched_py)} — extra in Python)")
        print(f"  {'#':>4} {'Mode':>5} {'Dir':>6} {'Entry DT':>18} {'Exit DT':>18} "
              f"{'EntryP':>10} {'ExitP':>10} {'PnL':>10} {'ExitR':>12}")
        print(f"  {'-' * 100}")
        for py in unmatched_py[:30]:  # limit output
            print(f"  {py['idx']:>4} {py['mode']:>5} {py['direction']:>6} "
                  f"{py['entry_dt'][:16]:>18} {py['exit_dt'][:16]:>18} "
                  f"${float(py['entry_price']):>9.2f} ${float(py['exit_price']):>9.2f} "
                  f"${float(py['pnl']):>+9.2f} {py['exit_reason']:>12}")
        if len(unmatched_py) > 30:
            print(f"  ... and {len(unmatched_py) - 30} more")
        extra_pnl = sum(t["pnl"] for t in unmatched_py)
        print(f"  {'TOTAL':>4} {'':>5} {'':>6} {'':>18} {'':>18} "
              f"{'':>10} {'':>10} ${float(extra_pnl):>+9.2f}")


def report_pnl_attribution(matched, unmatched_tv, unmatched_py, tv_pnl_total, py_pnl_total):
    """Attribute the total P&L gap to specific sources."""
    print_header("P&L GAP ATTRIBUTION")

    gap = float(py_pnl_total - tv_pnl_total)
    print(f"\n  Total gap: ${gap:>+,.2f} (Python={float(py_pnl_total):>+,.2f}, TV={float(tv_pnl_total):>+,.2f})")

    # 1. Matched trade divergence
    matched_tv_pnl = sum(float(tv["pnl"]) for tv, _ in matched)
    matched_py_pnl = sum(float(py["pnl"]) for _, py in matched)
    matched_delta = matched_py_pnl - matched_tv_pnl

    # 2. Missing TV trades (TV has, Python doesn't)
    missing_tv_pnl = sum(float(t["pnl"]) for t in unmatched_tv)

    # 3. Extra Python trades (Python has, TV doesn't)
    extra_py_pnl = sum(float(t["pnl"]) for t in unmatched_py)

    print(f"\n  {'Source':<40} {'Contribution':>15} {'% of Gap':>10}")
    print(f"  {'-' * 67}")
    print(f"  {'Matched trade divergence (same trades)':<40} ${matched_delta:>+14,.2f} "
          f"{matched_delta / gap * 100 if gap else 0:>9.1f}%")
    print(f"  {'Missing TV trades (not in Python)':<40} ${-missing_tv_pnl:>+14,.2f} "
          f"{-missing_tv_pnl / gap * 100 if gap else 0:>9.1f}%")
    print(f"  {'Extra Python trades (not in TV)':<40} ${extra_py_pnl:>+14,.2f} "
          f"{extra_py_pnl / gap * 100 if gap else 0:>9.1f}%")
    print(f"  {'-' * 67}")
    attributed = matched_delta - missing_tv_pnl + extra_py_pnl
    print(f"  {'Sum (should ≈ total gap)':<40} ${attributed:>+14,.2f}")

    # Breakdown within matched trades
    print_section("MATCHED TRADE DIVERGENCE BY EXIT REASON MAPPING")

    exit_reason_map = defaultdict(lambda: {"count": 0, "tv_pnl": 0.0, "py_pnl": 0.0})
    for tv, py in matched:
        key = f"{tv['exit_reason']} → {py['exit_reason']}"
        exit_reason_map[key]["count"] += 1
        exit_reason_map[key]["tv_pnl"] += float(tv["pnl"])
        exit_reason_map[key]["py_pnl"] += float(py["pnl"])

    print(f"\n  {'TV Exit → Py Exit':<35} {'Count':>6} {'TV PnL':>12} {'Py PnL':>12} {'Delta':>12}")
    print(f"  {'-' * 80}")
    for key, vals in sorted(exit_reason_map.items(), key=lambda x: x[1]["py_pnl"] - x[1]["tv_pnl"]):
        delta = vals["py_pnl"] - vals["tv_pnl"]
        print(f"  {key:<35} {vals['count']:>6} ${vals['tv_pnl']:>+11,.2f} "
              f"${vals['py_pnl']:>+11,.2f} ${delta:>+11,.2f}")


def report_daily_trade_timeline(tv_trades, py_trades):
    """Show which dates have trades in TV vs Python."""
    print_section("DAILY TRADE TIMELINE (first/last 30 dates)")

    tv_by_date = defaultdict(list)
    for t in tv_trades:
        d = t["entry_dt_et"][:10]
        tv_by_date[d].append(t)

    py_by_date = defaultdict(list)
    for t in py_trades:
        d = t["entry_dt"][:10]
        py_by_date[d].append(t)

    all_dates = sorted(set(list(tv_by_date.keys()) + list(py_by_date.keys())))

    print(f"\n  {'Date':<12} {'TV#':>4} {'TV Modes':>20} {'Py#':>4} {'Py Modes':>20} {'Match':>6}")
    print(f"  {'-' * 70}")

    for date in all_dates[:30]:
        tv_t = tv_by_date.get(date, [])
        py_t = py_by_date.get(date, [])
        tv_modes = ",".join(t["mode"] for t in tv_t)
        py_modes = ",".join(t["mode"] for t in py_t)
        match = "✓" if len(tv_t) == len(py_t) else "✗"
        print(f"  {date:<12} {len(tv_t):>4} {tv_modes:>20} {len(py_t):>4} {py_modes:>20} {match:>6}")

    if len(all_dates) > 30:
        print(f"  ... {len(all_dates) - 30} more dates ...")
        for date in all_dates[-10:]:
            tv_t = tv_by_date.get(date, [])
            py_t = py_by_date.get(date, [])
            tv_modes = ",".join(t["mode"] for t in tv_t)
            py_modes = ",".join(t["mode"] for t in py_t)
            match = "✓" if len(tv_t) == len(py_t) else "✗"
            print(f"  {date:<12} {len(tv_t):>4} {tv_modes:>20} {len(py_t):>4} {py_modes:>20} {match:>6}")


# ===================================================================
# Main
# ===================================================================

def main():
    print("Loading 5m bars...")
    all_bars = load_5m_bars(BARS_CSV)

    print("Parsing TV trades...")
    tv_trades = parse_tv_trades(TV_CSV)
    print(f"  {len(tv_trades)} TV trades parsed")

    # Determine date range from TV trades
    tv_dates = [t["entry_dt_et"][:10] for t in tv_trades]
    tv_start = min(tv_dates)
    tv_end = max(tv_dates)
    print(f"  TV date range (ET): {tv_start} to {tv_end}")

    # Add warmup buffer (60 trading days ≈ 3 months)
    # The strategy needs warmup bars before the first trade date
    warmup_start = None
    for b in all_bars:
        d = b.date[:10]
        if d < tv_start:
            warmup_start = d
        else:
            break

    # Filter bars: include warmup period before first TV trade + all bars through last TV trade
    # Use all available bars before tv_start for warmup, plus bars through tv_end
    bars = [b for b in all_bars if b.date[:10] <= tv_end]
    print(f"  Using {len(bars)} bars (through {tv_end}) with full warmup")

    print("Running Python backtest...")
    py_trades = run_python_backtest(bars)
    print(f"  {len(py_trades)} Python trades generated")

    # Filter Python trades to TV date range (trades may appear earlier due to different signals)
    py_trades_filtered = [t for t in py_trades if tv_start <= t["entry_dt"][:10] <= tv_end]
    print(f"  {len(py_trades_filtered)} Python trades in TV date range ({tv_start} to {tv_end})")

    # Match trades
    print("Matching trades...")
    matched, unmatched_tv, unmatched_py = match_trades(tv_trades, py_trades_filtered)
    print(f"  {len(matched)} matched, {len(unmatched_tv)} TV-only, {len(unmatched_py)} Py-only")

    # Reports
    report_summary(tv_trades, py_trades_filtered, matched, unmatched_tv, unmatched_py)
    report_mode_breakdown(tv_trades, py_trades_filtered)
    report_exit_reason_breakdown(tv_trades, py_trades_filtered)
    report_matched_divergence(matched)
    report_unmatched(unmatched_tv, unmatched_py)

    tv_pnl_total = sum(t["pnl"] for t in tv_trades)
    py_pnl_total = sum(t["pnl"] for t in py_trades_filtered)
    report_pnl_attribution(matched, unmatched_tv, unmatched_py, tv_pnl_total, py_pnl_total)
    report_daily_trade_timeline(tv_trades, py_trades_filtered)

    print(f"\n{'=' * 100}")
    print("  DONE")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
