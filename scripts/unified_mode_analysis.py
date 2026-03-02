"""Unified VWAP mode-by-mode analysis — compare Python to TV v11.2.

Runs the unified VWAP strategy (same config as tv_parity_backtest.py),
extracts the MODE from each trade's signal metadata, and produces:

1. Total trades by mode (PB, PS, ORB, REV)
2. Win rate by mode
3. PnL by mode
4. Day-by-day comparison against TV export

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/unified_mode_analysis.py
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

D = Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData

# Re-use build_unified_strategy from the parity backtest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv_parity_backtest import build_unified_strategy, load_5m_bars  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"

TV_CSV = Path("/Users/kelsayed/Downloads/VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv")

# TV known breakdown
TV_MODE_COUNTS = {"PB": 77, "PS": 19, "ORB": 6, "REV": 5}
TV_TOTAL = 107

INITIAL_CAPITAL = D("100000")
RISK_PER_TRADE = D("0.01")
SLIPPAGE_PCT = D("0.0002")


# ── TV CSV parsing ────────────────────────────────────────────────────


def load_tv_trades(csv_path: Path) -> list[dict]:
    """Parse TV export CSV into a list of trade dicts.

    Each trade in TV export has two rows: Entry and Exit.
    The entry row's Signal field starts with the mode (PB, PS, ORB, REV).
    """
    trades: list[dict] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by Trade #
    trade_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        trade_num = row["Trade #"].strip()
        trade_groups[trade_num].append(row)

    for trade_num, group_rows in sorted(trade_groups.items(), key=lambda x: int(x[0])):
        entry_row = None
        exit_row = None
        for r in group_rows:
            rtype = r["Type"].strip()
            if "Entry" in rtype:
                entry_row = r
            elif "Exit" in rtype:
                exit_row = r

        if entry_row is None:
            continue

        signal = entry_row.get("Signal", "")
        mode = signal.split("|")[0].strip() if signal else "UNK"

        # Extract direction from Type field
        entry_type = entry_row["Type"].strip()
        direction = "SHORT" if "short" in entry_type.lower() else "LONG"

        entry_dt = entry_row.get("Date and time", "").strip()
        entry_date = entry_dt[:10] if entry_dt else ""

        pnl_str = entry_row.get("Net P&L USD", "0").strip()
        try:
            pnl = float(pnl_str)
        except ValueError:
            pnl = 0.0

        exit_dt = exit_row.get("Date and time", "").strip() if exit_row else ""
        exit_date = exit_dt[:10] if exit_dt else ""
        exit_signal = exit_row.get("Signal", "").strip() if exit_row else ""

        trades.append({
            "trade_num": int(trade_num),
            "mode": mode,
            "direction": direction,
            "entry_datetime": entry_dt,
            "entry_date": entry_date,
            "exit_datetime": exit_dt,
            "exit_date": exit_date,
            "exit_signal": exit_signal,
            "pnl": pnl,
        })

    return trades


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    if not BARS_CSV.exists():
        print(f"ERROR: SPY 5m data not found at {BARS_CSV}")
        sys.exit(1)

    print("=" * 100)
    print("UNIFIED VWAP MODE ANALYSIS")
    print("=" * 100)

    # ── Load bars ─────────────────────────────────────────────────────
    print("\nLoading SPY 5m bars...")
    bars = load_5m_bars(BARS_CSV)
    print(f"  Loaded {len(bars)} bars ({bars[0].date[:10]} to {bars[-1].date[:10]})")

    # ── Run unified strategy ──────────────────────────────────────────
    unified_name, unified_strategy = build_unified_strategy()
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        slippage_pct=SLIPPAGE_PCT,
        fixed_capital=True,
    )

    print(f"\nRunning {unified_name}...")
    t0 = time.time()
    result = engine.run(unified_strategy, bars)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # ── Extract trades with modes ─────────────────────────────────────
    closed_trades = result.closed_trades
    trade_modes = result.trade_modes
    total_python = len(closed_trades)

    print(f"\n  Total Python trades: {total_python}")
    print(f"  Total PnL: ${float(result.total_pnl):+,.2f}")
    print(f"  Return: {float(result.total_return):+.2f}%")
    print(f"  Win rate: {float(result.win_rate):.1f}%")

    # ── Mode breakdown ────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("PER-MODE BREAKDOWN (Python Unified Strategy)")
    print(f"{'=' * 100}")

    # Build per-mode stats
    mode_stats: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "wins": 0, "losses": 0, "pnl": D("0"),
        "entry_dates": [],
    })

    for i, trade in enumerate(closed_trades):
        mode = trade_modes[i] if i < len(trade_modes) else "UNK"
        stats = mode_stats[mode]
        stats["count"] += 1
        if trade.is_win():
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["pnl"] += trade.profit_loss
        stats["entry_dates"].append(trade.entry_date[:10])

    print(f"\n{'Mode':<8} {'Py Count':>10} {'TV Count':>10} {'Delta':>8} "
          f"{'Win':>5} {'Loss':>5} {'Win%':>7} {'PnL':>12}")
    print("-" * 80)

    all_modes = sorted(set(list(mode_stats.keys()) + list(TV_MODE_COUNTS.keys())))
    total_py = 0
    total_tv = 0
    total_wins = 0
    total_losses = 0
    total_pnl = D("0")

    for mode in all_modes:
        stats = mode_stats.get(mode, {"count": 0, "wins": 0, "losses": 0, "pnl": D("0")})
        py_count = stats["count"]
        tv_count = TV_MODE_COUNTS.get(mode, 0)
        delta = py_count - tv_count
        wins = stats["wins"]
        losses = stats["losses"]
        wr = (wins / py_count * 100) if py_count > 0 else 0
        pnl = stats["pnl"]

        total_py += py_count
        total_tv += tv_count
        total_wins += wins
        total_losses += losses
        total_pnl += pnl

        delta_str = f"{delta:+d}"
        print(f"{mode:<8} {py_count:>10} {tv_count:>10} {delta_str:>8} "
              f"{wins:>5} {losses:>5} {wr:>6.1f}% ${float(pnl):>+10,.2f}")

    print("-" * 80)
    total_wr = (total_wins / total_py * 100) if total_py > 0 else 0
    total_delta = total_py - total_tv
    print(f"{'TOTAL':<8} {total_py:>10} {total_tv:>10} {total_delta:>+8d} "
          f"{total_wins:>5} {total_losses:>5} {total_wr:>6.1f}% ${float(total_pnl):>+10,.2f}")

    # ── Biggest gaps ──────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("MODE GAP ANALYSIS")
    print(f"{'=' * 100}")
    for mode in all_modes:
        stats = mode_stats.get(mode, {"count": 0})
        py_count = stats["count"]
        tv_count = TV_MODE_COUNTS.get(mode, 0)
        delta = py_count - tv_count
        if delta != 0:
            pct = (delta / tv_count * 100) if tv_count > 0 else float('inf')
            direction = "UNDER" if delta < 0 else "OVER"
            print(f"  {mode}: Python={py_count}, TV={tv_count} -> {direction}-trading by "
                  f"{abs(delta)} trades ({pct:+.0f}%)")

    # ── Load TV trades for day-by-day comparison ──────────────────────
    if not TV_CSV.exists():
        print(f"\nWARNING: TV CSV not found at {TV_CSV}")
        print("  Skipping day-by-day comparison.")
        return

    print(f"\n{'=' * 100}")
    print("DAY-BY-DAY COMPARISON (Python vs TV)")
    print(f"{'=' * 100}")

    tv_trades = load_tv_trades(TV_CSV)
    print(f"\n  Loaded {len(tv_trades)} TV trades from CSV")

    # TV mode verification
    tv_mode_counts: dict[str, int] = defaultdict(int)
    for t in tv_trades:
        tv_mode_counts[t["mode"]] += 1
    print(f"  TV modes from CSV: {dict(tv_mode_counts)}")

    # Build per-day data for both systems
    py_by_day: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tv_by_day: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i, trade in enumerate(closed_trades):
        mode = trade_modes[i] if i < len(trade_modes) else "UNK"
        day = trade.entry_date[:10]
        py_by_day[day][mode] += 1
        py_by_day[day]["_total"] += 1

    for t in tv_trades:
        day = t["entry_date"]
        mode = t["mode"]
        tv_by_day[day][mode] += 1
        tv_by_day[day]["_total"] += 1

    all_days = sorted(set(list(py_by_day.keys()) + list(tv_by_day.keys())))

    # Header
    print(f"\n{'Date':<12} {'Py Tot':>6} {'TV Tot':>6} {'Delta':>6} "
          f"{'Py PB':>5} {'TV PB':>5} "
          f"{'Py PS':>5} {'TV PS':>5} "
          f"{'Py ORB':>6} {'TV ORB':>6} "
          f"{'Py REV':>6} {'TV REV':>6} "
          f"{'Flag':>6}")
    print("-" * 110)

    divergent_days = []
    for day in all_days:
        py_d = py_by_day.get(day, {})
        tv_d = tv_by_day.get(day, {})

        py_total = py_d.get("_total", 0)
        tv_total = tv_d.get("_total", 0)
        delta = py_total - tv_total

        flag = ""
        if abs(delta) >= 2:
            flag = "***"
            divergent_days.append((day, py_total, tv_total, delta, dict(py_d), dict(tv_d)))
        elif abs(delta) == 1:
            flag = "*"

        print(f"{day:<12} {py_total:>6} {tv_total:>6} {delta:>+6} "
              f"{py_d.get('PB', 0):>5} {tv_d.get('PB', 0):>5} "
              f"{py_d.get('PS', 0):>5} {tv_d.get('PS', 0):>5} "
              f"{py_d.get('ORB', 0):>6} {tv_d.get('ORB', 0):>6} "
              f"{py_d.get('REV', 0):>6} {tv_d.get('REV', 0):>6} "
              f"{flag:>6}")

    # ── Summary of divergent days ─────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("DIVERGENT DAYS (|delta| >= 2)")
    print(f"{'=' * 100}")

    if not divergent_days:
        print("  No days with significant divergence (|delta| >= 2).")
    else:
        for day, py_t, tv_t, delta, py_modes, tv_modes in divergent_days:
            print(f"\n  {day}: Python={py_t}, TV={tv_t} (delta={delta:+d})")
            # Show mode breakdown
            for m in ["PB", "PS", "ORB", "REV"]:
                pm = py_modes.get(m, 0)
                tm = tv_modes.get(m, 0)
                if pm != 0 or tm != 0:
                    mode_delta = pm - tm
                    marker = " <--" if mode_delta != 0 else ""
                    print(f"    {m}: Py={pm}, TV={tm} (delta={mode_delta:+d}){marker}")

    # ── Days with Python-only or TV-only trades ───────────────────────
    py_only_days = [d for d in all_days if d in py_by_day and d not in tv_by_day]
    tv_only_days = [d for d in all_days if d in tv_by_day and d not in py_by_day]

    if py_only_days:
        print(f"\n  Python-only days ({len(py_only_days)}):")
        for d in py_only_days:
            modes = {k: v for k, v in py_by_day[d].items() if k != "_total"}
            print(f"    {d}: {py_by_day[d]['_total']} trades {dict(modes)}")

    if tv_only_days:
        print(f"\n  TV-only days ({len(tv_only_days)}):")
        for d in tv_only_days:
            modes = {k: v for k, v in tv_by_day[d].items() if k != "_total"}
            print(f"    {d}: {tv_by_day[d]['_total']} trades {dict(modes)}")

    # ── Per-mode day overlap analysis ─────────────────────────────────
    print(f"\n{'=' * 100}")
    print("PER-MODE DAY OVERLAP")
    print(f"{'=' * 100}")

    for mode in ["PB", "PS", "ORB", "REV"]:
        py_mode_days = set()
        tv_mode_days = set()
        for d in all_days:
            if py_by_day.get(d, {}).get(mode, 0) > 0:
                py_mode_days.add(d)
            if tv_by_day.get(d, {}).get(mode, 0) > 0:
                tv_mode_days.add(d)

        overlap = py_mode_days & tv_mode_days
        py_only = py_mode_days - tv_mode_days
        tv_only = tv_mode_days - py_mode_days

        print(f"\n  {mode}:")
        print(f"    Python days: {len(py_mode_days)}, TV days: {len(tv_mode_days)}")
        print(f"    Overlap: {len(overlap)} days")
        if py_only:
            print(f"    Python-only ({len(py_only)}): {sorted(py_only)}")
        if tv_only:
            print(f"    TV-only ({len(tv_only)}): {sorted(tv_only)}")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"  Python total: {total_py} trades")
    print(f"  TV total:     {TV_TOTAL} trades")
    print(f"  Gap:          {total_py - TV_TOTAL:+d} trades ({(total_py - TV_TOTAL) / TV_TOTAL * 100:+.1f}%)")
    print()
    for mode in ["PB", "PS", "ORB", "REV"]:
        py_c = mode_stats.get(mode, {"count": 0})["count"]
        tv_c = TV_MODE_COUNTS.get(mode, 0)
        gap = py_c - tv_c
        pct = (gap / tv_c * 100) if tv_c > 0 else 0
        print(f"  {mode}: Py={py_c:>3}, TV={tv_c:>3}, gap={gap:>+4} ({pct:>+6.1f}%)")

    print(f"\n  Divergent days (|delta| >= 2): {len(divergent_days)}")
    print(f"  Python-only days: {len(py_only_days)}")
    print(f"  TV-only days: {len(tv_only_days)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
