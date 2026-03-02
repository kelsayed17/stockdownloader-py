"""Exit Comparison — Python vs TradingView trade-by-trade analysis.

Matches Python unified VWAP strategy trades to TV export trades and
compares entry/exit prices, PnL, exit reasons, and holding periods
to identify where the exit management diverges.

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/exit_comparison.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

D = Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy, UnifiedVWAPConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"

TV_CSV = Path("/Users/kelsayed/Downloads/VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv")

TV_START_DATE = "2025-02-20"
TV_END_DATE = "2026-02-27"

INITIAL_CAPITAL = D("100000")
RISK_PER_TRADE = D("0.01")
SLIPPAGE_PCT = D("0.0002")

# Matching tolerance: trades within this many minutes are considered
# "same bar" for matching purposes.
MATCH_TOLERANCE_MINUTES = 15


# ══════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TVTrade:
    """A parsed TradingView trade (entry + exit paired)."""
    trade_num: int
    direction: str           # "LONG" or "SHORT"
    mode: str                # PB, PS, ORB, REV
    entry_datetime: str      # "YYYY-MM-DD HH:MM"
    exit_datetime: str
    entry_price: float
    exit_price: float
    pnl: float
    exit_signal: str         # TRAIL_TP, SX, LX, BE_TP, ORB_TRAIL, Flip, SL
    signal_metadata: str     # Raw signal string
    fav_excursion: float     # Favorable excursion USD
    adv_excursion: float     # Adverse excursion USD


@dataclass
class PyTrade:
    """A Python backtest trade."""
    direction: str           # "LONG" or "SHORT"
    mode: str                # PB, PS, ORB, REV
    entry_datetime: str
    exit_datetime: str
    entry_price: float
    exit_price: float
    pnl: float
    shares: int
    pnl_per_share: float     # PnL / shares for comparable sizing


@dataclass
class MatchedPair:
    """A matched TV+Python trade pair."""
    tv: TVTrade
    py: PyTrade
    entry_price_delta: float
    exit_price_delta: float
    pnl_delta_per_share: float
    tv_win: bool
    py_win: bool
    opposite_outcome: bool
    holding_bars_tv: float
    holding_bars_py: float


# ══════════════════════════════════════════════════════════════════════
# TV CSV parser
# ══════════════════════════════════════════════════════════════════════

def parse_tv_csv(csv_path: Path) -> list[TVTrade]:
    """Parse TradingView trade export CSV into paired trades."""
    entries: dict[int, dict] = {}
    exits: dict[int, dict] = {}

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trade_num = int(row["Trade #"])
            trade_type = row["Type"].strip()
            signal = row["Signal"].strip()
            price = float(row["Price USD"])
            pnl = float(row["Net P&L USD"])

            # Parse favorable/adverse excursion
            fav = float(row.get("Favorable excursion USD", 0) or 0)
            adv = float(row.get("Adverse excursion USD", 0) or 0)

            # Determine if entry or exit
            if "Entry" in trade_type:
                direction = "LONG" if "long" in trade_type.lower() else "SHORT"
                entries[trade_num] = {
                    "direction": direction,
                    "datetime": row["Date and time"].strip(),
                    "price": price,
                    "signal": signal,
                    "pnl": pnl,
                    "fav": fav,
                    "adv": adv,
                }
            elif "Exit" in trade_type or "Flip" in trade_type:
                exits[trade_num] = {
                    "datetime": row["Date and time"].strip(),
                    "price": price,
                    "signal": signal,
                    "pnl": pnl,
                    "fav": fav,
                    "adv": adv,
                }

    # Pair entries with exits
    trades: list[TVTrade] = []
    for num in sorted(entries.keys()):
        if num not in exits:
            continue
        entry = entries[num]
        ex = exits[num]

        # Parse mode from signal string
        sig = entry["signal"]
        mode = sig.split("|")[0] if "|" in sig else "UNKNOWN"

        trades.append(TVTrade(
            trade_num=num,
            direction=entry["direction"],
            mode=mode,
            entry_datetime=entry["datetime"],
            exit_datetime=ex["datetime"],
            entry_price=entry["price"],
            exit_price=ex["price"],
            pnl=ex["pnl"],
            exit_signal=ex["signal"],
            signal_metadata=entry["signal"],
            fav_excursion=ex["fav"],
            adv_excursion=ex["adv"],
        ))

    return trades


# ══════════════════════════════════════════════════════════════════════
# Python strategy runner
# ══════════════════════════════════════════════════════════════════════

def load_5m_bars(csv_path: Path) -> list[IntradayPriceData]:
    """Load SPY 5-minute OHLCV bars from CSV."""
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


def build_unified_strategy() -> UnifiedVWAPStrategy:
    """Build the unified strategy with TV-parity configs."""
    config = UnifiedVWAPConfig(
        max_day=2,
        spacing=3,
        be_trigger=D("0.5"),
        trail_buf=D("0.15"),
        trail_keep_tp=True,
        close_eod=True,
        circuit=3,
        day_loss=D("3.0"),
        adx_thresh=D("21"),
        allow_longs=True,
        allow_shorts=True,
    )

    pb_overrides = dict(
        pb_zone=D("0.5"), pb_body=D("0.15"), rr=D("1.4"),
        sl_atr=D("1.3"), sl_cap=D("1.50"), trend_bars=3,
        htf_align=True, ar_filter=True, ar_thresh=D("0.9"),
        ar_cap=D("1.15"), va_filter=True, va_min=D("-0.1"),
        cvd_long_filter=True, lrs_short_filter=True,
        lrs_thresh=D("0.08"), max_vxc=6,
        w_vol=3, w_sr=2, w_rsi=1, w_time=0, w_pq=1, w_box=0,
        min_score=3, min_score_long=5, pq_max_cross=3,
        no_friday_short=True, no_monday_long=True,
        trail_vwap=True, pb_vwap_bias=False, pb_tp_mode="rr",
    )

    ps_overrides = dict(
        ps_atr_pct=D("30.0"), ps_window=12, ps_rvol=D("1.0"),
        ps_engulf=D("0.35"), ps_sl_mode="Day Extreme",
        ps_sl_atr=D("1.5"), ps_sl_cap=D("2.50"),
        ps_tp_pct=D("75.0"), ps_sma_filter=False,
        ps_htf_align=False, ps_time_gate=False, ps_min_rr=D("0.3"),
    )

    orb_overrides = dict(
        orb_window=20, orb_rvol=D("2.0"),
        orb_sl_mode="OR Opposite", orb_sl_atr=D("1.5"),
        orb_sl_cap=D("2.50"), orb_vwap_align=True,
        orb_body_min=D("0.2"), orb_entry_mode="aggressive",
        orb_trail_atr=D("1.5"), orb_htf_align=False,
        orb_gap_filter=False, orb_adx_filter=False,
    )

    rev_overrides = dict(
        rev_band="2\u03c3", rev_body=D("0.20"),
        rev_sl_atr=D("1.0"), rev_sl_cap=D("1.50"),
        rev_shorts=False, rev_min_rr=D("0.3"),
        rev_tp_mode="vwap", rev_vwap_flat_tol=D("0.05"),
        rev_require_sr=False, rev_hug_limit=20,
        rev_can_trade_bar=11, min_score=3,
    )

    return UnifiedVWAPStrategy(
        config=config,
        pb_overrides=pb_overrides,
        ps_overrides=ps_overrides,
        orb_overrides=orb_overrides,
        rev_overrides=rev_overrides,
    )


def run_python_strategy(bars: list[IntradayPriceData]) -> list[PyTrade]:
    """Run the unified strategy and extract trade details."""
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        slippage_pct=SLIPPAGE_PCT,
        fixed_capital=True,
    )
    strategy = build_unified_strategy()
    result = engine.run(strategy, bars)

    py_trades: list[PyTrade] = []
    for i, trade in enumerate(result.closed_trades):
        mode = result.trade_modes[i] if i < len(result.trade_modes) else "UNK"
        shares = trade.shares
        pnl = float(trade.profit_loss)
        pnl_per_share = pnl / shares if shares > 0 else 0.0

        py_trades.append(PyTrade(
            direction=trade.direction.value,
            mode=mode,
            entry_datetime=trade.entry_date,
            exit_datetime=trade.exit_date or "",
            entry_price=float(trade.entry_price),
            exit_price=float(trade.exit_price) if trade.exit_price else 0.0,
            pnl=pnl,
            shares=shares,
            pnl_per_share=pnl_per_share,
        ))

    return py_trades


# ══════════════════════════════════════════════════════════════════════
# Trade matching
# ══════════════════════════════════════════════════════════════════════

def parse_dt(dt_str: str, *, to_et: bool = False) -> datetime:
    """Parse datetime string to datetime object.

    Parameters
    ----------
    to_et:
        If True, treat the input as Pacific Time and convert to ET
        (add 3 hours).  Used for TV CSV timestamps.
    """
    # TV format: "YYYY-MM-DD HH:MM" (Pacific Time, UTC-8)
    # Python format: "YYYY-MM-DD HH:MM:SS-05:00" (Eastern Time)
    dt_str = dt_str.replace("T", " ").strip()
    # Strip timezone offset if present (e.g., "-05:00", "+00:00")
    dt_str = re.sub(r'[+-]\d{2}:\d{2}$', '', dt_str)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(dt_str, fmt)
            if to_et:
                from datetime import timedelta
                dt = dt + timedelta(hours=3)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {dt_str}")


def match_trades(
    tv_trades: list[TVTrade],
    py_trades: list[PyTrade],
    tolerance_minutes: int = MATCH_TOLERANCE_MINUTES,
) -> tuple[list[MatchedPair], list[TVTrade], list[PyTrade]]:
    """Match TV trades to Python trades by date and approximate time.

    Returns (matched, unmatched_tv, unmatched_py).
    """
    matched: list[MatchedPair] = []
    used_py: set[int] = set()
    unmatched_tv: list[TVTrade] = []

    for tv in tv_trades:
        tv_dt = parse_dt(tv.entry_datetime, to_et=True)
        best_idx = -1
        best_delta = float("inf")

        for j, py in enumerate(py_trades):
            if j in used_py:
                continue
            # Must have same direction
            if py.direction != tv.direction:
                continue
            py_dt = parse_dt(py.entry_datetime)
            delta_min = abs((tv_dt - py_dt).total_seconds()) / 60.0
            if delta_min <= tolerance_minutes and delta_min < best_delta:
                best_delta = delta_min
                best_idx = j

        if best_idx >= 0:
            used_py.add(best_idx)
            py = py_trades[best_idx]

            # Compute deltas
            entry_price_delta = py.entry_price - tv.entry_price
            exit_price_delta = py.exit_price - tv.exit_price
            # Per-share PnL for TV (pnl is total)
            # TV qty is in the CSV but we use per-share for fair comparison
            tv_pnl_per_share = (tv.exit_price - tv.entry_price) if tv.direction == "LONG" else (tv.entry_price - tv.exit_price)
            pnl_delta_per_share = py.pnl_per_share - tv_pnl_per_share
            tv_win = tv.pnl > 0
            py_win = py.pnl > 0

            # Holding period in bars (5 min each)
            tv_entry_dt = parse_dt(tv.entry_datetime, to_et=True)
            tv_exit_dt = parse_dt(tv.exit_datetime, to_et=True)
            py_entry_dt = parse_dt(py.entry_datetime)
            py_exit_dt = parse_dt(py.exit_datetime)
            holding_bars_tv = max(1.0, (tv_exit_dt - tv_entry_dt).total_seconds() / 300.0)
            holding_bars_py = max(1.0, (py_exit_dt - py_entry_dt).total_seconds() / 300.0)

            matched.append(MatchedPair(
                tv=tv, py=py,
                entry_price_delta=entry_price_delta,
                exit_price_delta=exit_price_delta,
                pnl_delta_per_share=pnl_delta_per_share,
                tv_win=tv_win, py_win=py_win,
                opposite_outcome=(tv_win != py_win),
                holding_bars_tv=holding_bars_tv,
                holding_bars_py=holding_bars_py,
            ))
        else:
            unmatched_tv.append(tv)

    unmatched_py = [py for j, py in enumerate(py_trades) if j not in used_py]
    return matched, unmatched_tv, unmatched_py


# ══════════════════════════════════════════════════════════════════════
# Analysis and reporting
# ══════════════════════════════════════════════════════════════════════

def holding_bars_str(bars: float) -> str:
    """Format holding period in bars as human-readable string."""
    if bars < 12:
        return f"{bars:.0f} bars ({bars * 5:.0f} min)"
    hours = bars * 5 / 60
    return f"{bars:.0f} bars ({hours:.1f} hr)"


def analyze_and_report(
    matched: list[MatchedPair],
    unmatched_tv: list[TVTrade],
    unmatched_py: list[PyTrade],
    tv_trades: list[TVTrade],
    py_trades: list[PyTrade],
) -> None:
    """Print comprehensive analysis."""

    W = 90
    print("=" * W)
    print("EXIT COMPARISON: Python vs TradingView")
    print("=" * W)

    # ── 1. Overview ────────────────────────────────────────────────────
    print(f"\n{'OVERVIEW':^{W}}")
    print("-" * W)
    tv_wins = sum(1 for t in tv_trades if t.pnl > 0)
    py_wins = sum(1 for t in py_trades if t.pnl > 0)
    tv_wr = tv_wins / len(tv_trades) * 100 if tv_trades else 0
    py_wr = py_wins / len(py_trades) * 100 if py_trades else 0
    tv_total_pnl = sum(t.pnl for t in tv_trades)
    py_total_pnl = sum(t.pnl for t in py_trades)

    print(f"  TV trades:      {len(tv_trades):>6}  |  Win rate: {tv_wr:>5.1f}%  |  Total PnL: ${tv_total_pnl:>+10,.2f}")
    print(f"  Python trades:  {len(py_trades):>6}  |  Win rate: {py_wr:>5.1f}%  |  Total PnL: ${py_total_pnl:>+10,.2f}")
    print(f"  Matched pairs:  {len(matched):>6}")
    print(f"  Unmatched TV:   {len(unmatched_tv):>6}")
    print(f"  Unmatched Py:   {len(unmatched_py):>6}")

    # ── Mode breakdown ─────────────────────────────────────────────────
    print(f"\n{'MODE BREAKDOWN':^{W}}")
    print("-" * W)

    tv_modes = Counter(t.mode for t in tv_trades)
    py_modes = Counter(t.mode for t in py_trades)
    tv_mode_wins = Counter(t.mode for t in tv_trades if t.pnl > 0)
    py_mode_wins = Counter(t.mode for t in py_trades if t.pnl > 0)
    all_modes = sorted(set(list(tv_modes.keys()) + list(py_modes.keys())))

    print(f"  {'Mode':<8} {'TV Trades':>10} {'TV Wins':>8} {'TV WR%':>7} {'Py Trades':>10} {'Py Wins':>8} {'Py WR%':>7}")
    print(f"  {'-' * 60}")
    for mode in all_modes:
        tv_n = tv_modes.get(mode, 0)
        py_n = py_modes.get(mode, 0)
        tv_w = tv_mode_wins.get(mode, 0)
        py_w = py_mode_wins.get(mode, 0)
        tv_wr_m = tv_w / tv_n * 100 if tv_n else 0
        py_wr_m = py_w / py_n * 100 if py_n else 0
        print(f"  {mode:<8} {tv_n:>10} {tv_w:>8} {tv_wr_m:>6.1f}% {py_n:>10} {py_w:>8} {py_wr_m:>6.1f}%")

    if not matched:
        print("\n  NO MATCHED TRADES - cannot proceed with comparison.")
        return

    # ── 2. TV Exit Signal Distribution ─────────────────────────────────
    print(f"\n{'TV EXIT SIGNAL DISTRIBUTION':^{W}}")
    print("-" * W)

    tv_exit_counts = Counter(t.exit_signal for t in tv_trades)
    tv_exit_win_counts: dict[str, int] = defaultdict(int)
    tv_exit_pnl: dict[str, float] = defaultdict(float)
    for t in tv_trades:
        if t.pnl > 0:
            tv_exit_win_counts[t.exit_signal] += 1
        tv_exit_pnl[t.exit_signal] += t.pnl

    print(f"  {'Exit Signal':<14} {'Count':>6} {'Win':>5} {'WR%':>6} {'Total PnL':>12} {'Avg PnL':>10}")
    print(f"  {'-' * 55}")
    for sig, cnt in tv_exit_counts.most_common():
        wins = tv_exit_win_counts.get(sig, 0)
        wr = wins / cnt * 100 if cnt else 0
        total = tv_exit_pnl[sig]
        avg = total / cnt if cnt else 0
        print(f"  {sig:<14} {cnt:>6} {wins:>5} {wr:>5.1f}% ${total:>+10,.2f} ${avg:>+8,.2f}")

    # ── 3. Opposite outcome analysis ──────────────────────────────────
    print(f"\n{'OPPOSITE OUTCOME TRADES':^{W}}")
    print("-" * W)

    opposite = [m for m in matched if m.opposite_outcome]
    print(f"  Matched trades with opposite win/loss outcome: {len(opposite)} / {len(matched)}")

    if opposite:
        tv_win_py_lose = [m for m in opposite if m.tv_win and not m.py_win]
        py_win_tv_lose = [m for m in opposite if m.py_win and not m.tv_win]

        print(f"    TV wins, Python loses: {len(tv_win_py_lose)}")
        print(f"    Python wins, TV loses: {len(py_win_tv_lose)}")

        if tv_win_py_lose:
            print(f"\n  === TV WINS but PYTHON LOSES ({len(tv_win_py_lose)} trades) ===")
            avg_tv_pnl = sum(m.tv.pnl for m in tv_win_py_lose) / len(tv_win_py_lose)
            avg_py_pnl = sum(m.py.pnl for m in tv_win_py_lose) / len(tv_win_py_lose)
            avg_exit_delta = sum(abs(m.exit_price_delta) for m in tv_win_py_lose) / len(tv_win_py_lose)
            print(f"    Avg TV PnL (total):    ${avg_tv_pnl:>+10,.2f}")
            print(f"    Avg Py PnL (total):    ${avg_py_pnl:>+10,.2f}")
            print(f"    Avg |exit price delta|: ${avg_exit_delta:>.4f}")

            # Mode breakdown
            opp_modes = Counter(m.tv.mode for m in tv_win_py_lose)
            print(f"    By mode: {dict(opp_modes)}")

            # Exit signal breakdown
            opp_exits = Counter(m.tv.exit_signal for m in tv_win_py_lose)
            print(f"    TV exit signals: {dict(opp_exits)}")

            print(f"\n  Top 10 TV-win/Py-lose trades:")
            sorted_twpl = sorted(tv_win_py_lose, key=lambda m: m.tv.pnl, reverse=True)
            print(f"    {'Date':<18} {'Mode':<5} {'Dir':<6} {'TV Entry':>9} {'TV Exit':>9} {'Py Exit':>9} {'TV PnL':>10} {'Py PnL':>10} {'TV Exit Sig':<12} {'TV Bars':>7} {'Py Bars':>7}")
            print(f"    {'-' * 115}")
            for m in sorted_twpl[:10]:
                print(f"    {m.tv.entry_datetime:<18} {m.tv.mode:<5} {m.tv.direction:<6} "
                      f"${m.tv.entry_price:>8.2f} ${m.tv.exit_price:>8.2f} ${m.py.exit_price:>8.2f} "
                      f"${m.tv.pnl:>+9.2f} ${m.py.pnl:>+9.2f} {m.tv.exit_signal:<12} "
                      f"{m.holding_bars_tv:>6.0f} {m.holding_bars_py:>6.0f}")

        if py_win_tv_lose:
            print(f"\n  === PYTHON WINS but TV LOSES ({len(py_win_tv_lose)} trades) ===")
            avg_tv_pnl = sum(m.tv.pnl for m in py_win_tv_lose) / len(py_win_tv_lose)
            avg_py_pnl = sum(m.py.pnl for m in py_win_tv_lose) / len(py_win_tv_lose)
            print(f"    Avg TV PnL (total):   ${avg_tv_pnl:>+10,.2f}")
            print(f"    Avg Py PnL (total):   ${avg_py_pnl:>+10,.2f}")

    # ── 4. PnL comparison on matched trades ────────────────────────────
    print(f"\n{'PNL COMPARISON ON MATCHED TRADES':^{W}}")
    print("-" * W)

    # All matched
    matched_tv_pnl = sum(m.tv.pnl for m in matched)
    matched_py_pnl = sum(m.py.pnl for m in matched)
    avg_pnl_delta = sum(m.pnl_delta_per_share for m in matched) / len(matched)

    print(f"  Matched TV total PnL:  ${matched_tv_pnl:>+12,.2f}")
    print(f"  Matched Py total PnL:  ${matched_py_pnl:>+12,.2f}")
    print(f"  Delta (Py - TV):       ${matched_py_pnl - matched_tv_pnl:>+12,.2f}")
    print(f"  Avg per-share PnL delta: ${avg_pnl_delta:>+.4f}")

    # On matched WINNERS (both win)
    both_win = [m for m in matched if m.tv_win and m.py_win]
    if both_win:
        avg_tv_win_pps = sum((m.tv.exit_price - m.tv.entry_price) if m.tv.direction == "LONG" else (m.tv.entry_price - m.tv.exit_price) for m in both_win) / len(both_win)
        avg_py_win_pps = sum(m.py.pnl_per_share for m in both_win) / len(both_win)
        print(f"\n  Both-win trades ({len(both_win)}):")
        print(f"    Avg TV win/share:  ${avg_tv_win_pps:>+.4f}")
        print(f"    Avg Py win/share:  ${avg_py_win_pps:>+.4f}")
        print(f"    Delta:             ${avg_py_win_pps - avg_tv_win_pps:>+.4f}")

    # On matched LOSERS (both lose)
    both_lose = [m for m in matched if not m.tv_win and not m.py_win]
    if both_lose:
        avg_tv_lose_pps = sum((m.tv.exit_price - m.tv.entry_price) if m.tv.direction == "LONG" else (m.tv.entry_price - m.tv.exit_price) for m in both_lose) / len(both_lose)
        avg_py_lose_pps = sum(m.py.pnl_per_share for m in both_lose) / len(both_lose)
        print(f"\n  Both-lose trades ({len(both_lose)}):")
        print(f"    Avg TV loss/share: ${avg_tv_lose_pps:>+.4f}")
        print(f"    Avg Py loss/share: ${avg_py_lose_pps:>+.4f}")
        print(f"    Delta:             ${avg_py_lose_pps - avg_tv_lose_pps:>+.4f}")

    # ── 5. Holding period analysis ─────────────────────────────────────
    print(f"\n{'HOLDING PERIOD ANALYSIS':^{W}}")
    print("-" * W)

    avg_tv_bars = sum(m.holding_bars_tv for m in matched) / len(matched)
    avg_py_bars = sum(m.holding_bars_py for m in matched) / len(matched)
    py_exits_earlier = sum(1 for m in matched if m.holding_bars_py < m.holding_bars_tv)
    py_exits_later = sum(1 for m in matched if m.holding_bars_py > m.holding_bars_tv)
    py_exits_same = sum(1 for m in matched if m.holding_bars_py == m.holding_bars_tv)

    print(f"  Avg TV holding:    {holding_bars_str(avg_tv_bars)}")
    print(f"  Avg Py holding:    {holding_bars_str(avg_py_bars)}")
    print(f"  Delta (Py - TV):   {avg_py_bars - avg_tv_bars:>+.1f} bars ({(avg_py_bars - avg_tv_bars) * 5:>+.0f} min)")
    print(f"  Python exits EARLIER: {py_exits_earlier:>4} ({py_exits_earlier/len(matched)*100:.1f}%)")
    print(f"  Python exits LATER:   {py_exits_later:>4} ({py_exits_later/len(matched)*100:.1f}%)")
    print(f"  Same bar exit:        {py_exits_same:>4} ({py_exits_same/len(matched)*100:.1f}%)")

    # Holding period by TV exit signal
    print(f"\n  Holding period by TV exit signal:")
    by_exit_sig: dict[str, list[MatchedPair]] = defaultdict(list)
    for m in matched:
        by_exit_sig[m.tv.exit_signal].append(m)

    print(f"    {'Exit Signal':<14} {'N':>4} {'Avg TV Bars':>11} {'Avg Py Bars':>11} {'Delta':>8} {'Py Earlier%':>12}")
    print(f"    {'-' * 65}")
    for sig in sorted(by_exit_sig.keys()):
        pairs = by_exit_sig[sig]
        n = len(pairs)
        avg_tv = sum(m.holding_bars_tv for m in pairs) / n
        avg_py = sum(m.holding_bars_py for m in pairs) / n
        earlier_pct = sum(1 for m in pairs if m.holding_bars_py < m.holding_bars_tv) / n * 100
        print(f"    {sig:<14} {n:>4} {avg_tv:>10.1f} {avg_py:>10.1f} {avg_py - avg_tv:>+7.1f} {earlier_pct:>10.1f}%")

    # ── 6. Entry price comparison ──────────────────────────────────────
    print(f"\n{'ENTRY PRICE COMPARISON':^{W}}")
    print("-" * W)

    avg_entry_delta = sum(m.entry_price_delta for m in matched) / len(matched)
    avg_abs_entry_delta = sum(abs(m.entry_price_delta) for m in matched) / len(matched)
    max_entry_delta = max(abs(m.entry_price_delta) for m in matched)

    print(f"  Avg entry price delta (Py - TV):      ${avg_entry_delta:>+.4f}")
    print(f"  Avg |entry price delta|:               ${avg_abs_entry_delta:>.4f}")
    print(f"  Max |entry price delta|:               ${max_entry_delta:>.4f}")

    # ── 7. Exit price comparison ───────────────────────────────────────
    print(f"\n{'EXIT PRICE COMPARISON':^{W}}")
    print("-" * W)

    avg_exit_delta = sum(m.exit_price_delta for m in matched) / len(matched)
    avg_abs_exit_delta = sum(abs(m.exit_price_delta) for m in matched) / len(matched)
    max_exit_delta = max(abs(m.exit_price_delta) for m in matched)

    print(f"  Avg exit price delta (Py - TV):       ${avg_exit_delta:>+.4f}")
    print(f"  Avg |exit price delta|:                ${avg_abs_exit_delta:>.4f}")
    print(f"  Max |exit price delta|:                ${max_exit_delta:>.4f}")

    # ── 8. PB-mode deep dive (biggest gap) ─────────────────────────────
    print(f"\n{'PB MODE DEEP DIVE':^{W}}")
    print("-" * W)

    pb_matched = [m for m in matched if m.tv.mode == "PB"]
    if pb_matched:
        pb_tv_wins = sum(1 for m in pb_matched if m.tv_win)
        pb_py_wins = sum(1 for m in pb_matched if m.py_win)
        pb_opposite = sum(1 for m in pb_matched if m.opposite_outcome)
        pb_tv_pnl = sum(m.tv.pnl for m in pb_matched)
        pb_py_pnl = sum(m.py.pnl for m in pb_matched)

        print(f"  Matched PB trades: {len(pb_matched)}")
        print(f"  TV PB wins:   {pb_tv_wins}/{len(pb_matched)} = {pb_tv_wins/len(pb_matched)*100:.1f}%")
        print(f"  Py PB wins:   {pb_py_wins}/{len(pb_matched)} = {pb_py_wins/len(pb_matched)*100:.1f}%")
        print(f"  Opposite outcome: {pb_opposite}")
        print(f"  TV PB PnL:    ${pb_tv_pnl:>+10,.2f}")
        print(f"  Py PB PnL:    ${pb_py_pnl:>+10,.2f}")

        # PB exit signal breakdown for opposite outcomes
        pb_opp = [m for m in pb_matched if m.opposite_outcome and m.tv_win]
        if pb_opp:
            print(f"\n  PB trades where TV wins but Python loses ({len(pb_opp)}):")
            pb_opp_exits = Counter(m.tv.exit_signal for m in pb_opp)
            print(f"    TV exit signals: {dict(pb_opp_exits)}")

            avg_pb_tv_hold = sum(m.holding_bars_tv for m in pb_opp) / len(pb_opp)
            avg_pb_py_hold = sum(m.holding_bars_py for m in pb_opp) / len(pb_opp)
            print(f"    Avg TV holding: {holding_bars_str(avg_pb_tv_hold)}")
            print(f"    Avg Py holding: {holding_bars_str(avg_pb_py_hold)}")

            print(f"\n    Detail:")
            print(f"    {'Date':<18} {'Dir':<6} {'TV Entry':>9} {'TV Exit':>9} {'Py Exit':>9} {'TV PnL':>10} {'Py PnL':>10} {'TV Sig':<12}")
            print(f"    {'-' * 95}")
            for m in sorted(pb_opp, key=lambda m: m.tv.pnl, reverse=True):
                print(f"    {m.tv.entry_datetime:<18} {m.tv.direction:<6} "
                      f"${m.tv.entry_price:>8.2f} ${m.tv.exit_price:>8.2f} ${m.py.exit_price:>8.2f} "
                      f"${m.tv.pnl:>+9.2f} ${m.py.pnl:>+9.2f} {m.tv.exit_signal:<12}")
    else:
        print("  No matched PB trades.")

    # ── 9. Unmatched trade analysis ────────────────────────────────────
    print(f"\n{'UNMATCHED TRADES':^{W}}")
    print("-" * W)

    if unmatched_tv:
        print(f"\n  TV trades with NO Python match ({len(unmatched_tv)}):")
        um_tv_modes = Counter(t.mode for t in unmatched_tv)
        um_tv_pnl = sum(t.pnl for t in unmatched_tv)
        um_tv_wins = sum(1 for t in unmatched_tv if t.pnl > 0)
        print(f"    By mode: {dict(um_tv_modes)}")
        print(f"    Total PnL: ${um_tv_pnl:>+10,.2f}")
        print(f"    Wins: {um_tv_wins}/{len(unmatched_tv)}")
        print(f"\n    {'Date':<18} {'Mode':<5} {'Dir':<6} {'Entry':>9} {'Exit':>9} {'PnL':>10} {'Exit Sig':<12}")
        print(f"    {'-' * 75}")
        for t in unmatched_tv[:15]:
            print(f"    {t.entry_datetime:<18} {t.mode:<5} {t.direction:<6} "
                  f"${t.entry_price:>8.2f} ${t.exit_price:>8.2f} ${t.pnl:>+9.2f} {t.exit_signal:<12}")
        if len(unmatched_tv) > 15:
            print(f"    ... and {len(unmatched_tv) - 15} more")

    if unmatched_py:
        print(f"\n  Python trades with NO TV match ({len(unmatched_py)}):")
        um_py_modes = Counter(t.mode for t in unmatched_py)
        um_py_pnl = sum(t.pnl for t in unmatched_py)
        um_py_wins = sum(1 for t in unmatched_py if t.pnl > 0)
        print(f"    By mode: {dict(um_py_modes)}")
        print(f"    Total PnL: ${um_py_pnl:>+10,.2f}")
        print(f"    Wins: {um_py_wins}/{len(unmatched_py)}")
        print(f"\n    {'Date':<18} {'Mode':<5} {'Dir':<6} {'Entry':>9} {'Exit':>9} {'PnL':>10}")
        print(f"    {'-' * 65}")
        for t in unmatched_py[:15]:
            print(f"    {t.entry_datetime:<18} {t.mode:<5} {t.direction:<6} "
                  f"${t.entry_price:>8.2f} ${t.exit_price:>8.2f} ${t.pnl:>+9.2f}")
        if len(unmatched_py) > 15:
            print(f"    ... and {len(unmatched_py) - 15} more")

    # ── 10. Win-rate impact decomposition ──────────────────────────────
    print(f"\n{'WIN-RATE GAP DECOMPOSITION':^{W}}")
    print("-" * W)

    matched_tv_wr = sum(1 for m in matched if m.tv_win) / len(matched) * 100
    matched_py_wr = sum(1 for m in matched if m.py_win) / len(matched) * 100
    wr_gap = matched_py_wr - matched_tv_wr

    tv_win_py_lose_count = sum(1 for m in matched if m.tv_win and not m.py_win)
    py_win_tv_lose_count = sum(1 for m in matched if m.py_win and not m.tv_win)
    net_flips = tv_win_py_lose_count - py_win_tv_lose_count

    print(f"  Matched TV win rate:  {matched_tv_wr:.1f}%")
    print(f"  Matched Py win rate:  {matched_py_wr:.1f}%")
    print(f"  Win-rate gap:         {wr_gap:+.1f}%")
    print(f"  Net win->loss flips:  {net_flips} (TV wins that Python loses - vice versa)")

    # What fraction of the WR gap is explained by exit differences?
    if net_flips > 0:
        wr_from_flips = net_flips / len(matched) * 100
        print(f"  WR impact from flips: {wr_from_flips:+.1f}% "
              f"(explains {abs(wr_from_flips / wr_gap * 100):.0f}% of gap)" if wr_gap != 0 else "")

    # ── 11. PnL Gap Decomposition ──────────────────────────────────────
    print(f"\n{'PNL GAP DECOMPOSITION':^{W}}")
    print("-" * W)

    total_gap = tv_total_pnl - py_total_pnl
    print(f"  Total PnL gap (TV - Python): ${total_gap:>+12,.2f}")
    print()

    # Source 1: unmatched TV trades (trades TV takes that Python doesn't)
    um_tv_pnl_total = sum(t.pnl for t in unmatched_tv)
    um_tv_win_pnl = sum(t.pnl for t in unmatched_tv if t.pnl > 0)
    um_tv_lose_pnl = sum(t.pnl for t in unmatched_tv if t.pnl <= 0)
    print(f"  1. Unmatched TV trades (TV takes, Python skips): ${um_tv_pnl_total:>+10,.2f}")
    print(f"       Winners: {sum(1 for t in unmatched_tv if t.pnl > 0)} trades, ${um_tv_win_pnl:>+10,.2f}")
    print(f"       Losers:  {sum(1 for t in unmatched_tv if t.pnl <= 0)} trades, ${um_tv_lose_pnl:>+10,.2f}")

    # Source 2: unmatched Python trades (trades Python takes that TV doesn't)
    um_py_pnl_total = sum(t.pnl for t in unmatched_py)
    um_py_win_pnl = sum(t.pnl for t in unmatched_py if t.pnl > 0)
    um_py_lose_pnl = sum(t.pnl for t in unmatched_py if t.pnl <= 0)
    print(f"  2. Unmatched Python trades (Python takes, TV skips): ${um_py_pnl_total:>+10,.2f}")
    print(f"       Winners: {sum(1 for t in unmatched_py if t.pnl > 0)} trades, ${um_py_win_pnl:>+10,.2f}")
    print(f"       Losers:  {sum(1 for t in unmatched_py if t.pnl <= 0)} trades, ${um_py_lose_pnl:>+10,.2f}")

    # Source 3: matched trade PnL difference
    matched_gap = matched_tv_pnl - matched_py_pnl
    print(f"  3. Matched trade PnL gap (TV better on exits): ${matched_gap:>+10,.2f}")

    explained = um_tv_pnl_total - um_py_pnl_total + matched_gap
    print(f"\n  Sum: ${um_tv_pnl_total:>+10,.2f} (missed TV) "
          f"+ ${-um_py_pnl_total:>+10,.2f} (extra Py drag) "
          f"+ ${matched_gap:>+10,.2f} (exit gap) = ${explained:>+10,.2f}")
    print(f"  Actual gap: ${total_gap:>+10,.2f}  (check: {'OK' if abs(explained - total_gap) < 1 else 'MISMATCH'})")

    pct_from_missed = abs(um_tv_pnl_total) / total_gap * 100 if total_gap != 0 else 0
    pct_from_extra = abs(um_py_pnl_total) / total_gap * 100 if total_gap != 0 else 0
    pct_from_exits = abs(matched_gap) / total_gap * 100 if total_gap != 0 else 0
    print(f"\n  Contribution to gap:")
    print(f"    Missed TV trades:    {pct_from_missed:>5.1f}%")
    print(f"    Extra Python trades: {pct_from_extra:>5.1f}%")
    print(f"    Exit management:     {pct_from_exits:>5.1f}%")

    # ── 12. Summary / Diagnosis ────────────────────────────────────────
    print(f"\n{'DIAGNOSIS SUMMARY':^{W}}")
    print("=" * W)

    # Check if Python exits are systematically different
    entry_same_05 = sum(1 for m in matched if abs(m.entry_price_delta) < 0.05) / len(matched) * 100
    entry_same_25 = sum(1 for m in matched if abs(m.entry_price_delta) < 0.25) / len(matched) * 100
    exit_diff_10 = sum(1 for m in matched if abs(m.exit_price_delta) > 0.10) / len(matched) * 100
    exit_diff_50 = sum(1 for m in matched if abs(m.exit_price_delta) > 0.50) / len(matched) * 100

    print(f"  Entry prices within $0.05: {entry_same_05:.0f}% of matched trades")
    print(f"  Entry prices within $0.25: {entry_same_25:.0f}% of matched trades")
    print(f"  Exit prices differ >$0.10: {exit_diff_10:.0f}% of matched trades")
    print(f"  Exit prices differ >$0.50: {exit_diff_50:.0f}% of matched trades")

    if avg_py_bars > avg_tv_bars + 2:
        print(f"  --> Python holds {avg_py_bars - avg_tv_bars:.1f} bars LONGER on average")
        print(f"      This suggests Python trail/exit triggers are SLOWER to fire.")
    elif avg_py_bars < avg_tv_bars - 2:
        print(f"  --> Python exits {avg_tv_bars - avg_py_bars:.1f} bars EARLIER on average")
        print(f"      This suggests Python trail/exit triggers are TOO AGGRESSIVE.")
    else:
        print(f"  --> Holding periods are similar (delta = {avg_py_bars - avg_tv_bars:+.1f} bars)")

    # Trail TP analysis
    trail_tp_matched = [m for m in matched if m.tv.exit_signal == "TRAIL_TP"]
    if trail_tp_matched:
        trail_tp_py_wins = sum(1 for m in trail_tp_matched if m.py_win)
        trail_tp_tv_wins = sum(1 for m in trail_tp_matched if m.tv_win)
        print(f"\n  TRAIL_TP trades (TV's trailing profit exit):")
        print(f"    N={len(trail_tp_matched)}, TV wins={trail_tp_tv_wins}, Py wins={trail_tp_py_wins}")
        if trail_tp_tv_wins > trail_tp_py_wins:
            print(f"    --> Python misses {trail_tp_tv_wins - trail_tp_py_wins} wins that TV captures via trailing profit.")
            print(f"        This is {(trail_tp_tv_wins - trail_tp_py_wins)/len(matched)*100:.1f}% of all matched trades.")

    print("\n" + "=" * W)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    if not BARS_CSV.exists():
        print(f"ERROR: SPY 5m data not found at {BARS_CSV}")
        sys.exit(1)
    if not TV_CSV.exists():
        print(f"ERROR: TV CSV not found at {TV_CSV}")
        sys.exit(1)

    # ── Load TV trades ─────────────────────────────────────────────────
    print("Loading TV trades...")
    tv_trades = parse_tv_csv(TV_CSV)
    print(f"  Parsed {len(tv_trades)} TV trades")

    # ── Load 5m bars and filter to TV range ────────────────────────────
    print("Loading SPY 5m bars...")
    all_bars = load_5m_bars(BARS_CSV)
    bars = [b for b in all_bars if TV_START_DATE <= b.date[:10] <= TV_END_DATE]
    print(f"  Loaded {len(all_bars)} bars total, filtered to {len(bars)} "
          f"({bars[0].date[:10]} to {bars[-1].date[:10]})")

    # ── Run Python strategy ────────────────────────────────────────────
    print("Running Python unified VWAP strategy...")
    py_trades = run_python_strategy(bars)
    print(f"  Got {len(py_trades)} Python trades")

    # ── Match trades ───────────────────────────────────────────────────
    print("Matching trades...")
    matched, unmatched_tv, unmatched_py = match_trades(tv_trades, py_trades)
    print(f"  Matched: {len(matched)}, Unmatched TV: {len(unmatched_tv)}, Unmatched Py: {len(unmatched_py)}")

    # ── Analyze ────────────────────────────────────────────────────────
    analyze_and_report(matched, unmatched_tv, unmatched_py, tv_trades, py_trades)


if __name__ == "__main__":
    main()
