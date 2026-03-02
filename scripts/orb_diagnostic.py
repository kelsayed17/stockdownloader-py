"""ORB Diagnostic — instrument every gate on TV ORB trade dates.

Loads SPY 5m bars, runs the ORB strategy on the 6 dates where
TradingView produced ORB entries, and reports exactly which gate
rejects the entry on those bars.

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/orb_diagnostic.py
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

D = Decimal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.strategies.intraday.or_breakout import (
    ORBreakoutStrategy,
    ORBreakoutStrategyConfig,
)
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trade_mgmt import IntradayExitManager
from stockdownloader.strategies.intraday.trail import AtrChandelierTrail
from stockdownloader.strategies.intraday.session import BarContext, SessionState
from stockdownloader.indicators.intraday import candle_strength
from stockdownloader.core.math import ZERO

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"

# ── TV ORB trades (parsed from CSV) ──────────────────────────────────
# Format: (date_str, time_str_PT, direction, rvol, adx, htf_trend, signal_str)
TV_ORB_TRADES = [
    # Trade 3: 2025-02-21 07:05 PT → 10:05 ET = bar 8 from 09:30
    {
        "trade_id": 3,
        "date": "2025-02-21",
        "time_pt": "07:05",
        "time_et": "10:05",
        "direction": "short",
        "rvol": D("2.16"),
        "adx": D("21.5"),
        "htf": 1,
        "td": -2,
        "orb_level": "ORL",
        "ord_dist": D("0.69"),
        "price": D("608.16"),
    },
    # Trade 7: 2025-02-27 07:05 PT → 10:05 ET = bar 8
    {
        "trade_id": 7,
        "date": "2025-02-27",
        "time_pt": "07:05",
        "time_et": "10:05",
        "direction": "short",
        "rvol": D("2.65"),
        "adx": D("22.7"),
        "htf": -1,
        "td": 0,
        "orb_level": "ORL",
        "ord_dist": D("3.21"),
        "price": D("591.87"),
    },
    # Trade 44: 2025-08-20 06:55 PT → 09:55 ET = bar 6
    {
        "trade_id": 44,
        "date": "2025-08-20",
        "time_pt": "06:55",
        "time_et": "09:55",
        "direction": "short",
        "rvol": D("2.64"),
        "adx": D("28.8"),
        "htf": -1,
        "td": -2,
        "orb_level": "ORL",
        "ord_dist": D("2.60"),
        "price": D("635.48"),
    },
    # Trade 52: 2025-10-07 08:10 PT → 11:10 ET = bar 21
    {
        "trade_id": 52,
        "date": "2025-10-07",
        "time_pt": "08:10",
        "time_et": "11:10",
        "direction": "short",
        "rvol": D("2.87"),
        "adx": D("26.6"),
        "htf": -1,
        "td": -2,
        "orb_level": "ORL",
        "ord_dist": D("3.29"),
        "price": D("668.90"),
    },
    # Trade 100: 2026-02-20 07:05 PT → 10:05 ET = bar 8
    {
        "trade_id": 100,
        "date": "2026-02-20",
        "time_pt": "07:05",
        "time_et": "10:05",
        "direction": "long",
        "rvol": D("3.00"),
        "adx": D("17.2"),
        "htf": 1,
        "td": 2,
        "orb_level": "ORH",
        "ord_dist": D("1.61"),
        "price": D("687.02"),
    },
    # Trade 104: 2026-02-26 07:25 PT → 10:25 ET = bar 12
    {
        "trade_id": 104,
        "date": "2026-02-26",
        "time_pt": "07:25",
        "time_et": "10:25",
        "direction": "short",
        "rvol": D("2.34"),
        "adx": D("45.8"),
        "htf": -1,
        "td": -2,
        "orb_level": "ORL",
        "ord_dist": D("5.58"),
        "price": D("685.32"),
    },
]


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


def make_config() -> ORBreakoutStrategyConfig:
    """TV-parity ORB config from scripts/tv_parity_backtest.py."""
    return ORBreakoutStrategyConfig(
        allow_shorts=True,
        allow_longs=True,
        orb_window=20,
        orb_rvol=D("2.0"),
        orb_sl_mode="OR Opposite",
        orb_sl_atr=D("1.5"),
        orb_sl_cap=D("2.50"),
        orb_vwap_align=True,
        orb_body_min=D("0.2"),
        orb_entry_mode="aggressive",
        orb_trail_atr=D("1.5"),
        orb_htf_align=True,
        orb_gap_filter=True,
        be_trigger=D("0.5"),
        adx_thresh=D("21"),
        close_eod=True,
        circuit=3,
        day_loss=D("3.0"),
    )


def et_bar_number(time_et: str) -> int:
    """Convert ET time string (HH:MM) to bar_of_day (1-indexed from 09:30)."""
    h, m = map(int, time_et.split(":"))
    total_min = h * 60 + m
    open_min = 9 * 60 + 30  # 09:30 ET
    return (total_min - open_min) // 5 + 1


def diagnose_date(
    bars: list[IntradayPriceData],
    trade: dict,
    config: ORBreakoutStrategyConfig,
) -> None:
    """Run ORB gate-by-gate diagnosis for a single TV trade date."""
    target_date = trade["date"]
    expected_bar = et_bar_number(trade["time_et"])

    print(f"\n{'='*80}")
    print(f"TV Trade #{trade['trade_id']}: {target_date} {trade['time_et']} ET "
          f"(bar {expected_bar}) | {trade['direction'].upper()} | "
          f"Price={trade['price']} | RVOL={trade['rvol']} | ADX={trade['adx']} | "
          f"HTF={trade['htf']}")
    print(f"  ORB Level: {trade['orb_level']} | OR Dist: {trade['ord_dist']}")
    print(f"{'='*80}")

    # Filter bars to include warmup + target date
    target_bars = [b for b in bars if b.date[:10] <= target_date]
    if not target_bars:
        print(f"  ERROR: No bars found for date {target_date}")
        return

    # Build infra manually to instrument gates
    infra = IntradayInfra(config, IntradayExitManager(AtrChandelierTrail()))

    # Process all bars up to and including target date
    target_day_bars = []
    for i, bar in enumerate(target_bars):
        if bar.date[:10] == target_date:
            target_day_bars.append((i, bar))

        # Let infra process for warmup/state
        ctx = infra.on_new_bar(target_bars, i)

        # Only instrument gates on the target date
        if bar.date[:10] != target_date:
            # Still need to handle exits for position tracking
            if ctx is None and infra.state.in_position:
                infra.evaluate_exit(target_bars, i)
            continue

        state = infra.state
        bar_of_day = state.bar_count

        # Print OR building info for first few bars
        if bar_of_day <= config.or_bars:
            print(f"  Bar {bar_of_day:>2} ({bar.date[11:16]}) OR building: "
                  f"H={bar.high} L={bar.low} | "
                  f"OR_H={state.or_high} OR_L={state.or_low} or_done={state.or_done}")
            continue

        if bar_of_day == config.or_bars + 1:
            print(f"  --- OR Complete: H={state.or_high} L={state.or_low} "
                  f"Range={state.or_range} Dir={state.or_dir} ---")

        # If in position, skip entry analysis
        if ctx is None:
            if bar_of_day >= expected_bar - 1 and bar_of_day <= expected_bar + 1:
                print(f"  Bar {bar_of_day:>2} ({bar.date[11:16]}) SKIPPED - in position "
                      f"(mode={state.entry_mode})")
            continue

        # Only detail bars near the expected entry
        if bar_of_day < expected_bar - 2 or bar_of_day > expected_bar + 2:
            # But still check if ORB would fire on any bar in window
            if bar_of_day > config.or_bars and bar_of_day <= config.orb_window:
                # Quick check
                close_long = bar.close > state.or_high and bar.close > bar.open
                close_short = bar.close < state.or_low and bar.close < bar.open
                if close_long or close_short:
                    print(f"  Bar {bar_of_day:>2} ({bar.date[11:16]}) "
                          f"Close beyond OR! close={bar.close} "
                          f"{'LONG' if close_long else 'SHORT'} "
                          f"(but detailed analysis only near expected bar)")
            continue

        # ── Detailed gate-by-gate analysis ──
        print(f"\n  Bar {bar_of_day:>2} ({bar.date[11:16]}) DETAILED GATE ANALYSIS:")
        print(f"    Price: O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume}")

        # Gate 1: orb_enable
        g1 = config.orb_enable
        print(f"    [{'PASS' if g1 else 'FAIL'}] orb_enable = {config.orb_enable}")

        # Gate 2: fired_today
        g2 = not state.fired_today
        print(f"    [{'PASS' if g2 else 'FAIL'}] not fired_today = {not state.fired_today}")

        # Gate 3: or_done
        g3 = state.or_done
        print(f"    [{'PASS' if g3 else 'FAIL'}] or_done = {state.or_done}")

        # Gate 4: Window check (Python: bar_of_day > or_bars AND bar_of_day <= orb_window)
        g4_or_bars = bar_of_day > config.or_bars
        g4_window = bar_of_day <= config.orb_window
        g4 = g4_or_bars and g4_window
        print(f"    [{'PASS' if g4 else 'FAIL'}] window: bar_of_day({bar_of_day}) > or_bars({config.or_bars})={g4_or_bars} "
              f"AND bar_of_day({bar_of_day}) <= orb_window({config.orb_window})={g4_window}")

        # Gate 5: is_good_time (Python ONLY — PineScript ORB does NOT have this gate!)
        is_good_time = ctx.is_good_time
        g5_can_trade = bar_of_day >= config.can_trade_bar
        g5_eod = bar_of_day <= config.eod_bar
        g5_lunch = not (config.lunch_start <= bar_of_day <= config.lunch_end)
        print(f"    [{'PASS' if is_good_time else '** FAIL **'}] is_good_time (PYTHON-ONLY gate!):")
        print(f"      bar_of_day({bar_of_day}) >= can_trade_bar({config.can_trade_bar}) = {g5_can_trade}")
        print(f"      bar_of_day({bar_of_day}) <= eod_bar({config.eod_bar}) = {g5_eod}")
        print(f"      NOT lunch({config.lunch_start}-{config.lunch_end}) = {g5_lunch}")
        if not is_good_time:
            print(f"      >>> PineScript ORB does NOT check is_good_time/canTrade!")

        # Gate 6: Close beyond OR
        close_long = bar.close > state.or_high and bar.close > bar.open
        close_short = bar.close < state.or_low and bar.close < bar.open
        g6 = close_long or close_short
        direction = "LONG" if close_long else ("SHORT" if close_short else "NONE")
        print(f"    [{'PASS' if g6 else 'FAIL'}] close_beyond_OR: {direction}")
        print(f"      close({bar.close}) > or_high({state.or_high}) = {bar.close > state.or_high}")
        print(f"      close({bar.close}) < or_low({state.or_low}) = {bar.close < state.or_low}")
        print(f"      close({bar.close}) > open({bar.open}) = {bar.close > bar.open} (bullish)")
        print(f"      close({bar.close}) < open({bar.open}) = {bar.close < bar.open} (bearish)")

        # Gate 7: Body filter
        cs = candle_strength(bar, ctx.atr_val)
        g7 = cs.body_atr >= config.orb_body_min
        print(f"    [{'PASS' if g7 else 'FAIL'}] body_filter: body_atr({cs.body_atr:.4f}) >= "
              f"orb_body_min({config.orb_body_min})")

        # Gate 8: RVOL
        g8 = ctx.rel_vol >= config.orb_rvol
        print(f"    [{'PASS' if g8 else 'FAIL'}] rvol: rel_vol({ctx.rel_vol:.4f}) >= "
              f"orb_rvol({config.orb_rvol}) | TV says: {trade['rvol']}")

        # Gate 9: VWAP alignment
        vwap = ctx.vwap_bands.vwap
        if config.orb_vwap_align:
            if close_long:
                g9 = bar.close > vwap
                print(f"    [{'PASS' if g9 else 'FAIL'}] vwap_align (long): "
                      f"close({bar.close}) > vwap({vwap:.4f})")
            elif close_short:
                g9 = bar.close < vwap
                print(f"    [{'PASS' if g9 else 'FAIL'}] vwap_align (short): "
                      f"close({bar.close}) < vwap({vwap:.4f})")
            else:
                g9 = True
                print(f"    [SKIP] vwap_align: no close beyond OR")
        else:
            g9 = True
            print(f"    [SKIP] vwap_align: disabled")

        # Gate 10: Direction allowed
        go_long = close_long and config.allow_longs
        go_short = close_short and config.allow_shorts
        g10 = go_long or go_short
        print(f"    [{'PASS' if g10 else 'FAIL'}] direction: go_long={go_long} go_short={go_short}")

        # Gate 11: Gap filter (Python ONLY — PineScript ORB does NOT have this gate!)
        g11 = True
        if config.orb_gap_filter:
            if go_long and state.gap_dir < 0:
                g11 = False
            if go_short and state.gap_dir > 0:
                g11 = False
        print(f"    [{'PASS' if g11 else '** FAIL **'}] gap_filter (PYTHON-ONLY gate!): "
              f"gap_dir={state.gap_dir} | go_long={go_long} go_short={go_short}")
        if not g11:
            print(f"      >>> PineScript ORB does NOT check gap_filter!")

        # Gate 12: ADX filter (Python ONLY — PineScript ORB does NOT have this gate!)
        g12 = True
        if config.orb_adx_filter:
            g12 = ctx.adx_val >= config.adx_thresh
        print(f"    [{'PASS' if g12 else '** FAIL **'}] adx_filter (PYTHON-ONLY gate!): "
              f"adx({ctx.adx_val:.2f}) >= thresh({config.adx_thresh}) | "
              f"TV says ADX={trade['adx']}")
        if not g12:
            print(f"      >>> PineScript ORB does NOT check ADX for entry!")

        # Gate 13: HTF alignment (PineScript does NOT check this for ORB setup)
        g13 = True
        if config.orb_htf_align:
            if go_long and ctx.htf_trend < 0:
                g13 = False
            if go_short and ctx.htf_trend > 0:
                g13 = False
        print(f"    [{'PASS' if g13 else '** FAIL **'}] htf_align (PYTHON-ONLY gate!): "
              f"htf_trend={ctx.htf_trend} | go_long={go_long} go_short={go_short} | "
              f"TV says HTF={trade['htf']}")
        if not g13:
            print(f"      >>> PineScript ORB does NOT check HTF alignment!")

        # Summary
        all_gates = [g1, g2, g3, g4, is_good_time, g6, g7, g8, g9, g10, g11, g12, g13]
        all_pass = all(all_gates)
        python_only_gates = {
            "is_good_time": is_good_time,
            "gap_filter": g11,
            "adx_filter": g12,
            "htf_align": g13,
        }
        python_blockers = [k for k, v in python_only_gates.items() if not v]

        print(f"\n    {'>>> WOULD ENTER' if all_pass else '>>> BLOCKED'}")
        if not all_pass:
            failed = []
            gate_names = ["orb_enable", "not_fired", "or_done", "window",
                          "is_good_time", "close_beyond_OR", "body_filter",
                          "rvol", "vwap_align", "direction", "gap_filter",
                          "adx_filter", "htf_align"]
            for name, val in zip(gate_names, all_gates):
                if not val:
                    failed.append(name)
            print(f"    Failed gates: {failed}")
            if python_blockers:
                print(f"    PYTHON-ONLY blockers: {python_blockers}")
                print(f"    >>> These gates exist in Python but NOT in PineScript ORB!")


def run_backtest(bars: list[IntradayPriceData], config: ORBreakoutStrategyConfig) -> None:
    """Run the actual backtest to confirm 0 trades."""
    strategy = ORBreakoutStrategy(config=config)
    engine = IntradayBacktestEngine(
        initial_capital=D("100000"),
        risk_per_trade=D("0.01"),
        slippage_pct=D("0.0002"),
        fixed_capital=True,
    )
    result = engine.run(strategy, bars)
    print(f"\n{'='*80}")
    print("BACKTEST RESULT")
    print(f"{'='*80}")
    print(f"  Total trades: {result.total_trades}")
    print(f"  Total P&L: ${float(result.total_pnl):+,.2f}")
    print(f"  Win rate: {float(result.win_rate):.1f}%")


def main() -> None:
    if not BARS_CSV.exists():
        print(f"ERROR: SPY 5m data not found at {BARS_CSV}")
        sys.exit(1)

    print("=" * 80)
    print("ORB DIAGNOSTIC — Why does Python produce 0 ORB trades?")
    print("=" * 80)

    # Load data
    print("\nLoading SPY 5m bars...")
    all_bars = load_5m_bars(BARS_CSV)
    bars = [b for b in all_bars if b.date[:10] >= "2025-02-20"]
    print(f"  Loaded {len(all_bars)} total, filtered to {len(bars)} bars "
          f"({bars[0].date[:10]} to {bars[-1].date[:10]})")

    config = make_config()

    # Show config
    print(f"\n  Config:")
    print(f"    or_bars={config.or_bars} (OR period = {config.or_bars * 5} min)")
    print(f"    orb_window={config.orb_window} (last eligible bar)")
    print(f"    can_trade_bar={config.can_trade_bar} (part of is_good_time)")
    print(f"    lunch_start={config.lunch_start}, lunch_end={config.lunch_end}")
    print(f"    orb_rvol={config.orb_rvol}")
    print(f"    orb_body_min={config.orb_body_min}")
    print(f"    orb_vwap_align={config.orb_vwap_align}")
    print(f"    orb_gap_filter={config.orb_gap_filter}")
    print(f"    orb_adx_filter={config.orb_adx_filter}")
    print(f"    adx_thresh={config.adx_thresh}")
    print(f"    orb_htf_align={config.orb_htf_align}")

    # Show key difference
    print(f"\n  KEY DIFFERENCE vs PineScript:")
    print(f"    PineScript ORB checks: orbWindow = orDone AND barOfDay > i_orBars AND barOfDay <= i_orbWindow")
    print(f"    PineScript ORB does NOT check: canTrade(bar>=11), isGoodTime, lunch lull, gap, ADX, HTF")
    print(f"    Python ORB checks ALL of the above as extra gates!")

    # Expected bar_of_day for each TV trade
    print(f"\n  TV ORB trade bar_of_day values:")
    for t in TV_ORB_TRADES:
        b = et_bar_number(t["time_et"])
        blocked_by_can_trade = b < config.can_trade_bar
        blocked_by_lunch = config.lunch_start <= b <= config.lunch_end
        print(f"    Trade #{t['trade_id']}: {t['time_et']} ET = bar {b} "
              f"{'** BLOCKED by can_trade_bar(11) **' if blocked_by_can_trade else ''}"
              f"{'** BLOCKED by lunch lull **' if blocked_by_lunch else ''}")

    # Diagnose each TV trade date
    for trade in TV_ORB_TRADES:
        diagnose_date(bars, trade, config)

    # Run actual backtest
    run_backtest(bars, config)

    # Summary
    print(f"\n{'='*80}")
    print("ROOT CAUSE SUMMARY")
    print(f"{'='*80}")
    print("""
Python ORB strategy has gates that do NOT exist in PineScript ORB:

1. is_good_time gate (line 131 in or_breakout.py):
   - Requires bar_of_day >= can_trade_bar (11)
   - Blocks bars 4-10 (09:45 - 10:20 ET)
   - TV ORB trades fire on bars 6-8 (09:55 - 10:05 ET)
   - This ALONE blocks 4 of 6 TV ORB trades!

   PineScript ORB uses orbWindow = barOfDay > i_orBars AND barOfDay <= i_orbWindow
   No canTrade or isGoodTime check!

2. orb_adx_filter gate (line 174 in or_breakout.py):
   - Requires ADX >= adx_thresh (21)
   - TV ORB trade #100 has ADX=17.2, blocked

   PineScript ORB does NOT check ADX at all!

3. orb_gap_filter gate (line 167-170 in or_breakout.py):
   - Blocks long entries with gap_dir < 0, short entries with gap_dir > 0

   PineScript ORB does NOT check gap_filter!

4. orb_htf_align gate (line 178-184 in or_breakout.py):
   - Blocks entries against HTF trend

   PineScript ORB does NOT check HTF alignment!

FIX: Remove is_good_time, gap_filter, adx_filter, and htf_align from ORB,
     OR make them configurable with TV-parity defaults that disable them.
""")


if __name__ == "__main__":
    main()
