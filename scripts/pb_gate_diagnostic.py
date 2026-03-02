"""PB (Pullback) Gate Diagnostic — per-gate rejection funnel.

Instruments PullbackStrategy._evaluate_entry to count how many bars are
rejected at each gate.  Also compares against TradingView PB trade list
to identify which dates are missing and what gate blocks them.

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/pb_gate_diagnostic.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

D = Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.pullback import (
    PullbackStrategy,
    PullbackStrategyConfig,
)
from stockdownloader.strategies.intraday.session import BarContext
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    clamp_sl_dist,
    directional_sl_tp,
    make_entry_signal,
)
from stockdownloader.strategies.intraday.trail import VwapRatchetTrail
from stockdownloader.indicators.intraday import candle_strength
from stockdownloader.core.math import ZERO

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"
TV_CSV = Path("/Users/kelsayed/Downloads/VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv")

INITIAL_CAPITAL = D("100000")
RISK_PER_TRADE = D("0.01")
SLIPPAGE_PCT = D("0.0002")


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


def load_tv_pb_trades(csv_path: Path) -> list[dict]:
    """Extract PB entry trades from TV export CSV."""
    trades: list[dict] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signal = row.get("Signal", "")
            if signal.startswith("PB|"):
                dt_str = row.get("Date and time", "")
                # Only entry rows (Type starts with "Entry")
                trade_type = row.get("Type", "")
                if "Entry" in trade_type:
                    trades.append({
                        "trade_num": int(row["Trade #"]),
                        "datetime": dt_str,
                        "date": dt_str.split(" ")[0] if dt_str else "",
                        "time": dt_str.split(" ")[1] if " " in dt_str else "",
                        "signal": signal,
                        "direction": "L" if "Entry long" in trade_type else "S",
                        "price": D(row.get("Price USD", "0")),
                    })
    return trades


# ─── Gate names ──────────────────────────────────────────────────────────

GATE_NAMES = [
    "G0:  In-position (infra)",
    "G0b: Circuit-breaker/day-loss (infra)",
    "G1:  day_trades >= max_day OR not spaced",
    "G2:  ADX < thresh (not trending)",
    "G2b: bar_of_day < can_trade_bar",
    "G3:  trend_dir == 0 (no trend)",
    "G4:  not in VWAP zone (dist > zone_w)",
    "G5:  no candle confirm (no bull/bear at VWAP, no strong)",
    "G6:  direction filters (no go_long AND no go_short)",
    "G7:  VWAP crosses > max_vxc",
    "G8:  VWAP session bias filter",
    "G9a: HTF alignment filter",
    "G9b: AR filter (ATR ratio out of range)",
    "G9c: VA filter (VWAP accel < va_min)",
    "G9d: CVD long filter",
    "G9e: DOW filter (no_friday_short / no_monday_long)",
    "G9f: LRS short filter",
    "G9g: Direction zero after filter block",
    "G10: S/R hard filter (pb_require_sr)",
    "G11: Score < min_score (long) or < min_score (short)",
    "G12: SL dist is None",
    "G12b: VWAP TP mode RR < 0.3",
    "PASS: Signal returned",
]


class DiagnosticPullbackStrategy(PullbackStrategy):
    """PullbackStrategy subclass that counts per-gate rejections."""

    def __init__(self, **overrides):
        super().__init__(**overrides)
        # Global counters
        self.gate_counts: dict[str, int] = {name: 0 for name in GATE_NAMES}
        self.total_bars_evaluated = 0
        self.total_bars_in_position = 0
        self.total_bars_risk_blocked = 0

        # Per-date tracking: date -> list of (bar_time, gate_name)
        self.per_date_rejections: dict[str, list[tuple[str, str]]] = defaultdict(list)
        # Per-date tracking: date -> list of (bar_time, direction)
        self.per_date_passes: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        """Instrumented _evaluate_entry with per-gate counters."""
        self.total_bars_evaluated += 1
        c = self._c
        s = ctx.state
        bar_time = ctx.bar.date  # full datetime string
        trading_date = ctx.bar.trading_date

        # -- Gate 1: Spacing + day trade limit --
        ready = s.day_trades < c.max_day
        spaced = (ctx.bar_of_day - s.last_entry_bar) >= c.spacing or s.last_entry_bar <= 0
        if not ready or not spaced:
            gate = "G1:  day_trades >= max_day OR not spaced"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # -- Gate 2: ADX trending --
        is_trending = ctx.adx_val >= c.adx_thresh
        if not is_trending:
            gate = "G2:  ADX < thresh (not trending)"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # -- Gate 2b: can_trade_bar --
        if ctx.bar_of_day < c.can_trade_bar:
            gate = "G2b: bar_of_day < can_trade_bar"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # -- Gate 3: Trend direction --
        trend_dir = self._trend_dir(
            s.bull_bars, s.bear_bars, c.trend_bars,
            ctx.vwap_delta, ctx.atr_val,
        )
        if trend_dir == 0:
            gate = "G3:  trend_dir == 0 (no trend)"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # -- Gate 4: VWAP zone --
        vwap = ctx.vwap_bands.vwap
        band_width = ctx.vwap_bands.std_dev
        zone_w = band_width * c.pb_zone
        dist_vwap = abs(ctx.bar.close - vwap)
        if zone_w <= ZERO or dist_vwap > zone_w:
            gate = "G4:  not in VWAP zone (dist > zone_w)"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # -- Gate 5: Candle confirmation --
        cs = candle_strength(ctx.bar, ctx.atr_val)
        vwap_tol = ctx.atr_val * D("0.25")
        bull_candle = cs.bull_candle(c.pb_body)
        bear_candle = cs.bear_candle(c.pb_body)
        bull_at_vwap = ctx.bar.close >= (vwap - vwap_tol) and (bull_candle or cs.bull_wick)
        bear_at_vwap = ctx.bar.close <= (vwap + vwap_tol) and (bear_candle or cs.bear_wick)
        strong_bull = cs.is_bull and cs.body >= ctx.atr_val * D("0.3") and ctx.bar.close > vwap
        strong_bear = cs.is_bear and cs.body >= ctx.atr_val * D("0.3") and ctx.bar.close < vwap

        # -- Gate 6: Determine direction --
        go_long = (
            trend_dir >= 1
            and (bull_at_vwap or strong_bull)
            and c.allow_longs
        )
        go_short = (
            trend_dir <= -1
            and (bear_at_vwap or strong_bear)
            and c.allow_shorts
        )

        if not go_long and not go_short:
            # Determine if it's candle confirmation issue or direction issue
            any_candle = bull_at_vwap or strong_bull or bear_at_vwap or strong_bear
            if not any_candle:
                gate = "G5:  no candle confirm (no bull/bear at VWAP, no strong)"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
            else:
                gate = "G6:  direction filters (no go_long AND no go_short)"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # -- Gate 7: VWAP crosses cap --
        if s.vwap_crosses > c.max_vxc:
            gate = "G7:  VWAP crosses > max_vxc"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # -- Gate 8: VWAP session bias filter --
        if c.pb_vwap_bias and ctx.bar_of_day > 0:
            total = s.cum_bars_above_vwap + s.cum_bars_below_vwap
            if total > 0:
                ratio_above = D(str(s.cum_bars_above_vwap)) / D(str(total))
                ratio_below = D(str(s.cum_bars_below_vwap)) / D(str(total))
                if go_long and ratio_above < c.pb_vwap_bias_pct:
                    go_long = False
                if go_short and ratio_below < c.pb_vwap_bias_pct:
                    go_short = False
            if not go_long and not go_short:
                gate = "G8:  VWAP session bias filter"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
                return None

        # -- Gate 9: Filters block --
        # 9a: HTF alignment
        if c.htf_align:
            if go_long and ctx.htf_trend == -1:
                go_long = False
            if go_short and ctx.htf_trend == 1:
                go_short = False

        # Check if direction gone after HTF
        if not go_long and not go_short:
            gate = "G9a: HTF alignment filter"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # 9b: AR filter
        if c.ar_filter and ctx.atr_val > ZERO:
            ar = ctx.atr_fast / ctx.atr_val
            if not (c.ar_thresh <= ar <= c.ar_cap):
                gate = "G9b: AR filter (ATR ratio out of range)"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
                return None

        # 9c: VA filter
        if c.va_filter and ctx.vwap_accel < c.va_min:
            gate = "G9c: VA filter (VWAP accel < va_min)"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # 9d: CVD long filter
        pre_cvd_long = go_long
        if go_long and c.cvd_long_filter and ctx.cvd_norm <= ZERO:
            go_long = False

        # 9e: DOW filters
        pre_dow_short = go_short
        pre_dow_long = go_long
        if go_short and c.no_friday_short and ctx.dow == 4:
            go_short = False
        if go_long and c.no_monday_long and ctx.dow == 0:
            go_long = False

        # 9f: LRS short filter
        pre_lrs_short = go_short
        if go_short and c.lrs_short_filter and ctx.lrs_atr > c.lrs_thresh:
            go_short = False

        if not go_long and not go_short:
            # Determine which sub-filter killed it
            if pre_cvd_long and not go_long and not pre_dow_long:
                # CVD + DOW both could have killed long
                pass
            if pre_cvd_long and not go_long:
                gate = "G9d: CVD long filter"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
            elif pre_dow_short and not go_short:
                if c.no_friday_short and ctx.dow == 4:
                    gate = "G9e: DOW filter (no_friday_short / no_monday_long)"
                elif pre_lrs_short and not go_short:
                    gate = "G9f: LRS short filter"
                else:
                    gate = "G9e: DOW filter (no_friday_short / no_monday_long)"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
            elif pre_lrs_short and not go_short:
                gate = "G9f: LRS short filter"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
            elif pre_dow_long and not go_long:
                gate = "G9e: DOW filter (no_friday_short / no_monday_long)"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
            else:
                gate = "G9g: Direction zero after filter block"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # -- Gate 10: S/R hard filter --
        if c.pb_require_sr and not ctx.sr_any:
            gate = "G10: S/R hard filter (pb_require_sr)"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        # -- Gate 11: Confluence scoring --
        if ctx.tod_rvol < D("1.0"):
            pts_vol = c.w_vol
        elif ctx.tod_rvol < D("2.0"):
            pts_vol = 1
        else:
            pts_vol = 0
        pts_sr = c.w_sr if ctx.sr_score_count > 0 else 0
        pts_sr2 = 1 if ctx.sr_score_count >= 2 else 0
        pts_time = c.w_time if ctx.is_good_time else 0
        pts_pq = c.w_pq if ctx.clean_pb else 0

        if go_long:
            pts_rsi = c.w_rsi if D("45") <= ctx.rsi_val <= D("70") else 0
            pts_box = c.w_box if ctx.box_pos <= D("0.33") else 0
            score = pts_vol + pts_sr + pts_sr2 + pts_rsi + pts_time + pts_pq + pts_box
            if score < c.min_score_long:
                gate = "G11: Score < min_score (long) or < min_score (short)"
                self.gate_counts[gate] += 1
                detail = (f"  >> Score={score} < min_score_long={c.min_score_long} "
                          f"(vol={pts_vol} sr={pts_sr} sr2={pts_sr2} rsi={pts_rsi} "
                          f"time={pts_time} pq={pts_pq} box={pts_box} "
                          f"rvol={ctx.tod_rvol:.2f} sr_cnt={ctx.sr_score_count} "
                          f"rsi={ctx.rsi_val:.1f} clean_pb={ctx.clean_pb})")
                self.per_date_rejections[trading_date].append((bar_time, gate + detail))
                return None
        else:
            pts_rsi = c.w_rsi if D("30") <= ctx.rsi_val <= D("55") else 0
            pts_box = c.w_box if ctx.box_pos >= D("0.67") else 0
            score = pts_vol + pts_sr + pts_sr2 + pts_rsi + pts_time + pts_pq + pts_box
            if score < c.min_score:
                gate = "G11: Score < min_score (long) or < min_score (short)"
                self.gate_counts[gate] += 1
                detail = (f"  >> Score={score} < min_score={c.min_score} "
                          f"(vol={pts_vol} sr={pts_sr} sr2={pts_sr2} rsi={pts_rsi} "
                          f"time={pts_time} pq={pts_pq} box={pts_box} "
                          f"rvol={ctx.tod_rvol:.2f} sr_cnt={ctx.sr_score_count} "
                          f"rsi={ctx.rsi_val:.1f} clean_pb={ctx.clean_pb})")
                self.per_date_rejections[trading_date].append((bar_time, gate + detail))
                return None

        max_score = c.w_vol + c.w_sr + 1 + c.w_rsi + c.w_time + c.w_pq + c.w_box

        # -- Gate 12: SL dist --
        sl_dist = clamp_sl_dist(ctx.atr_val * c.sl_atr, c.sl_cap)
        if sl_dist is None:
            gate = "G12: SL dist is None"
            self.gate_counts[gate] += 1
            self.per_date_rejections[trading_date].append((bar_time, gate))
            return None

        if c.pb_tp_mode == "vwap":
            tp_price = vwap
            sl_price = ctx.bar.close - sl_dist if go_long else ctx.bar.close + sl_dist
            reward = abs(tp_price - ctx.bar.close)
            rr_actual = reward / sl_dist if sl_dist > ZERO else ZERO
            if rr_actual < D("0.3"):
                gate = "G12b: VWAP TP mode RR < 0.3"
                self.gate_counts[gate] += 1
                self.per_date_rejections[trading_date].append((bar_time, gate))
                return None
        else:
            tp_dist = sl_dist * c.rr
            sl_price, tp_price = directional_sl_tp(go_long, ctx.bar.close, sl_dist, tp_dist)

        label = ("Strong" if abs(trend_dir) >= 2 else "Soft") + " Trend PB"

        # -- PASS --
        direction = "L" if go_long else "S"
        self.gate_counts["PASS: Signal returned"] += 1
        self.per_date_passes[trading_date].append((bar_time, direction))

        return make_entry_signal(
            go_long=go_long,
            mode="PB",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=label,
        )


def main() -> None:
    if not BARS_CSV.exists():
        print(f"ERROR: SPY 5m data not found at {BARS_CSV}")
        sys.exit(1)

    print("=" * 100)
    print("PB (PULLBACK) GATE DIAGNOSTIC")
    print("=" * 100)

    # Load bars
    print("\nLoading SPY 5m bars...")
    bars = load_5m_bars(BARS_CSV)
    print(f"  Loaded {len(bars)} bars ({bars[0].date[:10]} to {bars[-1].date[:10]})")

    # Load TV PB trades
    tv_pb_trades = []
    if TV_CSV.exists():
        tv_pb_trades = load_tv_pb_trades(TV_CSV)
        print(f"  Loaded {len(tv_pb_trades)} PB entries from TV export")
    else:
        print(f"  WARNING: TV CSV not found at {TV_CSV}")

    # Build strategy with TV-parity config
    strategy = DiagnosticPullbackStrategy(
        allow_shorts=True,
        allow_longs=True,
        pb_zone=D("0.5"),
        pb_body=D("0.15"),
        rr=D("1.4"),
        sl_atr=D("1.3"),
        sl_cap=D("1.50"),
        adx_thresh=D("21"),
        trend_bars=3,
        htf_align=True,
        ar_filter=True,
        ar_thresh=D("0.9"),
        ar_cap=D("1.15"),
        va_filter=True,
        va_min=D("-0.1"),
        cvd_long_filter=True,
        lrs_short_filter=True,
        lrs_thresh=D("0.08"),
        max_vxc=6,
        w_vol=3,
        w_sr=2,
        w_rsi=1,
        w_time=0,
        w_pq=1,
        w_box=0,
        min_score=3,
        min_score_long=5,
        pq_max_cross=3,
        max_day=2,
        spacing=3,
        circuit=3,
        day_loss=D("3.0"),
        no_friday_short=True,
        no_monday_long=True,
        be_trigger=D("0.5"),
        trail_vwap=True,
        trail_buf=D("0.15"),
        trail_keep_tp=True,
        close_eod=True,
        pb_vwap_bias=True,
        pb_tp_mode="rr",
    )

    # Run backtest
    print("\nRunning PB backtest with gate instrumentation...")
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        slippage_pct=SLIPPAGE_PCT,
        fixed_capital=True,
    )
    result = engine.run(strategy, bars)
    print(f"  Backtest complete: {result.total_trades} trades, "
          f"PnL=${float(result.total_pnl):+,.2f}")

    # ── GATE REJECTION FUNNEL ────────────────────────────────────────
    print("\n" + "=" * 100)
    print("GATE REJECTION FUNNEL (cumulative — once rejected, doesn't reach later gates)")
    print("=" * 100)

    total_bars = len(bars)
    print(f"\nTotal 5m bars in dataset:                     {total_bars:>8}")
    print(f"Total bars where _evaluate_entry was called:   {strategy.total_bars_evaluated:>8}")
    print(f"  (Remainder = in-position or risk-blocked by infra layer)")
    print()

    print(f"{'Gate':<60} {'Count':>8} {'% of Evaluated':>15}")
    print("-" * 85)

    evaluated = strategy.total_bars_evaluated
    for gate_name in GATE_NAMES:
        count = strategy.gate_counts[gate_name]
        pct = (count / evaluated * 100) if evaluated > 0 else 0
        marker = " ***" if count > 0 and gate_name.startswith("PASS") else ""
        print(f"{gate_name:<60} {count:>8} {pct:>14.1f}%{marker}")

    # Verify sum
    gate_sum = sum(strategy.gate_counts.values())
    print(f"\n  Sum of all gate counts: {gate_sum} (should equal {evaluated})")
    if gate_sum != evaluated:
        print(f"  MISMATCH: diff = {evaluated - gate_sum}")

    # ── TV COMPARISON ────────────────────────────────────────────────
    if tv_pb_trades:
        print("\n" + "=" * 100)
        print("TV PB TRADE COMPARISON")
        print("=" * 100)

        # Group TV PB trades by date
        tv_dates: dict[str, list[dict]] = defaultdict(list)
        for t in tv_pb_trades:
            tv_dates[t["date"]].append(t)

        # Group Python PB passes by date
        py_dates: dict[str, list[tuple[str, str]]] = strategy.per_date_passes

        tv_date_set = set(tv_dates.keys())
        py_date_set = set(py_dates.keys())

        common = tv_date_set & py_date_set
        tv_only = sorted(tv_date_set - py_date_set)
        py_only = sorted(py_date_set - tv_date_set)

        print(f"\n  TV PB entries:       {len(tv_pb_trades)}")
        print(f"  Python PB passes:   {strategy.gate_counts['PASS: Signal returned']}")
        print(f"  Python actual trades:{result.total_trades}")
        print(f"  Dates with TV PB:    {len(tv_date_set)}")
        print(f"  Dates with PY PB:    {len(py_date_set)}")
        print(f"  Dates in common:     {len(common)}")
        print(f"  Dates TV-only:       {len(tv_only)}  (PB in TV but NOT in Python)")
        print(f"  Dates PY-only:       {len(py_only)}  (PB in Python but NOT in TV)")

        # ── TV-only dates: what gate blocked them? ────────────────
        if tv_only:
            print(f"\n{'='*100}")
            print("DATES WITH PB IN TV BUT NOT IN PYTHON — Gate analysis")
            print(f"{'='*100}")

            # Aggregate which gates block entries on TV-trade dates
            gate_block_counts: dict[str, int] = defaultdict(int)

            for date in tv_only:
                tv_entries = tv_dates[date]
                rejections = strategy.per_date_rejections.get(date, [])

                print(f"\n  DATE: {date}")
                for tv_entry in tv_entries:
                    print(f"    TV trade #{tv_entry['trade_num']}: "
                          f"{tv_entry['direction']} at {tv_entry['time']} "
                          f"@ ${tv_entry['price']}")
                    print(f"      Signal: {tv_entry['signal'][:120]}...")

                if rejections:
                    # Group rejections by gate
                    gate_summary: dict[str, int] = defaultdict(int)
                    for _time, gate in rejections:
                        # Strip detail info for summary
                        base_gate = gate.split("  >>")[0]
                        gate_summary[base_gate] += 1

                    print(f"    Python rejections on this date ({len(rejections)} total):")
                    for gate, cnt in sorted(gate_summary.items()):
                        print(f"      {gate}: {cnt} bars")
                        gate_block_counts[gate] += cnt

                    # Show rejections near TV entry times
                    for tv_entry in tv_entries:
                        tv_time = tv_entry["time"]
                        # Find rejections within +/- 2 bars (10 min) of TV time
                        nearby = []
                        for r_time, r_gate in rejections:
                            r_time_only = r_time.split(" ")[1] if " " in r_time else r_time
                            # Simple time proximity check
                            try:
                                tv_dt = datetime.strptime(tv_time, "%H:%M")
                                r_dt = datetime.strptime(r_time_only, "%H:%M")
                                diff_min = abs((tv_dt - r_dt).total_seconds()) / 60
                                if diff_min <= 15:
                                    nearby.append((r_time_only, r_gate))
                            except ValueError:
                                pass

                        if nearby:
                            print(f"    Rejections near TV entry {tv_time}:")
                            for r_time, r_gate in nearby:
                                print(f"      {r_time}: {r_gate}")
                        else:
                            print(f"    No rejections found near TV entry time {tv_time}")
                else:
                    print(f"    No bars evaluated on this date (all in-position/risk-blocked)")

            print(f"\n{'='*100}")
            print("AGGREGATE: Gates blocking TV-only dates")
            print(f"{'='*100}")
            for gate, cnt in sorted(gate_block_counts.items(), key=lambda x: -x[1]):
                print(f"  {gate}: {cnt}")

        # ── Common dates: do trade counts match? ──────────────────
        if common:
            print(f"\n{'='*100}")
            print("COMMON DATES — Trade count comparison")
            print(f"{'='*100}")
            mismatches = []
            for date in sorted(common):
                tv_count = len(tv_dates[date])
                py_count = len(py_dates[date])
                if tv_count != py_count:
                    mismatches.append((date, tv_count, py_count))
            if mismatches:
                print(f"\n  Dates where trade counts differ: {len(mismatches)}")
                print(f"  {'Date':<12} {'TV':>4} {'PY':>4} {'Delta':>6}")
                print(f"  {'-'*30}")
                total_tv_extra = 0
                for date, tv_cnt, py_cnt in mismatches:
                    delta = tv_cnt - py_cnt
                    total_tv_extra += max(0, delta)
                    print(f"  {date:<12} {tv_cnt:>4} {py_cnt:>4} {delta:>+6}")

                    # Show what gates blocked the extra TV entries
                    if delta > 0:
                        rejections = strategy.per_date_rejections.get(date, [])
                        tv_entries = tv_dates[date]
                        py_times = {t[0] for t in py_dates[date]}

                        # Find TV entry times NOT matched in Python
                        for tv_entry in tv_entries:
                            tv_time = tv_entry["time"]
                            tv_full = tv_entry["datetime"]
                            # Check if this TV entry has a nearby Python pass
                            matched = False
                            for py_time, _ in py_dates[date]:
                                py_t = py_time.split(" ")[1] if " " in py_time else py_time
                                try:
                                    tv_dt = datetime.strptime(tv_time, "%H:%M")
                                    py_dt = datetime.strptime(py_t, "%H:%M")
                                    if abs((tv_dt - py_dt).total_seconds()) <= 300:
                                        matched = True
                                        break
                                except ValueError:
                                    pass
                            if not matched:
                                print(f"    TV #{tv_entry['trade_num']} "
                                      f"({tv_entry['direction']}) at {tv_time} "
                                      f"NOT matched in Python")
                                # Find nearby rejections
                                for r_time, r_gate in rejections:
                                    r_t = r_time.split(" ")[1] if " " in r_time else r_time
                                    try:
                                        tv_dt = datetime.strptime(tv_time, "%H:%M")
                                        r_dt = datetime.strptime(r_t, "%H:%M")
                                        if abs((tv_dt - r_dt).total_seconds()) <= 600:
                                            print(f"      nearby rejection @ {r_t}: {r_gate}")
                                    except ValueError:
                                        pass

                print(f"\n  Total extra TV entries missing from Python: {total_tv_extra}")
            else:
                print(f"\n  All {len(common)} common dates have matching trade counts!")

        # ── PY-only dates ─────────────────────────────────────────
        if py_only:
            print(f"\n{'='*100}")
            print("DATES WITH PB IN PYTHON BUT NOT IN TV (potential false positives)")
            print(f"{'='*100}")
            for date in py_only:
                py_entries = py_dates[date]
                print(f"  {date}: {len(py_entries)} Python entries")
                for bar_time, direction in py_entries:
                    t = bar_time.split(" ")[1] if " " in bar_time else bar_time
                    print(f"    {t} {direction}")

    print(f"\n{'='*100}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
