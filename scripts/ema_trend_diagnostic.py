"""EMA/Trend Divergence Diagnostic — identify which indicators cause extra PB trades.

Focuses on:
1. EMA fast/slow values: Python's StreamingEMA vs TV's ta.ema()
2. bull_bars/bear_bars tracking: first-bar-of-day reset ordering
3. ATR values: Python's StreamingATR vs TV's ta.atr(14)
4. ADX values: Python's StreamingADX vs TV's ta.dmi(14,14)

For each FALSE POSITIVE (Python PB entry that TV does not take), dumps:
- Python's ema_fast, ema_slow, bull_bars, bear_bars, trendDir
- TV's ADX (from signal metadata) vs Python's ADX
- Python's ATR vs what TV uses for body_atr
- The specific gate(s) that might differ

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/ema_trend_diagnostic.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

D = Decimal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.core.math import TWO, ZERO
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.indicators.hub import IndicatorHub
from stockdownloader.indicators.intraday import candle_strength
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.pullback import (
    PullbackStrategy,
    PullbackStrategyConfig,
)
from stockdownloader.strategies.intraday.session import BarContext
from stockdownloader.strategies.intraday.trade_mgmt import IntradayExitManager
from stockdownloader.strategies.intraday.trail import VwapRatchetTrail

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"
TV_CSV = Path.home() / "Downloads" / "VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv"

TV_TO_ET_OFFSET = timedelta(hours=2, minutes=55)


# =====================================================================
# Data loading
# =====================================================================


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


def build_bar_index(bars: list[IntradayPriceData]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, bar in enumerate(bars):
        index[bar.date[:16]] = i
    return index


# =====================================================================
# TV trade parsing
# =====================================================================


def parse_tv_signal(signal: str) -> dict[str, str]:
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


@dataclass
class TVPBTrade:
    trade_num: int
    direction: str
    tv_datetime: str
    et_bar_label: str
    price: float
    signal_raw: str
    adx: float = 0.0
    trend_dir: int = 0
    ar: float = 0.0
    vwap_delta: float = 0.0
    tod_rvol: float = 0.0
    body_atr: float = 0.0


def load_tv_pb_trades(csv_path: Path) -> list[TVPBTrade]:
    trades: list[TVPBTrade] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trade_type = row.get("Type", "")
            if "Entry" not in trade_type:
                continue
            signal = row.get("Signal", "")
            if not signal.startswith("PB|"):
                continue

            parsed = parse_tv_signal(signal)
            trade_num = int(row.get("Trade #", "0"))
            direction = "S" if "short" in trade_type.lower() else "L"
            tv_dt = row.get("Date and time", "").strip()
            price = float(row.get("Price USD", "0"))
            dt = datetime.strptime(tv_dt, "%Y-%m-%d %H:%M")
            et_dt = dt + TV_TO_ET_OFFSET
            et_label = et_dt.strftime("%Y-%m-%d %H:%M")

            trades.append(TVPBTrade(
                trade_num=trade_num,
                direction=direction,
                tv_datetime=tv_dt,
                et_bar_label=et_label,
                price=price,
                signal_raw=signal,
                adx=float(parsed.get("ADX", "0")),
                trend_dir=int(parsed.get("TD", "0")),
                ar=float(parsed.get("AR", "0")),
                vwap_delta=float(parsed.get("VD", "0")),
                tod_rvol=float(parsed.get("TRV", "0")),
            ))
    return trades


# =====================================================================
# TV-parity PB config
# =====================================================================


def build_tv_parity_pb_config() -> PullbackStrategyConfig:
    return PullbackStrategyConfig(
        allow_shorts=True, allow_longs=True,
        pb_zone=D("0.5"), pb_body=D("0.15"),
        rr=D("1.4"), sl_atr=D("1.3"), sl_cap=D("1.50"),
        adx_thresh=D("21"), trend_bars=3,
        htf_align=True,
        ar_filter=True, ar_thresh=D("0.9"), ar_cap=D("1.15"),
        va_filter=True, va_min=D("-0.1"),
        cvd_long_filter=True, lrs_short_filter=True, lrs_thresh=D("0.08"),
        max_vxc=6, w_vol=3, w_sr=2, w_rsi=1, w_time=0, w_pq=1, w_box=0,
        min_score=3, min_score_long=5, pq_max_cross=3,
        max_day=2, spacing=3, circuit=3, day_loss=D("3.0"),
        no_friday_short=True, no_monday_long=True,
        be_trigger=D("0.5"), trail_vwap=True, trail_buf=D("0.15"),
        trail_keep_tp=True, close_eod=True,
        pb_vwap_bias=False, pb_tp_mode="rr",
    )


# =====================================================================
# Instrumented strategy that captures ALL BarContext details
# =====================================================================


@dataclass
class IndicatorSnapshot:
    """Full indicator state at a bar."""
    bar_label: str
    bar_idx: int
    direction: str
    close: float
    vwap: float
    # EMA
    ema_fast: float
    ema_slow: float
    bull_bars: int
    bear_bars: int
    trend_dir: int
    # ATR
    atr_val: float
    atr_fast: float
    ar: float
    # ADX
    adx_val: float
    # Candle
    body_atr: float
    is_bull: bool
    is_bear: bool
    # VWAP
    vwap_delta: float
    band_width: float
    dist_vwap: float
    zone_w: float
    # Other
    bar_of_day: int
    tod_rvol: float
    htf_trend: int
    cvd_norm: float
    lrs_atr: float
    rsi: float
    vwap_crosses: int
    score: int


class InstrumentedPB(PullbackStrategy):
    """PB strategy that records indicator snapshots for every entry."""

    def __init__(self, config: PullbackStrategyConfig):
        super().__init__(config=config)
        self.entries: list[IndicatorSnapshot] = []

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        signal = super()._evaluate_entry(ctx)
        if signal is not None and signal.action.name in ("ENTER_LONG", "ENTER_SHORT"):
            c = self._c
            bar = ctx.bar
            s = ctx.state
            go_long = signal.action.name == "ENTER_LONG"

            vwap = ctx.vwap_bands.vwap
            band_width = ctx.vwap_bands.std_dev * TWO
            zone_w = band_width * c.pb_zone
            dist_vwap = abs(bar.close - vwap)

            cs = candle_strength(bar, ctx.atr_val)
            ar = float(ctx.atr_fast / ctx.atr_val) if ctx.atr_val > ZERO else 0.0

            trend_dir = self._trend_dir(
                s.bull_bars, s.bear_bars, c.trend_bars,
                ctx.vwap_delta, ctx.atr_val,
            )

            # Confluence scoring
            if ctx.tod_rvol < D("1.0"):
                pts_vol = c.w_vol
            elif ctx.tod_rvol < D("2.0"):
                pts_vol = 1
            else:
                pts_vol = 0
            pts_sr = c.w_sr if ctx.sr_score_count > 0 else 0
            pts_sr2 = 1 if ctx.sr_score_count >= 2 else 0
            pts_pq = c.w_pq if ctx.clean_pb else 0
            if go_long:
                pts_rsi = c.w_rsi if D("45") <= ctx.rsi_val <= D("70") else 0
                pts_box = c.w_box if ctx.box_pos <= D("0.33") else 0
            else:
                pts_rsi = c.w_rsi if D("30") <= ctx.rsi_val <= D("55") else 0
                pts_box = c.w_box if ctx.box_pos >= D("0.67") else 0
            score = pts_vol + pts_sr + pts_sr2 + pts_rsi + c.w_time * int(ctx.is_good_time) + pts_pq + pts_box

            self.entries.append(IndicatorSnapshot(
                bar_label=bar.date[:16], bar_idx=0,
                direction="L" if go_long else "S",
                close=float(bar.close), vwap=float(vwap),
                ema_fast=float(ctx.ema_fast), ema_slow=float(ctx.ema_slow),
                bull_bars=s.bull_bars, bear_bars=s.bear_bars,
                trend_dir=trend_dir,
                atr_val=float(ctx.atr_val), atr_fast=float(ctx.atr_fast), ar=ar,
                adx_val=float(ctx.adx_val),
                body_atr=float(cs.body_atr), is_bull=cs.is_bull, is_bear=cs.is_bear,
                vwap_delta=float(ctx.vwap_delta),
                band_width=float(band_width), dist_vwap=float(dist_vwap),
                zone_w=float(zone_w),
                bar_of_day=ctx.bar_of_day,
                tod_rvol=float(ctx.tod_rvol),
                htf_trend=ctx.htf_trend,
                cvd_norm=float(ctx.cvd_norm), lrs_atr=float(ctx.lrs_atr),
                rsi=float(ctx.rsi_val),
                vwap_crosses=s.vwap_crosses,
                score=score,
            ))

        return signal


def run_pb(bars: list[IntradayPriceData], config: PullbackStrategyConfig) -> list[IndicatorSnapshot]:
    strategy = InstrumentedPB(config=config)
    for i in range(len(bars)):
        signal = strategy.evaluate(bars, i)
        if signal.action.name in ("ENTER_LONG", "ENTER_SHORT"):
            strategy.on_position_opened(signal.action.name == "ENTER_LONG")
            if strategy.entries:
                strategy.entries[-1].bar_idx = i
        elif signal.action.name in ("EXIT_LONG", "EXIT_SHORT"):
            strategy.on_position_closed()
    return strategy.entries


# =====================================================================
# Extract Python indicators at arbitrary bar (for non-entry bars)
# =====================================================================


def extract_at_bar(
    bars: list[IntradayPriceData],
    target_idx: int,
    config: PullbackStrategyConfig,
) -> IndicatorSnapshot | None:
    """Run infra through all bars up to target_idx, return snapshot."""
    infra = IntradayInfra(config, IntradayExitManager(VwapRatchetTrail()))

    ctx: BarContext | None = None
    for i in range(target_idx + 1):
        ctx = infra.on_new_bar(bars, i)

    if ctx is None:
        # If in position at target_idx, need indicator values anyway
        hub = infra.hub
        s = infra.state
        bar = bars[target_idx]
        atr_val = hub.atr(bars, target_idx, config.adx_len)
        atr_fast = hub.atr(bars, target_idx, 5)
        adx_result = hub.adx(bars, target_idx, config.adx_len)
        ema_fast = hub.ema(bars, target_idx, config.ema_fast)
        ema_slow = hub.ema(bars, target_idx, config.ema_slow)
        vwap_bands = hub.extended_session_vwap_bands(bars, target_idx)
        vwap = vwap_bands.vwap
        band_width = vwap_bands.std_dev * TWO
        zone_w = band_width * config.pb_zone
        dist_vwap = abs(bar.close - vwap)
        cs = candle_strength(bar, atr_val)
        ar = float(atr_fast / atr_val) if atr_val > ZERO else 0.0
        v_delta = hub.vwap_slope(bars, target_idx, config.slope_period)
        trend_dir = PullbackStrategy._trend_dir(
            s.bull_bars, s.bear_bars, config.trend_bars,
            v_delta, atr_val,
        )
        return IndicatorSnapshot(
            bar_label=bar.date[:16], bar_idx=target_idx,
            direction="?", close=float(bar.close), vwap=float(vwap),
            ema_fast=float(ema_fast), ema_slow=float(ema_slow),
            bull_bars=s.bull_bars, bear_bars=s.bear_bars,
            trend_dir=trend_dir,
            atr_val=float(atr_val), atr_fast=float(atr_fast), ar=ar,
            adx_val=float(adx_result.adx),
            body_atr=float(cs.body_atr), is_bull=cs.is_bull, is_bear=cs.is_bear,
            vwap_delta=float(v_delta),
            band_width=float(band_width), dist_vwap=float(dist_vwap),
            zone_w=float(zone_w),
            bar_of_day=s.bar_count, tod_rvol=0.0, htf_trend=0,
            cvd_norm=0.0, lrs_atr=0.0, rsi=0.0,
            vwap_crosses=s.vwap_crosses, score=0,
        )

    s = infra.state
    bar = bars[target_idx]
    hub = infra.hub
    atr_val = hub.atr(bars, target_idx, config.adx_len)
    atr_fast = hub.atr(bars, target_idx, 5)
    vwap_bands = hub.extended_session_vwap_bands(bars, target_idx)
    vwap = vwap_bands.vwap
    band_width = vwap_bands.std_dev * TWO
    zone_w = band_width * config.pb_zone
    dist_vwap = abs(bar.close - vwap)
    cs = candle_strength(bar, atr_val)
    ar = float(atr_fast / atr_val) if atr_val > ZERO else 0.0

    trend_dir = PullbackStrategy._trend_dir(
        s.bull_bars, s.bear_bars, config.trend_bars,
        ctx.vwap_delta, atr_val,
    )

    return IndicatorSnapshot(
        bar_label=bar.date[:16], bar_idx=target_idx,
        direction="?", close=float(bar.close), vwap=float(vwap),
        ema_fast=float(ctx.ema_fast), ema_slow=float(ctx.ema_slow),
        bull_bars=s.bull_bars, bear_bars=s.bear_bars,
        trend_dir=trend_dir,
        atr_val=float(ctx.atr_val), atr_fast=float(ctx.atr_fast), ar=ar,
        adx_val=float(ctx.adx_val),
        body_atr=float(cs.body_atr), is_bull=cs.is_bull, is_bear=cs.is_bear,
        vwap_delta=float(ctx.vwap_delta),
        band_width=float(band_width), dist_vwap=float(dist_vwap),
        zone_w=float(zone_w),
        bar_of_day=ctx.bar_of_day,
        tod_rvol=float(ctx.tod_rvol),
        htf_trend=ctx.htf_trend,
        cvd_norm=float(ctx.cvd_norm), lrs_atr=float(ctx.lrs_atr),
        rsi=float(ctx.rsi_val),
        vwap_crosses=s.vwap_crosses, score=0,
    )


# =====================================================================
# Matching
# =====================================================================


def match_entries(
    py_entries: list[IndicatorSnapshot],
    tv_trades: list[TVPBTrade],
    bar_index: dict[str, int],
) -> tuple[list[IndicatorSnapshot], list[IndicatorSnapshot], list[TVPBTrade]]:
    """Return (true_positives, false_positives, false_negatives)."""
    tv_bar_indices: dict[int, TVPBTrade] = {}
    for tv in tv_trades:
        idx = bar_index.get(tv.et_bar_label)
        if idx is not None:
            tv_bar_indices[idx] = tv

    matched_tv: set[int] = set()
    tps: list[IndicatorSnapshot] = []
    fps: list[IndicatorSnapshot] = []

    for py in py_entries:
        py_idx = bar_index.get(py.bar_label)
        if py_idx is None:
            fps.append(py)
            continue

        found = False
        for offset in (0, -1, 1):
            check_idx = py_idx + offset
            if check_idx in tv_bar_indices:
                candidate = tv_bar_indices[check_idx]
                if candidate.trade_num not in matched_tv:
                    matched_tv.add(candidate.trade_num)
                    tps.append(py)
                    found = True
                    break

        if not found:
            fps.append(py)

    fns: list[TVPBTrade] = [tv for tv in tv_trades if tv.trade_num not in matched_tv]
    return tps, fps, fns


# =====================================================================
# Main diagnostic
# =====================================================================


def main() -> None:
    print("=" * 100)
    print("EMA / TREND DIVERGENCE DIAGNOSTIC")
    print("=" * 100)

    if not BARS_CSV.exists():
        print(f"ERROR: {BARS_CSV} not found")
        sys.exit(1)
    if not TV_CSV.exists():
        print(f"ERROR: {TV_CSV} not found")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    bars = load_5m_bars(BARS_CSV)
    bar_index = build_bar_index(bars)
    tv_trades = load_tv_pb_trades(TV_CSV)
    print(f"  {len(bars)} bars, {len(tv_trades)} TV PB entries")

    # ── Run Python PB ─────────────────────────────────────────────────
    print("\n[2] Running Python PB (TV-parity config)...")
    config = build_tv_parity_pb_config()
    py_entries = run_pb(bars, config)
    print(f"  Python took {len(py_entries)} PB entries")

    # ── Match ─────────────────────────────────────────────────────────
    print("\n[3] Matching...")
    tps, fps, fns = match_entries(py_entries, tv_trades, bar_index)
    print(f"  TP: {len(tps)}  FP: {len(fps)}  FN: {len(fns)}")

    # ── Part A: EMA initialization check ─────────────────────────────
    print(f"\n{'='*100}")
    print("PART A: EMA INITIALIZATION / SMA SEEDING")
    print(f"{'='*100}")
    print("""
Python's StreamingEMA seeds with SMA(bars[0..period-1]), then applies
exponential smoothing from bar[period] onward. This matches TV's ta.ema()
which also seeds with SMA.

However, Python's EMA runs over ALL bars from bar 0 of the data (including
across session boundaries), while TV's ta.ema() also runs over all bars
from bar 0 of the chart. So EMA seeding should match.

KEY DIVERGENCE CANDIDATE: bull_bars/bear_bars reset ordering.
- TV: update bull/bear (lines 404-412), THEN reset on isNewDay (lines 414-416)
  -> On first bar of day: after update, reset to 0. Net: always 0 on bar 1.
- Python: state.reset() is called first, THEN update_trend() is called.
  -> On first bar of day: reset to 0, then increment. Net: 0 or 1 on bar 1.

So Python accumulates ONE EXTRA bar of EMA alignment on each new day.
With trend_bars=3, this means Python reaches "trending" status one bar earlier
than TV on each day.
""")

    # ── Part B: Compare indicators at FP bars ─────────────────────────
    print(f"\n{'='*100}")
    print(f"PART B: FALSE POSITIVE DETAIL (Python entries TV does NOT take)")
    print(f"{'='*100}")

    # Build TV trade lookup by ET bar label for ADX comparison
    tv_by_label: dict[str, TVPBTrade] = {}
    for tv in tv_trades:
        tv_by_label[tv.et_bar_label] = tv

    # For FPs, find nearest TV trade on same day to compare ADX
    fp_adx_deltas: list[float] = []
    fp_atr_deltas: list[float] = []
    fp_trend_issues: list[str] = []
    fp_body_issues: list[str] = []

    for i, fp in enumerate(fps[:20]):  # Show first 20
        print(f"\n  FP #{i+1}: {fp.bar_label} | {fp.direction} | close=${fp.close:.2f}")
        print(f"    EMA fast={fp.ema_fast:.4f}  slow={fp.ema_slow:.4f}  "
              f"ema_diff={'BULL' if fp.ema_fast > fp.ema_slow else 'BEAR'}")
        print(f"    bull_bars={fp.bull_bars}  bear_bars={fp.bear_bars}  "
              f"trendDir={fp.trend_dir}  (need {config.trend_bars}+ bars)")
        print(f"    ADX={fp.adx_val:.1f}  (thresh={float(config.adx_thresh)})")
        print(f"    ATR={fp.atr_val:.4f}  ATR_fast={fp.atr_fast:.4f}  AR={fp.ar:.3f}")
        print(f"    body_atr={fp.body_atr:.3f}  (thresh={float(config.pb_body)})")
        print(f"    VWAP delta={fp.vwap_delta:.4f}  dist_vwap={fp.dist_vwap:.4f}  zone_w={fp.zone_w:.4f}")
        print(f"    HTF={fp.htf_trend}  CVD={fp.cvd_norm:.4f}  LRS={fp.lrs_atr:.3f}")
        print(f"    VXC={fp.vwap_crosses}  score={fp.score}  bar_of_day={fp.bar_of_day}")

        # Check if this might be a borderline case
        issues: list[str] = []
        if fp.bull_bars == 3 or fp.bear_bars == 3:
            issues.append(f"MARGINAL_TREND: {'bull' if fp.bull_bars==3 else 'bear'}_bars exactly 3 (= trend_bars)")
        if fp.adx_val < 25:
            issues.append(f"LOW_ADX: {fp.adx_val:.1f}")
        if fp.body_atr < 0.20:
            issues.append(f"WEAK_CANDLE: body_atr={fp.body_atr:.3f}")
        if abs(fp.ar - 1.0) > 0.10:
            issues.append(f"AR_EDGE: {fp.ar:.3f}")

        if issues:
            print(f"    >> ISSUES: {', '.join(issues)}")
            fp_trend_issues.extend(issues)

    # ── Part C: ADX comparison for ALL TV entries ──────────────────────
    print(f"\n\n{'='*100}")
    print("PART C: ADX COMPARISON (Python vs TV for matched trades)")
    print(f"{'='*100}")

    # For each TP, compare Python ADX to TV ADX
    adx_deltas: list[float] = []
    vd_deltas: list[float] = []

    print(f"\n  {'Bar':<20} {'TV ADX':>8} {'Py ADX':>8} {'Delta':>8} "
          f"{'TV VD':>8} {'Py VD':>8} {'Delta':>8}")
    print(f"  {'-'*80}")

    for py in tps[:30]:
        # Find the matching TV trade
        tv_match = None
        py_idx = bar_index.get(py.bar_label)
        if py_idx is None:
            continue
        for offset in (0, -1, 1):
            check_label = bars[py_idx + offset].date[:16] if 0 <= py_idx + offset < len(bars) else None
            if check_label and check_label in tv_by_label:
                tv_match = tv_by_label[check_label]
                break
            # Also check direct
            if tv_by_label.get(py.bar_label):
                tv_match = tv_by_label[py.bar_label]
                break

        if tv_match:
            adx_d = py.adx_val - tv_match.adx
            vd_d = py.vwap_delta - tv_match.vwap_delta
            adx_deltas.append(adx_d)
            vd_deltas.append(vd_d)
            print(f"  {py.bar_label:<20} {tv_match.adx:>8.1f} {py.adx_val:>8.1f} {adx_d:>+8.1f} "
                  f"{tv_match.vwap_delta:>8.3f} {py.vwap_delta:>8.4f} {vd_d:>+8.4f}")

    if adx_deltas:
        print(f"\n  ADX Delta: mean={sum(adx_deltas)/len(adx_deltas):+.2f}  "
              f"std={max(0.01, (sum((d-sum(adx_deltas)/len(adx_deltas))**2 for d in adx_deltas)/len(adx_deltas))**0.5):.2f}  "
              f"max_abs={max(abs(d) for d in adx_deltas):.2f}")
    if vd_deltas:
        print(f"  VD  Delta: mean={sum(vd_deltas)/len(vd_deltas):+.4f}  "
              f"max_abs={max(abs(d) for d in vd_deltas):.4f}")

    # ── Part D: Specific FP gate analysis ─────────────────────────────
    print(f"\n\n{'='*100}")
    print("PART D: WHY TV REJECTS THESE ENTRIES — GATE ANALYSIS")
    print(f"{'='*100}")

    gate_counters: Counter = Counter()
    marginal_trend_count = 0
    adx_borderline_count = 0
    body_borderline_count = 0
    ar_borderline_count = 0

    for fp in fps:
        reasons: list[str] = []

        # Check if bull_bars or bear_bars is exactly threshold (borderline)
        if fp.direction == "L" and fp.bull_bars <= 4:
            reasons.append(f"bull_bars={fp.bull_bars} (marginal)")
            marginal_trend_count += 1
        if fp.direction == "S" and fp.bear_bars <= 4:
            reasons.append(f"bear_bars={fp.bear_bars} (marginal)")
            marginal_trend_count += 1

        # ADX near threshold
        if fp.adx_val < 25:
            reasons.append(f"ADX={fp.adx_val:.1f} (near thresh)")
            adx_borderline_count += 1

        # body_atr near threshold
        if fp.body_atr < 0.20:
            reasons.append(f"body_atr={fp.body_atr:.3f} (near thresh)")
            body_borderline_count += 1

        # AR borderline
        if fp.ar < 0.95 or fp.ar > 1.10:
            reasons.append(f"AR={fp.ar:.3f}")
            ar_borderline_count += 1

        for r in reasons:
            gate_counters[r.split("=")[0].split("(")[0].strip()] += 1

    print(f"\n  Total FP entries: {len(fps)}")
    print(f"\n  Gate analysis counts:")
    print(f"    Marginal trend (bull/bear_bars 3-4): {marginal_trend_count}/{len(fps)} "
          f"({marginal_trend_count/len(fps)*100:.0f}%)")
    print(f"    ADX borderline (<25):               {adx_borderline_count}/{len(fps)} "
          f"({adx_borderline_count/len(fps)*100:.0f}%)")
    print(f"    body_atr borderline (<0.20):         {body_borderline_count}/{len(fps)} "
          f"({body_borderline_count/len(fps)*100:.0f}%)")
    print(f"    AR borderline:                       {ar_borderline_count}/{len(fps)} "
          f"({ar_borderline_count/len(fps)*100:.0f}%)")

    # ── Part E: bull_bars distribution ────────────────────────────────
    print(f"\n\n{'='*100}")
    print("PART E: BULL/BEAR BARS DISTRIBUTION AT ENTRY")
    print(f"{'='*100}")

    tp_bull = [e.bull_bars for e in tps if e.direction == "L"]
    tp_bear = [e.bear_bars for e in tps if e.direction == "S"]
    fp_bull = [e.bull_bars for e in fps if e.direction == "L"]
    fp_bear = [e.bear_bars for e in fps if e.direction == "S"]

    print(f"\n  Long entries:")
    print(f"    TP bull_bars: {Counter(tp_bull).most_common()}")
    print(f"    FP bull_bars: {Counter(fp_bull).most_common()}")
    if tp_bull:
        print(f"    TP mean={sum(tp_bull)/len(tp_bull):.1f}")
    if fp_bull:
        print(f"    FP mean={sum(fp_bull)/len(fp_bull):.1f}")

    print(f"\n  Short entries:")
    print(f"    TP bear_bars: {Counter(tp_bear).most_common()}")
    print(f"    FP bear_bars: {Counter(fp_bear).most_common()}")
    if tp_bear:
        print(f"    TP mean={sum(tp_bear)/len(tp_bear):.1f}")
    if fp_bear:
        print(f"    FP mean={sum(fp_bear)/len(fp_bear):.1f}")

    # ── Part F: First-bar-of-day ordering test ────────────────────────
    print(f"\n\n{'='*100}")
    print("PART F: FIRST-BAR-OF-DAY ORDERING DIVERGENCE")
    print(f"{'='*100}")
    print("""
Critical test: Do any FP entries occur on bars where bull_bars or bear_bars
is EXACTLY trend_bars (3), suggesting the off-by-one from reset ordering
let them barely qualify?
""")

    fp_exactly_at_threshold = [
        fp for fp in fps
        if (fp.direction == "L" and fp.bull_bars == config.trend_bars)
        or (fp.direction == "S" and fp.bear_bars == config.trend_bars)
    ]

    print(f"  FPs where bull/bear_bars == {config.trend_bars} (exactly at threshold):")
    print(f"    Count: {len(fp_exactly_at_threshold)}/{len(fps)} ({len(fp_exactly_at_threshold)/len(fps)*100:.0f}%)")

    for fp in fp_exactly_at_threshold[:10]:
        print(f"    {fp.bar_label} | {fp.direction} | "
              f"bull={fp.bull_bars} bear={fp.bear_bars} | "
              f"bar_of_day={fp.bar_of_day} | ADX={fp.adx_val:.1f}")

    # ── Part G: FP entries at bars 4-6 (early, where off-by-one matters most)
    print(f"\n  FPs at early bars (bar_of_day 11-15, where off-by-one matters most):")
    early_fps = [fp for fp in fps if fp.bar_of_day <= 15]
    print(f"    Count: {len(early_fps)}/{len(fps)}")
    for fp in early_fps:
        print(f"    {fp.bar_label} | {fp.direction} | "
              f"bull={fp.bull_bars} bear={fp.bear_bars} | bar={fp.bar_of_day}")

    # ── Part H: ATR computation comparison ────────────────────────────
    print(f"\n\n{'='*100}")
    print("PART H: ATR COMPUTATION COMPARISON")
    print(f"{'='*100}")
    print("""
Python ATR uses Wilder smoothing: ATR = (prev_ATR * 13 + TR) / 14
TV ta.atr(14) also uses Wilder smoothing.

Python seed: SMA of first 14 TRs (bars 1..14).
TV seed: ta.atr seeds with SMA of first period TRs starting from bar 0.

Key difference: Python's StreamingATR starts seeding from bar 1 (index 1),
accumulating period TRs. TV starts from bar 0. But TR at bar 0 = high-low
(no prev close), same as Python.

The ATR lookback window is the main concern: Python computes from bar 0
of the entire dataset, including all history. TV also computes from bar 0
of the chart. If both use the same data range, ATR should converge.
""")

    # Show ATR at a few FP bars vs what we can infer from TV
    print(f"  ATR values at first 10 FP bars:")
    for fp in fps[:10]:
        print(f"    {fp.bar_label} | ATR={fp.atr_val:.4f} | body_atr={fp.body_atr:.3f} "
              f"(thresh={float(config.pb_body)})")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("SUMMARY OF FINDINGS")
    print(f"{'='*100}")
    print(f"""
  Python PB entries: {len(py_entries)}
  TV PB entries:     {len(tv_trades)}
  True Positives:    {len(tps)}
  False Positives:   {len(fps)} (Python enters, TV does NOT)
  False Negatives:   {len(fns)} (TV enters, Python does NOT)

  LIKELY ROOT CAUSES OF {len(fps)} FALSE POSITIVES:

  1. BULL/BEAR BARS RESET ORDERING (off-by-one on each new day):
     - {len(fp_exactly_at_threshold)}/{len(fps)} FPs have bull/bear_bars exactly at threshold ({config.trend_bars})
     - Python resets THEN updates: bar 1 gets bull_bars=1
     - TV updates THEN resets: bar 1 always has bull_bars=0
     - Effect: Python reaches trend threshold ~1 bar earlier each day

  2. ADX DIVERGENCE:
     - Mean ADX delta: {sum(adx_deltas)/len(adx_deltas):+.2f} (Python vs TV) for matched trades
     - {adx_borderline_count}/{len(fps)} FPs have ADX near threshold (<25)
     - Wilder smoothing seed/lookback window differences amplify

  3. ATR DIVERGENCE (affects body_atr gate):
     - {body_borderline_count}/{len(fps)} FPs have body_atr near threshold
     - ATR differences shift the body/ATR ratio up/down

  4. AR FILTER DIVERGENCE:
     - {ar_borderline_count}/{len(fps)} FPs have AR near threshold edges
""")

    print("=" * 100)


if __name__ == "__main__":
    main()
