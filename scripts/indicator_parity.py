"""Indicator-by-indicator comparison between Python and TradingView for PB trades.

Identifies exactly which indicators diverge and by how much for every TV PB
entry bar that Python misses.

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/indicator_parity.py

Requires:
    - TV export CSV at ~/Downloads/VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv
    - SPY 5-minute bars at data/SPY/5m_bars.csv
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

D = Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.strategies.intraday.pullback import (
    PullbackStrategy,
    PullbackStrategyConfig,
)
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.session import BarContext
from stockdownloader.strategies.intraday.trade_mgmt import IntradayExitManager
from stockdownloader.strategies.intraday.trail import VwapRatchetTrail
from stockdownloader.indicators.intraday import candle_strength
from stockdownloader.core.math import ZERO

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"
TV_CSV = Path.home() / "Downloads" / "VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv"

# ── TV export timezone ────────────────────────────────────────────────
# TV export timestamps are bar CLOSE time in US/Pacific.
# Python bars use bar OPEN time in US/Eastern.
# Pacific + 3h = Eastern, then -5min for close->open = net +2h55m.
TV_TO_ET_OFFSET = timedelta(hours=2, minutes=55)


# =====================================================================
# Part 1: Parse TV signal metadata
# =====================================================================


@dataclass
class TVTrade:
    """Parsed TV PB entry trade."""
    trade_num: int
    direction: str           # "L" or "S"
    tv_datetime: str         # raw TV timestamp (Pacific close time)
    et_bar_label: str        # computed Python bar label (ET open time)
    price: float
    signal_raw: str
    # Parsed signal fields
    mode: str = ""
    vol_score: int = 0       # V:
    sr_score: int = 0        # SR:
    sr_x: int = 0            # SRx:
    rsi_score: int = 0       # RSI:
    time_score: int = 0      # T:
    pq_score: int = 0        # PQ:
    box_score: int = 0       # BX:
    total_score: str = ""    # S:x/y
    rel_vol: float = 0.0     # RV:
    tod_rvol: float = 0.0    # TRV:
    adx: float = 0.0         # ADX:
    trend_dir: int = 0       # TD:
    rr: float = 0.0          # RR:
    vxc: int = 0             # VXC:
    sr_levels: str = ""      # SRL:
    ar: float = 0.0          # AR:
    vwap_delta: float = 0.0  # VD:
    cvd: float = 0.0         # CVD:
    box_pos: float = 0.0     # BOX:
    va: float = 0.0          # VA:
    htf_trend: int = 0       # HTF:
    age: int = 0             # AGE:
    dow: int = 0             # DOW:
    dhod: float = 0.0        # DHOD:
    lrs: float = 0.0         # LRS:


def parse_tv_signal(signal: str) -> dict[str, str]:
    """Parse a pipe-delimited TV signal into key-value pairs."""
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


def _safe_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def tv_time_to_et_bar_label(tv_dt_str: str) -> str:
    """Convert TV export timestamp (Pacific close time) to Python bar label (ET open time).

    TV exports timestamps as bar close time in US/Pacific.
    Python bars are labeled by bar open time in US/Eastern.
    The offset is: +3h for Pacific->Eastern, -5min for close->open = net +2h55m.
    """
    # Parse TV datetime (naive, in Pacific)
    dt = datetime.strptime(tv_dt_str.strip(), "%Y-%m-%d %H:%M")
    # Add offset to convert to ET bar label
    et_dt = dt + TV_TO_ET_OFFSET
    # Format to match Python bar labels (without timezone suffix)
    return et_dt.strftime("%Y-%m-%d %H:%M")


def load_tv_pb_trades(csv_path: Path) -> list[TVTrade]:
    """Load and parse all PB entry trades from TV export CSV."""
    trades: list[TVTrade] = []

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
            price = _safe_float(row.get("Price USD", "0"))

            et_label = tv_time_to_et_bar_label(tv_dt)

            trade = TVTrade(
                trade_num=trade_num,
                direction=direction,
                tv_datetime=tv_dt,
                et_bar_label=et_label,
                price=price,
                signal_raw=signal,
                mode=parsed.get("mode", "PB"),
                vol_score=_safe_int(parsed.get("V", "0")),
                sr_score=_safe_int(parsed.get("SR", "0")),
                sr_x=_safe_int(parsed.get("SRx", "0")),
                rsi_score=_safe_int(parsed.get("RSI", "0")),
                time_score=_safe_int(parsed.get("T", "0")),
                pq_score=_safe_int(parsed.get("PQ", "0")),
                box_score=_safe_int(parsed.get("BX", "0")),
                total_score=parsed.get("S", ""),
                rel_vol=_safe_float(parsed.get("RV", "0")),
                tod_rvol=_safe_float(parsed.get("TRV", "0")),
                adx=_safe_float(parsed.get("ADX", "0")),
                trend_dir=_safe_int(parsed.get("TD", "0")),
                rr=_safe_float(parsed.get("RR", "0")),
                vxc=_safe_int(parsed.get("VXC", "0")),
                sr_levels=parsed.get("SRL", ""),
                ar=_safe_float(parsed.get("AR", "0")),
                vwap_delta=_safe_float(parsed.get("VD", "0")),
                cvd=_safe_float(parsed.get("CVD", "0")),
                box_pos=_safe_float(parsed.get("BOX", "0")),
                va=_safe_float(parsed.get("VA", "0")),
                htf_trend=_safe_int(parsed.get("HTF", "0")),
                age=_safe_int(parsed.get("AGE", "0")),
                dow=_safe_int(parsed.get("DOW", "0")),
                lrs=_safe_float(parsed.get("LRS", "0")),
            )
            trades.append(trade)

    return trades


# =====================================================================
# Part 2: Load Python bars and build timestamp index
# =====================================================================


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


def build_bar_index(bars: list[IntradayPriceData]) -> dict[str, int]:
    """Build a map from 'YYYY-MM-DD HH:MM' -> bar index for fast lookups."""
    index: dict[str, int] = {}
    for i, bar in enumerate(bars):
        # bar.date is like '2025-02-20 09:30:00-05:00'
        # Extract 'YYYY-MM-DD HH:MM'
        key = bar.date[:16]
        index[key] = i
    return index


# =====================================================================
# Part 3: Run Python strategy and capture BarContext at TV entry bars
# =====================================================================


def build_tv_parity_pb_config() -> PullbackStrategyConfig:
    """Build PB config with TV-parity overrides (from tv_parity_backtest.py)."""
    return PullbackStrategyConfig(
        # Entry
        allow_shorts=True,
        allow_longs=True,
        pb_zone=D("0.5"),
        pb_body=D("0.15"),
        rr=D("1.4"),
        sl_atr=D("1.3"),
        sl_cap=D("1.50"),
        # Trend
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
        # Confluence
        w_vol=3,
        w_sr=2,
        w_rsi=1,
        w_time=0,
        w_pq=1,
        w_box=0,
        min_score=3,
        min_score_long=5,
        pq_max_cross=3,
        # Risk
        max_day=2,
        spacing=3,
        circuit=3,
        day_loss=D("3.0"),
        no_friday_short=True,
        no_monday_long=True,
        # Exit
        be_trigger=D("0.5"),
        trail_vwap=True,
        trail_buf=D("0.15"),
        trail_keep_tp=True,
        close_eod=True,
        # PB extras
        pb_vwap_bias=True,
        pb_tp_mode="rr",
    )


@dataclass
class PyIndicators:
    """Python-computed indicator values at a specific bar."""
    adx: float = 0.0
    atr: float = 0.0
    atr_fast: float = 0.0
    rsi: float = 0.0
    vwap: float = 0.0
    std_dev: float = 0.0
    close: float = 0.0
    dist_vwap: float = 0.0
    zone_w: float = 0.0
    in_zone: bool = False
    trend_dir: int = 0
    bull_bars: int = 0
    bear_bars: int = 0
    vwap_crosses: int = 0
    tod_rvol: float = 0.0
    rel_vol: float = 0.0
    htf_trend: int = 0
    ar: float = 0.0
    vwap_delta: float = 0.0
    vwap_accel: float = 0.0
    cvd_norm: float = 0.0
    lrs_atr: float = 0.0
    is_good_time: bool = False
    bar_of_day: int = 0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    sr_any: bool = False
    sr_score_count: int = 0
    clean_pb: bool = False
    box_pos: float = 0.0
    dow: int = 0
    cum_bars_above_vwap: int = 0
    cum_bars_below_vwap: int = 0
    day_trades: int = 0
    last_entry_bar: int = -100
    consec_losses: int = 0
    tripped: bool = False
    day_limited: bool = False
    in_position: bool = False
    # Candle analysis
    candle_body: float = 0.0
    is_bull: bool = False
    is_bear: bool = False
    bull_candle: bool = False
    bear_candle: bool = False
    bull_wick: bool = False
    bear_wick: bool = False
    # Confluence scores
    pts_vol: int = 0
    pts_sr: int = 0
    pts_sr2: int = 0
    pts_rsi: int = 0
    pts_time: int = 0
    pts_pq: int = 0
    pts_box: int = 0
    total_score: int = 0


def extract_indicators_at_bar(
    bars: list[IntradayPriceData],
    target_idx: int,
    config: PullbackStrategyConfig,
) -> PyIndicators | None:
    """Run IntradayInfra through all bars up to target_idx, return indicators.

    Uses a fresh IntradayInfra instance so indicator state is clean.
    We process every bar up to and including target_idx, capturing the
    BarContext at the target bar. If the strategy is in a position at
    that bar (ctx is None), we still extract raw indicator values.
    """
    infra = IntradayInfra(config, IntradayExitManager(VwapRatchetTrail()))

    ctx: BarContext | None = None
    for i in range(target_idx + 1):
        ctx = infra.on_new_bar(bars, i)

    # Even if ctx is None (in_position), extract what we can from infra's hub
    # and state
    s = infra.state
    hub = infra.hub
    bar = bars[target_idx]

    # Raw indicators from hub
    atr_val = hub.atr(bars, target_idx, config.adx_len)
    atr_fast = hub.atr(bars, target_idx, 5)
    adx_result = hub.adx(bars, target_idx, config.adx_len)
    adx_val = adx_result.adx
    rsi_val = hub.rsi(bars, target_idx, 10)
    ema_fast = hub.ema(bars, target_idx, config.ema_fast)
    ema_slow = hub.ema(bars, target_idx, config.ema_slow)
    v_delta = hub.vwap_slope(bars, target_idx, config.slope_period)
    v_accel = hub.vwap_acceleration(bars, target_idx, config.slope_period)
    lrs_val = hub.lrs_normalized(bars, target_idx, 15)
    cvd_val = hub.cvd_normalized(bars, target_idx)
    t_rvol = hub.tod_rvol(bars, target_idx, config.tod_days, config.bars_per_day)
    r_vol = hub.rel_vol(bars, target_idx)
    htf = hub.htf_ema_trend(bars, target_idx)
    vwap_bands = hub.extended_session_vwap_bands(bars, target_idx)

    vwap = vwap_bands.vwap
    std_dev = vwap_bands.std_dev
    zone_w = std_dev * config.pb_zone
    dist_vwap = abs(bar.close - vwap)
    in_zone = zone_w > ZERO and dist_vwap <= zone_w

    # Trend direction
    noise = atr_val * D("0.10")
    trend_dir = PullbackStrategy._trend_dir(
        s.bull_bars, s.bear_bars, config.trend_bars,
        v_delta, atr_val,
    )

    # AR ratio
    ar = float(atr_fast / atr_val) if atr_val > ZERO else 0.0

    # is_good_time
    bar_of_day = s.bar_count
    is_good_time = (
        bar_of_day >= config.can_trade_bar
        and bar_of_day <= config.eod_bar
        and not (config.lunch_start <= bar_of_day <= config.lunch_end)
    )

    # S/R
    from stockdownloader.indicators.intraday import compute_sr_score
    sr_any, sr_score_count = compute_sr_score(
        bar.close,
        pd_high=s.pd_high,
        pd_low=s.pd_low,
        pd_close=s.pd_close,
        or_high=s.or_high if s.or_done else ZERO,
        or_low=s.or_low if s.or_done else ZERO,
        pw_high=s.pw_high,
        pw_low=s.pw_low,
        prev_vwap=s.prev_vwap_close,
        avwap=ZERO,
        proximity_pct=config.sr_prox,
        sr_pdhlc=config.sr_pdhlc,
        sr_round=config.sr_round,
        sr_or=config.sr_or,
        sr_week_hl=config.sr_week_hl,
        sr_prev_vwap=config.sr_prev_vwap,
        sr_avwap=False,
    )

    # Box position
    pd_range = s.pd_high - s.pd_low
    if pd_range > ZERO:
        box_pos = max(ZERO, min(D("1"), (bar.close - s.pd_low) / pd_range))
    else:
        box_pos = D("0.5")

    clean_pb = s.vwap_crosses <= config.pq_max_cross
    dow = bar.datetime_parsed.weekday()

    # Candle analysis
    cs = candle_strength(bar, atr_val)
    vwap_tol = atr_val * D("0.25")
    bull_c = cs.bull_candle(config.pb_body)
    bear_c = cs.bear_candle(config.pb_body)

    # Confluence scoring (direction-dependent)
    # We compute for the TV direction
    if t_rvol < D("1.0"):
        pts_vol = config.w_vol
    elif t_rvol < D("2.0"):
        pts_vol = 1
    else:
        pts_vol = 0

    pts_sr = config.w_sr if sr_score_count > 0 else 0
    pts_sr2 = 1 if sr_score_count >= 2 else 0
    pts_time = config.w_time if is_good_time else 0
    pts_pq = config.w_pq if clean_pb else 0
    pts_rsi = 0  # set based on direction below
    pts_box = 0

    result = PyIndicators(
        adx=float(adx_val),
        atr=float(atr_val),
        atr_fast=float(atr_fast),
        rsi=float(rsi_val),
        vwap=float(vwap),
        std_dev=float(std_dev),
        close=float(bar.close),
        dist_vwap=float(dist_vwap),
        zone_w=float(zone_w),
        in_zone=in_zone,
        trend_dir=trend_dir,
        bull_bars=s.bull_bars,
        bear_bars=s.bear_bars,
        vwap_crosses=s.vwap_crosses,
        tod_rvol=float(t_rvol),
        rel_vol=float(r_vol),
        htf_trend=htf,
        ar=ar,
        vwap_delta=float(v_delta),
        vwap_accel=float(v_accel),
        cvd_norm=float(cvd_val),
        lrs_atr=float(lrs_val),
        is_good_time=is_good_time,
        bar_of_day=bar_of_day,
        ema_fast=float(ema_fast),
        ema_slow=float(ema_slow),
        sr_any=sr_any,
        sr_score_count=sr_score_count,
        clean_pb=clean_pb,
        box_pos=float(box_pos),
        dow=dow,
        cum_bars_above_vwap=s.cum_bars_above_vwap,
        cum_bars_below_vwap=s.cum_bars_below_vwap,
        day_trades=s.day_trades,
        last_entry_bar=s.last_entry_bar,
        consec_losses=s.consec_losses,
        tripped=s.tripped,
        day_limited=s.day_limited,
        in_position=s.in_position,
        candle_body=float(cs.body),
        is_bull=cs.is_bull,
        is_bear=cs.is_bear,
        bull_candle=bull_c,
        bear_candle=bear_c,
        bull_wick=cs.bull_wick,
        bear_wick=cs.bear_wick,
        pts_vol=pts_vol,
        pts_sr=pts_sr,
        pts_sr2=pts_sr2,
        pts_rsi=0,  # set below
        pts_time=pts_time,
        pts_pq=pts_pq,
        pts_box=0,
        total_score=0,
    )

    return result


def compute_confluence(py: PyIndicators, go_long: bool, config: PullbackStrategyConfig) -> None:
    """Compute direction-dependent confluence scores in-place."""
    rsi = D(str(py.rsi))
    box = D(str(py.box_pos))

    if go_long:
        py.pts_rsi = config.w_rsi if D("45") <= rsi <= D("70") else 0
        py.pts_box = config.w_box if box <= D("0.33") else 0
    else:
        py.pts_rsi = config.w_rsi if D("30") <= rsi <= D("55") else 0
        py.pts_box = config.w_box if box >= D("0.67") else 0

    py.total_score = (
        py.pts_vol + py.pts_sr + py.pts_sr2 + py.pts_rsi
        + py.pts_time + py.pts_pq + py.pts_box
    )


# =====================================================================
# Part 4: Gate evaluation — determine which gate blocks each trade
# =====================================================================


@dataclass
class GateResult:
    """Result of evaluating a single gate."""
    name: str
    passed: bool
    reason: str = ""


def evaluate_gates(
    py: PyIndicators,
    tv: TVTrade,
    config: PullbackStrategyConfig,
) -> list[GateResult]:
    """Evaluate all PB entry gates and return results for each.

    Order matches _evaluate_entry in pullback.py.
    """
    gates: list[GateResult] = []
    go_long = tv.direction == "L"
    go_short = tv.direction == "S"

    # G0: Risk — day trade limit
    ready = py.day_trades < config.max_day
    gates.append(GateResult(
        "G0: Day limit",
        ready,
        f"day_trades={py.day_trades} < max_day={config.max_day}" if ready
        else f"day_trades={py.day_trades} >= max_day={config.max_day}",
    ))

    # G0b: Risk — spacing
    spaced = (py.bar_of_day - py.last_entry_bar) >= config.spacing or py.last_entry_bar <= 0
    gates.append(GateResult(
        "G0b: Spacing",
        spaced,
        f"gap={py.bar_of_day - py.last_entry_bar} >= spacing={config.spacing}" if spaced
        else f"gap={py.bar_of_day - py.last_entry_bar} < spacing={config.spacing}",
    ))

    # G0c: In position
    gates.append(GateResult(
        "G0c: Not in position",
        not py.in_position,
        "free" if not py.in_position else "ALREADY IN POSITION",
    ))

    # G0d: Circuit breaker / day loss
    gates.append(GateResult(
        "G0d: Circuit breaker",
        not py.tripped and not py.day_limited,
        "OK" if not py.tripped and not py.day_limited
        else f"tripped={py.tripped} day_limited={py.day_limited}",
    ))

    # G1: can_trade_bar
    can_trade = py.bar_of_day >= config.can_trade_bar
    gates.append(GateResult(
        "G1: Can trade bar",
        can_trade,
        f"bar_of_day={py.bar_of_day} >= {config.can_trade_bar}" if can_trade
        else f"bar_of_day={py.bar_of_day} < {config.can_trade_bar}",
    ))

    # G2: ADX threshold
    adx_ok = py.adx >= float(config.adx_thresh)
    gates.append(GateResult(
        "G2: ADX threshold",
        adx_ok,
        f"ADX={py.adx:.1f} >= {config.adx_thresh}" if adx_ok
        else f"ADX={py.adx:.1f} < {config.adx_thresh}",
    ))

    # G3: Trend direction
    if go_long:
        trend_ok = py.trend_dir >= 1
    else:
        trend_ok = py.trend_dir <= -1
    gates.append(GateResult(
        "G3: Trend direction",
        trend_ok,
        f"trend_dir={py.trend_dir} (want {'>=1' if go_long else '<=-1'})" if trend_ok
        else f"trend_dir={py.trend_dir} (NEED {'>=1' if go_long else '<=-1'})",
    ))

    # G4: VWAP zone
    gates.append(GateResult(
        "G4: VWAP zone",
        py.in_zone,
        f"dist={py.dist_vwap:.4f} <= zone_w={py.zone_w:.4f}" if py.in_zone
        else f"dist={py.dist_vwap:.4f} > zone_w={py.zone_w:.4f} (OUTSIDE)",
    ))

    # G5: Candle confirmation
    vwap_tol = py.atr * 0.25
    if go_long:
        bull_at_vwap = py.close >= (py.vwap - vwap_tol) and (py.bull_candle or py.bull_wick)
        strong_bull = py.is_bull and py.candle_body >= py.atr * 0.3 and py.close > py.vwap
        candle_ok = bull_at_vwap or strong_bull
    else:
        bear_at_vwap = py.close <= (py.vwap + vwap_tol) and (py.bear_candle or py.bear_wick)
        strong_bear = py.is_bear and py.candle_body >= py.atr * 0.3 and py.close < py.vwap
        candle_ok = bear_at_vwap or strong_bear
    gates.append(GateResult(
        "G5: Candle confirmation",
        candle_ok,
        f"bull_candle={py.bull_candle} bull_wick={py.bull_wick} strong={'Y' if go_long and (py.is_bull and py.candle_body >= py.atr * 0.3) else 'N'}"
        if go_long else
        f"bear_candle={py.bear_candle} bear_wick={py.bear_wick} strong={'Y' if go_short and (py.is_bear and py.candle_body >= py.atr * 0.3) else 'N'}",
    ))

    # G6: VWAP crosses cap
    vxc_ok = py.vwap_crosses <= config.max_vxc
    gates.append(GateResult(
        "G6: VWAP crosses",
        vxc_ok,
        f"VXC={py.vwap_crosses} <= max={config.max_vxc}" if vxc_ok
        else f"VXC={py.vwap_crosses} > max={config.max_vxc}",
    ))

    # G7: VWAP session bias
    if config.pb_vwap_bias and py.bar_of_day > 0:
        total = py.cum_bars_above_vwap + py.cum_bars_below_vwap
        if total > 0:
            ratio_above = py.cum_bars_above_vwap / total
            ratio_below = py.cum_bars_below_vwap / total
            if go_long:
                bias_ok = ratio_above >= float(config.pb_vwap_bias_pct)
                gates.append(GateResult(
                    "G7: VWAP bias",
                    bias_ok,
                    f"ratio_above={ratio_above:.2f} >= {config.pb_vwap_bias_pct}" if bias_ok
                    else f"ratio_above={ratio_above:.2f} < {config.pb_vwap_bias_pct}",
                ))
            else:
                bias_ok = ratio_below >= float(config.pb_vwap_bias_pct)
                gates.append(GateResult(
                    "G7: VWAP bias",
                    bias_ok,
                    f"ratio_below={ratio_below:.2f} >= {config.pb_vwap_bias_pct}" if bias_ok
                    else f"ratio_below={ratio_below:.2f} < {config.pb_vwap_bias_pct}",
                ))
        else:
            gates.append(GateResult("G7: VWAP bias", True, "total=0 (skip)"))
    else:
        gates.append(GateResult("G7: VWAP bias", True, "disabled or bar_of_day=0"))

    # G8: HTF alignment
    if config.htf_align:
        if go_long:
            htf_ok = py.htf_trend != -1
        else:
            htf_ok = py.htf_trend != 1
        gates.append(GateResult(
            "G8: HTF alignment",
            htf_ok,
            f"HTF={py.htf_trend} (want {'!= -1' if go_long else '!= 1'})" if htf_ok
            else f"HTF={py.htf_trend} (BLOCKS {'long' if go_long else 'short'})",
        ))
    else:
        gates.append(GateResult("G8: HTF alignment", True, "disabled"))

    # G9: AR filter
    if config.ar_filter and py.atr > 0:
        ar_ok = float(config.ar_thresh) <= py.ar <= float(config.ar_cap)
        gates.append(GateResult(
            "G9: AR filter",
            ar_ok,
            f"AR={py.ar:.2f} in [{config.ar_thresh}, {config.ar_cap}]" if ar_ok
            else f"AR={py.ar:.2f} NOT in [{config.ar_thresh}, {config.ar_cap}]",
        ))
    else:
        gates.append(GateResult("G9: AR filter", True, "disabled or atr=0"))

    # G10: VA filter
    if config.va_filter:
        va_ok = py.vwap_accel >= float(config.va_min)
        gates.append(GateResult(
            "G10: VA filter",
            va_ok,
            f"VA={py.vwap_accel:.3f} >= {config.va_min}" if va_ok
            else f"VA={py.vwap_accel:.3f} < {config.va_min}",
        ))
    else:
        gates.append(GateResult("G10: VA filter", True, "disabled"))

    # G11: CVD long filter
    if go_long and config.cvd_long_filter:
        cvd_ok = py.cvd_norm > 0
        gates.append(GateResult(
            "G11: CVD long filter",
            cvd_ok,
            f"CVD={py.cvd_norm:.4f} > 0" if cvd_ok
            else f"CVD={py.cvd_norm:.4f} <= 0 (negative CVD blocks long)",
        ))
    else:
        gates.append(GateResult("G11: CVD long filter", True, "N/A (short or disabled)"))

    # G12: Day-of-week filters
    dow_ok = True
    dow_reason = "OK"
    if go_short and config.no_friday_short and py.dow == 4:
        dow_ok = False
        dow_reason = "no_friday_short blocks"
    if go_long and config.no_monday_long and py.dow == 0:
        dow_ok = False
        dow_reason = "no_monday_long blocks"
    gates.append(GateResult("G12: Day-of-week", dow_ok, dow_reason))

    # G13: LRS short filter
    if go_short and config.lrs_short_filter:
        lrs_ok = py.lrs_atr <= float(config.lrs_thresh)
        gates.append(GateResult(
            "G13: LRS short filter",
            lrs_ok,
            f"LRS={py.lrs_atr:.3f} <= {config.lrs_thresh}" if lrs_ok
            else f"LRS={py.lrs_atr:.3f} > {config.lrs_thresh} (uptrend blocks short)",
        ))
    else:
        gates.append(GateResult("G13: LRS short filter", True, "N/A (long or disabled)"))

    # G14: Confluence score
    compute_confluence(py, go_long, config)
    min_score = config.min_score_long if go_long else config.min_score
    score_ok = py.total_score >= min_score
    max_possible = config.w_vol + config.w_sr + 1 + config.w_rsi + config.w_time + config.w_pq + config.w_box
    gates.append(GateResult(
        "G14: Confluence score",
        score_ok,
        f"score={py.total_score}/{max_possible} >= min={min_score} "
        f"(V:{py.pts_vol} SR:{py.pts_sr} SRx:{py.pts_sr2} RSI:{py.pts_rsi} "
        f"T:{py.pts_time} PQ:{py.pts_pq} BX:{py.pts_box})"
        if score_ok else
        f"score={py.total_score}/{max_possible} < min={min_score} "
        f"(V:{py.pts_vol} SR:{py.pts_sr} SRx:{py.pts_sr2} RSI:{py.pts_rsi} "
        f"T:{py.pts_time} PQ:{py.pts_pq} BX:{py.pts_box})",
    ))

    return gates


# =====================================================================
# Part 5: Display and aggregate
# =====================================================================


@dataclass
class TradeComparison:
    """Complete comparison of one TV trade to Python indicators."""
    tv: TVTrade
    py: PyIndicators | None
    matched: bool             # Was the bar found in Python data?
    python_took_trade: bool   # Did Python also take this trade?
    gates: list[GateResult] = field(default_factory=list)
    blocking_gates: list[str] = field(default_factory=list)


def print_trade_comparison(comp: TradeComparison) -> None:
    """Print side-by-side comparison for a single trade."""
    tv = comp.tv
    py = comp.py

    status = "MATCHED" if comp.python_took_trade else "MISSED"
    print(f"\n{'='*90}")
    print(f"Trade #{tv.trade_num:>3} | {tv.tv_datetime} -> {tv.et_bar_label} ET | "
          f"PB|{tv.direction} | TV price: ${tv.price:.2f} | [{status}]")
    print(f"TV Signal: {tv.signal_raw[:100]}{'...' if len(tv.signal_raw) > 100 else ''}")
    print(f"{'='*90}")

    if py is None:
        print("  ** NO MATCHING BAR IN PYTHON DATA **")
        return

    # Indicator comparison table
    print(f"\n{'Indicator':<24} {'TV Value':>12} {'Python Value':>14} {'Delta':>10} {'Match?':>8}")
    print("-" * 72)

    rows = [
        ("ADX",        f"{tv.adx:.1f}",        f"{py.adx:.1f}",        tv.adx - py.adx),
        ("RVOL (tod)",  f"{tv.tod_rvol:.2f}",   f"{py.tod_rvol:.2f}",   tv.tod_rvol - py.tod_rvol),
        ("RVOL (rel)",  f"{tv.rel_vol:.2f}",    f"{py.rel_vol:.2f}",    tv.rel_vol - py.rel_vol),
        ("trend_dir",  f"{tv.trend_dir}",       f"{py.trend_dir}",      float(tv.trend_dir - py.trend_dir)),
        ("bull_bars",  f"{'-':>6}",             f"{py.bull_bars}",       None),
        ("bear_bars",  f"{'-':>6}",             f"{py.bear_bars}",       None),
        ("dist_vwap",  f"{'-':>6}",             f"{py.dist_vwap:.4f}",   None),
        ("zone_w",     f"{'-':>6}",             f"{py.zone_w:.4f}",      None),
        ("in_zone",    f"{'-':>6}",             f"{'Y' if py.in_zone else 'N'}",  None),
        ("VXC",        f"{tv.vxc}",             f"{py.vwap_crosses}",    float(tv.vxc - py.vwap_crosses)),
        ("HTF",        f"{tv.htf_trend}",       f"{py.htf_trend}",       float(tv.htf_trend - py.htf_trend)),
        ("AR",         f"{tv.ar:.2f}",          f"{py.ar:.2f}",          tv.ar - py.ar),
        ("VD (slope)", f"{tv.vwap_delta:.3f}",  f"{py.vwap_delta:.3f}",  tv.vwap_delta - py.vwap_delta),
        ("CVD",        f"{tv.cvd:.4f}",         f"{py.cvd_norm:.4f}",    tv.cvd - py.cvd_norm),
        ("BOX",        f"{tv.box_pos:.2f}",     f"{py.box_pos:.2f}",     tv.box_pos - py.box_pos),
        ("VA",         f"{tv.va:.3f}",          f"{py.vwap_accel:.3f}",  tv.va - py.vwap_accel),
        ("LRS",        f"{tv.lrs:.3f}",         f"{py.lrs_atr:.3f}",    tv.lrs - py.lrs_atr),
        # TV dayofweek: 1=Sun..7=Sat; Python weekday(): 0=Mon..6=Sun
        # Convert TV to Python convention: tv_dow-2 (mod 7)
        ("DOW",        f"{tv.dow}(TV)",         f"{py.dow}(Py)",         float((tv.dow - 2) % 7 - py.dow)),
        ("RSI",        f"{'-':>6}",             f"{py.rsi:.1f}",         None),
        # TV AGE = bar_of_day in Pine; Python bar_of_day = bar_count in session
        ("bar_of_day", f"{tv.age}(AGE)",        f"{py.bar_of_day}",      float(tv.age - py.bar_of_day)),
    ]

    for name, tv_val, py_val, delta in rows:
        if delta is not None:
            d_str = f"{delta:+.3f}"
            match = "Y" if abs(delta) < 1.0 else "N"
        else:
            d_str = "    -"
            match = "-"
        print(f"  {name:<22} {tv_val:>12} {py_val:>14} {d_str:>10} {match:>8}")

    # Confluence score comparison
    tv_score_parts = tv.total_score.split("/")
    tv_total = _safe_int(tv_score_parts[0]) if tv_score_parts else 0
    tv_max = _safe_int(tv_score_parts[1]) if len(tv_score_parts) > 1 else 0

    print(f"\n  {'Confluence':<22} {'TV':>12} {'Python':>14}")
    print(f"  {'-'*50}")
    print(f"  {'V (volume):':<22} {tv.vol_score:>12} {py.pts_vol:>14}")
    print(f"  {'SR (support/res):':<22} {tv.sr_score:>12} {py.pts_sr:>14}")
    print(f"  {'SRx (extra SR):':<22} {tv.sr_x:>12} {py.pts_sr2:>14}")
    print(f"  {'RSI:':<22} {tv.rsi_score:>12} {py.pts_rsi:>14}")
    print(f"  {'T (time):':<22} {tv.time_score:>12} {py.pts_time:>14}")
    print(f"  {'PQ (pullback qual):':<22} {tv.pq_score:>12} {py.pts_pq:>14}")
    print(f"  {'BX (box):':<22} {tv.box_score:>12} {py.pts_box:>14}")
    print(f"  {'TOTAL:':<22} {tv_total:>12} {py.total_score:>14}")

    # Gate evaluation
    if not comp.python_took_trade:
        print(f"\n  GATE EVALUATION:")
        for g in comp.gates:
            symbol = "PASS" if g.passed else "FAIL"
            print(f"    [{symbol}] {g.name}: {g.reason}")
        if comp.blocking_gates:
            print(f"\n  BLOCKING GATE(S): {', '.join(comp.blocking_gates)}")

    # Position / risk state
    if py.in_position:
        print(f"\n  ** PYTHON IN POSITION AT THIS BAR (cannot enter) **")
    if py.tripped:
        print(f"  ** CIRCUIT BREAKER TRIPPED **")
    if py.day_limited:
        print(f"  ** DAY LOSS LIMIT HIT **")


def print_aggregate_stats(comparisons: list[TradeComparison]) -> None:
    """Print aggregate divergence statistics."""
    missed = [c for c in comparisons if not c.python_took_trade and c.py is not None]
    matched = [c for c in comparisons if c.python_took_trade]
    unmatched_bars = [c for c in comparisons if c.py is None]

    print(f"\n\n{'='*90}")
    print(f"AGGREGATE STATISTICS")
    print(f"{'='*90}")

    print(f"\n  TV PB trades:           {len(comparisons)}")
    print(f"  Python matched:         {len(matched)}")
    print(f"  Python missed:          {len(missed)}")
    print(f"  No bar match:           {len(unmatched_bars)}")
    print(f"  Match rate:             {len(matched)/len(comparisons)*100:.1f}%")

    if not missed:
        print("\n  No missed trades to analyze!")
        return

    # Gate blocking frequency
    print(f"\n  {'GATE BLOCKING FREQUENCY (missed trades)':}")
    print(f"  {'-'*60}")
    gate_counter: Counter = Counter()
    for c in missed:
        for g in c.blocking_gates:
            gate_counter[g] += 1

    for gate, count in gate_counter.most_common():
        pct = count / len(missed) * 100
        print(f"    {gate:<40} {count:>4} ({pct:>5.1f}%)")

    # Top 3 blocking gates
    top3 = gate_counter.most_common(3)
    print(f"\n  TOP 3 BLOCKING GATES:")
    for i, (gate, count) in enumerate(top3, 1):
        print(f"    #{i}: {gate} ({count}/{len(missed)} = {count/len(missed)*100:.1f}%)")

    # Indicator divergence stats
    print(f"\n  {'INDICATOR DIVERGENCE (TV - Python for missed trades)':}")
    print(f"  {'-'*60}")
    print(f"  {'Indicator':<20} {'Mean Delta':>12} {'Max |Delta|':>14} {'Std Dev':>12}")
    print(f"  {'-'*60}")

    # Collect deltas for each indicator
    indicator_deltas: dict[str, list[float]] = defaultdict(list)
    for c in missed:
        if c.py is None:
            continue
        tv, py = c.tv, c.py
        indicator_deltas["ADX"].append(tv.adx - py.adx)
        indicator_deltas["RVOL_tod"].append(tv.tod_rvol - py.tod_rvol)
        indicator_deltas["RVOL_rel"].append(tv.rel_vol - py.rel_vol)
        indicator_deltas["trend_dir"].append(float(tv.trend_dir - py.trend_dir))
        indicator_deltas["VXC"].append(float(tv.vxc - py.vwap_crosses))
        indicator_deltas["HTF"].append(float(tv.htf_trend - py.htf_trend))
        indicator_deltas["AR"].append(tv.ar - py.ar)
        indicator_deltas["VD"].append(tv.vwap_delta - py.vwap_delta)
        indicator_deltas["CVD"].append(tv.cvd - py.cvd_norm)
        indicator_deltas["BOX"].append(tv.box_pos - py.box_pos)
        indicator_deltas["VA"].append(tv.va - py.vwap_accel)
        indicator_deltas["LRS"].append(tv.lrs - py.lrs_atr)
        indicator_deltas["DOW"].append(float((tv.dow - 2) % 7 - py.dow))
        indicator_deltas["bar_of_day"].append(float(tv.age - py.bar_of_day))

    for name, deltas in sorted(indicator_deltas.items()):
        if not deltas:
            continue
        import statistics as stat
        mean_d = sum(deltas) / len(deltas)
        max_abs = max(abs(d) for d in deltas)
        std_d = stat.stdev(deltas) if len(deltas) > 1 else 0.0
        print(f"  {name:<20} {mean_d:>+12.3f} {max_abs:>14.3f} {std_d:>12.3f}")

    # Confluence score comparison
    print(f"\n  {'CONFLUENCE SCORE COMPARISON (missed trades)':}")
    print(f"  {'-'*60}")

    score_deltas: dict[str, list[int]] = defaultdict(list)
    for c in missed:
        if c.py is None:
            continue
        tv, py = c.tv, c.py
        score_deltas["V"].append(tv.vol_score - py.pts_vol)
        score_deltas["SR"].append(tv.sr_score - py.pts_sr)
        score_deltas["SRx"].append(tv.sr_x - py.pts_sr2)
        score_deltas["RSI"].append(tv.rsi_score - py.pts_rsi)
        score_deltas["T"].append(tv.time_score - py.pts_time)
        score_deltas["PQ"].append(tv.pq_score - py.pts_pq)
        score_deltas["BX"].append(tv.box_score - py.pts_box)

    for name, deltas in sorted(score_deltas.items()):
        if not deltas:
            continue
        mean_d = sum(deltas) / len(deltas)
        mismatches = sum(1 for d in deltas if d != 0)
        print(f"    {name:<10} mean_delta={mean_d:>+.2f}  mismatches={mismatches}/{len(deltas)}")


# =====================================================================
# Part 3b: Run Python strategy through entire dataset and record entries
# =====================================================================


def run_python_pb_entries(
    bars: list[IntradayPriceData],
    config: PullbackStrategyConfig,
) -> set[str]:
    """Run PB strategy through all bars and return set of bar timestamps where
    Python takes a PB entry.

    Returns timestamps in 'YYYY-MM-DD HH:MM' format (matching bar_index keys).
    """
    strategy = PullbackStrategy(config=config)
    python_entries: set[str] = set()

    for i in range(len(bars)):
        signal = strategy.evaluate(bars, i)
        if signal.action.name in ("ENTER_LONG", "ENTER_SHORT") and signal.mode == "PB":
            python_entries.add(bars[i].date[:16])
            # Tell strategy the position was opened
            is_long = signal.action.name == "ENTER_LONG"
            strategy.on_position_opened(is_long)
        elif signal.action.name in ("EXIT_LONG", "EXIT_SHORT"):
            strategy.on_position_closed()

    return python_entries


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    print("=" * 90)
    print("INDICATOR PARITY: TV vs Python — PB Trade Analysis")
    print("=" * 90)

    # Validate paths
    if not BARS_CSV.exists():
        print(f"ERROR: SPY 5m data not found at {BARS_CSV}")
        sys.exit(1)
    if not TV_CSV.exists():
        print(f"ERROR: TV export not found at {TV_CSV}")
        sys.exit(1)

    # ── Part 1: Load TV PB trades ─────────────────────────────────────
    print("\n[1/5] Loading TV PB trades...")
    tv_trades = load_tv_pb_trades(TV_CSV)
    print(f"  Found {len(tv_trades)} PB entries in TV export")

    # ── Part 2: Load Python bars ──────────────────────────────────────
    print("\n[2/5] Loading SPY 5m bars...")
    bars = load_5m_bars(BARS_CSV)
    print(f"  Loaded {len(bars)} bars ({bars[0].date[:10]} to {bars[-1].date[:10]})")

    bar_index = build_bar_index(bars)
    print(f"  Built timestamp index with {len(bar_index)} unique timestamps")

    # ── Part 2b: Match TV bars to Python bars ─────────────────────────
    print("\n  Matching TV timestamps to Python bars...")
    match_count = 0
    miss_count = 0
    for tv in tv_trades:
        if tv.et_bar_label in bar_index:
            match_count += 1
        else:
            miss_count += 1
            print(f"    MISS: TV #{tv.trade_num} {tv.tv_datetime} -> {tv.et_bar_label} (no bar)")
    print(f"  Matched: {match_count}/{len(tv_trades)}, Missed: {miss_count}")

    # ── Part 3: Run Python PB strategy ────────────────────────────────
    print("\n[3/5] Running Python PB strategy (full dataset)...")
    config = build_tv_parity_pb_config()
    python_entries = run_python_pb_entries(bars, config)
    print(f"  Python took {len(python_entries)} PB entries")

    # ── Part 4: Compute indicators and compare ────────────────────────
    print("\n[4/5] Computing Python indicators at each TV entry bar...")
    comparisons: list[TradeComparison] = []

    for i, tv in enumerate(tv_trades):
        idx = bar_index.get(tv.et_bar_label)
        if idx is None:
            comparisons.append(TradeComparison(
                tv=tv, py=None, matched=False, python_took_trade=False,
            ))
            continue

        # Did Python take a trade at this bar?
        python_took = tv.et_bar_label in python_entries

        # Compute Python indicators at this bar
        py = extract_indicators_at_bar(bars, idx, config)

        # Evaluate gates
        go_long = tv.direction == "L"
        gates: list[GateResult] = []
        blocking: list[str] = []
        if py is not None:
            gates = evaluate_gates(py, tv, config)
            blocking = [g.name for g in gates if not g.passed]

        comp = TradeComparison(
            tv=tv,
            py=py,
            matched=True,
            python_took_trade=python_took,
            gates=gates,
            blocking_gates=blocking,
        )
        comparisons.append(comp)

        # Progress
        if (i + 1) % 10 == 0 or i == len(tv_trades) - 1:
            print(f"  Processed {i+1}/{len(tv_trades)} trades...")

    # ── Part 4b: Print per-trade comparisons (missed only) ────────────
    print(f"\n[4b/5] Per-trade comparison (MISSED trades only):")
    missed_count = 0
    for comp in comparisons:
        if not comp.python_took_trade:
            print_trade_comparison(comp)
            missed_count += 1

    print(f"\n  Total missed trades shown: {missed_count}")

    # Also show matched trades summary
    matched_trades = [c for c in comparisons if c.python_took_trade]
    if matched_trades:
        print(f"\n  Trades Python DID take ({len(matched_trades)}):")
        for c in matched_trades:
            print(f"    #{c.tv.trade_num:>3} | {c.tv.tv_datetime} | PB|{c.tv.direction} | "
                  f"TV ADX={c.tv.adx:.1f} Py ADX={c.py.adx:.1f}" if c.py else
                  f"    #{c.tv.trade_num:>3} | {c.tv.tv_datetime} | PB|{c.tv.direction}")

    # ── Part 5: Aggregate stats ───────────────────────────────────────
    print_aggregate_stats(comparisons)

    print(f"\n{'='*90}")
    print("DONE")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
