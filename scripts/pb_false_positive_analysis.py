"""PB False-Positive Analysis — compare Python PB entries vs TradingView.

Runs the PB standalone strategy (TV-parity config) over the full dataset,
loads TV PB trades from the export CSV, and categorises each entry as:

- TRUE POSITIVE:  Python and TV both enter on the same bar (+/- 1 bar)
- FALSE POSITIVE: Python enters but TV does not
- FALSE NEGATIVE: TV enters but Python does not

For FALSE POSITIVES, dumps key indicator/gate values to identify what
is letting Python take trades that TV rejects.

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/pb_false_positive_analysis.py
"""
from __future__ import annotations

import csv
import statistics as stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

D = Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.core.math import TWO, ZERO
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.indicators.intraday import candle_strength
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.pullback import (
    PullbackStrategy,
    PullbackStrategyConfig,
)
from stockdownloader.strategies.intraday.session import BarContext
from stockdownloader.strategies.intraday.trade_mgmt import IntradayExitManager
from stockdownloader.strategies.intraday.trail import VwapRatchetTrail

# ── Paths ───────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"
TV_CSV = Path.home() / "Downloads" / "VWAP_v11.2_PS_SPY_AMEX_SPY_2026-02-27_695e3.csv"

# TV export timestamps are bar CLOSE time in US/Pacific.
# Python bars use bar OPEN time in US/Eastern.
# Pacific + 3h = Eastern, then -5min for close->open = net +2h55m.
TV_TO_ET_OFFSET = timedelta(hours=2, minutes=55)


# =====================================================================
# Data loading
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
    """Build map from 'YYYY-MM-DD HH:MM' -> bar index."""
    index: dict[str, int] = {}
    for i, bar in enumerate(bars):
        key = bar.date[:16]
        index[key] = i
    return index


# =====================================================================
# TV trade parsing
# =====================================================================


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


def tv_time_to_et_bar_label(tv_dt_str: str) -> str:
    """Convert TV export timestamp (Pacific close time) to Python bar label (ET open time)."""
    dt = datetime.strptime(tv_dt_str.strip(), "%Y-%m-%d %H:%M")
    et_dt = dt + TV_TO_ET_OFFSET
    return et_dt.strftime("%Y-%m-%d %H:%M")


@dataclass
class TVPBTrade:
    """Parsed TV PB entry trade."""
    trade_num: int
    direction: str           # "L" or "S"
    tv_datetime: str
    et_bar_label: str
    price: float
    signal_raw: str
    # Parsed signal fields
    vol_score: int = 0
    sr_score: int = 0
    sr_x: int = 0
    rsi_score: int = 0
    time_score: int = 0
    pq_score: int = 0
    box_score: int = 0
    total_score: str = ""
    rel_vol: float = 0.0
    tod_rvol: float = 0.0
    adx: float = 0.0
    trend_dir: int = 0
    rr: float = 0.0
    vxc: int = 0
    sr_levels: str = ""
    ar: float = 0.0
    vwap_delta: float = 0.0
    cvd: float = 0.0
    box_pos: float = 0.0
    va: float = 0.0
    htf_trend: int = 0
    age: int = 0
    dow: int = 0
    dhod: float = 0.0
    lrs: float = 0.0


def load_tv_pb_trades(csv_path: Path) -> list[TVPBTrade]:
    """Load and parse all PB entry trades from TV export CSV."""
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
            price = _safe_float(row.get("Price USD", "0"))
            et_label = tv_time_to_et_bar_label(tv_dt)

            trades.append(TVPBTrade(
                trade_num=trade_num,
                direction=direction,
                tv_datetime=tv_dt,
                et_bar_label=et_label,
                price=price,
                signal_raw=signal,
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
                dhod=_safe_float(parsed.get("DHOD", "0")),
                lrs=_safe_float(parsed.get("LRS", "0")),
            ))
    return trades


# =====================================================================
# TV-parity PB config (mirrors tv_parity_backtest.py)
# =====================================================================


def build_tv_parity_pb_config() -> PullbackStrategyConfig:
    return PullbackStrategyConfig(
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
        pb_vwap_bias=False,        # TV: no VWAP session bias filter
        pb_tp_mode="rr",
    )


# =====================================================================
# Run Python PB strategy and capture entries with indicator snapshots
# =====================================================================


@dataclass
class PyEntry:
    """A Python PB entry with indicator snapshot."""
    bar_label: str          # 'YYYY-MM-DD HH:MM'
    bar_idx: int
    direction: str          # 'L' or 'S'
    # Indicators
    adx: float = 0.0
    trend_dir: int = 0
    bull_bars: int = 0
    bear_bars: int = 0
    dist_vwap: float = 0.0
    zone_w: float = 0.0
    band_width: float = 0.0
    body: float = 0.0
    is_bull: bool = False
    is_bear: bool = False
    bull_candle: bool = False
    bear_candle: bool = False
    bull_wick: bool = False
    bear_wick: bool = False
    htf_trend: int = 0
    cvd_norm: float = 0.0
    lrs_atr: float = 0.0
    tod_rvol: float = 0.0
    rel_vol: float = 0.0
    ar: float = 0.0
    vwap_delta: float = 0.0
    vwap_accel: float = 0.0
    vwap_crosses: int = 0
    rsi: float = 0.0
    sr_score_count: int = 0
    clean_pb: bool = False
    box_pos: float = 0.0
    dow: int = 0
    bar_of_day: int = 0
    # Confluence
    pts_vol: int = 0
    pts_sr: int = 0
    pts_sr2: int = 0
    pts_rsi: int = 0
    pts_time: int = 0
    pts_pq: int = 0
    pts_box: int = 0
    total_score: int = 0
    # Close / VWAP
    close: float = 0.0
    vwap: float = 0.0


class InstrumentedPullbackStrategy(PullbackStrategy):
    """PB strategy that records indicator snapshots for every entry taken."""

    def __init__(self, config: PullbackStrategyConfig):
        super().__init__(config=config)
        self.entries: list[PyEntry] = []

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        signal = super()._evaluate_entry(ctx)
        if signal is not None and signal.action.name in ("ENTER_LONG", "ENTER_SHORT"):
            c = self._c
            bar = ctx.bar
            s = ctx.state

            vwap = ctx.vwap_bands.vwap
            band_width = ctx.vwap_bands.std_dev * TWO
            zone_w = band_width * c.pb_zone
            dist_vwap = abs(bar.close - vwap)

            cs = candle_strength(bar, ctx.atr_val)

            go_long = signal.action.name == "ENTER_LONG"

            # Trend dir
            trend_dir = self._trend_dir(
                s.bull_bars, s.bear_bars, c.trend_bars,
                ctx.vwap_delta, ctx.atr_val,
            )

            # AR
            ar = float(ctx.atr_fast / ctx.atr_val) if ctx.atr_val > ZERO else 0.0

            # Confluence (recompute to capture per-component)
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
            else:
                pts_rsi = c.w_rsi if D("30") <= ctx.rsi_val <= D("55") else 0
                pts_box = c.w_box if ctx.box_pos >= D("0.67") else 0
            total_score = pts_vol + pts_sr + pts_sr2 + pts_rsi + pts_time + pts_pq + pts_box

            entry = PyEntry(
                bar_label=bar.date[:16],
                bar_idx=0,  # set externally
                direction="L" if go_long else "S",
                adx=float(ctx.adx_val),
                trend_dir=trend_dir,
                bull_bars=s.bull_bars,
                bear_bars=s.bear_bars,
                dist_vwap=float(dist_vwap),
                zone_w=float(zone_w),
                band_width=float(band_width),
                body=float(cs.body),
                is_bull=cs.is_bull,
                is_bear=cs.is_bear,
                bull_candle=cs.bull_candle(c.pb_body),
                bear_candle=cs.bear_candle(c.pb_body),
                bull_wick=cs.bull_wick,
                bear_wick=cs.bear_wick,
                htf_trend=ctx.htf_trend,
                cvd_norm=float(ctx.cvd_norm),
                lrs_atr=float(ctx.lrs_atr),
                tod_rvol=float(ctx.tod_rvol),
                rel_vol=float(ctx.rel_vol),
                ar=ar,
                vwap_delta=float(ctx.vwap_delta),
                vwap_accel=float(ctx.vwap_accel),
                vwap_crosses=s.vwap_crosses,
                rsi=float(ctx.rsi_val),
                sr_score_count=ctx.sr_score_count,
                clean_pb=ctx.clean_pb,
                box_pos=float(ctx.box_pos),
                dow=ctx.dow,
                bar_of_day=ctx.bar_of_day,
                pts_vol=pts_vol,
                pts_sr=pts_sr,
                pts_sr2=pts_sr2,
                pts_rsi=pts_rsi,
                pts_time=pts_time,
                pts_pq=pts_pq,
                pts_box=pts_box,
                total_score=total_score,
                close=float(bar.close),
                vwap=float(vwap),
            )
            self.entries.append(entry)

        return signal


def run_python_pb(bars: list[IntradayPriceData], config: PullbackStrategyConfig) -> list[PyEntry]:
    """Run PB strategy and return all entry snapshots."""
    strategy = InstrumentedPullbackStrategy(config=config)

    for i in range(len(bars)):
        signal = strategy.evaluate(bars, i)
        if signal.action.name in ("ENTER_LONG", "ENTER_SHORT"):
            strategy.on_position_opened(signal.action.name == "ENTER_LONG")
            # set bar_idx on last entry
            if strategy.entries:
                strategy.entries[-1].bar_idx = i
        elif signal.action.name in ("EXIT_LONG", "EXIT_SHORT"):
            strategy.on_position_closed()

    return strategy.entries


# =====================================================================
# Matching logic
# =====================================================================


@dataclass
class MatchResult:
    category: str  # 'TP', 'FP', 'FN'
    py_entry: PyEntry | None = None
    tv_trade: TVPBTrade | None = None


def match_entries(
    py_entries: list[PyEntry],
    tv_trades: list[TVPBTrade],
    bar_index: dict[str, int],
) -> list[MatchResult]:
    """Match Python entries to TV entries with +/- 1 bar tolerance.

    Returns list of TP, FP, FN results.
    """
    results: list[MatchResult] = []

    # Build sets for quick lookup
    # For tolerance matching, build index from TV label -> list of TV trades
    tv_by_label: dict[str, list[TVPBTrade]] = defaultdict(list)
    for tv in tv_trades:
        tv_by_label[tv.et_bar_label].append(tv)

    # Build set of TV bar indices for tolerance matching
    tv_bar_indices: dict[int, TVPBTrade] = {}
    for tv in tv_trades:
        idx = bar_index.get(tv.et_bar_label)
        if idx is not None:
            tv_bar_indices[idx] = tv

    # Track which TV trades have been matched
    matched_tv: set[int] = set()  # TV trade_num
    matched_py: set[int] = set()  # index into py_entries

    # Pass 1: Match Python entries to TV (finding TPs and FPs)
    for pi, py in enumerate(py_entries):
        py_idx = bar_index.get(py.bar_label)
        if py_idx is None:
            continue

        # Check +/- 1 bar tolerance
        found_tv: TVPBTrade | None = None
        for offset in (0, -1, 1):
            check_idx = py_idx + offset
            if check_idx in tv_bar_indices:
                candidate = tv_bar_indices[check_idx]
                if candidate.trade_num not in matched_tv:
                    found_tv = candidate
                    break

        if found_tv is not None:
            results.append(MatchResult("TP", py_entry=py, tv_trade=found_tv))
            matched_tv.add(found_tv.trade_num)
            matched_py.add(pi)
        else:
            results.append(MatchResult("FP", py_entry=py, tv_trade=None))

    # Pass 2: Find TV trades with no Python match (FNs)
    for tv in tv_trades:
        if tv.trade_num not in matched_tv:
            results.append(MatchResult("FN", py_entry=None, tv_trade=tv))

    return results


# =====================================================================
# Indicator extraction at a specific bar (for FN analysis)
# =====================================================================


def extract_indicators_at_bar(
    bars: list[IntradayPriceData],
    target_idx: int,
    config: PullbackStrategyConfig,
) -> PyEntry | None:
    """Run IntradayInfra through all bars up to target_idx, return indicator snapshot."""
    infra = IntradayInfra(config, IntradayExitManager(VwapRatchetTrail()))

    ctx: BarContext | None = None
    for i in range(target_idx + 1):
        ctx = infra.on_new_bar(bars, i)

    if ctx is None:
        return None

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

    return PyEntry(
        bar_label=bar.date[:16],
        bar_idx=target_idx,
        direction="?",
        adx=float(ctx.adx_val),
        trend_dir=trend_dir,
        bull_bars=s.bull_bars,
        bear_bars=s.bear_bars,
        dist_vwap=float(dist_vwap),
        zone_w=float(zone_w),
        band_width=float(band_width),
        body=float(cs.body),
        is_bull=cs.is_bull,
        is_bear=cs.is_bear,
        bull_candle=cs.bull_candle(config.pb_body),
        bear_candle=cs.bear_candle(config.pb_body),
        bull_wick=cs.bull_wick,
        bear_wick=cs.bear_wick,
        htf_trend=ctx.htf_trend,
        cvd_norm=float(ctx.cvd_norm),
        lrs_atr=float(ctx.lrs_atr),
        tod_rvol=float(ctx.tod_rvol),
        rel_vol=float(ctx.rel_vol),
        ar=ar,
        vwap_delta=float(ctx.vwap_delta),
        vwap_accel=float(ctx.vwap_accel),
        vwap_crosses=s.vwap_crosses,
        rsi=float(ctx.rsi_val),
        sr_score_count=ctx.sr_score_count,
        clean_pb=ctx.clean_pb,
        box_pos=float(ctx.box_pos),
        dow=ctx.dow,
        bar_of_day=ctx.bar_of_day,
        close=float(bar.close),
        vwap=float(vwap),
    )


# =====================================================================
# Display helpers
# =====================================================================


def print_entry_detail(label: str, py: PyEntry, config: PullbackStrategyConfig) -> None:
    """Print detailed indicator dump for one entry."""
    max_score = config.w_vol + config.w_sr + 1 + config.w_rsi + config.w_time + config.w_pq + config.w_box
    print(f"  {label}")
    print(f"    bar: {py.bar_label}  dir: {py.direction}  close: ${py.close:.2f}  vwap: ${py.vwap:.2f}")
    print(f"    --- Trend ---")
    print(f"      trend_dir={py.trend_dir}  bull_bars={py.bull_bars}  bear_bars={py.bear_bars}")
    print(f"      ADX={py.adx:.1f}  HTF={py.htf_trend}")
    print(f"    --- VWAP Zone ---")
    print(f"      dist_vwap={py.dist_vwap:.4f}  zone_w={py.zone_w:.4f}  band_width={py.band_width:.4f}")
    print(f"      VXC={py.vwap_crosses}  VD(slope)={py.vwap_delta:.3f}  VA(accel)={py.vwap_accel:.3f}")
    print(f"    --- Candle ---")
    print(f"      body={py.body:.4f}  is_bull={py.is_bull}  is_bear={py.is_bear}")
    print(f"      bull_candle={py.bull_candle}  bear_candle={py.bear_candle}")
    print(f"      bull_wick={py.bull_wick}  bear_wick={py.bear_wick}")
    print(f"    --- Filters ---")
    print(f"      AR={py.ar:.3f}  CVD={py.cvd_norm:.4f}  LRS={py.lrs_atr:.3f}")
    print(f"      todRVOL={py.tod_rvol:.2f}  relVOL={py.rel_vol:.2f}")
    print(f"    --- Confluence ({py.total_score}/{max_score}) ---")
    print(f"      V:{py.pts_vol} SR:{py.pts_sr} SRx:{py.pts_sr2} RSI:{py.pts_rsi}"
          f" T:{py.pts_time} PQ:{py.pts_pq} BX:{py.pts_box}")
    print(f"    --- Context ---")
    print(f"      RSI={py.rsi:.1f}  BOX={py.box_pos:.2f}  DOW={py.dow}  bar_of_day={py.bar_of_day}")
    print(f"      SR_count={py.sr_score_count}  clean_pb={py.clean_pb}")


def compare_tp_fp_distributions(
    tp_entries: list[PyEntry],
    fp_entries: list[PyEntry],
    config: PullbackStrategyConfig,
) -> None:
    """Compare indicator distributions between TP and FP entries."""
    if not tp_entries or not fp_entries:
        return

    print(f"\n{'='*100}")
    print("INDICATOR DISTRIBUTION: TRUE POSITIVES vs FALSE POSITIVES")
    print(f"{'='*100}")

    fields = [
        ("ADX", lambda e: e.adx),
        ("trend_dir", lambda e: float(e.trend_dir)),
        ("bull_bars", lambda e: float(e.bull_bars)),
        ("bear_bars", lambda e: float(e.bear_bars)),
        ("dist_vwap", lambda e: e.dist_vwap),
        ("zone_w", lambda e: e.zone_w),
        ("band_width", lambda e: e.band_width),
        ("body", lambda e: e.body),
        ("VXC", lambda e: float(e.vwap_crosses)),
        ("VD(slope)", lambda e: e.vwap_delta),
        ("VA(accel)", lambda e: e.vwap_accel),
        ("AR", lambda e: e.ar),
        ("CVD", lambda e: e.cvd_norm),
        ("LRS", lambda e: e.lrs_atr),
        ("HTF", lambda e: float(e.htf_trend)),
        ("todRVOL", lambda e: e.tod_rvol),
        ("relVOL", lambda e: e.rel_vol),
        ("RSI", lambda e: e.rsi),
        ("BOX", lambda e: e.box_pos),
        ("bar_of_day", lambda e: float(e.bar_of_day)),
        ("total_score", lambda e: float(e.total_score)),
    ]

    print(f"\n{'Indicator':<16} {'TP mean':>10} {'TP std':>10} {'FP mean':>10} {'FP std':>10} "
          f"{'Delta mean':>12} {'Significant':>12}")
    print("-" * 82)

    significant_indicators: list[tuple[str, float]] = []

    for name, fn in fields:
        tp_vals = [fn(e) for e in tp_entries]
        fp_vals = [fn(e) for e in fp_entries]

        tp_mean = sum(tp_vals) / len(tp_vals)
        fp_mean = sum(fp_vals) / len(fp_vals)
        tp_std = stat.stdev(tp_vals) if len(tp_vals) > 1 else 0.0
        fp_std = stat.stdev(fp_vals) if len(fp_vals) > 1 else 0.0

        delta = fp_mean - tp_mean

        # Determine significance: delta > 0.5 * combined_std
        combined_std = (tp_std + fp_std) / 2 if (tp_std + fp_std) > 0 else 1.0
        effect_size = abs(delta) / combined_std if combined_std > 0 else 0.0
        significant = "***" if effect_size > 0.5 else ("**" if effect_size > 0.3 else "")

        if effect_size > 0.3:
            significant_indicators.append((name, effect_size))

        print(f"{name:<16} {tp_mean:>10.3f} {tp_std:>10.3f} {fp_mean:>10.3f} {fp_std:>10.3f} "
              f"{delta:>+12.3f} {significant:>12}")

    # Confluence component comparison
    print(f"\n{'Confluence Component':<20} {'TP mean':>10} {'FP mean':>10} {'Delta':>10}")
    print("-" * 54)
    for name, fn in [
        ("V(vol)", lambda e: float(e.pts_vol)),
        ("SR", lambda e: float(e.pts_sr)),
        ("SRx", lambda e: float(e.pts_sr2)),
        ("RSI", lambda e: float(e.pts_rsi)),
        ("T(time)", lambda e: float(e.pts_time)),
        ("PQ", lambda e: float(e.pts_pq)),
        ("BX(box)", lambda e: float(e.pts_box)),
    ]:
        tp_mean = sum(fn(e) for e in tp_entries) / len(tp_entries)
        fp_mean = sum(fn(e) for e in fp_entries) / len(fp_entries)
        delta = fp_mean - tp_mean
        print(f"{name:<20} {tp_mean:>10.3f} {fp_mean:>10.3f} {delta:>+10.3f}")

    # Direction distribution
    print(f"\n{'Direction':<12} {'TP':>10} {'FP':>10}")
    print("-" * 34)
    tp_long = sum(1 for e in tp_entries if e.direction == "L")
    fp_long = sum(1 for e in fp_entries if e.direction == "L")
    tp_short = sum(1 for e in tp_entries if e.direction == "S")
    fp_short = sum(1 for e in fp_entries if e.direction == "S")
    print(f"{'Long':<12} {tp_long:>10} ({tp_long/len(tp_entries)*100:.0f}%) "
          f"{fp_long:>6} ({fp_long/len(fp_entries)*100:.0f}%)")
    print(f"{'Short':<12} {tp_short:>10} ({tp_short/len(tp_entries)*100:.0f}%) "
          f"{fp_short:>6} ({fp_short/len(fp_entries)*100:.0f}%)")

    # DOW distribution
    print(f"\n{'DOW':<12} {'TP':>10} {'FP':>10}")
    print("-" * 34)
    for d in range(5):
        tp_d = sum(1 for e in tp_entries if e.dow == d)
        fp_d = sum(1 for e in fp_entries if e.dow == d)
        dow_name = ["Mon", "Tue", "Wed", "Thu", "Fri"][d]
        tp_pct = tp_d / len(tp_entries) * 100
        fp_pct = fp_d / len(fp_entries) * 100
        print(f"{dow_name:<12} {tp_d:>10} ({tp_pct:.0f}%) {fp_d:>6} ({fp_pct:.0f}%)")

    # Summary of most significant differences
    if significant_indicators:
        significant_indicators.sort(key=lambda x: -x[1])
        print(f"\n{'='*80}")
        print("TOP DISCRIMINATING INDICATORS (effect size > 0.3)")
        print(f"{'='*80}")
        for name, effect in significant_indicators:
            print(f"  {name:<20} effect_size = {effect:.3f}")


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    print("=" * 100)
    print("PB FALSE-POSITIVE ANALYSIS — Python entries that TV does NOT take")
    print("=" * 100)

    # Validate paths
    if not BARS_CSV.exists():
        print(f"ERROR: SPY 5m data not found at {BARS_CSV}")
        sys.exit(1)
    if not TV_CSV.exists():
        print(f"ERROR: TV export not found at {TV_CSV}")
        sys.exit(1)

    # ── Step 1: Load data ─────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    bars = load_5m_bars(BARS_CSV)
    print(f"  Loaded {len(bars)} bars ({bars[0].date[:10]} to {bars[-1].date[:10]})")

    bar_index = build_bar_index(bars)
    print(f"  Built timestamp index: {len(bar_index)} unique keys")

    tv_trades = load_tv_pb_trades(TV_CSV)
    print(f"  Loaded {len(tv_trades)} PB entries from TV export")

    # ── Step 2: Run Python PB strategy ────────────────────────────────
    print("\n[2/5] Running Python PB strategy (TV-parity config)...")
    config = build_tv_parity_pb_config()
    py_entries = run_python_pb(bars, config)
    print(f"  Python took {len(py_entries)} PB entries")

    # ── Step 3: Match entries ─────────────────────────────────────────
    print("\n[3/5] Matching Python vs TV entries (+/- 1 bar tolerance)...")
    results = match_entries(py_entries, tv_trades, bar_index)

    tp = [r for r in results if r.category == "TP"]
    fp = [r for r in results if r.category == "FP"]
    fn = [r for r in results if r.category == "FN"]

    print(f"\n  SUMMARY")
    print(f"  {'='*50}")
    print(f"  Python PB entries:  {len(py_entries)}")
    print(f"  TV PB entries:      {len(tv_trades)}")
    print(f"  TRUE POSITIVES:     {len(tp):>4}  (both enter)")
    print(f"  FALSE POSITIVES:    {len(fp):>4}  (Python only -- TV does NOT take)")
    print(f"  FALSE NEGATIVES:    {len(fn):>4}  (TV only -- Python does NOT take)")
    print(f"  Python precision:   {len(tp)/len(py_entries)*100:.1f}%  ({len(tp)}/{len(py_entries)})"
          if py_entries else "  Python precision: N/A")
    print(f"  Python recall:      {len(tp)/len(tv_trades)*100:.1f}%  ({len(tp)}/{len(tv_trades)})"
          if tv_trades else "  Python recall: N/A")

    # ── Step 4: Print FALSE POSITIVES ─────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"FALSE POSITIVES: {len(fp)} entries Python takes but TV does NOT")
    print(f"{'='*100}")

    for i, r in enumerate(fp):
        py = r.py_entry
        assert py is not None
        print(f"\n  FP #{i+1}: {py.bar_label} | {py.direction} | close=${py.close:.2f}")
        print_entry_detail(f"FP #{i+1}", py, config)
        print()

    # ── Step 4b: Print FALSE NEGATIVES ────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"FALSE NEGATIVES: {len(fn)} entries TV takes but Python does NOT")
    print(f"{'='*100}")

    for i, r in enumerate(fn):
        tv = r.tv_trade
        assert tv is not None
        print(f"\n  FN #{i+1}: TV #{tv.trade_num} | {tv.et_bar_label} | {tv.direction} | "
              f"TV price=${tv.price:.2f}")
        print(f"    TV Signal: {tv.signal_raw[:120]}{'...' if len(tv.signal_raw) > 120 else ''}")

        # Extract Python indicators at this bar
        idx = bar_index.get(tv.et_bar_label)
        if idx is not None:
            py_snap = extract_indicators_at_bar(bars, idx, config)
            if py_snap is not None:
                py_snap.direction = tv.direction
                print_entry_detail(f"  Python indicators at FN bar", py_snap, config)
        else:
            print(f"    ** Bar not found in Python data **")

    # ── Step 5: Statistical comparison ────────────────────────────────
    tp_entries = [r.py_entry for r in tp if r.py_entry is not None]
    fp_entries = [r.py_entry for r in fp if r.py_entry is not None]

    compare_tp_fp_distributions(tp_entries, fp_entries, config)

    # ── Step 5b: FP date distribution ─────────────────────────────────
    print(f"\n{'='*100}")
    print("FALSE POSITIVE DATE DISTRIBUTION")
    print(f"{'='*100}")

    fp_dates: Counter[str] = Counter()
    for r in fp:
        if r.py_entry:
            date = r.py_entry.bar_label[:10]
            fp_dates[date] += 1

    print(f"\n  {'Date':<14} {'FP Count':>10}")
    print(f"  {'-'*26}")
    for date, count in fp_dates.most_common():
        print(f"  {date:<14} {count:>10}")

    # ── Step 5c: FP time-of-day distribution ──────────────────────────
    print(f"\n{'='*100}")
    print("FALSE POSITIVE TIME-OF-DAY DISTRIBUTION")
    print(f"{'='*100}")

    fp_hours: Counter[str] = Counter()
    for r in fp:
        if r.py_entry:
            time_str = r.py_entry.bar_label[11:13]  # HH
            fp_hours[time_str] += 1

    tp_hours: Counter[str] = Counter()
    for r in tp:
        if r.py_entry:
            time_str = r.py_entry.bar_label[11:13]
            tp_hours[time_str] += 1

    all_hours = sorted(set(fp_hours.keys()) | set(tp_hours.keys()))
    print(f"\n  {'Hour':<8} {'TP':>6} {'FP':>6} {'FP ratio':>10}")
    print(f"  {'-'*32}")
    for h in all_hours:
        tp_c = tp_hours.get(h, 0)
        fp_c = fp_hours.get(h, 0)
        total = tp_c + fp_c
        ratio = fp_c / total * 100 if total > 0 else 0
        print(f"  {h}:xx     {tp_c:>6} {fp_c:>6} {ratio:>9.0f}%")

    # ── Step 5d: Key hypothesis tests ─────────────────────────────────
    print(f"\n{'='*100}")
    print("KEY HYPOTHESES FOR FALSE POSITIVES")
    print(f"{'='*100}")

    if fp_entries and tp_entries:
        # H1: Are FPs happening in weaker trends?
        tp_adx = [e.adx for e in tp_entries]
        fp_adx = [e.adx for e in fp_entries]
        print(f"\n  H1: FPs in weaker trends?")
        print(f"    TP ADX: mean={sum(tp_adx)/len(tp_adx):.1f}, "
              f"min={min(tp_adx):.1f}, median={sorted(tp_adx)[len(tp_adx)//2]:.1f}")
        print(f"    FP ADX: mean={sum(fp_adx)/len(fp_adx):.1f}, "
              f"min={min(fp_adx):.1f}, median={sorted(fp_adx)[len(fp_adx)//2]:.1f}")

        # H2: Are FPs at the edge of the VWAP zone?
        tp_zone_ratio = [e.dist_vwap / e.zone_w if e.zone_w > 0 else 0 for e in tp_entries]
        fp_zone_ratio = [e.dist_vwap / e.zone_w if e.zone_w > 0 else 0 for e in fp_entries]
        print(f"\n  H2: FPs at edge of VWAP zone (dist/zone_w)?")
        print(f"    TP: mean={sum(tp_zone_ratio)/len(tp_zone_ratio):.3f}, "
              f"max={max(tp_zone_ratio):.3f}")
        print(f"    FP: mean={sum(fp_zone_ratio)/len(fp_zone_ratio):.3f}, "
              f"max={max(fp_zone_ratio):.3f}")

        # H3: Are FPs at lower confluence scores?
        tp_scores = [e.total_score for e in tp_entries]
        fp_scores = [e.total_score for e in fp_entries]
        print(f"\n  H3: FPs at lower confluence?")
        print(f"    TP score: mean={sum(tp_scores)/len(tp_scores):.1f}, "
              f"min={min(tp_scores)}, "
              f"distribution: {Counter(tp_scores).most_common()}")
        print(f"    FP score: mean={sum(fp_scores)/len(fp_scores):.1f}, "
              f"min={min(fp_scores)}, "
              f"distribution: {Counter(fp_scores).most_common()}")

        # H4: Are FPs with different VWAP crosses count?
        tp_vxc = [e.vwap_crosses for e in tp_entries]
        fp_vxc = [e.vwap_crosses for e in fp_entries]
        print(f"\n  H4: FPs with more VWAP crosses?")
        print(f"    TP VXC: mean={sum(tp_vxc)/len(tp_vxc):.1f}, "
              f"distribution: {Counter(tp_vxc).most_common()}")
        print(f"    FP VXC: mean={sum(fp_vxc)/len(fp_vxc):.1f}, "
              f"distribution: {Counter(fp_vxc).most_common()}")

        # H5: Are FPs from specific candle patterns?
        tp_bull_candle = sum(1 for e in tp_entries if e.bull_candle)
        tp_bull_wick = sum(1 for e in tp_entries if e.bull_wick and not e.bull_candle)
        tp_bear_candle = sum(1 for e in tp_entries if e.bear_candle)
        tp_bear_wick = sum(1 for e in tp_entries if e.bear_wick and not e.bear_candle)
        fp_bull_candle = sum(1 for e in fp_entries if e.bull_candle)
        fp_bull_wick = sum(1 for e in fp_entries if e.bull_wick and not e.bull_candle)
        fp_bear_candle = sum(1 for e in fp_entries if e.bear_candle)
        fp_bear_wick = sum(1 for e in fp_entries if e.bear_wick and not e.bear_candle)
        print(f"\n  H5: FP candle patterns?")
        print(f"    TP: bull_candle={tp_bull_candle} bull_wick_only={tp_bull_wick} "
              f"bear_candle={tp_bear_candle} bear_wick_only={tp_bear_wick}")
        print(f"    FP: bull_candle={fp_bull_candle} bull_wick_only={fp_bull_wick} "
              f"bear_candle={fp_bear_candle} bear_wick_only={fp_bear_wick}")

        # H6: Are FPs coming from position/spacing interactions?
        tp_bod = [e.bar_of_day for e in tp_entries]
        fp_bod = [e.bar_of_day for e in fp_entries]
        print(f"\n  H6: FPs at different bar_of_day?")
        print(f"    TP: mean={sum(tp_bod)/len(tp_bod):.1f}, min={min(tp_bod)}, max={max(tp_bod)}")
        print(f"    FP: mean={sum(fp_bod)/len(fp_bod):.1f}, min={min(fp_bod)}, max={max(fp_bod)}")

        # H7: LRS values
        tp_lrs = [e.lrs_atr for e in tp_entries]
        fp_lrs = [e.lrs_atr for e in fp_entries]
        print(f"\n  H7: FP LRS values?")
        print(f"    TP LRS: mean={sum(tp_lrs)/len(tp_lrs):.3f}")
        print(f"    FP LRS: mean={sum(fp_lrs)/len(fp_lrs):.3f}")

        # H8: CVD values
        tp_cvd = [e.cvd_norm for e in tp_entries]
        fp_cvd = [e.cvd_norm for e in fp_entries]
        print(f"\n  H8: FP CVD values?")
        print(f"    TP CVD: mean={sum(tp_cvd)/len(tp_cvd):.4f}")
        print(f"    FP CVD: mean={sum(fp_cvd)/len(fp_cvd):.4f}")

    print(f"\n\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
