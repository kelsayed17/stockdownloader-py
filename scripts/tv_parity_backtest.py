"""TV-Parity Backtest — match PineScript v11.2 configs exactly.

Usage:
    DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/tv_parity_backtest.py

Runs PB, PS, ORB, REV with overrides matched to TradingView VWAP v11.2
defaults and compares against the TV trade export (~107 trades, $10,892 PnL).
"""
from __future__ import annotations

import csv
import sys
import time
from decimal import Decimal
from pathlib import Path

D = Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.strategies.intraday.pullback import PullbackStrategy
from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy
from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy
from stockdownloader.strategies.intraday.reversal import ReversalStrategy
from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy, UnifiedVWAPConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"

# ── TV benchmark ───────────────────────────────────────────────────
TV_TRADES = 107
TV_PNL = 10892.02
TV_RETURN = 10.89  # %

INITIAL_CAPITAL = D("100000")
RISK_PER_TRADE = D("0.01")     # 1% risk
SLIPPAGE_PCT = D("0.0002")     # 2 bps


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


def compute_buy_and_hold(bars: list[IntradayPriceData]) -> Decimal:
    if not bars:
        return D("0")
    return ((bars[-1].close - bars[0].close) / bars[0].close) * 100


def make_engine() -> IntradayBacktestEngine:
    return IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        slippage_pct=SLIPPAGE_PCT,
        fixed_capital=True,  # match TV's fixed-cap sizing
    )


def safe_float(fn, default=0.0):
    try:
        return float(fn())
    except Exception:
        return default


def run_strategy(name: str, strategy, bars: list[IntradayPriceData]) -> dict:
    engine = make_engine()
    t0 = time.time()
    result = engine.run(strategy, bars)
    elapsed = time.time() - t0

    return {
        "name": name,
        "total_return_pct": float(result.total_return),
        "total_trades": result.total_trades,
        "win_rate_pct": float(result.win_rate),
        "profit_factor": safe_float(lambda: result.profit_factor),
        "max_drawdown_pct": float(result.max_drawdown),
        "sharpe": safe_float(result.sharpe_ratio),
        "sortino": safe_float(result.sortino_ratio),
        "calmar": safe_float(result.calmar_ratio),
        "avg_win": safe_float(lambda: result.average_win),
        "avg_loss": safe_float(lambda: result.average_loss),
        "total_pnl": float(result.total_pnl),
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "elapsed_sec": round(elapsed, 1),
    }


def build_tv_parity_strategies() -> list[tuple[str, object]]:
    """Build strategies with configs matched to PineScript v11.2."""
    strategies: list[tuple[str, object]] = []

    # ── PB (Pullback) — TV-parity ──────────────────────────────────
    strategies.append((
        "PB (TV-parity)",
        PullbackStrategy(
            # Entry
            allow_shorts=True,          # TV: i_shorts=true (CRITICAL — was False)
            allow_longs=True,
            pb_zone=D("0.5"),           # TV: i_pbZone=0.5
            pb_body=D("0.15"),          # TV: i_pbBody=0.15 (was 0.20)
            rr=D("1.4"),               # TV: i_rr=1.4 (was 1.8)
            sl_atr=D("1.3"),           # TV: i_slATR=1.3 ✓
            sl_cap=D("1.50"),          # TV: i_slCap=1.50 ✓
            # Trend
            adx_thresh=D("21"),        # TV: i_adxThresh=21 (was 22)
            trend_bars=3,              # TV: i_trendBars=3 (was 7)
            htf_align=True,            # TV: i_htfAlign=true ✓
            ar_filter=True,            # TV: i_arFilter=true (was False)
            ar_thresh=D("0.9"),        # TV: i_arThresh=0.9 ✓
            ar_cap=D("1.15"),          # TV: i_arCap=1.15 (was 1.30)
            va_filter=True,            # TV: i_vaFilter=true (was False)
            va_min=D("-0.1"),          # TV: i_vaMin=-0.1 ✓
            cvd_long_filter=True,      # TV: i_cvdLongFilter=true ✓
            lrs_short_filter=True,     # TV: i_lrsShortFilter=true ✓
            lrs_thresh=D("0.08"),      # TV: i_lrsThresh=0.08 ✓
            max_vxc=6,                 # TV: i_maxVXC=6 (was 4)
            # Confluence
            w_vol=3,                   # TV: i_wVol=3 (was 2)
            w_sr=2,                    # TV: i_wSR=2 ✓
            w_rsi=1,                   # TV: i_wRSI=1 ✓
            w_time=0,                  # TV: i_wTime=0 ✓
            w_pq=1,                    # TV: i_wPQ=1 ✓
            w_box=0,                   # TV: i_wBox=0 ✓
            min_score=3,               # TV: i_minScore=3 (was 4)
            min_score_long=5,          # TV: i_minScoreL=5 (was 4)
            pq_max_cross=3,            # TV: i_pqMaxCross=3 ✓
            # Risk
            max_day=2,                 # TV: i_maxDay=2 (was 1)
            spacing=3,                 # TV: i_spacing=3 (was 5)
            circuit=3,                 # TV: i_circuit=3 ✓
            day_loss=D("3.0"),         # TV: i_dayLoss=3.0 ✓
            no_friday_short=True,      # TV: i_noFriday=true ✓
            no_monday_long=True,       # TV: i_noMondayLong=true (was False)
            # Exit
            be_trigger=D("0.5"),       # TV: i_beTrigger=0.5 (was 0.7)
            trail_vwap=True,           # TV: i_trailVWAP=true ✓
            trail_buf=D("0.15"),       # TV: i_trailBuf=0.15 ✓
            trail_keep_tp=True,        # TV: i_trailKeepTP=true ✓
            close_eod=True,            # TV: i_closeEOD=true ✓
            # PB extras
            pb_vwap_bias=False,        # TV: no VWAP session bias filter
            pb_tp_mode="rr",
        ),
    ))

    # ── PS (Pattern Scalp) — TV-parity ─────────────────────────────
    strategies.append((
        "PS (TV-parity)",
        PatternScalpStrategy(
            allow_shorts=True,
            allow_longs=True,
            ps_atr_pct=D("30.0"),      # TV: i_psATRPct=30.0 (was 20.0)
            ps_window=12,              # TV: i_psWindow=12 (was 35)
            ps_rvol=D("1.0"),          # TV: i_psRVOL=1.0 (was 1.5)
            ps_engulf=D("0.35"),       # TV: i_psEngulf=0.35 (was 0.25)
            ps_sl_mode="Day Extreme",  # TV: i_psSLMode="Day Extreme"
            ps_sl_atr=D("1.5"),        # TV: i_psSLATR=1.5 (was 1.3)
            ps_sl_cap=D("2.50"),       # TV: i_psSLCap=2.50 (was 1.50)
            ps_tp_pct=D("75.0"),       # TV: i_psTPPct=75.0 ✓
            ps_sma_filter=False,       # TV: i_psSMAFilter=false (was True)
            ps_htf_align=False,        # TV: no HTF alignment on PS (was True)
            ps_time_gate=False,        # TV: no is_good_time gate on PS
            ps_min_rr=D("0.3"),        # TV: no explicit min RR (was 1.5)
            be_trigger=D("0.5"),       # TV: i_beTrigger=0.5
            close_eod=True,
            circuit=3,
            day_loss=D("3.0"),
        ),
    ))

    # ── ORB (OR Breakout) — TV-parity ──────────────────────────────
    strategies.append((
        "ORB (TV-parity)",
        ORBreakoutStrategy(
            allow_shorts=True,
            allow_longs=True,
            orb_window=20,             # TV: i_orbWindow=20 (was 30)
            orb_rvol=D("2.0"),         # TV: i_orbRVOL=2.0 ✓
            orb_sl_mode="OR Opposite", # TV: i_orbSLMode="OR Opposite"
            orb_sl_atr=D("1.5"),       # TV: i_orbSLATR=1.5 ✓
            orb_sl_cap=D("2.50"),      # TV: i_orbSLCap=2.50 (was 2.00)
            orb_vwap_align=True,       # TV: i_orbVWAPAlign=true ✓
            orb_body_min=D("0.2"),     # TV: i_orbBodyMin=0.2 (was 0.25)
            orb_entry_mode="aggressive", # TV: immediate on breakout
            orb_trail_atr=D("1.5"),    # TV: i_orbTrailATR=1.5 (was 0.8)
            orb_htf_align=True,
            orb_gap_filter=True,
            be_trigger=D("0.5"),       # TV: i_beTrigger=0.5
            adx_thresh=D("21"),        # TV doesn't strictly enforce ADX for ORB
            close_eod=True,
            circuit=3,
            day_loss=D("3.0"),
        ),
    ))

    # ── REV (Reversal) — TV-parity ─────────────────────────────────
    strategies.append((
        "REV (TV-parity)",
        ReversalStrategy(
            allow_longs=True,
            rev_band="2σ",             # TV: i_revBand="2σ" (was "1σ")
            rev_body=D("0.20"),        # TV: i_revBody=0.20 (was 0.15)
            rev_sl_atr=D("1.0"),       # TV: i_revSLATR=1.0 ✓
            rev_sl_cap=D("1.50"),      # TV: i_revSLCap=1.50 ✓
            rev_shorts=False,          # TV: i_revShorts=false ✓
            rev_min_rr=D("0.3"),       # TV: i_revMinRR=0.3 (was 1.0)
            rev_tp_mode="vwap",        # TV: TP at VWAP ✓
            be_trigger=D("0.5"),       # TV: i_beTrigger=0.5
            adx_thresh=D("21"),        # TV: i_adxThresh=21
            min_score=3,               # TV: i_minScore=3 (was 5)
            rev_vwap_flat_tol=D("0.05"),  # TV: vwapDelta <= atr*0.05 (was 0.10)
            rev_require_sr=False,      # TV: soft scoring only, no hard SR filter
            rev_hug_limit=20,          # TV: notHugging limit=20 (was 30)
            rev_can_trade_bar=11,      # TV: canTrade=bar 11+ (was 8)
            close_eod=True,
            circuit=3,
            day_loss=D("3.0"),
            spacing=3,                 # TV: i_spacing=3
        ),
    ))

    # ── ORR (OR Reversal) — DISABLED in TV v11.2 ──────────────────
    # i_orrEnable = false, so we skip it entirely

    return strategies


def build_unified_strategy() -> tuple[str, UnifiedVWAPStrategy]:
    """Build a UnifiedVWAPStrategy with TV-parity configs.

    Shared infra/exit/trail fields go into :class:`UnifiedVWAPConfig`;
    mode-specific fields go into per-mode override dicts.
    """
    config = UnifiedVWAPConfig(
        # Shared risk / exit / trail
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

    # ── PB overrides (mode-specific fields from PB standalone) ────
    pb_overrides = dict(
        pb_zone=D("0.5"),
        pb_body=D("0.15"),
        rr=D("1.4"),
        sl_atr=D("1.3"),
        sl_cap=D("1.50"),
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
        no_friday_short=True,
        no_monday_long=True,
        trail_vwap=True,
        pb_vwap_bias=False,        # TV: no VWAP session bias filter
        pb_tp_mode="rr",
    )

    # ── PS overrides (mode-specific fields from PS standalone) ────
    ps_overrides = dict(
        ps_atr_pct=D("30.0"),
        ps_window=12,
        ps_rvol=D("1.0"),
        ps_engulf=D("0.35"),
        ps_sl_mode="Day Extreme",
        ps_sl_atr=D("1.5"),
        ps_sl_cap=D("2.50"),
        ps_tp_pct=D("75.0"),
        ps_sma_filter=False,
        ps_htf_align=False,
        ps_time_gate=False,
        ps_min_rr=D("0.3"),
    )

    # ── ORB overrides (mode-specific fields from ORB standalone) ──
    orb_overrides = dict(
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
    )

    # ── REV overrides (mode-specific fields from REV standalone) ──
    rev_overrides = dict(
        rev_band="2\u03c3",
        rev_body=D("0.20"),
        rev_sl_atr=D("1.0"),
        rev_sl_cap=D("1.50"),
        rev_shorts=False,
        rev_min_rr=D("0.3"),
        rev_tp_mode="vwap",
        rev_vwap_flat_tol=D("0.05"),
        rev_require_sr=False,
        rev_hug_limit=20,
        rev_can_trade_bar=11,
        min_score=3,
    )

    strategy = UnifiedVWAPStrategy(
        config=config,
        pb_overrides=pb_overrides,
        ps_overrides=ps_overrides,
        orb_overrides=orb_overrides,
        rev_overrides=rev_overrides,
    )
    return ("Unified VWAP (TV-parity)", strategy)


def main() -> None:
    if not BARS_CSV.exists():
        print(f"ERROR: SPY 5m data not found at {BARS_CSV}")
        sys.exit(1)

    print("=" * 90)
    print("TV-PARITY BACKTEST — PineScript v11.2 Config Match")
    print("=" * 90)

    # Load data
    print("\nLoading SPY 5m bars...")
    bars = load_5m_bars(BARS_CSV)
    print(f"  Loaded {len(bars)} bars ({bars[0].date[:10]} to {bars[-1].date[:10]})")

    bnh_return = compute_buy_and_hold(bars)
    print(f"  SPY Buy-and-Hold return: {bnh_return:.2f}%")

    # TV benchmark
    print(f"\n  TV BENCHMARK: {TV_TRADES} trades | ${TV_PNL:,.2f} PnL | {TV_RETURN:.2f}% return")

    # Build strategies
    strats = build_tv_parity_strategies()
    print(f"\nRunning {len(strats)} strategies with TV-parity configs...\n")

    # Run backtests
    results: list[dict] = []
    for i, (name, strat) in enumerate(strats, 1):
        print(f"  [{i}/{len(strats)}] {name}...", end=" ", flush=True)
        try:
            r = run_strategy(name, strat, bars)
            results.append(r)
            beat_tv = "✓" if r["total_pnl"] > 0 else "✗"
            print(f"Return={r['total_return_pct']:+.2f}% | "
                  f"PnL=${r['total_pnl']:+,.2f} | "
                  f"Trades={r['total_trades']} ({r['winning_trades']}W/{r['losing_trades']}L) | "
                  f"Win={r['win_rate_pct']:.1f}% | "
                  f"Sharpe={r['sharpe']:.2f} | "
                  f"MaxDD={r['max_drawdown_pct']:.2f}% | "
                  f"{r['elapsed_sec']}s {beat_tv}")
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            results.append({
                "name": name, "total_return_pct": 0.0, "total_trades": 0,
                "win_rate_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "total_pnl": 0.0, "winning_trades": 0,
                "losing_trades": 0, "elapsed_sec": 0.0,
            })

    # ── Unified strategy ────────────────────────────────────────────
    unified_name, unified_strategy = build_unified_strategy()
    print(f"\n  [Unified] {unified_name}...", end=" ", flush=True)
    try:
        unified_result = run_strategy(unified_name, unified_strategy, bars)
        results.append(unified_result)
        beat_tv = "\u2713" if unified_result["total_pnl"] > 0 else "\u2717"
        print(f"Return={unified_result['total_return_pct']:+.2f}% | "
              f"PnL=${unified_result['total_pnl']:+,.2f} | "
              f"Trades={unified_result['total_trades']} "
              f"({unified_result['winning_trades']}W/{unified_result['losing_trades']}L) | "
              f"Win={unified_result['win_rate_pct']:.1f}% | "
              f"Sharpe={unified_result['sharpe']:.2f} | "
              f"MaxDD={unified_result['max_drawdown_pct']:.2f}% | "
              f"{unified_result['elapsed_sec']}s {beat_tv}")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        unified_result = {
            "name": unified_name, "total_return_pct": 0.0, "total_trades": 0,
            "win_rate_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "total_pnl": 0.0, "winning_trades": 0,
            "losing_trades": 0, "elapsed_sec": 0.0,
        }
        results.append(unified_result)

    # ── Combined metrics (standalone strategies only, excluding unified) ─
    standalone_results = [r for r in results if r["name"] != unified_name]
    combined_pnl = sum(r["total_pnl"] for r in standalone_results)
    combined_trades = sum(r["total_trades"] for r in standalone_results)
    combined_wins = sum(r["winning_trades"] for r in standalone_results)
    combined_losses = sum(r["losing_trades"] for r in standalone_results)
    combined_wr = (combined_wins / combined_trades * 100) if combined_trades > 0 else 0
    combined_return = combined_pnl / float(INITIAL_CAPITAL) * 100

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("PER-STRATEGY RESULTS")
    print("=" * 90)
    print(f"\n{'Strategy':<22} {'Return%':>8} {'PnL$':>11} {'Trades':>7} {'W/L':>8} "
          f"{'Win%':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>7}")
    print("-" * 95)

    for r in results:
        wl = f"{r['winning_trades']}/{r['losing_trades']}"
        print(f"{r['name']:<22} {r['total_return_pct']:>+7.2f}% "
              f"${r['total_pnl']:>+10,.2f} {r['total_trades']:>7} {wl:>8} "
              f"{r['win_rate_pct']:>5.1f}% {r['profit_factor']:>6.2f} "
              f"{r['sharpe']:>7.2f} {r['max_drawdown_pct']:>6.2f}%")

    print("-" * 95)
    wl_c = f"{combined_wins}/{combined_losses}"
    print(f"{'COMBINED':<22} {combined_return:>+7.2f}% "
          f"${combined_pnl:>+10,.2f} {combined_trades:>7} {wl_c:>8} "
          f"{combined_wr:>5.1f}%")

    # ── TV Comparison ──────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("TV COMPARISON")
    print(f"{'=' * 90}")
    print(f"\n{'Metric':<25} {'TradingView':>15} {'Python':>15} {'Delta':>15}")
    print("-" * 72)
    print(f"{'Total Trades':<25} {TV_TRADES:>15} {combined_trades:>15} {combined_trades - TV_TRADES:>+15}")
    print(f"{'Total P&L ($)':<25} ${TV_PNL:>14,.2f} ${combined_pnl:>14,.2f} ${combined_pnl - TV_PNL:>+14,.2f}")
    print(f"{'Return (%)':<25} {TV_RETURN:>14.2f}% {combined_return:>14.2f}% {combined_return - TV_RETURN:>+14.2f}%")
    print(f"{'Win Rate (%)':<25} {'~70':>14}% {combined_wr:>14.1f}%")

    pnl_match = abs(combined_pnl - TV_PNL) / TV_PNL * 100
    print(f"\n  P&L match: {100 - pnl_match:.1f}% (delta = ${abs(combined_pnl - TV_PNL):,.2f})")

    if pnl_match <= 30:
        print("  ✓ CONFIGS EXPLAIN THE GAP (within 30% of TV P&L)")
    elif pnl_match <= 50:
        print("  ⚠ PARTIAL MATCH — configs help but cross-mode interaction may be needed")
    else:
        print("  ✗ SIGNIFICANT DIVERGENCE — unified multi-mode strategy likely needed")

    # ── Save CSV ───────────────────────────────────────────────────
    out_csv = DATA_DIR / "SPY" / "tv_parity_backtest_results.csv"
    with open(out_csv, "w", newline="") as f:
        fields = ["name", "total_return_pct", "total_trades", "winning_trades",
                  "losing_trades", "win_rate_pct", "profit_factor",
                  "max_drawdown_pct", "sharpe", "sortino", "calmar",
                  "avg_win", "avg_loss", "total_pnl", "elapsed_sec"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})
        # Combined row
        writer.writerow({
            "name": "COMBINED",
            "total_return_pct": combined_return,
            "total_trades": combined_trades,
            "winning_trades": combined_wins,
            "losing_trades": combined_losses,
            "win_rate_pct": combined_wr,
            "total_pnl": combined_pnl,
        })

    print(f"\n  Results saved to {out_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
