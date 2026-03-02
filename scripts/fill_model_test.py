"""Test fill model configurations to isolate impact on TV parity.

Tests 4 configurations:
  1. Baseline: same-bar fill, exit at bar.close (original)
  2. Next-bar entry only: entry at next bar open, exit at bar.close
  3. Trigger exit only: same-bar fill, exit at SL/TP/trail level
  4. Both: next-bar fill + trigger exits
"""
from __future__ import annotations

import csv
import sys
import time
from decimal import Decimal
from pathlib import Path

D = Decimal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy, UnifiedVWAPConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BARS_CSV = DATA_DIR / "SPY" / "5m_bars.csv"

TV_START_DATE = "2025-02-20"
TV_END_DATE = "2026-02-27"
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
        rev_band="\u03c3".join(["2", ""]), rev_body=D("0.20"), rev_sl_atr=D("1.0"),
        rev_sl_cap=D("1.50"), rev_shorts=False, rev_min_rr=D("0.3"),
        rev_tp_mode="vwap", rev_vwap_flat_tol=D("0.05"), rev_require_sr=False,
        rev_hug_limit=20, rev_can_trade_bar=11, min_score=3,
    )
    return UnifiedVWAPStrategy(
        config=config, pb_overrides=pb_overrides, ps_overrides=ps_overrides,
        orb_overrides=orb_overrides, rev_overrides=rev_overrides,
    )


def run_config(label: str, bars: list[IntradayPriceData],
               next_bar_fill: bool, trigger_exit_fill: bool) -> None:
    strategy = build_unified()
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        slippage_pct=SLIPPAGE_PCT,
        fixed_capital=True,
        next_bar_fill=next_bar_fill,
        trigger_exit_fill=trigger_exit_fill,
    )
    t0 = time.time()
    result = engine.run(strategy, bars)
    elapsed = time.time() - t0

    print(f"  {label:<35} | Trades={result.total_trades:>4} "
          f"({result.winning_trades}W/{result.losing_trades}L) | "
          f"Win={float(result.win_rate):>5.1f}% | "
          f"PnL=${float(result.total_pnl):>+10,.2f} | "
          f"Return={float(result.total_return):>+6.2f}% | {elapsed:.1f}s")


def main() -> None:
    print("Loading bars...")
    all_bars = load_5m_bars(BARS_CSV)
    bars = [b for b in all_bars if TV_START_DATE <= b.date[:10] <= TV_END_DATE]
    print(f"  {len(bars)} bars ({bars[0].date[:10]} to {bars[-1].date[:10]})")
    print(f"\n  TV BENCHMARK: 107 trades | $10,892 PnL | 10.89% return | ~70% WR\n")

    configs = [
        ("1. Baseline (same-bar, close exit)", False, False),
        ("2. Next-bar entry only", True, False),
        ("3. Trigger exit only", False, True),
        ("4. Both (next-bar + trigger exit)", True, True),
    ]

    for label, nbf, tef in configs:
        run_config(label, bars, nbf, tef)


if __name__ == "__main__":
    main()
