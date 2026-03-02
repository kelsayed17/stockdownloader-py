#!/usr/bin/env python3
"""Mega combinatorial backtest: every strategy × every config × every permutation.

Tests all 13 intraday + 7 daily strategy families across hundreds of parameter
combinations against SPY 5-minute bar data (Feb 2024 – Feb 2026).
"""
from __future__ import annotations

import csv
import logging
import sys
import time
import traceback
from decimal import Decimal
from pathlib import Path

# -- Ensure src is on path --
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.backtesting.engines.daily import BacktestEngine
from stockdownloader.core.models.price import IntradayPriceData, PriceData
from stockdownloader.core.math import ZERO

logging.disable(logging.CRITICAL)  # suppress all log noise

# ======================================================================
# Data loading
# ======================================================================
DATA_DIR = ROOT / "data" / "SPY"
CSV_5M = DATA_DIR / "5m_bars.csv"
RESULTS_CSV = DATA_DIR / "backtest_results_full.csv"

D = Decimal


def load_5m_bars() -> list[IntradayPriceData]:
    """Load SPY 5-minute bars from CSV."""
    bars: list[IntradayPriceData] = []
    with open(CSV_5M) as f:
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


def aggregate_to_daily(bars_5m: list[IntradayPriceData]) -> list[PriceData]:
    """Aggregate 5-minute bars into daily OHLCV bars."""
    from collections import OrderedDict
    daily: OrderedDict[str, dict] = OrderedDict()
    for b in bars_5m:
        d = b.date[:10]
        if d not in daily:
            daily[d] = {
                "open": b.open, "high": b.high, "low": b.low,
                "close": b.close, "volume": b.volume,
            }
        else:
            rec = daily[d]
            if b.high > rec["high"]:
                rec["high"] = b.high
            if b.low < rec["low"]:
                rec["low"] = b.low
            rec["close"] = b.close
            rec["volume"] += b.volume
    result = []
    for date_str, rec in daily.items():
        result.append(PriceData(
            date=date_str,
            open=rec["open"],
            high=rec["high"],
            low=rec["low"],
            close=rec["close"],
            adj_close=rec["close"],
            volume=rec["volume"],
        ))
    return result


# ======================================================================
# Result extraction
# ======================================================================

def extract_result(result, name: str, config_label: str, timeframe: str) -> dict:
    """Extract key metrics from a BacktestResult."""
    try:
        sharpe = float(result.sharpe_ratio())
    except Exception:
        sharpe = 0.0
    try:
        sortino = float(result.sortino_ratio())
    except Exception:
        sortino = 0.0
    try:
        calmar = float(result.calmar_ratio())
    except Exception:
        calmar = 0.0
    return {
        "strategy": name,
        "config": config_label,
        "timeframe": timeframe,
        "total_return_pct": float(result.total_return),
        "total_pnl": float(result.total_pnl),
        "total_trades": result.total_trades,
        "win_rate_pct": float(result.win_rate),
        "profit_factor": float(result.profit_factor),
        "max_drawdown_pct": float(result.max_drawdown),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "avg_win": float(result.average_win),
        "avg_loss": float(result.average_loss),
    }


# ======================================================================
# Intraday strategy configurations
# ======================================================================

def get_intraday_configs() -> list[tuple[str, str, dict]]:
    """Return (strategy_class_name, config_label, overrides) tuples."""
    configs = []

    # --- MACD+OBV (13 configs) ---
    base = "MACDOBVStrategy"
    configs.append((base, "default", {}))
    configs.append((base, "long-only", {"allow_shorts": False}))
    configs.append((base, "short-only", {"allow_longs": False}))
    configs.append((base, "tight-sl", {"sl_mult": D("1.0"), "sl_cap": D("1.5")}))
    configs.append((base, "wide-sl", {"sl_mult": D("2.0"), "sl_cap": D("3.0")}))
    configs.append((base, "high-rr", {"rr_ratio": D("2.0")}))
    configs.append((base, "low-rr", {"rr_ratio": D("1.0")}))
    configs.append((base, "no-circuit", {"circuit": 99, "day_loss": D("99")}))
    configs.append((base, "max2-trades", {"max_day": 2}))
    configs.append((base, "max6-trades", {"max_day": 6, "spacing": 2}))
    configs.append((base, "obv3", {"obv_smooth": 3}))
    configs.append((base, "obv10", {"obv_smooth": 10}))
    configs.append((base, "no-be", {"be_trigger": D("99")}))
    configs.append((base, "macd-8-21-5", {"macd_fast": 8, "macd_slow": 21, "macd_signal": 5}))
    configs.append((base, "macd-5-13-3", {"macd_fast": 5, "macd_slow": 13, "macd_signal": 3}))

    # --- SMA Cross 2021 (12 configs) ---
    base = "SMACross2021Strategy"
    configs.append((base, "default-20/21", {}))
    configs.append((base, "both-dirs", {"allow_shorts": True}))
    configs.append((base, "sma-5/20", {"sma_short": 5, "sma_long": 20}))
    configs.append((base, "sma-9/21", {"sma_short": 9, "sma_long": 21}))
    configs.append((base, "sma-10/30", {"sma_short": 10, "sma_long": 30}))
    configs.append((base, "sma-10/50", {"sma_short": 10, "sma_long": 50}))
    configs.append((base, "sma-50/200", {"sma_short": 50, "sma_long": 200}))
    configs.append((base, "sma-3/8", {"sma_short": 3, "sma_long": 8}))
    configs.append((base, "tight-sl", {"sl_mult": D("1.0"), "sl_cap": D("1.5")}))
    configs.append((base, "wide-sl", {"sl_mult": D("2.0"), "sl_cap": D("3.0")}))
    configs.append((base, "high-rr", {"rr_ratio": D("2.0")}))
    configs.append((base, "no-be", {"be_trigger": D("99")}))

    # --- MACD Optimized (12 configs) ---
    base = "MACDOptimizedStrategy"
    configs.append((base, "default-8/35/5", {}))
    configs.append((base, "long-only", {"allow_shorts": False}))
    configs.append((base, "short-only", {"allow_longs": False}))
    configs.append((base, "macd-12/26/9", {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}))
    configs.append((base, "macd-5/20/9", {"macd_fast": 5, "macd_slow": 20, "macd_signal": 9}))
    configs.append((base, "macd-3/10/3", {"macd_fast": 3, "macd_slow": 10, "macd_signal": 3}))
    configs.append((base, "tight-sl", {"sl_mult": D("1.0"), "sl_cap": D("1.5")}))
    configs.append((base, "wide-sl", {"sl_mult": D("2.0"), "sl_cap": D("3.0")}))
    configs.append((base, "high-rr", {"rr_ratio": D("2.0")}))
    configs.append((base, "low-rr", {"rr_ratio": D("1.0")}))
    configs.append((base, "no-circuit", {"circuit": 99, "day_loss": D("99")}))
    configs.append((base, "no-be", {"be_trigger": D("99")}))

    # --- Intraday Momentum (10 configs) ---
    base = "IntradayMomentumStrategy"
    configs.append((base, "momentum-default", {"mode": "momentum"}))
    configs.append((base, "reversal-default", {"mode": "reversal"}))
    configs.append((base, "momentum-long-only", {"mode": "momentum", "allow_shorts": False}))
    configs.append((base, "reversal-long-only", {"mode": "reversal", "allow_shorts": False}))
    configs.append((base, "momentum-r1-pos", {"mode": "momentum", "r1_threshold": D("0.002")}))
    configs.append((base, "momentum-r1-neg", {"mode": "momentum", "r1_threshold": D("-0.002")}))
    configs.append((base, "momentum-early-entry", {"mode": "momentum", "entry_bar": 60}))
    configs.append((base, "momentum-late-entry", {"mode": "momentum", "entry_bar": 75}))
    configs.append((base, "reversal-early", {"mode": "reversal", "entry_bar": 60}))
    configs.append((base, "reversal-late", {"mode": "reversal", "entry_bar": 75}))

    # --- DMI+VWAP (11 configs) ---
    base = "DmiVwapStrategy"
    configs.append((base, "default", {}))
    configs.append((base, "adx-20", {"adx_threshold": D("20")}))
    configs.append((base, "adx-30", {"adx_threshold": D("30")}))
    configs.append((base, "adx-rising", {"require_adx_rising": True}))
    configs.append((base, "rr-1.5", {"rr": D("1.5")}))
    configs.append((base, "rr-2.5", {"rr": D("2.5")}))
    configs.append((base, "rr-3.0", {"rr": D("3.0")}))
    configs.append((base, "dmi-10", {"dmi_period": 10}))
    configs.append((base, "dmi-20", {"dmi_period": 20}))
    configs.append((base, "tight-sl", {"sl_atr_mult": D("1.0")}))
    configs.append((base, "wide-sl", {"sl_atr_mult": D("2.0")}))

    # --- OR Breakout (10 configs) ---
    base = "ORBreakoutStrategy"
    configs.append((base, "default", {}))
    configs.append((base, "or-15", {"orb_window": 15}))
    configs.append((base, "or-45", {"orb_window": 45}))
    configs.append((base, "or-60", {"orb_window": 60}))
    configs.append((base, "rvol-1.5", {"orb_rvol": D("1.5")}))
    configs.append((base, "rvol-3.0", {"orb_rvol": D("3.0")}))
    configs.append((base, "no-vwap-filter", {"orb_vwap_align": False}))
    configs.append((base, "no-adx-filter", {"orb_adx_filter": False}))
    configs.append((base, "no-gap-filter", {"orb_gap_filter": False}))
    configs.append((base, "all-filters-off", {"orb_vwap_align": False, "orb_adx_filter": False, "orb_gap_filter": False}))

    # --- OR Reversal (8 configs) ---
    base = "ORReversalStrategy"
    configs.append((base, "default", {}))
    configs.append((base, "prox-0.4", {"orr_prox": D("0.4")}))
    configs.append((base, "prox-0.8", {"orr_prox": D("0.8")}))
    configs.append((base, "or-30", {"orr_window": 30}))
    configs.append((base, "no-break-req", {"orr_require_break": False}))
    configs.append((base, "no-filters", {"orr_adx_filter": False, "orr_gap_filter": False}))
    configs.append((base, "long-only", {"allow_shorts": False}))
    configs.append((base, "tight-sl", {"orr_sl_atr": D("0.3")}))

    # --- Reversal (8 configs) ---
    base = "ReversalStrategy"
    configs.append((base, "default", {}))
    configs.append((base, "band-2s", {"rev_band": "2σ"}))
    configs.append((base, "with-shorts", {"rev_shorts": True}))
    configs.append((base, "tp-rr", {"rev_tp_mode": "rr", "rev_rr": D("2.0")}))
    configs.append((base, "no-sr-req", {"rev_require_sr": False}))
    configs.append((base, "low-score", {"min_score": 3}))
    configs.append((base, "tight-sl", {"rev_sl_atr": D("0.7"), "rev_sl_cap": D("1.0")}))
    configs.append((base, "wide-sl", {"rev_sl_atr": D("1.5"), "rev_sl_cap": D("2.5")}))

    # --- Pattern Scalp (8 configs) ---
    base = "PatternScalpStrategy"
    configs.append((base, "default", {}))
    configs.append((base, "rvol-1.0", {"ps_rvol": D("1.0")}))
    configs.append((base, "rvol-2.0", {"ps_rvol": D("2.0")}))
    configs.append((base, "no-sma-filter", {"ps_sma_filter": False}))
    configs.append((base, "high-rr", {"ps_min_rr": D("2.0")}))
    configs.append((base, "tight-sl", {"ps_sl_atr": D("1.0"), "ps_sl_cap": D("1.0")}))
    configs.append((base, "wide-sl", {"ps_sl_atr": D("1.8"), "ps_sl_cap": D("2.5")}))
    configs.append((base, "no-htf", {"ps_htf_align": False}))

    # --- Pullback (10 configs) ---
    base = "PullbackStrategy"
    configs.append((base, "default", {}))
    configs.append((base, "rr-2.5", {"rr": D("2.5")}))
    configs.append((base, "rr-1.0", {"rr": D("1.0")}))
    configs.append((base, "no-htf", {"htf_align": False}))
    configs.append((base, "no-vwap-bias", {"pb_vwap_bias": False}))
    configs.append((base, "no-cvd-filter", {"cvd_long_filter": False}))
    configs.append((base, "no-lrs-filter", {"lrs_short_filter": False}))
    configs.append((base, "all-filters-off", {"htf_align": False, "pb_vwap_bias": False, "cvd_long_filter": False, "lrs_short_filter": False}))
    configs.append((base, "tight-sl", {"sl_atr": D("1.0"), "sl_cap": D("1.5")}))
    configs.append((base, "wide-sl", {"sl_atr": D("1.8"), "sl_cap": D("3.0")}))

    # --- AVWAP Pullback (8 configs) ---
    base = "AVWAPPullbackStrategy"
    configs.append((base, "default", {}))
    configs.append((base, "rr-1.5", {"avwap_rr": D("1.5")}))
    configs.append((base, "rr-3.0", {"avwap_rr": D("3.0")}))
    configs.append((base, "no-vwap-agree", {"avwap_session_vwap_agree": False}))
    configs.append((base, "no-htf", {"avwap_htf_align": False}))
    configs.append((base, "with-shorts", {"avwap_shorts": True}))
    configs.append((base, "tight-sl", {"avwap_sl_atr": D("1.0"), "avwap_sl_cap": D("1.5")}))
    configs.append((base, "wide-sl", {"avwap_sl_atr": D("1.8"), "avwap_sl_cap": D("3.0")}))

    # --- SMC Structure (8 configs) ---
    base = "SMCStructureStrategy"
    configs.append((base, "default", {}))
    configs.append((base, "no-bos-req", {"smc_require_bos": False}))
    configs.append((base, "no-sweep", {"smc_sweep_entry": False}))
    configs.append((base, "vwap-agree", {"smc_vwap_agree": True}))
    configs.append((base, "rr-2.0", {"smc_rr": D("2.0")}))
    configs.append((base, "with-shorts", {"smc_shorts": True}))
    configs.append((base, "tight-sl", {"smc_sl_atr": D("1.5"), "smc_sl_cap": D("1.0")}))
    configs.append((base, "low-score", {"min_score": 2}))

    return configs


def get_daily_configs() -> list[tuple[str, str, dict]]:
    """Return (strategy_class_name, config_label, constructor_kwargs) tuples."""
    configs = []

    # --- MACD (8 param combos) ---
    for fast, slow, sig in [
        (12, 26, 9), (8, 35, 5), (5, 20, 9), (3, 10, 3),
        (8, 21, 5), (12, 26, 5), (5, 13, 3), (8, 17, 9),
    ]:
        configs.append(("MACDStrategy", f"macd-{fast}/{slow}/{sig}",
                        {"fast_period": fast, "slow_period": slow, "signal_period": sig}))

    # --- SMA Crossover (10 param combos) ---
    for short_p, long_p in [
        (20, 21), (5, 20), (9, 21), (10, 30), (10, 50),
        (20, 50), (50, 200), (20, 100), (5, 10), (3, 8),
    ]:
        configs.append(("SMACrossoverStrategy", f"sma-{short_p}/{long_p}",
                        {"short_period": short_p, "long_period": long_p}))

    # --- RSI (6 param combos) ---
    for period, oversold, overbought in [
        (14, 30, 70), (7, 30, 70), (14, 20, 80), (7, 20, 80),
        (21, 40, 60), (5, 25, 75),
    ]:
        configs.append(("RSIStrategy", f"rsi-{period}/{oversold}/{overbought}",
                        {"period": period, "oversold": oversold, "overbought": overbought}))

    # --- BB+RSI (6 configs) ---
    for bb_p, bb_std, rsi_p, rsi_os, rsi_ob in [
        (20, 2.0, 14, 30, 70), (20, 2.0, 7, 30, 70), (20, 1.5, 14, 30, 70),
        (20, 2.5, 14, 30, 70), (10, 2.0, 14, 30, 70), (20, 2.0, 14, 20, 80),
    ]:
        configs.append(("BollingerBandRSIStrategy",
                        f"bb{bb_p}-std{bb_std}-rsi{rsi_p}/{rsi_os}/{rsi_ob}",
                        {"bb_period": bb_p, "bb_std_dev": bb_std, "rsi_period": rsi_p,
                         "rsi_oversold": rsi_os, "rsi_overbought": rsi_ob}))

    # --- Breakout (4 configs) ---
    for bb_p, sq_lb, vol_m in [
        (20, 120, 1.5), (20, 60, 1.5), (20, 120, 2.0), (10, 120, 1.5),
    ]:
        configs.append(("BreakoutStrategy",
                        f"bb{bb_p}-sq{sq_lb}-vol{vol_m}",
                        {"bb_period": bb_p, "squeeze_lookback": sq_lb, "volume_multiplier": vol_m}))

    # --- Momentum Confluence (4 configs) ---
    for fast, slow, sig, ema_tf in [
        (12, 26, 9, 200), (12, 26, 9, 50), (8, 21, 5, 200), (5, 13, 3, 100),
    ]:
        configs.append(("MomentumConfluenceStrategy",
                        f"mc-{fast}/{slow}/{sig}-ema{ema_tf}",
                        {"fast_ema": fast, "slow_ema": slow, "signal_period": sig,
                         "ema_trend_filter": ema_tf}))

    # --- Multi-Indicator (5 threshold combos) ---
    for buy_t, sell_t in [(4, 4), (3, 3), (5, 5), (3, 5), (5, 3)]:
        configs.append(("MultiIndicatorStrategy",
                        f"multi-b{buy_t}/s{sell_t}",
                        {"buy_threshold": buy_t, "sell_threshold": sell_t}))

    return configs


# ======================================================================
# Strategy factory
# ======================================================================

def make_intraday_strategy(class_name: str, overrides: dict):
    """Instantiate an intraday strategy by class name with overrides."""
    from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
    from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
    from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
    from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumStrategy
    from stockdownloader.strategies.intraday.dmi_vwap import DmiVwapStrategy
    from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy
    from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategy
    from stockdownloader.strategies.intraday.reversal import ReversalStrategy
    from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy
    from stockdownloader.strategies.intraday.pullback import PullbackStrategy
    from stockdownloader.strategies.intraday.avwap_pullback import AVWAPPullbackStrategy
    from stockdownloader.strategies.intraday.smc_structure import SMCStructureStrategy

    CLASSES = {
        "MACDOBVStrategy": MACDOBVStrategy,
        "SMACross2021Strategy": SMACross2021Strategy,
        "MACDOptimizedStrategy": MACDOptimizedStrategy,
        "IntradayMomentumStrategy": IntradayMomentumStrategy,
        "DmiVwapStrategy": DmiVwapStrategy,
        "ORBreakoutStrategy": ORBreakoutStrategy,
        "ORReversalStrategy": ORReversalStrategy,
        "ReversalStrategy": ReversalStrategy,
        "PatternScalpStrategy": PatternScalpStrategy,
        "PullbackStrategy": PullbackStrategy,
        "AVWAPPullbackStrategy": AVWAPPullbackStrategy,
        "SMCStructureStrategy": SMCStructureStrategy,
    }
    cls = CLASSES[class_name]
    return cls(**overrides)


def make_daily_strategy(class_name: str, kwargs: dict):
    """Instantiate a daily strategy by class name with kwargs."""
    from stockdownloader.strategies.daily.simple import MACDStrategy, SMACrossoverStrategy, RSIStrategy
    from stockdownloader.strategies.daily.bollinger_rsi import BollingerBandRSIStrategy
    from stockdownloader.strategies.daily.breakout import BreakoutStrategy
    from stockdownloader.strategies.daily.momentum import MomentumConfluenceStrategy
    from stockdownloader.strategies.daily.multi_indicator import MultiIndicatorStrategy

    CLASSES = {
        "MACDStrategy": MACDStrategy,
        "SMACrossoverStrategy": SMACrossoverStrategy,
        "RSIStrategy": RSIStrategy,
        "BollingerBandRSIStrategy": BollingerBandRSIStrategy,
        "BreakoutStrategy": BreakoutStrategy,
        "MomentumConfluenceStrategy": MomentumConfluenceStrategy,
        "MultiIndicatorStrategy": MultiIndicatorStrategy,
    }
    cls = CLASSES[class_name]
    return cls(**kwargs)


# ======================================================================
# Main runner
# ======================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("MEGA COMBINATORIAL SPY BACKTEST")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading 5-minute bars...")
    bars_5m = load_5m_bars()
    print(f"  Loaded {len(bars_5m)} 5-minute bars")
    print(f"  Range: {bars_5m[0].date} → {bars_5m[-1].date}")

    first_close = bars_5m[0].close
    last_close = bars_5m[-1].close
    bnh_return = float((last_close - first_close) / first_close * 100)
    print(f"  SPY Buy & Hold: {first_close} → {last_close} = {bnh_return:+.2f}%")

    print("\n[2/4] Aggregating to daily bars...")
    daily_bars = aggregate_to_daily(bars_5m)
    print(f"  {len(daily_bars)} trading days")

    # Prepare results
    all_results: list[dict] = []
    INITIAL_CAPITAL = D("100000")

    # ===== INTRADAY STRATEGIES =====
    print("\n[3/4] Running INTRADAY strategy permutations...")
    intraday_configs = get_intraday_configs()
    print(f"  {len(intraday_configs)} intraday configurations")

    # Risk levels to test
    risk_levels = [
        ("risk-0.5%", D("0.005")),
        ("risk-1%", D("0.01")),
        ("risk-2%", D("0.02")),
    ]

    total_intraday = len(intraday_configs) * len(risk_levels)
    done = 0
    errors = 0

    for class_name, config_label, overrides in intraday_configs:
        for risk_label, risk_pct in risk_levels:
            done += 1
            full_label = f"{config_label} | {risk_label}"
            try:
                strategy = make_intraday_strategy(class_name, overrides)
                engine = IntradayBacktestEngine(
                    initial_capital=INITIAL_CAPITAL,
                    risk_per_trade=risk_pct,
                    slippage_pct=D("0.0002"),
                )
                result = engine.run(strategy, bars_5m)
                row = extract_result(result, class_name, full_label, "5m")
                all_results.append(row)
                if done % 20 == 0 or done == total_intraday:
                    elapsed = time.time() - t0
                    print(f"  [{done}/{total_intraday}] {elapsed:.0f}s | "
                          f"{class_name} {config_label} {risk_label}: "
                          f"{row['total_return_pct']:+.2f}% | "
                          f"{row['total_trades']} trades | "
                          f"WR {row['win_rate_pct']:.1f}%")
            except Exception as e:
                errors += 1
                if done % 20 == 0:
                    print(f"  [{done}/{total_intraday}] ERROR {class_name} {config_label}: {e}")

    print(f"\n  Intraday complete: {done - errors} succeeded, {errors} errors")

    # ===== DAILY STRATEGIES =====
    print("\n[4/4] Running DAILY strategy permutations...")
    daily_configs = get_daily_configs()
    print(f"  {len(daily_configs)} daily configurations")

    slippage_levels = [
        ("slip-0%", D("0")),
        ("slip-0.1%", D("0.001")),
        ("slip-0.2%", D("0.002")),
    ]

    total_daily = len(daily_configs) * len(slippage_levels)
    done_d = 0
    errors_d = 0

    for class_name, config_label, kwargs in daily_configs:
        for slip_label, slip_pct in slippage_levels:
            done_d += 1
            full_label = f"{config_label} | {slip_label}"
            try:
                strategy = make_daily_strategy(class_name, kwargs)
                engine = BacktestEngine(
                    initial_capital=INITIAL_CAPITAL,
                    commission=D("0"),
                    slippage_pct=slip_pct,
                )
                result = engine.run(strategy, daily_bars)
                row = extract_result(result, class_name, full_label, "daily")
                all_results.append(row)
                if done_d % 15 == 0 or done_d == total_daily:
                    elapsed = time.time() - t0
                    print(f"  [{done_d}/{total_daily}] {elapsed:.0f}s | "
                          f"{class_name} {config_label} {slip_label}: "
                          f"{row['total_return_pct']:+.2f}% | "
                          f"{row['total_trades']} trades | "
                          f"WR {row['win_rate_pct']:.1f}%")
            except Exception as e:
                errors_d += 1
                if done_d % 15 == 0:
                    print(f"  [{done_d}/{total_daily}] ERROR {class_name} {config_label}: {e}")

    print(f"\n  Daily complete: {done_d - errors_d} succeeded, {errors_d} errors")

    # ===== ENGINE CONFIG SWEEP on top performers =====
    # Sort by total_return and take top 15 for engine config sweep
    sorted_results = sorted(all_results, key=lambda r: r["total_return_pct"], reverse=True)
    top_n = sorted_results[:15]

    print(f"\n[BONUS] Engine config sweep on top {len(top_n)} performers...")
    engine_configs = [
        ("fixed-cap", {"fixed_capital": True}),
        ("vol-scale", {"vol_scale": True, "vol_lookback": 60}),
        ("dd-throttle", {"dd_throttle": True}),
        ("fixed+vol", {"fixed_capital": True, "vol_scale": True, "vol_lookback": 60}),
        ("all-risk-mgmt", {"fixed_capital": True, "vol_scale": True, "vol_lookback": 60, "dd_throttle": True}),
    ]

    sweep_done = 0
    for top_row in top_n:
        strategy_name = top_row["strategy"]
        config_label = top_row["config"].split(" | ")[0]
        timeframe = top_row["timeframe"]

        # Find original config
        if timeframe == "5m":
            orig_configs = get_intraday_configs()
            match = [(cn, cl, ov) for cn, cl, ov in orig_configs
                     if cn == strategy_name and cl == config_label]
        else:
            orig_configs = get_daily_configs()
            match = [(cn, cl, kw) for cn, cl, kw in orig_configs
                     if cn == strategy_name and cl == config_label]

        if not match:
            continue

        _, _, orig_params = match[0]

        for eng_label, eng_kwargs in engine_configs:
            sweep_done += 1
            full_label = f"{config_label} | {eng_label}"
            try:
                if timeframe == "5m":
                    strategy = make_intraday_strategy(strategy_name, orig_params)
                    engine = IntradayBacktestEngine(
                        initial_capital=INITIAL_CAPITAL,
                        risk_per_trade=D("0.01"),
                        slippage_pct=D("0.0002"),
                        **eng_kwargs,
                    )
                    result = engine.run(strategy, bars_5m)
                else:
                    strategy = make_daily_strategy(strategy_name, orig_params)
                    engine = BacktestEngine(
                        initial_capital=INITIAL_CAPITAL,
                        commission=D("0"),
                        slippage_pct=D("0.001"),
                        **{k: v for k, v in eng_kwargs.items()
                           if k in ("fixed_capital",)},  # Daily engine supports fewer options
                    )
                    result = engine.run(strategy, daily_bars)
                row = extract_result(result, strategy_name, full_label, timeframe)
                all_results.append(row)
            except Exception:
                pass

    print(f"  Engine sweep: {sweep_done} combinations tested")

    # ===== SAVE RESULTS =====
    print(f"\nTotal combinations tested: {len(all_results)}")

    # Write CSV
    fieldnames = list(all_results[0].keys())
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Results saved to: {RESULTS_CSV}")

    # ===== LEADERBOARD =====
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\n" + "=" * 120)
    print("LEADERBOARD — TOP 30 (sorted by Total Return)")
    print("=" * 120)
    header = (f"{'Rank':>4} {'Strategy':<30} {'Config':<35} {'TF':>5} "
              f"{'Return%':>9} {'Trades':>6} {'WR%':>6} {'PF':>6} "
              f"{'Sharpe':>7} {'MaxDD%':>7}")
    print(header)
    print("-" * 120)

    for rank, row in enumerate(sorted_results[:30], 1):
        print(f"{rank:4d} {row['strategy']:<30} {row['config']:<35} {row['timeframe']:>5} "
              f"{row['total_return_pct']:>+9.2f} {row['total_trades']:>6} "
              f"{row['win_rate_pct']:>6.1f} {row['profit_factor']:>6.2f} "
              f"{row['sharpe']:>7.2f} {row['max_drawdown_pct']:>7.2f}")

    print("\n" + "=" * 120)
    print("BOTTOM 10 (worst performers)")
    print("=" * 120)
    print(header)
    print("-" * 120)
    worst = sorted(all_results, key=lambda r: r["total_return_pct"])[:10]
    for rank, row in enumerate(worst, 1):
        print(f"{rank:4d} {row['strategy']:<30} {row['config']:<35} {row['timeframe']:>5} "
              f"{row['total_return_pct']:>+9.2f} {row['total_trades']:>6} "
              f"{row['win_rate_pct']:>6.1f} {row['profit_factor']:>6.2f} "
              f"{row['sharpe']:>7.2f} {row['max_drawdown_pct']:>7.2f}")

    # ===== SUMMARY STATS =====
    returns = [r["total_return_pct"] for r in all_results]
    profitable = [r for r in all_results if r["total_return_pct"] > 0]
    beat_bnh = [r for r in all_results if r["total_return_pct"] > bnh_return]
    positive_sharpe = [r for r in all_results if r["sharpe"] > 0]

    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(f"Total combinations tested: {len(all_results)}")
    print(f"SPY Buy & Hold return:     {bnh_return:+.2f}%")
    print(f"Profitable strategies:     {len(profitable)} / {len(all_results)} ({len(profitable)/len(all_results)*100:.1f}%)")
    print(f"Beat Buy & Hold:           {len(beat_bnh)} / {len(all_results)} ({len(beat_bnh)/len(all_results)*100:.1f}%)")
    print(f"Positive Sharpe ratio:     {len(positive_sharpe)} / {len(all_results)} ({len(positive_sharpe)/len(all_results)*100:.1f}%)")
    print(f"Best return:               {max(returns):+.2f}%")
    print(f"Worst return:              {min(returns):+.2f}%")
    print(f"Median return:             {sorted(returns)[len(returns)//2]:+.2f}%")
    print(f"Mean return:               {sum(returns)/len(returns):+.2f}%")

    # Best by category
    print(f"\n{'=' * 80}")
    print("BEST BY STRATEGY FAMILY")
    print(f"{'=' * 80}")
    from collections import defaultdict
    by_family = defaultdict(list)
    for r in all_results:
        by_family[r["strategy"]].append(r)

    for family in sorted(by_family.keys()):
        best = max(by_family[family], key=lambda r: r["total_return_pct"])
        worst = min(by_family[family], key=lambda r: r["total_return_pct"])
        n = len(by_family[family])
        n_profit = sum(1 for r in by_family[family] if r["total_return_pct"] > 0)
        print(f"\n  {family} ({n} configs, {n_profit} profitable)")
        print(f"    Best:  {best['config']:<40} {best['total_return_pct']:>+8.2f}% | WR {best['win_rate_pct']:.1f}% | PF {best['profit_factor']:.2f} | Sharpe {best['sharpe']:.2f}")
        print(f"    Worst: {worst['config']:<40} {worst['total_return_pct']:>+8.2f}% | WR {worst['win_rate_pct']:.1f}% | PF {worst['profit_factor']:.2f} | Sharpe {worst['sharpe']:.2f}")

    # Best risk-adjusted (Sharpe > 0, > 10 trades)
    quality = [r for r in all_results if r["sharpe"] > 0 and r["total_trades"] >= 10]
    if quality:
        print(f"\n{'=' * 80}")
        print("TOP 10 RISK-ADJUSTED (Sharpe > 0, ≥10 trades, sorted by Sharpe)")
        print(f"{'=' * 80}")
        quality_sorted = sorted(quality, key=lambda r: r["sharpe"], reverse=True)[:10]
        for rank, row in enumerate(quality_sorted, 1):
            print(f"  {rank:2d}. {row['strategy']:<25} {row['config']:<35} "
                  f"Sharpe {row['sharpe']:>6.2f} | {row['total_return_pct']:>+8.2f}% | "
                  f"{row['total_trades']} trades | WR {row['win_rate_pct']:.1f}% | "
                  f"PF {row['profit_factor']:.2f}")

    print(f"\nDone. Full results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
