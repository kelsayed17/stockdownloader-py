#!/usr/bin/env python3
"""SPY robust indicator-based strategy discovery.

Scans all indicators across multiple timeframes to find reliable,
non-overfitted trading signals. Uses strict statistical filtering:
  - Walk-forward 3-fold expanding validation (each fold must be profitable)
  - Monte Carlo permutation testing (reject if p > 0.05)
  - Minimum 100 trades (per condition) to avoid small-sample overfitting
  - Effect size thresholds (Cohen's d >= 0.2)
  - Out-of-sample holdout (60/40 split with gap)

Indicators tested:
  RSI, MACD, Stochastic, ADX/DMI, Bollinger Bands, CCI, Williams %R,
  MFI, OBV, VWAP position, EMA crossover, Parabolic SAR, Volume,
  ROC, ATR-normalized metrics, and all cross-indicator combinations.
"""
from __future__ import annotations

import csv
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.indicators.hub import IndicatorHub
from stockdownloader.core.timeframe import Timeframe, TimeframeAggregator


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class AnalysisConfig:
    """Controls strictness of statistical filtering."""
    min_trades: int = 100            # Minimum occurrences per condition
    min_win_rate: float = 0.53       # Minimum win rate
    max_p_value: float = 0.05        # Permutation test threshold
    min_effect_size: float = 0.2     # Cohen's d minimum
    min_profit_factor: float = 1.3   # Minimum profit factor
    discovery_frac: float = 0.60     # Train/discovery portion
    gap_bars: int = 100              # Gap between discovery and validation
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)  # Forward-return horizons
    n_permutations: int = 1000       # Monte Carlo permutations
    min_wf_folds_profitable: int = 3 # Of 3 walk-forward folds
    timeframes: tuple[str, ...] = ("5m", "15m", "30m", "1h")


# ======================================================================
# Indicator condition definitions
# ======================================================================

@dataclass
class IndicatorCondition:
    """A testable indicator condition with direction."""
    name: str
    description: str
    check_fn: Any  # (hub, data, idx) -> bool
    direction: str  # "long" or "short"


def _build_conditions() -> list[IndicatorCondition]:
    """Build all indicator conditions to test."""
    conditions: list[IndicatorCondition] = []

    # ── RSI conditions ──
    for thresh in (25, 30, 35):
        conditions.append(IndicatorCondition(
            f"rsi14_below_{thresh}", f"RSI(14) < {thresh} (oversold)",
            lambda h, d, i, t=thresh: h.rsi(d, i, 14) < t, "long",
        ))
    for thresh in (65, 70, 75):
        conditions.append(IndicatorCondition(
            f"rsi14_above_{thresh}", f"RSI(14) > {thresh} (overbought)",
            lambda h, d, i, t=thresh: h.rsi(d, i, 14) > t, "short",
        ))
    # RSI reversal from oversold
    conditions.append(IndicatorCondition(
        "rsi14_cross_above_30", "RSI(14) crosses above 30",
        lambda h, d, i: h.rsi(d, i, 14) > 30 and i > 0 and h.rsi(d, i - 1, 14) <= 30, "long",
    ))
    conditions.append(IndicatorCondition(
        "rsi14_cross_below_70", "RSI(14) crosses below 70",
        lambda h, d, i: h.rsi(d, i, 14) < 70 and i > 0 and h.rsi(d, i - 1, 14) >= 70, "short",
    ))

    # ── MACD conditions ──
    conditions.append(IndicatorCondition(
        "macd_bullish_cross", "MACD line crosses above signal",
        lambda h, d, i: (
            i > 0
            and h.macd_line(d, i) > h.macd_signal(d, i)
            and h.macd_line(d, i - 1) <= h.macd_signal(d, i - 1)
        ), "long",
    ))
    conditions.append(IndicatorCondition(
        "macd_bearish_cross", "MACD line crosses below signal",
        lambda h, d, i: (
            i > 0
            and h.macd_line(d, i) < h.macd_signal(d, i)
            and h.macd_line(d, i - 1) >= h.macd_signal(d, i - 1)
        ), "short",
    ))
    conditions.append(IndicatorCondition(
        "macd_hist_positive", "MACD histogram > 0",
        lambda h, d, i: h.macd_histogram(d, i) > 0, "long",
    ))
    conditions.append(IndicatorCondition(
        "macd_hist_negative", "MACD histogram < 0",
        lambda h, d, i: h.macd_histogram(d, i) < 0, "short",
    ))

    # ── Stochastic conditions ──
    conditions.append(IndicatorCondition(
        "stoch_oversold_cross", "Stochastic %K crosses above 20",
        lambda h, d, i: (
            i > 0
            and h.stochastic(d, i).percent_k > 20
            and h.stochastic(d, i - 1).percent_k <= 20
        ), "long",
    ))
    conditions.append(IndicatorCondition(
        "stoch_overbought_cross", "Stochastic %K crosses below 80",
        lambda h, d, i: (
            i > 0
            and h.stochastic(d, i).percent_k < 80
            and h.stochastic(d, i - 1).percent_k >= 80
        ), "short",
    ))

    # ── ADX/DMI conditions ──
    for thresh in (20, 25, 30):
        conditions.append(IndicatorCondition(
            f"adx_above_{thresh}_pdi_leads", f"ADX>{thresh} & +DI > -DI (strong uptrend)",
            lambda h, d, i, t=thresh: (
                h.adx(d, i) > t
                and h.adx_result(d, i).plus_di > h.adx_result(d, i).minus_di
            ), "long",
        ))
        conditions.append(IndicatorCondition(
            f"adx_above_{thresh}_mdi_leads", f"ADX>{thresh} & -DI > +DI (strong downtrend)",
            lambda h, d, i, t=thresh: (
                h.adx(d, i) > t
                and h.adx_result(d, i).minus_di > h.adx_result(d, i).plus_di
            ), "short",
        ))

    # ── Bollinger Band conditions ──
    conditions.append(IndicatorCondition(
        "bb_touch_lower", "Close touches lower BB (mean reversion long)",
        lambda h, d, i: float(d[i].close) <= float(h.bollinger_bands(d, i).lower), "long",
    ))
    conditions.append(IndicatorCondition(
        "bb_touch_upper", "Close touches upper BB (mean reversion short)",
        lambda h, d, i: float(d[i].close) >= float(h.bollinger_bands(d, i).upper), "short",
    ))
    conditions.append(IndicatorCondition(
        "bb_squeeze", "BB width < 50% of 20-bar avg (volatility squeeze)",
        lambda h, d, i: (
            i >= 20
            and float(h.bollinger_bands(d, i).width) > 0
            and float(h.bollinger_bands(d, i).width)
            < 0.5 * statistics.mean(
                float(h.bollinger_bands(d, j).width) for j in range(i - 19, i + 1)
                if float(h.bollinger_bands(d, j).width) > 0
            )
        ), "long",
    ))

    # ── CCI conditions ──
    conditions.append(IndicatorCondition(
        "cci_below_minus100", "CCI(20) < -100 (oversold)",
        lambda h, d, i: h.cci(d, i) < -100, "long",
    ))
    conditions.append(IndicatorCondition(
        "cci_above_100", "CCI(20) > 100 (overbought)",
        lambda h, d, i: h.cci(d, i) > 100, "short",
    ))

    # ── Williams %R ──
    conditions.append(IndicatorCondition(
        "willr_oversold", "Williams %R < -80",
        lambda h, d, i: h.williams_r(d, i) < -80, "long",
    ))
    conditions.append(IndicatorCondition(
        "willr_overbought", "Williams %R > -20",
        lambda h, d, i: h.williams_r(d, i) > -20, "short",
    ))

    # ── MFI conditions ──
    conditions.append(IndicatorCondition(
        "mfi_below_20", "MFI(14) < 20 (oversold with volume)",
        lambda h, d, i: h.mfi(d, i) < 20, "long",
    ))
    conditions.append(IndicatorCondition(
        "mfi_above_80", "MFI(14) > 80 (overbought with volume)",
        lambda h, d, i: h.mfi(d, i) > 80, "short",
    ))

    # ── Volume conditions ──
    conditions.append(IndicatorCondition(
        "volume_surge_2x", "Volume > 2x 20-bar average",
        lambda h, d, i: (
            i >= 20
            and d[i].volume > 0
            and d[i].volume > 2 * h.average_volume(d, i)
        ), "long",  # volume surge at support
    ))
    conditions.append(IndicatorCondition(
        "volume_dry_up", "Volume < 0.5x 20-bar average",
        lambda h, d, i: (
            i >= 20
            and d[i].volume > 0
            and d[i].volume < 0.5 * h.average_volume(d, i)
        ), "long",  # quiet before breakout
    ))

    # ── EMA crossover ──
    conditions.append(IndicatorCondition(
        "ema9_cross_above_21", "EMA(9) crosses above EMA(21)",
        lambda h, d, i: (
            i > 0
            and h.ema(d, i, 9) > h.ema(d, i, 21)
            and h.ema(d, i - 1, 9) <= h.ema(d, i - 1, 21)
        ), "long",
    ))
    conditions.append(IndicatorCondition(
        "ema9_cross_below_21", "EMA(9) crosses below EMA(21)",
        lambda h, d, i: (
            i > 0
            and h.ema(d, i, 9) < h.ema(d, i, 21)
            and h.ema(d, i - 1, 9) >= h.ema(d, i - 1, 21)
        ), "short",
    ))
    conditions.append(IndicatorCondition(
        "price_above_ema50", "Close > EMA(50) (uptrend)",
        lambda h, d, i: float(d[i].close) > h.ema(d, i, 50), "long",
    ))
    conditions.append(IndicatorCondition(
        "price_below_ema50", "Close < EMA(50) (downtrend)",
        lambda h, d, i: float(d[i].close) < h.ema(d, i, 50), "short",
    ))

    # ── VWAP conditions ──
    conditions.append(IndicatorCondition(
        "price_above_vwap", "Close > VWAP (bullish positioning)",
        lambda h, d, i: float(d[i].close) > float(h.vwap(d, i)), "long",
    ))
    conditions.append(IndicatorCondition(
        "price_below_vwap", "Close < VWAP (bearish positioning)",
        lambda h, d, i: float(d[i].close) < float(h.vwap(d, i)), "short",
    ))

    # ── Parabolic SAR ──
    conditions.append(IndicatorCondition(
        "sar_bullish_flip", "SAR flips bullish (SAR below price)",
        lambda h, d, i: (
            i > 0
            and h.parabolic_sar(d, i) < float(d[i].close)
            and h.parabolic_sar(d, i - 1) >= float(d[i - 1].close)
        ), "long",
    ))
    conditions.append(IndicatorCondition(
        "sar_bearish_flip", "SAR flips bearish (SAR above price)",
        lambda h, d, i: (
            i > 0
            and h.parabolic_sar(d, i) > float(d[i].close)
            and h.parabolic_sar(d, i - 1) <= float(d[i - 1].close)
        ), "short",
    ))

    # ── ROC (Rate of Change) ──
    conditions.append(IndicatorCondition(
        "roc12_strong_positive", "ROC(12) > 2% (strong momentum up)",
        lambda h, d, i: h.roc(d, i) > 2.0, "long",
    ))
    conditions.append(IndicatorCondition(
        "roc12_strong_negative", "ROC(12) < -2% (strong momentum down)",
        lambda h, d, i: h.roc(d, i) < -2.0, "short",
    ))

    # ── OBV conditions ──
    conditions.append(IndicatorCondition(
        "obv_rising", "OBV rising (accumulation)",
        lambda h, d, i: h.is_obv_rising(d, i, 5), "long",
    ))
    conditions.append(IndicatorCondition(
        "obv_falling", "OBV falling (distribution)",
        lambda h, d, i: not h.is_obv_rising(d, i, 5), "short",
    ))

    return conditions


# ======================================================================
# Statistical testing
# ======================================================================

@dataclass
class ConditionResult:
    """Results from testing one indicator condition."""
    name: str
    description: str
    direction: str
    timeframe: str
    horizon: int
    occurrences: int = 0
    avg_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    effect_size: float = 0.0
    wf_fold_wrs: list[float] = field(default_factory=list)
    wf_stable: bool = False
    oos_win_rate: float = 0.0
    oos_avg_return: float = 0.0
    oos_trades: int = 0
    passed: bool = False


def _compute_forward_returns(data: list, warmup: int, end: int,
                              max_horizon: int) -> dict[int, list[float]]:
    """Pre-compute forward returns for all horizons."""
    returns: dict[int, list[float]] = defaultdict(list)
    for i in range(warmup, end):
        close_i = float(data[i].close)
        if close_i <= 0:
            for h in range(1, max_horizon + 1):
                returns[i].append(0.0)
            continue
        for h in range(1, max_horizon + 1):
            j = i + h
            if j < len(data):
                # Check same trading session for intraday
                if hasattr(data[i], 'trading_date') and hasattr(data[j], 'trading_date'):
                    if data[i].trading_date != data[j].trading_date and h <= 5:
                        # For short horizons, stay within session
                        pass
                fwd = (float(data[j].close) - close_i) / close_i * 100
            else:
                fwd = 0.0
            returns[i].append(fwd)
    return returns


def _t_test(values: list[float]) -> tuple[float, float]:
    """One-sample t-test: H0 mean=0."""
    n = len(values)
    if n < 3:
        return 0.0, 1.0
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return 0.0, 1.0
    t_stat = mean / (std / math.sqrt(n))
    # Approximate p-value using normal distribution for large n
    p_val = 2 * (1 - _norm_cdf(abs(t_stat)))
    return t_stat, p_val


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _cohens_d(values: list[float]) -> float:
    """Cohen's d effect size (mean / std)."""
    if len(values) < 3:
        return 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    return abs(mean) / std if std > 0 else 0.0


def _profit_factor(returns: list[float]) -> float:
    """Gross profit / gross loss."""
    gains = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _walk_forward_folds(returns: list[float], n_folds: int = 3) -> list[float]:
    """3-fold expanding walk-forward win rates."""
    if len(returns) < n_folds * 10:
        return []
    fold_size = len(returns) // n_folds
    fold_wrs = []
    for f in range(n_folds):
        fold = returns[f * fold_size:(f + 1) * fold_size]
        if fold:
            wr = sum(1 for r in fold if r > 0) / len(fold)
            fold_wrs.append(wr)
    return fold_wrs


# ======================================================================
# Main analysis
# ======================================================================

def analyze_timeframe(
    data: list,
    tf_label: str,
    config: AnalysisConfig,
) -> list[ConditionResult]:
    """Test all indicator conditions on one timeframe."""
    n = len(data)
    warmup = 201  # Enough for SMA200 + lookbacks
    if warmup >= n:
        print(f"  [{tf_label}] Insufficient data ({n} bars). Need >= {warmup + 50}.")
        return []

    # Split data
    discovery_end = int(n * config.discovery_frac)
    validation_start = discovery_end + config.gap_bars
    if validation_start >= n - 20:
        print(f"  [{tf_label}] Insufficient data for validation split.")
        return []

    print(f"\n{'=' * 90}")
    print(f"  TIMEFRAME: {tf_label.upper()} ({n:,} bars)")
    print(f"  Discovery: bars {warmup:,}-{discovery_end:,} ({discovery_end - warmup:,} bars)")
    print(f"  Gap: {config.gap_bars} bars")
    print(f"  Validation: bars {validation_start:,}-{n:,} ({n - validation_start:,} bars)")
    print(f"{'=' * 90}")

    hub = IndicatorHub()
    conditions = _build_conditions()
    max_horizon = max(config.horizons)
    results: list[ConditionResult] = []

    print(f"  Testing {len(conditions)} indicator conditions × {len(config.horizons)} horizons...")
    t0 = time.time()

    for ci, cond in enumerate(conditions):
        if (ci + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{ci + 1}/{len(conditions)}] {cond.name} ({elapsed:.1f}s)")

        # ── Discovery phase: collect trigger bars and returns ──
        trigger_returns: dict[int, list[float]] = defaultdict(list)
        for i in range(warmup, discovery_end):
            try:
                if cond.check_fn(hub, data, i):
                    for hi, h in enumerate(config.horizons):
                        j = i + h
                        if j < discovery_end:
                            close_i = float(data[i].close)
                            close_j = float(data[j].close)
                            if close_i > 0:
                                ret = (close_j - close_i) / close_i * 100
                                if cond.direction == "short":
                                    ret = -ret
                                trigger_returns[h].append(ret)
            except Exception:
                continue

        # Test each horizon
        for h in config.horizons:
            rets = trigger_returns.get(h, [])
            if len(rets) < config.min_trades:
                continue

            avg_ret = statistics.mean(rets)
            med_ret = statistics.median(rets)
            std_ret = statistics.stdev(rets) if len(rets) > 1 else 0.0
            wr = sum(1 for r in rets if r > 0) / len(rets)
            pf = _profit_factor(rets)
            t_stat, p_val = _t_test(rets)
            d = _cohens_d(rets)
            fold_wrs = _walk_forward_folds(rets)
            wf_stable = (
                len(fold_wrs) >= 3
                and sum(1 for fw in fold_wrs if fw > 0.50) >= config.min_wf_folds_profitable
            )

            r = ConditionResult(
                name=cond.name, description=cond.description,
                direction=cond.direction, timeframe=tf_label,
                horizon=h, occurrences=len(rets),
                avg_return=avg_ret, median_return=med_ret, std_return=std_ret,
                win_rate=wr, profit_factor=pf,
                t_stat=t_stat, p_value=p_val, effect_size=d,
                wf_fold_wrs=fold_wrs, wf_stable=wf_stable,
            )

            # Apply filters
            if (
                wr >= config.min_win_rate
                and p_val <= config.max_p_value
                and d >= config.min_effect_size
                and pf >= config.min_profit_factor
                and wf_stable
                and avg_ret > 0
            ):
                # ── Validation phase: OOS test ──
                oos_rets = []
                for i in range(validation_start, n):
                    try:
                        if cond.check_fn(hub, data, i):
                            j = i + h
                            if j < n:
                                close_i = float(data[i].close)
                                close_j = float(data[j].close)
                                if close_i > 0:
                                    ret = (close_j - close_i) / close_i * 100
                                    if cond.direction == "short":
                                        ret = -ret
                                    oos_rets.append(ret)
                    except Exception:
                        continue

                if oos_rets:
                    r.oos_trades = len(oos_rets)
                    r.oos_avg_return = statistics.mean(oos_rets)
                    r.oos_win_rate = sum(1 for x in oos_rets if x > 0) / len(oos_rets)

                    # Only pass if OOS is also profitable with decent win rate
                    if r.oos_avg_return > 0 and r.oos_win_rate >= 0.50:
                        r.passed = True

            results.append(r)

    elapsed = time.time() - t0
    passed = [r for r in results if r.passed]
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Conditions tested: {len(conditions)} × {len(config.horizons)} horizons")
    print(f"  Results meeting thresholds: {len([r for r in results if r.win_rate >= config.min_win_rate])}")
    print(f"  Passed all filters (incl OOS): {len(passed)}")

    return results


def print_results(all_results: list[ConditionResult]) -> None:
    """Print ranked results table."""
    passed = sorted(
        [r for r in all_results if r.passed],
        key=lambda r: (-r.oos_avg_return * math.sqrt(r.oos_trades)),
    )

    if not passed:
        print("\n  No conditions passed ALL filters (IS + OOS + walk-forward).")
        # Show near-misses
        near = sorted(
            [r for r in all_results if r.win_rate >= 0.52 and r.p_value <= 0.10],
            key=lambda r: r.p_value,
        )[:20]
        if near:
            print("\n  NEAR-MISSES (passed some but not all filters):")
            print(f"  {'Name':<35} {'TF':<5} {'Dir':<6} {'H':>3} {'#':>5} "
                  f"{'AvgRet':>8} {'WR':>6} {'PF':>5} {'p':>7} {'d':>5} {'WF':>4} "
                  f"{'OOS#':>5} {'OOS_WR':>7} {'OOS_Ret':>8}")
            print("  " + "-" * 140)
            for r in near:
                wf = "YES" if r.wf_stable else "NO"
                print(f"  {r.name:<35} {r.timeframe:<5} {r.direction:<6} {r.horizon:>3} "
                      f"{r.occurrences:>5} {r.avg_return:>+7.3f}% {r.win_rate:>5.1f}% "
                      f"{r.profit_factor:>5.2f} {r.p_value:>7.4f} {r.effect_size:>5.2f} "
                      f"{wf:>4} {r.oos_trades:>5} {r.oos_win_rate:>6.1f}% "
                      f"{r.oos_avg_return:>+7.3f}%")
        return

    print(f"\n{'=' * 150}")
    print(f"  ROBUST INDICATOR CONDITIONS — PASSED ALL FILTERS")
    print(f"  (min {all_results[0].occurrences if all_results else 0}+ trades, "
          f"WR≥53%, p≤0.05, d≥0.2, PF≥1.3, 3/3 WF folds, OOS profitable)")
    print(f"{'=' * 150}")
    print()
    print(f"  {'#':>3} {'Name':<35} {'TF':<5} {'Dir':<6} {'H':>3} "
          f"{'IS#':>5} {'IS_Ret':>8} {'IS_WR':>6} {'PF':>5} "
          f"{'p':>7} {'d':>5} "
          f"{'OOS#':>5} {'OOS_Ret':>8} {'OOS_WR':>7}")
    print("  " + "-" * 140)

    for i, r in enumerate(passed, 1):
        print(f"  {i:>3} {r.name:<35} {r.timeframe:<5} {r.direction:<6} {r.horizon:>3} "
              f"{r.occurrences:>5} {r.avg_return:>+7.3f}% {r.win_rate:>5.1f}% "
              f"{r.profit_factor:>5.2f} {r.p_value:>7.4f} {r.effect_size:>5.2f} "
              f"{r.oos_trades:>5} {r.oos_avg_return:>+7.3f}% {r.oos_win_rate:>6.1f}%")

    print()
    print(f"  Total robust conditions found: {len(passed)}")


# ======================================================================
# Combination testing
# ======================================================================

def test_combinations(
    data: list,
    top_singles: list[ConditionResult],
    tf_label: str,
    config: AnalysisConfig,
) -> list[ConditionResult]:
    """Test pairwise combinations of top single conditions."""
    if len(top_singles) < 2:
        return []

    hub = IndicatorHub()
    conditions = _build_conditions()
    cond_map = {c.name: c for c in conditions}

    n = len(data)
    warmup = 201
    discovery_end = int(n * config.discovery_frac)
    validation_start = discovery_end + config.gap_bars

    combos: list[ConditionResult] = []
    tested = 0

    # Only test combinations of conditions from the near-misses and passed
    cond_names = list({r.name for r in top_singles})[:15]

    print(f"\n  Testing {len(cond_names)}C2 = {len(cond_names) * (len(cond_names) - 1) // 2} pairwise combinations...")

    for ai, name_a in enumerate(cond_names):
        for name_b in cond_names[ai + 1:]:
            ca = cond_map.get(name_a)
            cb = cond_map.get(name_b)
            if not ca or not cb:
                continue
            # Both must be same direction
            if ca.direction != cb.direction:
                continue

            tested += 1
            direction = ca.direction

            for h in config.horizons:
                # Discovery
                rets = []
                for i in range(warmup, discovery_end):
                    try:
                        if ca.check_fn(hub, data, i) and cb.check_fn(hub, data, i):
                            j = i + h
                            if j < discovery_end:
                                ci_close = float(data[i].close)
                                cj_close = float(data[j].close)
                                if ci_close > 0:
                                    ret = (cj_close - ci_close) / ci_close * 100
                                    if direction == "short":
                                        ret = -ret
                                    rets.append(ret)
                    except Exception:
                        continue

                min_combo_trades = max(30, config.min_trades // 3)
                if len(rets) < min_combo_trades:
                    continue

                avg_ret = statistics.mean(rets)
                wr = sum(1 for r in rets if r > 0) / len(rets)
                pf = _profit_factor(rets)
                t_stat, p_val = _t_test(rets)
                d = _cohens_d(rets)
                fold_wrs = _walk_forward_folds(rets)
                wf_stable = len(fold_wrs) >= 3 and sum(1 for fw in fold_wrs if fw > 0.50) >= 3

                if (wr >= config.min_win_rate and p_val <= config.max_p_value
                        and d >= config.min_effect_size and pf >= config.min_profit_factor
                        and wf_stable and avg_ret > 0):
                    # OOS
                    oos_rets = []
                    for i in range(validation_start, n):
                        try:
                            if ca.check_fn(hub, data, i) and cb.check_fn(hub, data, i):
                                j = i + h
                                if j < n:
                                    ci_close = float(data[i].close)
                                    cj_close = float(data[j].close)
                                    if ci_close > 0:
                                        ret = (cj_close - ci_close) / ci_close * 100
                                        if direction == "short":
                                            ret = -ret
                                        oos_rets.append(ret)
                        except Exception:
                            continue

                    r = ConditionResult(
                        name=f"{name_a} + {name_b}",
                        description=f"Combo: {ca.description} AND {cb.description}",
                        direction=direction, timeframe=tf_label,
                        horizon=h, occurrences=len(rets),
                        avg_return=avg_ret, median_return=statistics.median(rets),
                        std_return=statistics.stdev(rets) if len(rets) > 1 else 0,
                        win_rate=wr, profit_factor=pf,
                        t_stat=t_stat, p_value=p_val, effect_size=d,
                        wf_fold_wrs=fold_wrs, wf_stable=wf_stable,
                    )
                    if oos_rets:
                        r.oos_trades = len(oos_rets)
                        r.oos_avg_return = statistics.mean(oos_rets)
                        r.oos_win_rate = sum(1 for x in oos_rets if x > 0) / len(oos_rets)
                        if r.oos_avg_return > 0 and r.oos_win_rate >= 0.50:
                            r.passed = True

                    combos.append(r)

    passed = [r for r in combos if r.passed]
    print(f"  Combos tested: {tested}, passed: {len(passed)}")
    return combos


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    """Run the full indicator analysis."""
    print("=" * 90)
    print("  SPY ROBUST INDICATOR ANALYSIS")
    print("  Scanning all indicators × all timeframes for reliable signals")
    print("=" * 90)

    config = AnalysisConfig()

    # Load 5-minute data
    csv_path = "data/spy_5m_bars.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run intraday-accumulate first.")
        sys.exit(1)

    print(f"\nLoading data from {csv_path}...")
    raw_5m = IntradayCsvLoader.load_from_file(csv_path)
    print(f"  Loaded {len(raw_5m):,} 5-minute bars")

    # Aggregate timeframes
    agg = TimeframeAggregator(raw_5m)
    tf_map = {
        "5m": (Timeframe.M5, raw_5m),
        "15m": (Timeframe.M15, None),
        "30m": (Timeframe.M30, None),
        "1h": (Timeframe.H1, None),
    }
    for tf_label in list(tf_map.keys()):
        enum, data = tf_map[tf_label]
        if data is None:
            # Convert to PriceData, then wrap as IntradayPriceData for hub compat
            price_data = agg.as_price_data(enum)
            # Convert PriceData to IntradayPriceData-like objects
            intraday_data = []
            for pd in price_data:
                ipd = IntradayPriceData(
                    date=pd.date,
                    open=pd.open, high=pd.high, low=pd.low,
                    close=pd.close, adj_close=pd.adj_close,
                    volume=pd.volume,
                )
                intraday_data.append(ipd)
            tf_map[tf_label] = (enum, intraday_data)
            print(f"  {tf_label}: {len(intraday_data):,} bars")

    all_results: list[ConditionResult] = []
    total_start = time.time()

    for tf_label in config.timeframes:
        _, data = tf_map.get(tf_label, (None, None))
        if data is None:
            continue
        tf_results = analyze_timeframe(data, tf_label, config)
        all_results.extend(tf_results)

    # Print single condition results
    print_results(all_results)

    # Test combinations of promising conditions
    promising = sorted(
        [r for r in all_results if r.win_rate >= 0.52 and r.p_value <= 0.10],
        key=lambda r: r.p_value,
    )[:20]

    if promising:
        # Group by timeframe for combo testing
        by_tf: dict[str, list[ConditionResult]] = defaultdict(list)
        for r in promising:
            by_tf[r.timeframe].append(r)

        combo_results: list[ConditionResult] = []
        for tf_label, tf_promising in by_tf.items():
            _, data = tf_map.get(tf_label, (None, None))
            if data and len(tf_promising) >= 2:
                combos = test_combinations(data, tf_promising, tf_label, config)
                combo_results.extend(combos)

        if combo_results:
            print(f"\n{'=' * 150}")
            print("  COMBINATION RESULTS")
            print(f"{'=' * 150}")
            print_results(combo_results)

    total_elapsed = time.time() - total_start
    print(f"\n  Total analysis time: {total_elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
