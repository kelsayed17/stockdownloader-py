#!/usr/bin/env python3
"""SPY 5-Minute VWAP Strategy Optimizer.

4-phase autonomous optimizer to find the best VWAP-based intraday strategy
for SPY with high win rate, good trade count, and profitable P&L.

Phase 1: Baseline all 6 VWAP strategies with default configs
Phase 2: Build 5 experimental variants (RVOL, time filter, combos)
Phase 3: Greedy parameter sweep on top 3
Phase 4: Walk-forward validation (5 rolling windows)
"""
from __future__ import annotations

import dataclasses
import math
import sys
import time
from collections import defaultdict, deque
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_scoring import score as default_score
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.core.models.trade import HOLD, IntradayAction, IntradaySignal
from stockdownloader.strategies.base import IntradayTradingStrategy
from stockdownloader.strategies.intraday.pullback import PullbackStrategy
from stockdownloader.strategies.intraday.pullback import PullbackStrategyConfig
from stockdownloader.strategies.intraday.reversal import ReversalStrategy
from stockdownloader.strategies.intraday.reversal import ReversalStrategyConfig

from stockdownloader.strategies.loader import ensure_registered
ensure_registered()

from stockdownloader.strategies.registry import StrategyRegistry

# ======================================================================
# Constants
# ======================================================================

CAPITAL = Decimal("100000")
RISK = Decimal("0.01")
SEP = "=" * 110
THIN = "-" * 110


def _s2(v: Decimal) -> str:
    return str(v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _pct(v: Decimal) -> str:
    return f"{v.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}%"


# ======================================================================
# Custom WR-Weighted Scoring Function
# ======================================================================


def wr_weighted_score(result: BacktestResult, trading_days: int = 0) -> float:
    """Composite score that prioritizes win rate 75% more than default.

    Formula:
        (WR * 35) + (PF * 25) + (Sharpe * 25) - (MaxDD * 15) + trade_bonus
    """
    wr = float(result.win_rate) / 100.0
    pf = min(float(result.profit_factor), 5.0)
    sharpe = float(result.sharpe_ratio(trading_days_per_year=252 * 78))
    dd = float(result.max_drawdown) / 100.0

    # Trade count penalty / bonus
    trade_penalty = 0.0
    if result.total_trades < 5:
        trade_penalty = 100.0
    elif result.total_trades < 20:
        trade_penalty = (20 - result.total_trades) * 3.0

    trade_bonus = 0.0
    if result.total_trades > 0:
        trade_bonus = min(math.log2(result.total_trades) * 1.5, 5.0)

    base = (wr * 35) + (pf * 25) + (sharpe * 25) - (dd * 15)
    return base - trade_penalty + trade_bonus


# ======================================================================
# Wrapper Classes (experimental filters, no production code changes)
# ======================================================================


class RvolGatedWrapper(IntradayTradingStrategy):
    """Wraps any IntradayTradingStrategy, gating entries on RVOL >= threshold.

    RVOL is computed as: current bar volume / 20-period SMA of
    same-time-of-day volume from prior sessions.
    """

    def __init__(
        self,
        inner: IntradayTradingStrategy,
        rvol_threshold: float = 1.5,
        lookback: int = 20,
        display_name: str | None = None,
    ) -> None:
        self._inner = inner
        self._rvol_thresh = rvol_threshold
        self._lookback = lookback
        self._display_name = display_name or f"RVOL({rvol_threshold:.1f})+{inner.name}"
        # time_slot -> deque of volumes (rolling window)
        self._vol_by_time: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=lookback)
        )

    @property
    def name(self) -> str:
        return self._display_name

    @property
    def warmup_period(self) -> int:
        return self._inner.warmup_period

    def on_session_start(self, trading_date: str) -> None:
        self._inner.on_session_start(trading_date)

    def on_position_opened(self, is_long: bool) -> None:
        self._inner.on_position_opened(is_long)

    def on_position_closed(self) -> None:
        self._inner.on_position_closed()

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        signal = self._inner.evaluate(data, current_index)

        # Only gate entry signals; exits and holds pass through
        if signal.action not in (
            IntradayAction.ENTER_LONG,
            IntradayAction.ENTER_SHORT,
        ):
            # Still record volume for RVOL calculation
            self._record_volume(data, current_index)
            return signal

        # Compute RVOL
        bar = data[current_index]
        time_slot = bar.date[11:16]  # "HH:MM"
        vol = float(bar.volume)

        history = self._vol_by_time[time_slot]
        if len(history) >= 5:  # need at least 5 observations
            avg_vol = sum(history) / len(history)
            rvol = vol / avg_vol if avg_vol > 0 else 0.0
        else:
            rvol = 0.0  # not enough history, block signal

        self._record_volume(data, current_index)

        if rvol >= self._rvol_thresh:
            return signal
        return HOLD

    def _record_volume(
        self, data: list[IntradayPriceData], index: int
    ) -> None:
        bar = data[index]
        time_slot = bar.date[11:16]
        self._vol_by_time[time_slot].append(float(bar.volume))


class TimeFilterWrapper(IntradayTradingStrategy):
    """Wraps any strategy, only allowing entry signals during specified
    time windows (ET).  Exits always pass through.
    """

    def __init__(
        self,
        inner: IntradayTradingStrategy,
        windows: list[tuple[int, int, int, int]] | None = None,
        display_name: str | None = None,
    ) -> None:
        self._inner = inner
        # Default: 10:00-11:30 and 14:00-15:30 ET
        self._windows = windows or [(10, 0, 11, 30), (14, 0, 15, 30)]
        self._display_name = display_name or f"TimeFilt+{inner.name}"

    @property
    def name(self) -> str:
        return self._display_name

    @property
    def warmup_period(self) -> int:
        return self._inner.warmup_period

    def on_session_start(self, trading_date: str) -> None:
        self._inner.on_session_start(trading_date)

    def on_position_opened(self, is_long: bool) -> None:
        self._inner.on_position_opened(is_long)

    def on_position_closed(self) -> None:
        self._inner.on_position_closed()

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        signal = self._inner.evaluate(data, current_index)

        if signal.action not in (
            IntradayAction.ENTER_LONG,
            IntradayAction.ENTER_SHORT,
        ):
            return signal

        bar = data[current_index]
        # Parse time from date string "YYYY-MM-DD HH:MM:SS"
        try:
            time_part = bar.date[11:16]  # "HH:MM"
            hh, mm = int(time_part[:2]), int(time_part[3:5])
        except (ValueError, IndexError):
            return HOLD

        bar_minutes = hh * 60 + mm
        for sh, sm, eh, em in self._windows:
            start_min = sh * 60 + sm
            end_min = eh * 60 + em
            if start_min <= bar_minutes <= end_min:
                return signal

        return HOLD


# ======================================================================
# Helpers
# ======================================================================


def run_backtest(
    strategy: IntradayTradingStrategy,
    data: list[IntradayPriceData],
) -> BacktestResult:
    engine = IntradayBacktestEngine(CAPITAL, RISK)
    return engine.run(strategy, data)


def print_result_row(
    label: str,
    result: BacktestResult,
    trading_days: int,
    score_fn=wr_weighted_score,
) -> float:
    s = score_fn(result, trading_days=trading_days)
    pnl = float(result.total_pnl)
    sign = "+" if pnl >= 0 else ""
    print(
        f"  {label:<35s} P/L:{sign}${pnl:>10,.2f}  "
        f"WR:{_s2(result.win_rate):>6}%  "
        f"PF:{str(result.profit_factor):>5}  "
        f"MaxDD:{_s2(result.max_drawdown):>6}%  "
        f"Trades:{result.total_trades:>4d}  "
        f"Score:{s:>7.1f}"
    )
    return s


def get_trading_days(data: list[IntradayPriceData]) -> int:
    return len({bar.date[:10] for bar in data})


# ======================================================================
# Phase 1: Baseline All VWAP Strategies
# ======================================================================


def phase1_baseline(
    data: list[IntradayPriceData], trading_days: int
) -> list[tuple[str, Any, BacktestResult, float]]:
    """Run all 6 VWAP strategies with default configs. Returns sorted results."""
    print(f"\n{SEP}")
    print("  PHASE 1: BASELINE — ALL 6 VWAP STRATEGIES (default configs)")
    print(SEP)

    strategies: list[tuple[str, IntradayTradingStrategy]] = []

    # Registry-based strategies
    for reg_name in ["vwap-pullback", "vwap-reversal", "vwap-orb", "vwap-orr", "avwap-pullback"]:
        try:
            entry = StrategyRegistry.get(reg_name)
            strat = entry.factory()
            strategies.append((entry.display_name, strat))
        except Exception as exc:
            print(f"  WARNING: could not load {reg_name}: {exc}")

    # DMI+VWAP (direct import, not in registry)
    try:
        from stockdownloader.strategies.intraday.dmi_vwap import DmiVwapStrategy
        strategies.append(("DMI+VWAP", DmiVwapStrategy()))
    except Exception as exc:
        print(f"  WARNING: could not load DMI+VWAP: {exc}")

    results: list[tuple[str, Any, BacktestResult, float]] = []

    print(f"\n  {'Strategy':<35s} {'P/L':>14s}  {'WR':>7s}  {'PF':>5s}  "
          f"{'MaxDD':>7s}  {'Trades':>6s}  {'Score':>7s}")
    print(f"  {THIN}")

    for name, strat in strategies:
        t0 = time.time()
        result = run_backtest(strat, data)
        elapsed = time.time() - t0
        score = print_result_row(name, result, trading_days)
        results.append((name, strat, result, score))

    # Sort by score descending
    results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n  Top baseline: {results[0][0]} (Score={results[0][3]:.1f})")
    return results


# ======================================================================
# Phase 2: Experimental Variants
# ======================================================================


def phase2_variants(
    data: list[IntradayPriceData], trading_days: int
) -> list[tuple[str, Any, BacktestResult, float]]:
    """Build and test 5 experimental variants targeting high WR."""
    print(f"\n\n{SEP}")
    print("  PHASE 2: EXPERIMENTAL VARIANTS — Targeting High Win Rate")
    print(SEP)

    variants: list[tuple[str, IntradayTradingStrategy]] = []

    # Variant A: Relaxed Pullback (disable restrictive filters, max 3 trades/day)
    cfg_a = PullbackStrategyConfig(
        allow_shorts=False,
        max_day=3,
        htf_align=False,        # remove HTF alignment filter
        ar_filter=False,         # remove AR ratio filter
        va_filter=False,         # remove VA acceleration filter
        cvd_long_filter=False,   # remove CVD filter
        lrs_short_filter=False,  # remove LRS filter
        pb_vwap_bias=False,      # remove VWAP bias filter
        min_score=3,             # lower confluence threshold
        trend_bars=5,            # easier trend requirement
    )
    variants.append(("A: Relaxed PB (3/day)", PullbackStrategy(config=cfg_a)))

    # Variant B: Relaxed Pullback + Time Filter (best windows only)
    cfg_b = PullbackStrategyConfig(
        allow_shorts=False,
        max_day=3,
        htf_align=False,
        ar_filter=False,
        va_filter=False,
        cvd_long_filter=False,
        lrs_short_filter=False,
        pb_vwap_bias=False,
        min_score=3,
        trend_bars=5,
    )
    inner_b = PullbackStrategy(config=cfg_b)
    variants.append((
        "B: Relaxed PB + TimeFilt",
        TimeFilterWrapper(
            inner_b,
            windows=[(10, 0, 11, 30), (14, 0, 15, 30)],
            display_name="B: Relaxed PB TimeFilt",
        ),
    ))

    # Variant C: High-WR Pullback (keep some filters for quality, long-only)
    cfg_c = PullbackStrategyConfig(
        allow_shorts=False,
        max_day=2,
        min_score=4,
        rr=Decimal("2.0"),
        sl_atr=Decimal("1.5"),
        htf_align=False,
        ar_filter=False,
        va_filter=False,
        cvd_long_filter=True,    # keep CVD for quality
        lrs_short_filter=False,
        pb_vwap_bias=True,       # keep VWAP bias
    )
    variants.append(("C: Balanced PB (2/day)", PullbackStrategy(config=cfg_c)))

    # Variant D: OR Reversal Long-Only (best baseline was ORR)
    from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategy
    from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategyConfig
    cfg_d = ORReversalStrategyConfig(
        allow_shorts=False,       # long-only
        max_day=2,
    )
    variants.append(("D: ORR Long-Only (2/day)", ORReversalStrategy(config=cfg_d)))

    # Variant E: OR Reversal Long-Only + Time Filter
    cfg_e = ORReversalStrategyConfig(
        allow_shorts=False,
        max_day=2,
    )
    inner_e = ORReversalStrategy(config=cfg_e)
    variants.append((
        "E: ORR LO + TimeFilt",
        TimeFilterWrapper(
            inner_e,
            windows=[(10, 0, 11, 30), (14, 0, 15, 30)],
            display_name="E: ORR LO TimeFilt",
        ),
    ))

    # Variant F: Reversal Long-Only, max 2/day
    cfg_f = ReversalStrategyConfig(
        allow_shorts=False,
        max_day=2,
        min_score=4,             # slightly lower threshold
    )
    variants.append(("F: REV LO (2/day)", ReversalStrategy(config=cfg_f)))

    # Variant G: Reversal + Time Filter
    cfg_g = ReversalStrategyConfig(
        allow_shorts=False,
        max_day=2,
        min_score=4,
    )
    inner_g = ReversalStrategy(config=cfg_g)
    variants.append((
        "G: REV LO + TimeFilt",
        TimeFilterWrapper(
            inner_g,
            windows=[(10, 0, 11, 30), (14, 0, 15, 30)],
            display_name="G: REV LO TimeFilt",
        ),
    ))

    # Variant H: Relaxed PB + RVOL gate (lower threshold)
    cfg_h = PullbackStrategyConfig(
        allow_shorts=False,
        max_day=3,
        htf_align=False,
        ar_filter=False,
        va_filter=False,
        cvd_long_filter=False,
        lrs_short_filter=False,
        pb_vwap_bias=False,
        min_score=3,
        trend_bars=5,
    )
    inner_h = PullbackStrategy(config=cfg_h)
    variants.append((
        "H: Relaxed PB + RVOL",
        RvolGatedWrapper(
            inner_h,
            rvol_threshold=1.2,  # lower RVOL threshold
            display_name="H: Relaxed PB RVOL",
        ),
    ))

    # ── Pine Script-Calibrated Variants ────────────────────────────────

    # Variant I: Pine-Calibrated PB (matches v10.9.2 GOLD params)
    cfg_i = PullbackStrategyConfig(
        allow_shorts=False,
        adx_thresh=Decimal("21"),       # Pine v10+ uses 21 (Python: 22)
        rr=Decimal("1.4"),              # Pine: 1.4 (Python: 1.8)
        sl_atr=Decimal("1.3"),          # Pine: 1.3 (same)
        sl_cap=Decimal("1.50"),         # Pine: $1.50 (Python: $2.00)
        be_trigger=Decimal("0.5"),      # Pine: 0.5R (Python: 0.7R)
        trend_bars=5,                    # Pine: 5 (Python: 7)
        min_score=3,                     # Pine: 3 (Python: 4)
        max_day=3,
        htf_align=True,                  # Keep — biggest WR filter
        cvd_long_filter=True,            # Keep — CVD>0 = 83% WR
        ar_filter=True,
        va_filter=True,
        lrs_short_filter=True,
        pb_vwap_bias=True,
        trail_buf=Decimal("0.15"),       # Pine: 0.15 ATR (Python: 0.10)
    )
    variants.append(("I: Pine-Cal PB (GOLD)", PullbackStrategy(config=cfg_i)))

    # Variant J: Pine PB Relaxed (same params, fewer filters for more trades)
    cfg_j = PullbackStrategyConfig(
        allow_shorts=False,
        adx_thresh=Decimal("21"),
        rr=Decimal("1.4"),
        sl_atr=Decimal("1.3"),
        sl_cap=Decimal("1.50"),
        be_trigger=Decimal("0.5"),
        trend_bars=5,
        min_score=3,
        max_day=3,
        htf_align=False,                 # OFF — more trades
        cvd_long_filter=True,            # Keep CVD
        ar_filter=False,                 # OFF — more trades
        va_filter=False,                 # OFF — more trades
        lrs_short_filter=False,
        pb_vwap_bias=False,              # OFF — more trades
        trail_buf=Decimal("0.15"),
    )
    variants.append(("J: Pine PB Relaxed", PullbackStrategy(config=cfg_j)))

    # Variant K: Pine ORR Long (OR Reversal with Pine-calibrated proximity)
    cfg_k = ORReversalStrategyConfig(
        allow_shorts=False,
        adx_thresh=Decimal("21"),
        orr_sl_atr=Decimal("0.5"),
        orr_prox=Decimal("0.2"),        # Pine: 0.2 ATR proximity (Python: 0.6)
        max_day=2,
    )
    variants.append(("K: Pine ORR LO (prox=0.2)", ORReversalStrategy(config=cfg_k)))

    # Variant L: Pine ORB with RVOL >= 2.0 gate (75% WR in Pine)
    from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy
    from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategyConfig
    cfg_l = ORBreakoutStrategyConfig(
        allow_shorts=False,
        max_day=2,
    )
    inner_l = ORBreakoutStrategy(config=cfg_l)
    variants.append((
        "L: Pine ORB RVOL>=2.0",
        RvolGatedWrapper(
            inner_l,
            rvol_threshold=2.0,
            display_name="L: Pine ORB RVOL2",
        ),
    ))

    # Variant M: Pine PB w_vol=0 (test volume scoring impact)
    cfg_m = PullbackStrategyConfig(
        allow_shorts=False,
        adx_thresh=Decimal("21"),
        rr=Decimal("1.4"),
        sl_atr=Decimal("1.3"),
        sl_cap=Decimal("1.50"),
        be_trigger=Decimal("0.5"),
        trend_bars=5,
        min_score=3,
        max_day=3,
        w_vol=0,                         # Disable volume scoring entirely
        htf_align=True,
        cvd_long_filter=True,
        ar_filter=True,
        va_filter=True,
        lrs_short_filter=True,
        pb_vwap_bias=True,
        trail_buf=Decimal("0.15"),
    )
    variants.append(("M: Pine PB no-vol-score", PullbackStrategy(config=cfg_m)))

    results: list[tuple[str, Any, BacktestResult, float]] = []

    print(f"\n  {'Variant':<35s} {'P/L':>14s}  {'WR':>7s}  {'PF':>5s}  "
          f"{'MaxDD':>7s}  {'Trades':>6s}  {'Score':>7s}")
    print(f"  {THIN}")

    for name, strat in variants:
        result = run_backtest(strat, data)
        score = print_result_row(name, result, trading_days)
        results.append((name, strat, result, score))

    results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n  Top variant: {results[0][0]} (Score={results[0][3]:.1f})")
    return results


# ======================================================================
# Phase 3: Greedy Parameter Sweep
# ======================================================================

# Parameter grids for Pullback-based strategies
PB_PARAM_GRID: dict[str, list[Any]] = {
    "rr": [Decimal("1.3"), Decimal("1.4"), Decimal("1.5"), Decimal("1.8"), Decimal("2.0"), Decimal("2.5")],
    "sl_atr": [Decimal("1.0"), Decimal("1.3"), Decimal("1.5"), Decimal("1.8"), Decimal("2.0")],
    "sl_cap": [Decimal("1.50"), Decimal("2.00"), Decimal("2.50"), Decimal("3.00")],
    "pb_zone": [Decimal("0.3"), Decimal("0.5"), Decimal("0.7"), Decimal("1.0")],
    "adx_thresh": [Decimal("18"), Decimal("20"), Decimal("21"), Decimal("22"), Decimal("25")],
    "trend_bars": [5, 7, 9, 12],
    "min_score": [3, 4, 5, 6],
    "spacing": [3, 5, 7, 10],
    "max_day": [1, 2, 3, 5],
    "be_trigger": [Decimal("0.3"), Decimal("0.5"), Decimal("0.7"), Decimal("1.0")],
    "trail_buf": [Decimal("0.10"), Decimal("0.15"), Decimal("0.20"), Decimal("0.30")],
}

# Parameter grids for Reversal-based strategies
REV_PARAM_GRID: dict[str, list[Any]] = {
    "rev_sl_atr": [Decimal("0.8"), Decimal("1.0"), Decimal("1.3"), Decimal("1.5")],
    "rev_body": [Decimal("0.10"), Decimal("0.15"), Decimal("0.20"), Decimal("0.25")],
    "adx_thresh": [Decimal("18"), Decimal("20"), Decimal("21"), Decimal("22"), Decimal("25")],
    "min_score": [3, 4, 5, 6],
    "spacing": [5, 7, 10, 15],
    "max_day": [1, 2, 3, 5],
    "be_trigger": [Decimal("0.3"), Decimal("0.5"), Decimal("0.7"), Decimal("1.0")],
}

# Parameter grids for OR Reversal strategies
ORR_PARAM_GRID: dict[str, list[Any]] = {
    "orr_sl_atr": [Decimal("0.3"), Decimal("0.5"), Decimal("0.7"), Decimal("1.0")],
    "orr_sl_cap": [Decimal("1.50"), Decimal("2.00"), Decimal("2.50"), Decimal("3.00")],
    "orr_prox": [Decimal("0.4"), Decimal("0.6"), Decimal("0.8"), Decimal("1.0")],
    "adx_thresh": [Decimal("18"), Decimal("20"), Decimal("22"), Decimal("25")],
    "min_score": [3, 4, 5, 6],
    "spacing": [3, 5, 7, 10],
    "max_day": [1, 2, 3, 5],
}


def _is_pullback_based(name: str) -> bool:
    name_lower = name.lower()
    return ("pullback" in name_lower or "pb" in name_lower) and "orr" not in name_lower


def _is_reversal_based(name: str) -> bool:
    name_lower = name.lower()
    return ("reversal" in name_lower or "rev" in name_lower) and "orr" not in name_lower and "or " not in name_lower


def _is_or_reversal_based(name: str) -> bool:
    name_lower = name.lower()
    return "or reversal" in name_lower or "orr" in name_lower


def _extract_inner_config(strat: IntradayTradingStrategy):
    """Extract the config dataclass from a strategy (possibly wrapped)."""
    inner = strat
    # Unwrap wrappers
    while hasattr(inner, "_inner"):
        inner = inner._inner
    if hasattr(inner, "_c"):
        return inner._c
    if hasattr(inner, "_cfg"):
        return inner._cfg
    return None


def _rebuild_strategy(
    name: str,
    original_strat: IntradayTradingStrategy,
    new_config,
) -> IntradayTradingStrategy:
    """Rebuild a strategy (possibly wrapped) with a new config."""
    # Determine the wrapping chain
    wrappers: list[tuple[type, dict]] = []
    inner = original_strat
    while isinstance(inner, (RvolGatedWrapper, TimeFilterWrapper)):
        if isinstance(inner, RvolGatedWrapper):
            wrappers.append((RvolGatedWrapper, {
                "rvol_threshold": inner._rvol_thresh,
                "lookback": inner._lookback,
                "display_name": inner._display_name,
            }))
            inner = inner._inner
        elif isinstance(inner, TimeFilterWrapper):
            wrappers.append((TimeFilterWrapper, {
                "windows": inner._windows,
                "display_name": inner._display_name,
            }))
            inner = inner._inner

    # Rebuild the core strategy
    from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategy
    if isinstance(inner, PullbackStrategy):
        core = PullbackStrategy(config=new_config)
    elif isinstance(inner, ORReversalStrategy):
        core = ORReversalStrategy(config=new_config)
    elif isinstance(inner, ReversalStrategy):
        core = ReversalStrategy(config=new_config)
    else:
        # Can't rebuild, return original
        return original_strat

    # Re-wrap in reverse order
    result = core
    for wrapper_cls, kwargs in reversed(wrappers):
        result = wrapper_cls(result, **kwargs)

    return result


def phase3_sweep(
    top3: list[tuple[str, Any, BacktestResult, float]],
    data: list[IntradayPriceData],
    trading_days: int,
) -> list[tuple[str, Any, BacktestResult, float, dict]]:
    """Greedy parameter sweep on top 3 configs from Phase 2."""
    print(f"\n\n{SEP}")
    print("  PHASE 3: GREEDY PARAMETER SWEEP — Top 3 Configs")
    print(SEP)

    optimized: list[tuple[str, Any, BacktestResult, float, dict]] = []

    for rank, (name, strat, baseline_result, baseline_score) in enumerate(top3, 1):
        print(f"\n  --- #{rank}: {name} ---")
        print(f"  Baseline: P/L=${_s2(baseline_result.total_pnl)}  "
              f"WR={_s2(baseline_result.win_rate)}%  "
              f"PF={baseline_result.profit_factor}  "
              f"Trades={baseline_result.total_trades}  "
              f"Score={baseline_score:.1f}")

        config = _extract_inner_config(strat)
        if config is None:
            print(f"  SKIP: cannot extract config for {name}")
            optimized.append((name, strat, baseline_result, baseline_score, {}))
            continue

        # Choose parameter grid
        if _is_or_reversal_based(name):
            param_grid = ORR_PARAM_GRID
        elif _is_pullback_based(name):
            param_grid = PB_PARAM_GRID
        elif _is_reversal_based(name):
            param_grid = REV_PARAM_GRID
        else:
            # Try pullback grid by default (most strategies use similar params)
            param_grid = PB_PARAM_GRID

        best_overrides: dict = {}

        for param, values in param_grid.items():
            if not hasattr(config, param):
                continue

            current_val = getattr(config, param)
            best_for_param = (current_val, baseline_score)
            print(f"\n  Testing {param} (default={current_val}):")

            for val in values:
                if val == current_val:
                    continue
                try:
                    trial_config = dataclasses.replace(config, **{param: val})
                    trial_strat = _rebuild_strategy(name, strat, trial_config)
                except (TypeError, ValueError) as exc:
                    continue

                result = run_backtest(trial_strat, data)
                s = wr_weighted_score(result, trading_days=trading_days)
                pnl = float(result.total_pnl)
                improved = " *" if s > best_for_param[1] else ""
                sign = "+" if pnl >= 0 else ""
                print(
                    f"    {param}={str(val):<12s} P/L:{sign}${pnl:>10,.2f} "
                    f"WR:{_s2(result.win_rate):>5}% PF:{str(result.profit_factor):>5} "
                    f"Trades:{result.total_trades:>4d} Score:{s:>7.1f}{improved}"
                )
                if s > best_for_param[1]:
                    best_for_param = (val, s)

            if best_for_param[0] != current_val:
                best_overrides[param] = best_for_param[0]
                print(f"    >>> Best: {param}={best_for_param[0]}")

        # Greedy sequential validation
        if best_overrides:
            print(f"\n  {THIN}")
            print(f"  GREEDY SEQUENTIAL VALIDATION for {name}:")
            print(f"  {THIN}")
            for p, v in sorted(best_overrides.items()):
                print(f"    {p}: {getattr(config, p)} -> {v}")

            greedy_config = config
            greedy_score = baseline_score
            accepted_overrides: dict = {}

            for p, v in best_overrides.items():
                try:
                    trial = dataclasses.replace(greedy_config, **{p: v})
                    trial_strat = _rebuild_strategy(name, strat, trial)
                    result = run_backtest(trial_strat, data)
                    s = wr_weighted_score(result, trading_days=trading_days)
                    if s > greedy_score:
                        greedy_config = trial
                        greedy_score = s
                        accepted_overrides[p] = v
                        print(f"    Applied {p}={v} -> Score={s:.1f} (accepted)")
                    else:
                        print(f"    Tried   {p}={v} -> Score={s:.1f} (rejected, conflicts)")
                except (TypeError, ValueError):
                    pass

            final_strat = _rebuild_strategy(name, strat, greedy_config)
            final_result = run_backtest(final_strat, data)
            final_score = wr_weighted_score(final_result, trading_days=trading_days)
            delta_pnl = final_result.total_pnl - baseline_result.total_pnl
            sign = "+" if delta_pnl >= 0 else ""

            print(f"\n  Optimized: P/L=${_s2(final_result.total_pnl)}  "
                  f"WR={_s2(final_result.win_rate)}%  "
                  f"PF={final_result.profit_factor}  "
                  f"Trades={final_result.total_trades}  "
                  f"Score={final_score:.1f}")
            print(f"  Baseline:  P/L=${_s2(baseline_result.total_pnl)}  "
                  f"WR={_s2(baseline_result.win_rate)}%  "
                  f"PF={baseline_result.profit_factor}  "
                  f"Trades={baseline_result.total_trades}  "
                  f"Score={baseline_score:.1f}")
            print(f"  Delta:     {sign}${_s2(delta_pnl)}")

            optimized.append((
                f"{name} (optimized)",
                final_strat,
                final_result,
                final_score,
                accepted_overrides,
            ))
        else:
            print(f"\n  No improvements found — keeping baseline.")
            optimized.append((name, strat, baseline_result, baseline_score, {}))

    optimized.sort(key=lambda x: x[3], reverse=True)
    return optimized


# ======================================================================
# Phase 4: Walk-Forward Validation
# ======================================================================


def phase4_walkforward(
    top3: list[tuple[str, Any, BacktestResult, float, dict]],
    data: list[IntradayPriceData],
) -> None:
    """5-window walk-forward validation on the top 3 optimized configs."""
    print(f"\n\n{SEP}")
    print("  PHASE 4: WALK-FORWARD VALIDATION (5 rolling windows)")
    print(SEP)

    total_bars = len(data)
    n_windows = 5
    # Each window: 70% IS, 30% OOS
    # Windows overlap by shifting ~15% of total data each step
    window_size = int(total_bars * 0.5)  # each window covers 50% of data
    step_size = int(total_bars * 0.125)   # shift by 12.5% each step
    is_ratio = 0.7

    for rank, (name, strat, full_result, full_score, overrides) in enumerate(top3, 1):
        print(f"\n  --- #{rank}: {name} ---")
        print(f"  Full-sample: P/L=${_s2(full_result.total_pnl)}  "
              f"WR={_s2(full_result.win_rate)}%  "
              f"PF={full_result.profit_factor}  "
              f"Trades={full_result.total_trades}  "
              f"Score={full_score:.1f}")

        config = _extract_inner_config(strat)
        if config is None:
            print(f"  SKIP: cannot extract config")
            continue

        is_scores: list[float] = []
        oos_scores: list[float] = []
        is_wrs: list[float] = []
        oos_wrs: list[float] = []
        is_pnls: list[float] = []
        oos_pnls: list[float] = []

        print(f"\n  {'Window':<10s} {'IS P/L':>12s}  {'IS WR':>7s}  {'IS Score':>9s}  "
              f"{'OOS P/L':>12s}  {'OOS WR':>7s}  {'OOS Score':>10s}  {'Degrad':>7s}")
        print(f"  {THIN}")

        for w in range(n_windows):
            start = w * step_size
            end = min(start + window_size, total_bars)
            if end - start < 200:  # need minimum bars
                continue

            split = start + int((end - start) * is_ratio)
            is_data = data[start:split]
            oos_data = data[split:end]

            if len(is_data) < 100 or len(oos_data) < 50:
                continue

            # Rebuild fresh strategy for each window (reset state)
            is_strat = _rebuild_strategy(name, strat, config)
            oos_strat = _rebuild_strategy(name, strat, config)

            is_result = run_backtest(is_strat, is_data)
            oos_result = run_backtest(oos_strat, oos_data)

            is_td = get_trading_days(is_data)
            oos_td = get_trading_days(oos_data)

            is_s = wr_weighted_score(is_result, is_td)
            oos_s = wr_weighted_score(oos_result, oos_td)

            degrad = oos_s / is_s if is_s != 0 else 0.0

            is_scores.append(is_s)
            oos_scores.append(oos_s)
            is_wrs.append(float(is_result.win_rate))
            oos_wrs.append(float(oos_result.win_rate))
            is_pnls.append(float(is_result.total_pnl))
            oos_pnls.append(float(oos_result.total_pnl))

            is_sign = "+" if is_result.total_pnl >= 0 else ""
            oos_sign = "+" if oos_result.total_pnl >= 0 else ""
            print(
                f"  W{w+1:<8d} {is_sign}${float(is_result.total_pnl):>10,.2f}  "
                f"{_s2(is_result.win_rate):>5}%  {is_s:>9.1f}  "
                f"{oos_sign}${float(oos_result.total_pnl):>10,.2f}  "
                f"{_s2(oos_result.win_rate):>5}%  {oos_s:>10.1f}  "
                f"{degrad:>6.2f}x"
            )

        if is_scores and oos_scores:
            avg_is = sum(is_scores) / len(is_scores)
            avg_oos = sum(oos_scores) / len(oos_scores)
            avg_degrad = avg_oos / avg_is if avg_is != 0 else 0.0
            avg_is_wr = sum(is_wrs) / len(is_wrs) if is_wrs else 0
            avg_oos_wr = sum(oos_wrs) / len(oos_wrs) if oos_wrs else 0
            avg_is_pnl = sum(is_pnls) / len(is_pnls) if is_pnls else 0
            avg_oos_pnl = sum(oos_pnls) / len(oos_pnls) if oos_pnls else 0

            print(f"\n  AVERAGES:")
            print(f"    IS  — Score: {avg_is:.1f}  WR: {avg_is_wr:.1f}%  "
                  f"P/L: ${avg_is_pnl:,.2f}")
            print(f"    OOS — Score: {avg_oos:.1f}  WR: {avg_oos_wr:.1f}%  "
                  f"P/L: ${avg_oos_pnl:,.2f}")
            print(f"    Degradation Ratio: {avg_degrad:.2f}x", end="")

            if avg_degrad < 0.3:
                print("  *** SEVERELY OVERFITTED ***")
            elif avg_degrad < 0.5:
                print("  ** LIKELY OVERFITTED **")
            elif avg_degrad < 0.7:
                print("  * MODERATE OVERFIT *")
            elif avg_degrad < 0.9:
                print("  (mild overfit)")
            else:
                print("  (robust)")

            # Check OOS profitability across windows
            oos_profitable = sum(1 for p in oos_pnls if p > 0)
            print(f"    OOS Profitable Windows: {oos_profitable}/{len(oos_pnls)}")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    total_start = time.time()

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/spy_5m_bars.csv"
    print(f"\n  Loading data from {csv_path}...", end="", flush=True)
    data = IntradayCsvLoader.load_from_file(csv_path)
    trading_days = get_trading_days(data)
    print(f" {len(data):,} bars, {trading_days} sessions")

    # ── Phase 1: Baseline ──
    baseline_results = phase1_baseline(data, trading_days)

    # ── Phase 2: Experimental Variants ──
    variant_results = phase2_variants(data, trading_days)

    # ── Combine Phase 1 + Phase 2 and pick top 3 ──
    all_results = [
        (name, strat, result, score)
        for name, strat, result, score in baseline_results + variant_results
    ]
    all_results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n\n{SEP}")
    print("  COMBINED LEADERBOARD (Phase 1 + Phase 2)")
    print(SEP)
    print(f"\n  {'Rank':<6s} {'Strategy':<35s} {'P/L':>14s}  {'WR':>7s}  "
          f"{'PF':>5s}  {'MaxDD':>7s}  {'Trades':>6s}  {'Score':>7s}")
    print(f"  {THIN}")

    for i, (name, strat, result, score) in enumerate(all_results, 1):
        pnl = float(result.total_pnl)
        sign = "+" if pnl >= 0 else ""
        marker = " <<<" if i <= 3 else ""
        print(
            f"  {i:<6d} {name:<35s} {sign}${pnl:>10,.2f}  "
            f"{_s2(result.win_rate):>5}%  "
            f"{str(result.profit_factor):>5}  "
            f"{_s2(result.max_drawdown):>5}%  "
            f"{result.total_trades:>6d}  "
            f"{score:>7.1f}{marker}"
        )

    top3_for_sweep = all_results[:3]

    # ── Phase 3: Greedy Parameter Sweep ──
    optimized = phase3_sweep(top3_for_sweep, data, trading_days)

    # ── Phase 4: Walk-Forward Validation ──
    phase4_walkforward(optimized, data)

    # ── Final Summary ──
    elapsed = time.time() - total_start
    print(f"\n\n{SEP}")
    print(f"  FINAL RESULTS — SPY 5-Min VWAP Strategy Optimizer")
    print(SEP)
    print(f"\n  {'Rank':<6s} {'Strategy':<40s} {'P/L':>14s}  {'WR':>7s}  "
          f"{'PF':>5s}  {'Trades':>6s}  {'Score':>7s}  {'Overrides'}")
    print(f"  {THIN}")

    for i, (name, strat, result, score, overrides) in enumerate(optimized, 1):
        pnl = float(result.total_pnl)
        sign = "+" if pnl >= 0 else ""
        ovr_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else "none"
        print(
            f"  {i:<6d} {name:<40s} {sign}${pnl:>10,.2f}  "
            f"{_s2(result.win_rate):>5}%  "
            f"{str(result.profit_factor):>5}  "
            f"{result.total_trades:>6d}  "
            f"{score:>7.1f}  {ovr_str}"
        )

    winner = optimized[0]
    print(f"\n  WINNER: {winner[0]}")
    print(f"    P/L:    ${_s2(winner[2].total_pnl)}")
    print(f"    WR:     {_s2(winner[2].win_rate)}%")
    print(f"    PF:     {winner[2].profit_factor}")
    print(f"    MaxDD:  {_s2(winner[2].max_drawdown)}%")
    print(f"    Trades: {winner[2].total_trades}")
    print(f"    Score:  {winner[3]:.1f}")
    if winner[4]:
        print(f"    Optimized params: {winner[4]}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
