"""Intraday-specific indicators (non-VWAP).

Time-of-day RVOL, linear regression slope, VWAP slope/acceleration,
daily aggregation, HTF resampling, candle pattern detection, S/R
proximity scoring.

VWAP-related data structures and functions live in volume.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.core.math import ZERO, quantize
from stockdownloader.indicators.core import (
    _compute_session_vwap_core,
    _find_session_start,
    ema as _ema,
)
from stockdownloader.indicators.volatility import atr as _atr

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.core.models.price import PriceData

_HALF = Decimal("0.5")
_THREE = Decimal("3")

__all__ = [
    "tod_rvol",
    "linear_regression_slope",
    "lrs_normalized",
    "vwap_slope",
    "vwap_acceleration",
    "_session_vwap_at",
    "aggregate_to_daily",
    "daily_atr_prior",
    "resample_to_htf",
    "htf_ema_trend",
    "CandleStrength",
    "candle_strength",
    "is_hammer",
    "is_inv_hammer",
    "is_bull_engulfing",
    "is_bear_engulfing",
    "near_level",
    "compute_sr_score",
    "_round_to_5",
    "rel_vol",
]


# =========================================================================
# TIME-OF-DAY RELATIVE VOLUME
# =========================================================================

def tod_rvol(
    data: Sequence[PriceData],
    end_index: int,
    lookback_days: int = 10,
    bars_per_day: int = 78,
) -> Decimal:
    """Time-of-day relative volume.

    Compares the current bar's volume to the average volume at the same
    bar-of-day position across the last *lookback_days* sessions.
    Mirrors the PineScript ``todRVOL`` calculation.
    """
    if end_index < 0:
        return Decimal("1")

    current_vol = Decimal(str(data[end_index].volume))
    if current_vol <= ZERO:
        return Decimal("1")

    total = ZERO
    count = 0
    for d in range(1, lookback_days + 1):
        offset = d * bars_per_day
        idx = end_index - offset
        if idx >= 0:
            total += Decimal(str(data[idx].volume))
            count += 1

    if count == 0 or total == ZERO:
        return Decimal("1")

    avg = total / Decimal(str(count))
    if avg <= ZERO:
        return Decimal("1")

    return quantize(current_vol / avg)


# =========================================================================
# LINEAR REGRESSION SLOPE
# =========================================================================

def linear_regression_slope(
    data: Sequence[PriceData],
    end_index: int,
    period: int = 15,
) -> Decimal:
    """Linear regression slope of close prices over *period* bars.

    Returns the raw slope (price change per bar).  Mirrors the PineScript
    ``ta.linreg(close, 15, 0) - ta.linreg(close, 15, 1)`` calculation.
    """
    if end_index < period - 1:
        return ZERO

    n = Decimal(str(period))
    sum_x = ZERO
    sum_y = ZERO
    sum_xy = ZERO
    sum_x2 = ZERO

    for i in range(period):
        x = Decimal(str(i))
        y = data[end_index - period + 1 + i].close
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denom = n * sum_x2 - sum_x * sum_x
    if denom == ZERO:
        return ZERO

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return quantize(slope)

def lrs_normalized(
    data: Sequence[PriceData],
    end_index: int,
    period: int = 15,
    atr_period: int = 14,
    *,
    _atr_fn: Callable[[Sequence[PriceData], int, int], Decimal] | None = None,
) -> Decimal:
    """LRS normalized by ATR (PineScript ``lrSlopeATR``).

    Parameters
    ----------
    _atr_fn:
        Optional ATR callable for dependency injection.  When provided,
        ``_atr_fn(data, end_index, atr_period)`` is used instead of the
        raw :func:`~indicators.volatility.atr`.  The :class:`IndicatorHub`
        uses this to route ATR lookups through its cache.
    """
    slope = linear_regression_slope(data, end_index, period)
    atr_val = (_atr_fn or _atr)(data, end_index, atr_period)
    if atr_val <= ZERO:
        return ZERO
    return quantize(slope / atr_val)


# =========================================================================
# VWAP SLOPE & ACCELERATION
# =========================================================================

def vwap_slope(
    data: Sequence[PriceData],
    end_index: int,
    lookback: int = 5,
    *,
    _vwap_fn: Callable[[Sequence[PriceData], int], Decimal] | None = None,
) -> Decimal:
    """Change in session VWAP over *lookback* bars.

    Mirrors PineScript ``ta.change(vwapLine, i_slopePer)``.

    Parameters
    ----------
    _vwap_fn:
        Optional VWAP callable for dependency injection.  When provided,
        ``_vwap_fn(data, index)`` is used instead of :func:`_session_vwap_at`.
        The :class:`IndicatorHub` uses this to route VWAP lookups through
        its cache.
    """
    if end_index < lookback:
        return ZERO

    vfn = _vwap_fn or _session_vwap_at
    cur = vfn(data, end_index)
    prev = vfn(data, end_index - lookback)
    return quantize(cur - prev)

def vwap_acceleration(
    data: Sequence[PriceData],
    end_index: int,
    lookback: int = 5,
    atr_period: int = 14,
    *,
    _atr_fn: Callable[[Sequence[PriceData], int, int], Decimal] | None = None,
    _slope_fn: Callable[[Sequence[PriceData], int, int], Decimal] | None = None,
) -> Decimal:
    """VWAP acceleration normalized by ATR.

    ``(currentSlope - priorSlope) / ATR`` -- mirrors PineScript ``vwapAccel``.

    Parameters
    ----------
    _atr_fn:
        Optional ATR callable for dependency injection.
    _slope_fn:
        Optional VWAP-slope callable; signature
        ``(data, end_index, lookback) -> Decimal``.  The
        :class:`IndicatorHub` uses this to route slope lookups through
        its cache.
    """
    if end_index < lookback * 2:
        return ZERO

    sfn = _slope_fn or (lambda d, i, lb: vwap_slope(d, i, lb))
    cur_slope = sfn(data, end_index, lookback)
    prev_slope = sfn(data, end_index - lookback, lookback)
    atr_val = (_atr_fn or _atr)(data, end_index, atr_period)
    if atr_val <= ZERO:
        return ZERO
    return quantize((cur_slope - prev_slope) / atr_val)

def _session_vwap_at(data: Sequence[PriceData], end_index: int) -> Decimal:
    """Quick session VWAP at a single index (no bands).

    Delegates to :func:`_compute_session_vwap_core` for consistent
    quantization of the typical price (TP).
    """
    if end_index < 0:
        return ZERO
    vwap_val, _ = _compute_session_vwap_core(data, end_index)
    return vwap_val


# =========================================================================
# DAILY DATA AGGREGATION & DAILY ATR
# =========================================================================

def aggregate_to_daily(
    data: Sequence[PriceData],
) -> list[PriceData]:
    """Aggregate intraday bars into daily OHLCV bars.

    Groups by the first 10 characters of ``date`` (trading date).
    Returns a list of :class:`PriceData` with one entry per session,
    sorted chronologically.
    """
    from stockdownloader.core.models.price import PriceData as PD

    if not data:
        return []

    daily: list[PD] = []
    cur_date = data[0].date[:10]
    o = data[0].open
    h = data[0].high
    lo = data[0].low
    c = data[0].close
    vol = data[0].volume

    for i in range(1, len(data)):
        bar = data[i]
        d = bar.date[:10]
        if d != cur_date:
            daily.append(PD(date=cur_date, open=o, high=h, low=lo,
                            close=c, adj_close=c, volume=vol))
            cur_date = d
            o = bar.open
            h = bar.high
            lo = bar.low
            c = bar.close
            vol = bar.volume
        else:
            h = max(h, bar.high)
            lo = min(lo, bar.low)
            c = bar.close
            vol += bar.volume

    daily.append(PD(date=cur_date, open=o, high=h, low=lo,
                     close=c, adj_close=c, volume=vol))
    return daily

def daily_atr_prior(
    daily_bars: Sequence[PriceData],
    trading_date: str,
    period: int = 14,
) -> Decimal:
    """ATR(*period*) as of the trading day immediately before *trading_date*.

    *daily_bars* must be pre-aggregated (see :func:`aggregate_to_daily`).
    """
    # Find index of the day before trading_date
    idx = -1
    for i, bar in enumerate(daily_bars):
        if bar.date[:10] >= trading_date:
            break
        idx = i

    if idx < 0:
        return ZERO

    return _atr(daily_bars, idx, period)


# =========================================================================
# HIGHER-TIMEFRAME RESAMPLING (5m -> 15m)
# =========================================================================

def resample_to_htf(
    data: Sequence[PriceData],
    end_index: int,
    factor: int = 3,
) -> list[PriceData]:
    """Resample 5-minute bars to *factor*-bar candles (default 15-minute).

    Only complete groups within the same session are returned.
    """
    from stockdownloader.core.models.price import PriceData as PD

    if end_index < 0:
        return []

    session_start = _find_session_start(data, end_index)
    count = end_index - session_start + 1
    complete = count // factor

    result: list[PD] = []
    for g in range(complete):
        base = session_start + g * factor
        o = data[base].open
        h = data[base].high
        lo = data[base].low
        c = data[base + factor - 1].close
        vol = 0
        for k in range(factor):
            bar = data[base + k]
            h = max(h, bar.high)
            lo = min(lo, bar.low)
            vol += bar.volume
        result.append(PD(date=data[base].date, open=o, high=h, low=lo,
                          close=c, adj_close=c, volume=vol))
    return result

def htf_ema_trend(
    data: Sequence[PriceData],
    end_index: int,
    fast_period: int = 9,
    slow_period: int = 21,
    htf_factor: int = 3,
) -> int:
    """15-minute EMA trend direction: +1 (bull), -1 (bear), 0 (neutral).

    Resamples 5m bars to 15m, then compares EMA(*fast*) vs EMA(*slow*).
    """
    htf_bars = resample_to_htf(data, end_index, htf_factor)
    if len(htf_bars) < slow_period + 1:
        return 0

    htf_idx = len(htf_bars) - 1
    fast_val = _ema(htf_bars, htf_idx, fast_period)
    slow_val = _ema(htf_bars, htf_idx, slow_period)

    if fast_val > slow_val:
        return 1
    if fast_val < slow_val:
        return -1
    return 0


# =========================================================================
# CANDLE ANALYSIS HELPERS
# =========================================================================

@dataclass(frozen=True, slots=True)
class CandleStrength:
    """Pre-computed candle body and wick metrics normalized by ATR.

    Avoids repeating the ``body / atr``, wick-ratio, and bull/bear checks
    that appear in pullback, reversal, and breakout modes.
    """

    body: Decimal
    bar_range: Decimal
    body_atr: Decimal
    is_bull: bool
    is_bear: bool
    bull_wick: bool
    bear_wick: bool

    def bull_candle(self, min_body_atr: Decimal) -> bool:
        """Bullish candle with body/ATR at or above *min_body_atr*."""
        return self.is_bull and self.body_atr >= min_body_atr

    def bear_candle(self, min_body_atr: Decimal) -> bool:
        """Bearish candle with body/ATR at or above *min_body_atr*."""
        return self.is_bear and self.body_atr >= min_body_atr

def candle_strength(
    bar: PriceData,
    atr_val: Decimal,
    wick_ratio: Decimal = Decimal("0.6"),
) -> CandleStrength:
    """Compute ATR-normalized candle metrics for *bar*.

    Parameters
    ----------
    wick_ratio:
        Fraction of bar range the close must be in the upper (bull) or
        lower (bear) portion to qualify as a wick signal.  Default 0.6
        matches the threshold used across PB and REV modes.

    Returns a :class:`CandleStrength` with body, range, body/ATR ratio,
    bull/bear flags, and wick signals.
    """
    body = abs(bar.close - bar.open)
    bar_range = bar.high - bar.low
    body_atr = body / atr_val if atr_val > ZERO else ZERO
    is_bull = bar.close > bar.open
    is_bear = bar.close < bar.open
    bw = (
        bar_range > ZERO
        and is_bull
        and bar.close > bar.low + bar_range * wick_ratio
    )
    ew = (
        bar_range > ZERO
        and is_bear
        and bar.close < bar.high - bar_range * wick_ratio
    )
    return CandleStrength(
        body=body,
        bar_range=bar_range,
        body_atr=body_atr,
        is_bull=is_bull,
        is_bear=is_bear,
        bull_wick=bw,
        bear_wick=ew,
    )


# =========================================================================
# CANDLE PATTERN DETECTION
# =========================================================================

def is_hammer(bar: PriceData, atr_val: Decimal) -> bool:
    """Bullish hammer: lower wick >= 60% of range, close > open, body > 0.05 ATR."""
    bar_range = bar.high - bar.low
    if bar_range <= ZERO:
        return False
    lower_wick = min(bar.close, bar.open) - bar.low
    body = abs(bar.close - bar.open)
    return (
        lower_wick >= bar_range * Decimal("0.6")
        and bar.close > bar.open
        and body >= atr_val * Decimal("0.05")
    )

def is_inv_hammer(bar: PriceData, atr_val: Decimal) -> bool:
    """Bearish inverted hammer: upper wick >= 60% of range, close < open."""
    bar_range = bar.high - bar.low
    if bar_range <= ZERO:
        return False
    upper_wick = bar.high - max(bar.close, bar.open)
    body = abs(bar.close - bar.open)
    return (
        upper_wick >= bar_range * Decimal("0.6")
        and bar.close < bar.open
        and body >= atr_val * Decimal("0.05")
    )

def is_bull_engulfing(
    current: PriceData,
    previous: PriceData,
    engulf_min: Decimal = Decimal("0.35"),
) -> bool:
    """Bullish engulfing pattern."""
    if previous.close >= previous.open:
        return False
    if current.close <= current.open:
        return False

    prior_body = abs(previous.close - previous.open)
    cur_body = abs(current.close - current.open)
    bar_range = current.high - current.low

    return (
        current.close > previous.open
        and current.open <= previous.close
        and cur_body > prior_body
        and prior_body > ZERO
        and bar_range > ZERO
        and (current.close - current.open) / bar_range >= engulf_min
    )

def is_bear_engulfing(
    current: PriceData,
    previous: PriceData,
    engulf_min: Decimal = Decimal("0.35"),
) -> bool:
    """Bearish engulfing pattern."""
    if previous.close <= previous.open:
        return False
    if current.close >= current.open:
        return False

    prior_body = abs(previous.close - previous.open)
    cur_body = abs(current.close - current.open)
    bar_range = current.high - current.low

    return (
        current.close < previous.open
        and current.open >= previous.close
        and cur_body > prior_body
        and prior_body > ZERO
        and bar_range > ZERO
        and (current.open - current.close) / bar_range >= engulf_min
    )


# =========================================================================
# S/R PROXIMITY
# =========================================================================

def near_level(
    price: Decimal,
    level: Decimal,
    proximity_pct: Decimal = Decimal("0.35"),
) -> bool:
    """True if *price* is within *proximity_pct* % of *level*."""
    if level <= ZERO or price <= ZERO:
        return False
    return abs(price - level) / price * Decimal("100") <= proximity_pct

def compute_sr_score(
    close: Decimal,
    *,
    pd_high: Decimal = ZERO,
    pd_low: Decimal = ZERO,
    pd_close: Decimal = ZERO,
    or_high: Decimal = ZERO,
    or_low: Decimal = ZERO,
    pw_high: Decimal = ZERO,
    pw_low: Decimal = ZERO,
    prev_vwap: Decimal = ZERO,
    avwap: Decimal = ZERO,
    proximity_pct: Decimal = Decimal("0.35"),
    sr_pdhlc: bool = True,
    sr_round: bool = True,
    sr_or: bool = True,
    sr_week_hl: bool = False,
    sr_prev_vwap: bool = False,
    sr_avwap: bool = False,
) -> tuple[bool, int]:
    """Check S/R proximity and return (any_near, count_for_scoring).

    *count_for_scoring* excludes PDH and PDC (display-only in PineScript),
    matching the v11.2 ``srCount_score`` logic.
    """
    score_count = 0

    # Scoring levels (PDL, OR H/L, PW H/L, prev VWAP, AVWAP)
    if sr_pdhlc and near_level(close, pd_low, proximity_pct):
        score_count += 1
    if sr_or and near_level(close, or_high, proximity_pct):
        score_count += 1
    if sr_or and near_level(close, or_low, proximity_pct):
        score_count += 1
    if sr_week_hl and near_level(close, pw_high, proximity_pct):
        score_count += 1
    if sr_week_hl and near_level(close, pw_low, proximity_pct):
        score_count += 1
    if sr_prev_vwap and near_level(close, prev_vwap, proximity_pct):
        score_count += 1
    if sr_avwap and near_level(close, avwap, proximity_pct):
        score_count += 1

    # Display-only levels (PDH, PDC, round $5)
    any_near = score_count > 0
    if sr_pdhlc and near_level(close, pd_high, proximity_pct):
        any_near = True
    if sr_pdhlc and near_level(close, pd_close, proximity_pct):
        any_near = True
    if sr_round:
        rnd = _round_to_5(close)
        if near_level(close, rnd, proximity_pct):
            any_near = True

    return any_near, score_count

def _round_to_5(price: Decimal) -> Decimal:
    """Round price to nearest $5."""
    return (price / Decimal("5")).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * Decimal("5")


# =========================================================================
# RELATIVE VOLUME (simple SMA-based, not time-of-day)
# =========================================================================

def rel_vol(
    data: Sequence[PriceData],
    end_index: int,
    period: int = 20,
) -> Decimal:
    """Current bar volume / SMA(volume, period).  PineScript ``relVol``."""
    if end_index < period:
        return Decimal("1")

    total = ZERO
    for i in range(end_index - period + 1, end_index + 1):
        total += Decimal(str(data[i].volume))
    avg = total / Decimal(str(period))
    if avg <= ZERO:
        return Decimal("1")
    return quantize(Decimal(str(data[end_index].volume)) / avg)
