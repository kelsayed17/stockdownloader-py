"""Intraday-specific indicator calculations for the VWAP strategy.

Supplements :mod:`technical_indicators` with indicators that only make sense
for intraday strategies: extended VWAP bands, time-of-day RVOL, cumulative
volume delta, linear regression slope, VWAP slope / acceleration,
higher-timeframe resampling, candle-pattern detection, and S/R proximity.

All functions follow the same convention as :mod:`technical_indicators`:
accept ``Sequence[PriceData]`` with an ``end_index`` parameter, return
``Decimal`` values quantized to ``SCALE`` decimal places.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.moving_average_calculator import ema as _ema
from stockdownloader.util.technical_indicators import atr as _atr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.price_data import PriceData

SCALE = 10
_ZERO = Decimal("0")
_HALF = Decimal("0.5")
_THREE = Decimal("3")


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP)


def _find_session_start(data: Sequence[PriceData], end_index: int) -> int:
    """Walk backward to find the first bar of the current trading session."""
    current_day = data[end_index].date[:10]
    start = end_index
    while start > 0 and data[start - 1].date[:10] == current_day:
        start -= 1
    return start


# =========================================================================
# EXTENDED SESSION VWAP BANDS
# =========================================================================


@dataclass(frozen=True)
class ExtendedSessionVWAP:
    """Session VWAP with 0.5 / 1 / 1.5 / 2 / 3 sigma bands."""

    vwap: Decimal
    std_dev: Decimal
    upper_05: Decimal
    lower_05: Decimal
    upper_1: Decimal
    lower_1: Decimal
    upper_15: Decimal
    lower_15: Decimal
    upper_2: Decimal
    lower_2: Decimal
    upper_3: Decimal
    lower_3: Decimal


_EMPTY_VWAP = ExtendedSessionVWAP(
    _ZERO, _ZERO, _ZERO, _ZERO, _ZERO, _ZERO,
    _ZERO, _ZERO, _ZERO, _ZERO, _ZERO, _ZERO,
)


def extended_session_vwap_bands(
    data: Sequence[PriceData], end_index: int
) -> ExtendedSessionVWAP:
    """Session VWAP with 0.5 / 1 / 1.5 / 2 / 3 sigma bands."""
    if end_index < 0:
        return _EMPTY_VWAP

    session_start = _find_session_start(data, end_index)

    sum_tpv = _ZERO
    sum_vol = _ZERO
    tps: list[Decimal] = []
    vols: list[Decimal] = []

    for i in range(session_start, end_index + 1):
        bar = data[i]
        tp = _quantize((bar.high + bar.low + bar.close) / _THREE)
        vol = Decimal(str(bar.volume))
        tps.append(tp)
        vols.append(vol)
        sum_tpv += tp * vol
        sum_vol += vol

    if sum_vol == _ZERO:
        return _EMPTY_VWAP

    vwap_val = _quantize(sum_tpv / sum_vol)

    # Weighted standard deviation
    sum_var = _ZERO
    for tp, vol in zip(tps, vols):
        diff = tp - vwap_val
        sum_var += diff * diff * vol

    variance = float(sum_var / sum_vol)
    std_float = math.sqrt(max(0.0, variance))
    std_val = _quantize(Decimal(str(std_float)))

    s05 = _quantize(std_val * Decimal("0.5"))
    s15 = _quantize(std_val * Decimal("1.5"))
    s2 = _quantize(std_val * Decimal("2"))
    s3 = _quantize(std_val * _THREE)

    return ExtendedSessionVWAP(
        vwap=vwap_val,
        std_dev=std_val,
        upper_05=_quantize(vwap_val + s05),
        lower_05=_quantize(vwap_val - s05),
        upper_1=_quantize(vwap_val + std_val),
        lower_1=_quantize(vwap_val - std_val),
        upper_15=_quantize(vwap_val + s15),
        lower_15=_quantize(vwap_val - s15),
        upper_2=_quantize(vwap_val + s2),
        lower_2=_quantize(vwap_val - s2),
        upper_3=_quantize(vwap_val + s3),
        lower_3=_quantize(vwap_val - s3),
    )


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
    if current_vol <= _ZERO:
        return Decimal("1")

    total = _ZERO
    count = 0
    for d in range(1, lookback_days + 1):
        offset = d * bars_per_day
        idx = end_index - offset
        if idx >= 0:
            total += Decimal(str(data[idx].volume))
            count += 1

    if count == 0 or total == _ZERO:
        return Decimal("1")

    avg = total / Decimal(str(count))
    if avg <= _ZERO:
        return Decimal("1")

    return _quantize(current_vol / avg)


# =========================================================================
# CUMULATIVE VOLUME DELTA PROXY
# =========================================================================


def cvd_session(
    data: Sequence[PriceData],
    end_index: int,
) -> Decimal:
    """Cumulative volume delta proxy for the current session.

    For each bar: ``vDelta = (close - low) / (high - low)`` (clamped to 0.5
    when range is zero).  Session CVD = ``Sum((vDelta - 0.5) * volume)``.

    This is the same approximation as the PineScript ``cumVD``.
    """
    if end_index < 0:
        return _ZERO

    session_start = _find_session_start(data, end_index)
    cum = _ZERO

    for i in range(session_start, end_index + 1):
        bar = data[i]
        bar_range = bar.high - bar.low
        if bar_range > _ZERO:
            v_delta = (bar.close - bar.low) / bar_range
        else:
            v_delta = _HALF
        cum += (v_delta - _HALF) * Decimal(str(bar.volume))

    return _quantize(cum)


def cvd_normalized(
    data: Sequence[PriceData],
    end_index: int,
    vol_sma_period: int = 20,
) -> Decimal:
    """CVD normalized by ``volSMA * 20`` (PineScript ``cumVDnorm``)."""
    cvd = cvd_session(data, end_index)

    if end_index < vol_sma_period:
        return _ZERO

    total = _ZERO
    for i in range(end_index - vol_sma_period + 1, end_index + 1):
        total += Decimal(str(data[i].volume))
    vol_sma = total / Decimal(str(vol_sma_period))

    denom = vol_sma * Decimal("20")
    if denom <= _ZERO:
        return _ZERO

    return _quantize(cvd / denom)


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
        return _ZERO

    n = Decimal(str(period))
    sum_x = _ZERO
    sum_y = _ZERO
    sum_xy = _ZERO
    sum_x2 = _ZERO

    for i in range(period):
        x = Decimal(str(i))
        y = data[end_index - period + 1 + i].close
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denom = n * sum_x2 - sum_x * sum_x
    if denom == _ZERO:
        return _ZERO

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return _quantize(slope)


def lrs_normalized(
    data: Sequence[PriceData],
    end_index: int,
    period: int = 15,
    atr_period: int = 14,
) -> Decimal:
    """LRS normalized by ATR (PineScript ``lrSlopeATR``)."""
    slope = linear_regression_slope(data, end_index, period)
    atr_val = _atr(data, end_index, atr_period)
    if atr_val <= _ZERO:
        return _ZERO
    return _quantize(slope / atr_val)


# =========================================================================
# VWAP SLOPE & ACCELERATION
# =========================================================================


def vwap_slope(
    data: Sequence[PriceData],
    end_index: int,
    lookback: int = 5,
) -> Decimal:
    """Change in session VWAP over *lookback* bars.

    Mirrors PineScript ``ta.change(vwapLine, i_slopePer)``.
    """
    if end_index < lookback:
        return _ZERO

    cur = _session_vwap_at(data, end_index)
    prev = _session_vwap_at(data, end_index - lookback)
    return _quantize(cur - prev)


def vwap_acceleration(
    data: Sequence[PriceData],
    end_index: int,
    lookback: int = 5,
    atr_period: int = 14,
) -> Decimal:
    """VWAP acceleration normalized by ATR.

    ``(currentSlope - priorSlope) / ATR`` — mirrors PineScript ``vwapAccel``.
    """
    if end_index < lookback * 2:
        return _ZERO

    cur_slope = vwap_slope(data, end_index, lookback)
    prev_slope = vwap_slope(data, end_index - lookback, lookback)
    atr_val = _atr(data, end_index, atr_period)
    if atr_val <= _ZERO:
        return _ZERO
    return _quantize((cur_slope - prev_slope) / atr_val)


def _session_vwap_at(data: Sequence[PriceData], end_index: int) -> Decimal:
    """Quick session VWAP at a single index (no bands)."""
    if end_index < 0:
        return _ZERO
    session_start = _find_session_start(data, end_index)
    sum_tpv = _ZERO
    sum_vol = _ZERO
    for i in range(session_start, end_index + 1):
        bar = data[i]
        tp = (bar.high + bar.low + bar.close) / _THREE
        vol = Decimal(str(bar.volume))
        sum_tpv += tp * vol
        sum_vol += vol
    if sum_vol == _ZERO:
        return _ZERO
    return _quantize(sum_tpv / sum_vol)


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
    from stockdownloader.model.price_data import PriceData as PD

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
        return _ZERO

    return _atr(daily_bars, idx, period)


# =========================================================================
# HIGHER-TIMEFRAME RESAMPLING (5m → 15m)
# =========================================================================


def resample_to_htf(
    data: Sequence[PriceData],
    end_index: int,
    factor: int = 3,
) -> list[PriceData]:
    """Resample 5-minute bars to *factor*-bar candles (default 15-minute).

    Only complete groups within the same session are returned.
    """
    from stockdownloader.model.price_data import PriceData as PD

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
# CANDLE PATTERN DETECTION
# =========================================================================


def is_hammer(bar: PriceData, atr_val: Decimal) -> bool:
    """Bullish hammer: lower wick >= 60% of range, close > open, body > 0.05 ATR."""
    bar_range = bar.high - bar.low
    if bar_range <= _ZERO:
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
    if bar_range <= _ZERO:
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
        and prior_body > _ZERO
        and bar_range > _ZERO
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
        and prior_body > _ZERO
        and bar_range > _ZERO
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
    if level <= _ZERO or price <= _ZERO:
        return False
    return abs(price - level) / price * Decimal("100") <= proximity_pct


def compute_sr_score(
    close: Decimal,
    *,
    pd_high: Decimal = _ZERO,
    pd_low: Decimal = _ZERO,
    pd_close: Decimal = _ZERO,
    or_high: Decimal = _ZERO,
    or_low: Decimal = _ZERO,
    pw_high: Decimal = _ZERO,
    pw_low: Decimal = _ZERO,
    prev_vwap: Decimal = _ZERO,
    proximity_pct: Decimal = Decimal("0.35"),
    sr_pdhlc: bool = True,
    sr_round: bool = True,
    sr_or: bool = True,
    sr_week_hl: bool = False,
    sr_prev_vwap: bool = False,
) -> tuple[bool, int]:
    """Check S/R proximity and return (any_near, count_for_scoring).

    *count_for_scoring* excludes PDH and PDC (display-only in PineScript),
    matching the v11.2 ``srCount_score`` logic.
    """
    score_count = 0

    # Scoring levels (PDL, OR H/L, PW H/L, prev VWAP)
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

    total = _ZERO
    for i in range(end_index - period + 1, end_index + 1):
        total += Decimal(str(data[i].volume))
    avg = total / Decimal(str(period))
    if avg <= _ZERO:
        return Decimal("1")
    return _quantize(Decimal(str(data[end_index].volume)) / avg)
