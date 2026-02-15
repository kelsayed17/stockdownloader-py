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

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.moving_average_calculator import ema as _ema
from stockdownloader.util.technical_indicators import (
    _compute_session_vwap_core,
    _find_session_start,
    atr as _atr,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.price_data import PriceData

SCALE = 10
_HALF = Decimal("0.5")
_THREE = Decimal("3")

def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP)

# =========================================================================
# EXTENDED SESSION VWAP BANDS
# =========================================================================

@dataclass(frozen=True, slots=True)
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

    def band_pair(self, label: str = "2σ") -> tuple[Decimal, Decimal]:
        """Return ``(upper, lower)`` band pair for the given sigma label.

        Supported labels: ``'0.5σ'``, ``'1σ'``, ``'1.5σ'``, ``'2σ'``, ``'3σ'``.
        Defaults to ``'2σ'`` if the label is unrecognized.
        """
        suffix_map: dict[str, str] = {
            "0.5σ": "_05",
            "1σ": "_1",
            "1.5σ": "_15",
            "2σ": "_2",
            "3σ": "_3",
        }
        sfx = suffix_map.get(label, "_2")
        return (
            getattr(self, f"upper{sfx}"),
            getattr(self, f"lower{sfx}"),
        )

_EMPTY_VWAP = ExtendedSessionVWAP(
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
)

def extended_session_vwap_bands(
    data: Sequence[PriceData], end_index: int
) -> ExtendedSessionVWAP:
    """Session VWAP with 0.5 / 1 / 1.5 / 2 / 3 sigma bands.

    Delegates core VWAP computation to
    :func:`~stockdownloader.util.technical_indicators._compute_session_vwap_core`.
    """
    if end_index < 0:
        return _EMPTY_VWAP

    vwap_val, std_val = _compute_session_vwap_core(data, end_index)

    if vwap_val == ZERO and std_val == ZERO:
        return _EMPTY_VWAP

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
        return ZERO

    session_start = _find_session_start(data, end_index)
    cum = ZERO

    for i in range(session_start, end_index + 1):
        bar = data[i]
        bar_range = bar.high - bar.low
        if bar_range > ZERO:
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
        return ZERO

    total = ZERO
    for i in range(end_index - vol_sma_period + 1, end_index + 1):
        total += Decimal(str(data[i].volume))
    vol_sma = total / Decimal(str(vol_sma_period))

    denom = vol_sma * Decimal("20")
    if denom <= ZERO:
        return ZERO

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
    return _quantize(slope)

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
        raw :func:`~technical_indicators.atr`.  The :class:`IndicatorHub`
        uses this to route ATR lookups through its cache.
    """
    slope = linear_regression_slope(data, end_index, period)
    atr_val = (_atr_fn or _atr)(data, end_index, atr_period)
    if atr_val <= ZERO:
        return ZERO
    return _quantize(slope / atr_val)

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
    return _quantize(cur - prev)

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

    ``(currentSlope - priorSlope) / ATR`` — mirrors PineScript ``vwapAccel``.

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
    return _quantize((cur_slope - prev_slope) / atr_val)

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
        return ZERO

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

    total = ZERO
    for i in range(end_index - period + 1, end_index + 1):
        total += Decimal(str(data[i].volume))
    avg = total / Decimal(str(period))
    if avg <= ZERO:
        return Decimal("1")
    return _quantize(Decimal(str(data[end_index].volume)) / avg)
