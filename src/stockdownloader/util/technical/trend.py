"""Trend indicators: ADX/DMI, Parabolic SAR, Ichimoku, VWAP, Fibonacci, Support/Resistance."""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import HUNDRED, TWO, ZERO
from stockdownloader.util.technical._helpers import (
    _deduplicate_levels,
    _period_midpoint,
    _quantize,
    true_range,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

__all__ = [
    "ADXResult",
    "IchimokuCloud",
    "FibonacciLevels",
    "SupportResistance",
    "SessionVWAP",
    "adx",
    "parabolic_sar",
    "is_sar_bullish",
    "ichimoku",
    "fibonacci_retracement",
    "support_resistance",
    "vwap",
    "session_vwap",
    "session_vwap_bands",
    "_find_session_start",
    "_compute_session_vwap_core",
]


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass(frozen=True, slots=True)
class ADXResult:
    """Average Directional Index result: ADX value, +DI, and -DI."""
    adx: Decimal
    plus_di: Decimal
    minus_di: Decimal

@dataclass(frozen=True, slots=True)
class IchimokuCloud:
    """Ichimoku Cloud components."""
    tenkan_sen: Decimal       # Conversion Line (9-period)
    kijun_sen: Decimal        # Base Line (26-period)
    senkou_span_a: Decimal    # Leading Span A
    senkou_span_b: Decimal    # Leading Span B (52-period)
    chikou_span: Decimal      # Lagging Span
    price_above_cloud: bool

@dataclass(frozen=True, slots=True)
class FibonacciLevels:
    """Fibonacci retracement levels derived from a swing high/low."""
    high: Decimal
    low: Decimal
    level_236: Decimal
    level_382: Decimal
    level_500: Decimal
    level_618: Decimal
    level_786: Decimal

@dataclass(frozen=True, slots=True)
class SupportResistance:
    """Detected support and resistance price levels."""
    support_levels: list[Decimal]
    resistance_levels: list[Decimal]


# =========================================================================
# AVERAGE DIRECTIONAL INDEX (ADX)
# =========================================================================

def adx(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> ADXResult:
    """Calculate ADX with +DI and -DI."""
    if end_index < period * 2:
        return ADXResult(ZERO, ZERO, ZERO)

    start_idx = max(1, end_index - period * 3)
    period_bd = Decimal(str(period))

    # Seed with sum of first *period* values
    smooth_plus_dm = ZERO
    smooth_minus_dm = ZERO
    smooth_tr = ZERO

    seed_end = min(start_idx + period, end_index + 1)
    for i in range(start_idx, seed_end):
        if i <= 0:
            continue
        high = data[i].high
        low = data[i].low
        prev_high = data[i - 1].high
        prev_low = data[i - 1].low

        plus_dm = high - prev_high
        minus_dm = prev_low - low

        if plus_dm > ZERO and plus_dm > minus_dm:
            smooth_plus_dm += plus_dm
        if minus_dm > ZERO and minus_dm > plus_dm:
            smooth_minus_dm += minus_dm
        smooth_tr += true_range(data, i)

    # Wilder smoothing for remaining bars
    dx_values: list[Decimal] = []

    for i in range(seed_end, end_index + 1):
        if i <= 0:
            continue
        high = data[i].high
        low = data[i].low
        prev_high = data[i - 1].high
        prev_low = data[i - 1].low

        plus_dm = high - prev_high
        minus_dm = prev_low - low

        cur_plus_dm = ZERO
        cur_minus_dm = ZERO

        if plus_dm > ZERO and plus_dm > minus_dm:
            cur_plus_dm = plus_dm
        if minus_dm > ZERO and minus_dm > plus_dm:
            cur_minus_dm = minus_dm

        smooth_plus_dm = smooth_plus_dm - _quantize(smooth_plus_dm / period_bd) + cur_plus_dm
        smooth_minus_dm = smooth_minus_dm - _quantize(smooth_minus_dm / period_bd) + cur_minus_dm
        smooth_tr = smooth_tr - _quantize(smooth_tr / period_bd) + true_range(data, i)

        if smooth_tr != ZERO:
            p_di = _quantize(smooth_plus_dm / smooth_tr) * HUNDRED
            m_di = _quantize(smooth_minus_dm / smooth_tr) * HUNDRED
            di_sum = p_di + m_di
            if di_sum != ZERO:
                dx = _quantize(abs(p_di - m_di) / di_sum) * HUNDRED
                dx_values.append(dx)

    # ADX = average of DX values
    adx_value = ZERO
    if dx_values:
        adx_period = min(period, len(dx_values))
        total = ZERO
        for i in range(len(dx_values) - adx_period, len(dx_values)):
            total += dx_values[i]
        adx_value = _quantize(total / Decimal(str(adx_period)))

    # Current +DI and -DI
    plus_di = ZERO
    minus_di = ZERO
    if smooth_tr != ZERO:
        plus_di = _quantize(smooth_plus_dm / smooth_tr) * HUNDRED
        minus_di = _quantize(smooth_minus_dm / smooth_tr) * HUNDRED

    return ADXResult(adx_value, plus_di, minus_di)


# =========================================================================
# PARABOLIC SAR
# =========================================================================

def parabolic_sar(
    data: Sequence[PriceData],
    end_index: int,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> Decimal:
    """Calculate Parabolic SAR at the given index.

    Uses the standard Wilder method with configurable acceleration factors.

    Parameters
    ----------
    af_start:
        Initial acceleration factor (default 0.02).
    af_step:
        Acceleration factor increment per new extreme point (default 0.02).
    af_max:
        Maximum acceleration factor (default 0.20).
    """
    if end_index < 2:
        return data[0].low

    af = af_start
    max_af = af_max

    is_up_trend = data[1].close > data[0].close
    sar = float(data[0].low) if is_up_trend else float(data[0].high)
    ep = float(data[1].high) if is_up_trend else float(data[1].low)

    for i in range(2, end_index + 1):
        high = float(data[i].high)
        low = float(data[i].low)

        sar = sar + af * (ep - sar)

        if is_up_trend:
            sar = min(sar, float(data[i - 1].low), float(data[i - 2].low))
            if low < sar:
                # Flip to downtrend
                is_up_trend = False
                sar = ep
                ep = low
                af = af_step
            else:
                if high > ep:
                    ep = high
                    af = min(af + af_step, max_af)
        else:
            sar = max(sar, float(data[i - 1].high), float(data[i - 2].high))
            if high > sar:
                # Flip to uptrend
                is_up_trend = True
                sar = ep
                ep = high
                af = af_step
            else:
                if low < ep:
                    ep = low
                    af = min(af + af_step, max_af)

    return Decimal(str(sar)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

def is_sar_bullish(
    data: Sequence[PriceData],
    end_index: int,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> bool:
    """Return ``True`` if SAR indicates uptrend (SAR below price)."""
    sar_val = parabolic_sar(data, end_index, af_start, af_step, af_max)
    return data[end_index].close > sar_val


# =========================================================================
# VWAP (Volume-Weighted Average Price)
# =========================================================================

def vwap(
    data: Sequence[PriceData], end_index: int, lookback: int = 20
) -> Decimal:
    """Calculate VWAP over a lookback period.

    VWAP = Sum(TP * Volume) / Sum(Volume) where TP = (High + Low + Close) / 3.
    """
    start_idx = max(0, end_index - lookback + 1)

    sum_tpv = ZERO
    sum_vol = ZERO

    for i in range(start_idx, end_index + 1):
        tp = (data[i].high + data[i].low + data[i].close) / Decimal('3')
        vol = Decimal(str(data[i].volume))
        sum_tpv += tp * vol
        sum_vol += vol

    if sum_vol == ZERO:
        return ZERO
    return _quantize(sum_tpv / sum_vol)


# =========================================================================
# FIBONACCI RETRACEMENT
# =========================================================================

def fibonacci_retracement(
    data: Sequence[PriceData], end_index: int, lookback: int = 50
) -> FibonacciLevels:
    """Calculate Fibonacci retracement levels from the swing high/low within a lookback period."""
    start_idx = max(0, end_index - lookback + 1)

    highest = ZERO
    lowest = Decimal("Infinity")

    for i in range(start_idx, end_index + 1):
        if data[i].high > highest:
            highest = data[i].high
        if data[i].low < lowest:
            lowest = data[i].low

    fib_range = highest - lowest

    def _level(ratio: str) -> Decimal:
        return (highest - fib_range * Decimal(ratio)).quantize(
            Decimal('0.0001'), rounding=ROUND_HALF_UP
        )

    return FibonacciLevels(
        high=highest,
        low=lowest,
        level_236=_level('0.236'),
        level_382=_level('0.382'),
        level_500=_level('0.500'),
        level_618=_level('0.618'),
        level_786=_level('0.786'),
    )


# =========================================================================
# ICHIMOKU CLOUD
# =========================================================================

def ichimoku(data: Sequence[PriceData], end_index: int) -> IchimokuCloud:
    """Calculate Ichimoku Cloud components."""
    if end_index < 52:
        return IchimokuCloud(ZERO, ZERO, ZERO, ZERO, ZERO, False)

    tenkan = _period_midpoint(data, end_index, 9)
    kijun = _period_midpoint(data, end_index, 26)
    senkou_a = _quantize((tenkan + kijun) / TWO)

    # Senkou Span B uses 52-period midpoint
    senkou_b = _period_midpoint(data, end_index, 52)

    # Chikou Span = current close (plotted 26 periods back)
    chikou = data[end_index].close

    price = data[end_index].close
    above_cloud = price > max(senkou_a, senkou_b)

    return IchimokuCloud(tenkan, kijun, senkou_a, senkou_b, chikou, above_cloud)


# =========================================================================
# SUPPORT & RESISTANCE
# =========================================================================

def support_resistance(
    data: Sequence[PriceData],
    end_index: int,
    lookback: int,
    window: int,
) -> SupportResistance:
    """Detect support and resistance levels using swing points.

    A swing high requires *window* bars on each side to be lower.
    A swing low requires *window* bars on each side to be higher.
    """
    start_idx = max(window, end_index - lookback)
    end_limit = min(end_index - window, end_index)

    supports: list[Decimal] = []
    resistances: list[Decimal] = []

    current_price = data[end_index].close

    for i in range(start_idx, end_limit + 1):
        is_swing_high = True
        is_swing_low = True

        for j in range(1, window + 1):
            if i - j < 0 or i + j > end_index:
                is_swing_high = False
                is_swing_low = False
                break
            if data[i].high <= data[i - j].high or data[i].high <= data[i + j].high:
                is_swing_high = False
            if data[i].low >= data[i - j].low or data[i].low >= data[i + j].low:
                is_swing_low = False

        if is_swing_high:
            level = data[i].high
            if level > current_price:
                resistances.append(level)
        if is_swing_low:
            level = data[i].low
            if level < current_price:
                supports.append(level)

    # Sort: supports descending (nearest first), resistances ascending (nearest first)
    supports.sort(reverse=True)
    resistances.sort()

    # Deduplicate levels within 1% of each other
    supports = _deduplicate_levels(supports, current_price)
    resistances = _deduplicate_levels(resistances, current_price)

    return SupportResistance(supports, resistances)


# =========================================================================
# SESSION VWAP (Cumulative intraday VWAP with standard-deviation bands)
# =========================================================================

@dataclass(frozen=True, slots=True)
class SessionVWAP:
    """Session VWAP with standard-deviation bands."""

    vwap: Decimal
    std_dev: Decimal
    upper_1: Decimal
    lower_1: Decimal
    upper_05: Decimal
    lower_05: Decimal

def _find_session_start(data: Sequence[PriceData], end_index: int) -> int:
    """Walk backward from *end_index* to find the first bar of the current
    trading session.  A new session starts when the first-10-chars of
    ``date`` change (i.e. a new calendar day)."""
    current_day = data[end_index].date[:10]
    start = end_index
    while start > 0 and data[start - 1].date[:10] == current_day:
        start -= 1
    return start

def _compute_session_vwap_core(
    data: Sequence[PriceData],
    end_index: int,
) -> tuple[Decimal, Decimal]:
    """Core two-pass session VWAP computation.

    Returns ``(vwap, std_dev)``.  Both values are ``ZERO`` when the
    session has no volume.

    This is the single source of truth for the VWAP accumulation logic
    used by :func:`session_vwap`, :func:`session_vwap_bands`, and the
    extended/intraday variants.
    """
    session_start = _find_session_start(data, end_index)

    sum_tpv = ZERO
    sum_vol = ZERO
    tps: list[Decimal] = []
    vols: list[Decimal] = []

    for i in range(session_start, end_index + 1):
        tp = _quantize(
            (data[i].high + data[i].low + data[i].close) / Decimal('3')
        )
        vol = Decimal(str(data[i].volume))
        tps.append(tp)
        vols.append(vol)
        sum_tpv += tp * vol
        sum_vol += vol

    if sum_vol == ZERO:
        return ZERO, ZERO

    vwap_val = _quantize(sum_tpv / sum_vol)

    # Weighted standard deviation
    sum_var = ZERO
    for tp, vol in zip(tps, vols, strict=True):
        diff = tp - vwap_val
        sum_var += diff * diff * vol

    variance = float(sum_var / sum_vol)
    std_float = math.sqrt(max(0.0, variance))
    std_val = _quantize(Decimal(str(std_float)))

    return vwap_val, std_val

def session_vwap(
    data: Sequence[PriceData], end_index: int
) -> Decimal:
    """Calculate cumulative session VWAP, resetting at each new trading day.

    Walks backward from *end_index* to find the session start, then
    cumulates ``TP * Volume / Volume`` forward.
    """
    vwap_val, _ = _compute_session_vwap_core(data, end_index)
    return vwap_val

def session_vwap_bands(
    data: Sequence[PriceData], end_index: int
) -> SessionVWAP:
    """Calculate session VWAP with standard-deviation bands.

    sigma = sqrt( Sum((TP - VWAP)^2 * Volume) / Sum(Volume) )

    Returns a :class:`SessionVWAP` with VWAP, std_dev and +-1/+-0.5 sigma
    bands.
    """
    vwap_val, std_val = _compute_session_vwap_core(data, end_index)

    if vwap_val == ZERO and std_val == ZERO:
        return SessionVWAP(ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)

    half_std = _quantize(std_val * Decimal('0.5'))

    return SessionVWAP(
        vwap=vwap_val,
        std_dev=std_val,
        upper_1=_quantize(vwap_val + std_val),
        lower_1=_quantize(vwap_val - std_val),
        upper_05=_quantize(vwap_val + half_std),
        lower_05=_quantize(vwap_val - half_std),
    )
