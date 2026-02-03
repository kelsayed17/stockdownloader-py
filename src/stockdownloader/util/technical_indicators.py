"""Comprehensive technical indicator calculations for stock analysis.

Provides Bollinger Bands, Stochastic Oscillator, ATR, OBV, ADX, Parabolic SAR,
Williams %R, CCI, VWAP, Fibonacci Retracement, ROC, MFI, Ichimoku Cloud,
RSI, MACD, support/resistance detection, and average volume.

All functions operate on ``list[PriceData]`` with an *end_index* parameter to
allow incremental calculation during backtesting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.moving_average_calculator import sma as _sma, ema as _ema

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

SCALE = 10
_HUNDRED = Decimal('100')
_TWO = Decimal('2')
_ZERO = Decimal('0')

# =========================================================================
# DATA CLASSES (converted from Java record inner classes)
# =========================================================================


@dataclass(frozen=True)
class BollingerBands:
    """Bollinger Band values: upper, middle (SMA), lower, and width."""
    upper: Decimal
    middle: Decimal
    lower: Decimal
    width: Decimal


@dataclass(frozen=True)
class Stochastic:
    """Stochastic Oscillator values: %K and %D."""
    percent_k: Decimal
    percent_d: Decimal


@dataclass(frozen=True)
class ADXResult:
    """Average Directional Index result: ADX value, +DI, and -DI."""
    adx: Decimal
    plus_di: Decimal
    minus_di: Decimal


@dataclass(frozen=True)
class IchimokuCloud:
    """Ichimoku Cloud components."""
    tenkan_sen: Decimal       # Conversion Line (9-period)
    kijun_sen: Decimal        # Base Line (26-period)
    senkou_span_a: Decimal    # Leading Span A
    senkou_span_b: Decimal    # Leading Span B (52-period)
    chikou_span: Decimal      # Lagging Span
    price_above_cloud: bool


@dataclass(frozen=True)
class FibonacciLevels:
    """Fibonacci retracement levels derived from a swing high/low."""
    high: Decimal
    low: Decimal
    level_236: Decimal
    level_382: Decimal
    level_500: Decimal
    level_618: Decimal
    level_786: Decimal


@dataclass(frozen=True)
class SupportResistance:
    """Detected support and resistance price levels."""
    support_levels: list[Decimal]
    resistance_levels: list[Decimal]


# =========================================================================
# BOLLINGER BANDS
# =========================================================================


def bollinger_bands(
    data: Sequence[PriceData],
    end_index: int,
    period: int = 20,
    num_std_dev: float = 2.0,
) -> BollingerBands:
    """Calculate Bollinger Bands.

    Middle = SMA(*period*), Upper/Lower = Middle +/- *num_std_dev* * StdDev.
    """
    if end_index < period - 1:
        return BollingerBands(_ZERO, _ZERO, _ZERO, _ZERO)

    mid = _sma(data, end_index, period)
    std_dev = standard_deviation(data, end_index, period)
    deviation = std_dev * Decimal(str(num_std_dev))

    upper = mid + deviation
    lower = mid - deviation
    width = upper - lower

    return BollingerBands(upper, mid, lower, width)


def bollinger_percent_b(
    data: Sequence[PriceData],
    end_index: int,
    period: int = 20,
) -> Decimal:
    """Calculate Bollinger Band %B: (Price - Lower) / (Upper - Lower).

    Values > 1 indicate above upper band, < 0 indicate below lower band.
    """
    bb = bollinger_bands(data, end_index, period, 2.0)
    band_range = bb.upper - bb.lower
    if band_range == _ZERO:
        return _ZERO
    return _quantize((data[end_index].close - bb.lower) / band_range)


# =========================================================================
# STOCHASTIC OSCILLATOR
# =========================================================================


def stochastic(
    data: Sequence[PriceData],
    end_index: int,
    k_period: int = 14,
    d_period: int = 3,
) -> Stochastic:
    """Calculate Stochastic Oscillator (%K and %D)."""
    if end_index < k_period - 1:
        return Stochastic(_ZERO, _ZERO)

    percent_k = _calculate_percent_k(data, end_index, k_period)

    # %D = SMA of recent %K values
    sum_k = _ZERO
    count = 0
    for i in range(max(k_period - 1, end_index - d_period + 1), end_index + 1):
        sum_k += _calculate_percent_k(data, i, k_period)
        count += 1

    percent_d = _quantize(sum_k / Decimal(str(count))) if count > 0 else _ZERO

    return Stochastic(percent_k, percent_d)


def _calculate_percent_k(
    data: Sequence[PriceData], end_index: int, period: int
) -> Decimal:
    highest_high = _ZERO
    lowest_low = Decimal(str(float('inf')))

    for i in range(end_index - period + 1, end_index + 1):
        high = data[i].high
        low = data[i].low
        if high > highest_high:
            highest_high = high
        if low < lowest_low:
            lowest_low = low

    hl_range = highest_high - lowest_low
    if hl_range == _ZERO:
        return _ZERO

    return _quantize((data[end_index].close - lowest_low) / hl_range) * _HUNDRED


# =========================================================================
# AVERAGE TRUE RANGE (ATR)
# =========================================================================


def atr(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> Decimal:
    """Calculate Average True Range using Wilder smoothing.

    TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|).
    """
    if end_index < period:
        return _ZERO

    start_index = max(1, end_index - period * 2)

    # Seed ATR as simple average of first *period* TRs
    atr_val = _ZERO
    count = 0
    for i in range(start_index, min(start_index + period, end_index + 1)):
        atr_val += true_range(data, i)
        count += 1
    if count == 0:
        return _ZERO
    atr_val = _quantize(atr_val / Decimal(str(count)))

    multiplier = _quantize(Decimal('1') / Decimal(str(period)))
    one_minus_mult = Decimal('1') - multiplier

    for i in range(start_index + count, end_index + 1):
        atr_val = _quantize(true_range(data, i) * multiplier + atr_val * one_minus_mult)

    return atr_val


def true_range(data: Sequence[PriceData], index: int) -> Decimal:
    """Calculate the True Range for a single bar."""
    if index <= 0:
        return data[index].high - data[index].low

    high = data[index].high
    low = data[index].low
    prev_close = data[index - 1].close

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    return max(tr1, tr2, tr3)


# =========================================================================
# ON-BALANCE VOLUME (OBV)
# =========================================================================


def obv(data: Sequence[PriceData], end_index: int) -> Decimal:
    """Calculate On-Balance Volume.

    If close > prevClose: OBV += volume; if close < prevClose: OBV -= volume.
    """
    obv_val = _ZERO
    for i in range(1, end_index + 1):
        if data[i].close > data[i - 1].close:
            obv_val += Decimal(str(data[i].volume))
        elif data[i].close < data[i - 1].close:
            obv_val -= Decimal(str(data[i].volume))
    return obv_val


def is_obv_rising(
    data: Sequence[PriceData], end_index: int, lookback: int
) -> bool:
    """Return ``True`` if OBV is rising over *lookback* bars."""
    if end_index < lookback:
        return False
    current = obv(data, end_index)
    previous = obv(data, end_index - lookback)
    return current > previous


# =========================================================================
# AVERAGE DIRECTIONAL INDEX (ADX)
# =========================================================================


def adx(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> ADXResult:
    """Calculate ADX with +DI and -DI."""
    if end_index < period * 2:
        return ADXResult(_ZERO, _ZERO, _ZERO)

    start_idx = max(1, end_index - period * 3)
    period_bd = Decimal(str(period))

    # Seed with sum of first *period* values
    smooth_plus_dm = _ZERO
    smooth_minus_dm = _ZERO
    smooth_tr = _ZERO

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

        if plus_dm > _ZERO and plus_dm > minus_dm:
            smooth_plus_dm += plus_dm
        if minus_dm > _ZERO and minus_dm > plus_dm:
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

        cur_plus_dm = _ZERO
        cur_minus_dm = _ZERO

        if plus_dm > _ZERO and plus_dm > minus_dm:
            cur_plus_dm = plus_dm
        if minus_dm > _ZERO and minus_dm > plus_dm:
            cur_minus_dm = minus_dm

        smooth_plus_dm = smooth_plus_dm - _quantize(smooth_plus_dm / period_bd) + cur_plus_dm
        smooth_minus_dm = smooth_minus_dm - _quantize(smooth_minus_dm / period_bd) + cur_minus_dm
        smooth_tr = smooth_tr - _quantize(smooth_tr / period_bd) + true_range(data, i)

        if smooth_tr != _ZERO:
            p_di = _quantize(smooth_plus_dm / smooth_tr) * _HUNDRED
            m_di = _quantize(smooth_minus_dm / smooth_tr) * _HUNDRED
            di_sum = p_di + m_di
            if di_sum != _ZERO:
                dx = _quantize(abs(p_di - m_di) / di_sum) * _HUNDRED
                dx_values.append(dx)

    # ADX = average of DX values
    adx_value = _ZERO
    if dx_values:
        adx_period = min(period, len(dx_values))
        total = _ZERO
        for i in range(len(dx_values) - adx_period, len(dx_values)):
            total += dx_values[i]
        adx_value = _quantize(total / Decimal(str(adx_period)))

    # Current +DI and -DI
    plus_di = _ZERO
    minus_di = _ZERO
    if smooth_tr != _ZERO:
        plus_di = _quantize(smooth_plus_dm / smooth_tr) * _HUNDRED
        minus_di = _quantize(smooth_minus_dm / smooth_tr) * _HUNDRED

    return ADXResult(adx_value, plus_di, minus_di)


# =========================================================================
# PARABOLIC SAR
# =========================================================================


def parabolic_sar(data: Sequence[PriceData], end_index: int) -> Decimal:
    """Calculate Parabolic SAR at the given index.

    Uses the standard Wilder method: AF starts at 0.02, increments by 0.02
    per new extreme point, max 0.20.
    """
    if end_index < 2:
        return data[0].low

    af = 0.02
    max_af = 0.20
    af_step = 0.02

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


def is_sar_bullish(data: Sequence[PriceData], end_index: int) -> bool:
    """Return ``True`` if SAR indicates uptrend (SAR below price)."""
    sar_val = parabolic_sar(data, end_index)
    return data[end_index].close > sar_val


# =========================================================================
# WILLIAMS %R
# =========================================================================


def williams_r(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> Decimal:
    """Calculate Williams %R.

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    """
    if end_index < period - 1:
        return _ZERO

    highest_high = _ZERO
    lowest_low = Decimal(str(float('inf')))

    for i in range(end_index - period + 1, end_index + 1):
        if data[i].high > highest_high:
            highest_high = data[i].high
        if data[i].low < lowest_low:
            lowest_low = data[i].low

    hl_range = highest_high - lowest_low
    if hl_range == _ZERO:
        return _ZERO

    return _quantize((highest_high - data[end_index].close) / hl_range) * Decimal('-100')


# =========================================================================
# COMMODITY CHANNEL INDEX (CCI)
# =========================================================================


def cci(
    data: Sequence[PriceData], end_index: int, period: int = 20
) -> Decimal:
    """Calculate CCI.

    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    where TP = (High + Low + Close) / 3
    """
    if end_index < period - 1:
        return _ZERO

    tp_values: list[Decimal] = []
    sum_tp = _ZERO

    for i in range(period):
        idx = end_index - period + 1 + i
        tp = _quantize(
            (data[idx].high + data[idx].low + data[idx].close) / Decimal('3')
        )
        tp_values.append(tp)
        sum_tp += tp

    sma_tp = _quantize(sum_tp / Decimal(str(period)))

    # Mean deviation
    sum_dev = _ZERO
    for tp in tp_values:
        sum_dev += abs(tp - sma_tp)
    mean_dev = _quantize(sum_dev / Decimal(str(period)))

    constant = Decimal('0.015')
    divisor = constant * mean_dev

    if divisor == _ZERO:
        return _ZERO

    current_tp = tp_values[-1]
    return _quantize((current_tp - sma_tp) / divisor)


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

    sum_tpv = _ZERO
    sum_vol = _ZERO

    for i in range(start_idx, end_index + 1):
        tp = _quantize(
            (data[i].high + data[i].low + data[i].close) / Decimal('3')
        )
        vol = Decimal(str(data[i].volume))
        sum_tpv += tp * vol
        sum_vol += vol

    if sum_vol == _ZERO:
        return _ZERO
    return _quantize(sum_tpv / sum_vol)


# =========================================================================
# FIBONACCI RETRACEMENT
# =========================================================================


def fibonacci_retracement(
    data: Sequence[PriceData], end_index: int, lookback: int = 50
) -> FibonacciLevels:
    """Calculate Fibonacci retracement levels from the swing high/low within a lookback period."""
    start_idx = max(0, end_index - lookback + 1)

    highest = _ZERO
    lowest = Decimal(str(float('inf')))

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
# RATE OF CHANGE (ROC)
# =========================================================================


def roc(
    data: Sequence[PriceData], end_index: int, period: int = 12
) -> Decimal:
    """Calculate Rate of Change.

    ROC = ((Close - Close_n) / Close_n) * 100
    """
    if end_index < period:
        return _ZERO

    current_close = data[end_index].close
    past_close = data[end_index - period].close

    if past_close == _ZERO:
        return _ZERO

    return _quantize((current_close - past_close) / past_close) * _HUNDRED


# =========================================================================
# MONEY FLOW INDEX (MFI)
# =========================================================================


def mfi(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> Decimal:
    """Calculate Money Flow Index (volume-weighted RSI).

    MFI = 100 - (100 / (1 + Money Ratio))
    where Money Ratio = Positive Money Flow / Negative Money Flow.
    """
    if end_index < period:
        return _ZERO

    positive_flow = _ZERO
    negative_flow = _ZERO

    for i in range(end_index - period + 1, end_index + 1):
        tp = _quantize(
            (data[i].high + data[i].low + data[i].close) / Decimal('3')
        )
        prev_tp = _quantize(
            (data[i - 1].high + data[i - 1].low + data[i - 1].close) / Decimal('3')
        )

        money_flow = tp * Decimal(str(data[i].volume))

        if tp > prev_tp:
            positive_flow += money_flow
        elif tp < prev_tp:
            negative_flow += money_flow

    if negative_flow == _ZERO:
        return _HUNDRED if positive_flow > _ZERO else _ZERO

    money_ratio = _quantize(positive_flow / negative_flow)
    return _HUNDRED - _quantize(_HUNDRED / (Decimal('1') + money_ratio))


# =========================================================================
# ICHIMOKU CLOUD
# =========================================================================


def ichimoku(data: Sequence[PriceData], end_index: int) -> IchimokuCloud:
    """Calculate Ichimoku Cloud components."""
    if end_index < 52:
        return IchimokuCloud(_ZERO, _ZERO, _ZERO, _ZERO, _ZERO, False)

    tenkan = _period_midpoint(data, end_index, 9)
    kijun = _period_midpoint(data, end_index, 26)
    senkou_a = _quantize((tenkan + kijun) / _TWO)

    # Senkou Span B uses 52-period midpoint
    senkou_b = _period_midpoint(data, end_index, 52)

    # Chikou Span = current close (plotted 26 periods back)
    chikou = data[end_index].close

    price = data[end_index].close
    above_cloud = price > max(senkou_a, senkou_b)

    return IchimokuCloud(tenkan, kijun, senkou_a, senkou_b, chikou, above_cloud)


def _period_midpoint(
    data: Sequence[PriceData], end_index: int, period: int
) -> Decimal:
    highest = _ZERO
    lowest = Decimal(str(float('inf')))

    for i in range(end_index - period + 1, end_index + 1):
        if data[i].high > highest:
            highest = data[i].high
        if data[i].low < lowest:
            lowest = data[i].low

    return _quantize((highest + lowest) / _TWO)


# =========================================================================
# RSI
# =========================================================================


def rsi(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> Decimal:
    """Calculate Relative Strength Index at the given index."""
    if end_index < period + 1:
        return Decimal('50')

    avg_gain = _ZERO
    avg_loss = _ZERO

    for i in range(end_index - period + 1, end_index + 1):
        change = data[i].close - data[i - 1].close
        if change > _ZERO:
            avg_gain += change
        else:
            avg_loss += abs(change)

    period_bd = Decimal(str(period))
    avg_gain = _quantize(avg_gain / period_bd)
    avg_loss = _quantize(avg_loss / period_bd)

    if avg_loss == _ZERO:
        return _HUNDRED

    rs = _quantize(avg_gain / avg_loss)
    return _HUNDRED - (_HUNDRED / (Decimal('1') + rs)).quantize(
        Decimal('0.000001'), rounding=ROUND_HALF_UP
    )


# =========================================================================
# MACD
# =========================================================================


def macd_line(
    data: Sequence[PriceData],
    end_index: int,
    fast: int = 12,
    slow: int = 26,
) -> Decimal:
    """Calculate MACD line value (fast EMA - slow EMA)."""
    if end_index < slow:
        return _ZERO
    return _ema(data, end_index, fast) - _ema(data, end_index, slow)


def macd_signal(
    data: Sequence[PriceData],
    end_index: int,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Decimal:
    """Calculate MACD signal line."""
    if end_index < slow + signal:
        return _ZERO

    multiplier = Decimal(str(2.0 / (signal + 1)))
    one_minus_mult = Decimal('1') - multiplier

    start_idx = max(slow, end_index - signal + 1)

    total = _ZERO
    count = 0
    for i in range(start_idx, min(start_idx + signal, end_index + 1)):
        total += macd_line(data, i, fast, slow)
        count += 1
    if count == 0:
        return _ZERO

    signal_ema = _quantize(total / Decimal(str(count)))

    for i in range(start_idx + count, end_index + 1):
        macd_val = macd_line(data, i, fast, slow)
        signal_ema = _quantize(macd_val * multiplier + signal_ema * one_minus_mult)

    return signal_ema


def macd_histogram(
    data: Sequence[PriceData],
    end_index: int,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Decimal:
    """Calculate MACD histogram (MACD line - signal line)."""
    return macd_line(data, end_index, fast, slow) - macd_signal(
        data, end_index, fast, slow, signal
    )


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


def _deduplicate_levels(
    levels: list[Decimal], reference: Decimal
) -> list[Decimal]:
    if not levels:
        return levels
    tolerance = reference * Decimal('0.01')
    deduped: list[Decimal] = [levels[0]]

    for i in range(1, len(levels)):
        too_close = False
        for existing in deduped:
            if abs(levels[i] - existing) < tolerance:
                too_close = True
                break
        if not too_close:
            deduped.append(levels[i])
    return deduped


# =========================================================================
# AVERAGE VOLUME
# =========================================================================


def average_volume(
    data: Sequence[PriceData], end_index: int, period: int
) -> Decimal:
    """Calculate average volume over a period."""
    if end_index < period - 1:
        return _ZERO
    total = _ZERO
    for i in range(end_index - period + 1, end_index + 1):
        total += Decimal(str(data[i].volume))
    return _quantize(total / Decimal(str(period)))


# =========================================================================
# HELPERS
# =========================================================================


def standard_deviation(
    data: Sequence[PriceData], end_index: int, period: int
) -> Decimal:
    """Calculate standard deviation of close prices over a period."""
    if end_index < period - 1:
        return _ZERO

    total = _ZERO
    for i in range(end_index - period + 1, end_index + 1):
        total += data[i].close
    mean = _quantize(total / Decimal(str(period)))

    sum_sq_diff = _ZERO
    for i in range(end_index - period + 1, end_index + 1):
        diff = data[i].close - mean
        sum_sq_diff += diff * diff

    variance = float(_quantize(sum_sq_diff / Decimal(str(period))))
    return Decimal(str(math.sqrt(variance))).quantize(
        Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _quantize(value: Decimal) -> Decimal:
    """Quantize *value* to *SCALE* decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP)
