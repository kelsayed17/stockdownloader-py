"""Momentum indicators: RSI, MACD, Stochastic, MFI, CCI, Williams %R, ROC."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.moving_average_calculator import ema as _ema
from stockdownloader.util.big_decimal_math import HUNDRED, ZERO
from stockdownloader.util.technical._helpers import _quantize

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from stockdownloader.model.price_data import PriceData

__all__ = [
    "Stochastic",
    "stochastic",
    "rsi",
    "macd_line",
    "macd_signal",
    "_macd_signal_from_lines",
    "macd_histogram",
    "roc",
    "mfi",
    "williams_r",
    "cci",
]


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass(frozen=True, slots=True)
class Stochastic:
    """Stochastic Oscillator values: %K and %D."""
    percent_k: Decimal
    percent_d: Decimal


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
        return Stochastic(ZERO, ZERO)

    percent_k = _calculate_percent_k(data, end_index, k_period)

    # %D = SMA of recent %K values
    sum_k = ZERO
    count = 0
    for i in range(max(k_period - 1, end_index - d_period + 1), end_index + 1):
        sum_k += _calculate_percent_k(data, i, k_period)
        count += 1

    percent_d = _quantize(sum_k / Decimal(str(count))) if count > 0 else ZERO

    return Stochastic(percent_k, percent_d)

def _calculate_percent_k(
    data: Sequence[PriceData], end_index: int, period: int
) -> Decimal:
    highest_high = ZERO
    lowest_low = Decimal("Infinity")

    for i in range(end_index - period + 1, end_index + 1):
        high = data[i].high
        low = data[i].low
        if high > highest_high:
            highest_high = high
        if low < lowest_low:
            lowest_low = low

    hl_range = highest_high - lowest_low
    if hl_range == ZERO:
        return ZERO

    return _quantize((data[end_index].close - lowest_low) / hl_range) * HUNDRED


# =========================================================================
# RSI
# =========================================================================

def rsi(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> Decimal:
    """Calculate Relative Strength Index at the given index."""
    if end_index < period + 1:
        return Decimal('50')

    avg_gain = ZERO
    avg_loss = ZERO

    for i in range(end_index - period + 1, end_index + 1):
        change = data[i].close - data[i - 1].close
        if change > ZERO:
            avg_gain += change
        else:
            avg_loss += abs(change)

    period_bd = Decimal(str(period))
    avg_gain = _quantize(avg_gain / period_bd)
    avg_loss = _quantize(avg_loss / period_bd)

    if avg_loss == ZERO:
        return HUNDRED

    rs = _quantize(avg_gain / avg_loss)
    return HUNDRED - (HUNDRED / (Decimal('1') + rs)).quantize(
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
        return ZERO
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
        return ZERO

    multiplier = Decimal(str(2.0 / (signal + 1)))
    one_minus_mult = Decimal('1') - multiplier

    start_idx = max(slow, end_index - signal + 1)

    total = ZERO
    count = 0
    for i in range(start_idx, min(start_idx + signal, end_index + 1)):
        total += macd_line(data, i, fast, slow)
        count += 1
    if count == 0:
        return ZERO

    signal_ema = _quantize(total / Decimal(str(count)))

    for i in range(start_idx + count, end_index + 1):
        macd_val = macd_line(data, i, fast, slow)
        signal_ema = _quantize(macd_val * multiplier + signal_ema * one_minus_mult)

    return signal_ema

def _macd_signal_from_lines(
    data: Sequence[PriceData],
    end_index: int,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    *,
    line_fn: Callable[[Sequence[PriceData], int, int, int], Decimal] | None = None,
) -> Decimal:
    """MACD signal line with injectable MACD-line lookup.

    Identical to :func:`macd_signal` but calls *line_fn* instead of
    :func:`macd_line` for each intermediate MACD-line value.  When
    *line_fn* is ``None`` (default), falls back to :func:`macd_line`.

    The :class:`IndicatorHub` uses this to route sub-lookups through
    its cache, avoiding redundant EMA recomputation.
    """
    _line = line_fn or macd_line

    if end_index < slow + signal:
        return ZERO

    multiplier = Decimal(str(2.0 / (signal + 1)))
    one_minus_mult = Decimal('1') - multiplier

    start_idx = max(slow, end_index - signal + 1)

    total = ZERO
    count = 0
    for i in range(start_idx, min(start_idx + signal, end_index + 1)):
        total += _line(data, i, fast, slow)
        count += 1
    if count == 0:
        return ZERO

    signal_ema = _quantize(total / Decimal(str(count)))

    for i in range(start_idx + count, end_index + 1):
        macd_val = _line(data, i, fast, slow)
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
# RATE OF CHANGE (ROC)
# =========================================================================

def roc(
    data: Sequence[PriceData], end_index: int, period: int = 12
) -> Decimal:
    """Calculate Rate of Change.

    ROC = ((Close - Close_n) / Close_n) * 100
    """
    if end_index < period:
        return ZERO

    current_close = data[end_index].close
    past_close = data[end_index - period].close

    if past_close == ZERO:
        return ZERO

    return _quantize((current_close - past_close) / past_close) * HUNDRED


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
        return ZERO

    positive_flow = ZERO
    negative_flow = ZERO

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

    if negative_flow == ZERO:
        return HUNDRED if positive_flow > ZERO else ZERO

    money_ratio = _quantize(positive_flow / negative_flow)
    return HUNDRED - _quantize(HUNDRED / (Decimal('1') + money_ratio))


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
        return ZERO

    highest_high = ZERO
    lowest_low = Decimal("Infinity")

    for i in range(end_index - period + 1, end_index + 1):
        if data[i].high > highest_high:
            highest_high = data[i].high
        if data[i].low < lowest_low:
            lowest_low = data[i].low

    hl_range = highest_high - lowest_low
    if hl_range == ZERO:
        return ZERO

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
        return ZERO

    tp_values: list[Decimal] = []
    sum_tp = ZERO

    for i in range(period):
        idx = end_index - period + 1 + i
        tp = _quantize(
            (data[idx].high + data[idx].low + data[idx].close) / Decimal('3')
        )
        tp_values.append(tp)
        sum_tp += tp

    sma_tp = _quantize(sum_tp / Decimal(str(period)))

    # Mean deviation
    sum_dev = ZERO
    for tp in tp_values:
        sum_dev += abs(tp - sma_tp)
    mean_dev = _quantize(sum_dev / Decimal(str(period)))

    constant = Decimal('0.015')
    divisor = constant * mean_dev

    if divisor == ZERO:
        return ZERO

    current_tp = tp_values[-1]
    return _quantize((current_tp - sma_tp) / divisor)
