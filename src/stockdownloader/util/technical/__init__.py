"""Technical indicators organized by category.

Sub-modules:
- volatility: Bollinger Bands, ATR, standard deviation
- momentum: RSI, MACD, Stochastic, MFI, CCI, Williams %R, ROC, OBV, average volume
- trend: ADX/DMI, Parabolic SAR, Ichimoku, VWAP, Fibonacci, Support/Resistance

All public symbols are re-exported here so that
``from stockdownloader.util.technical import <name>`` works for every indicator.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.math import ZERO

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

# =========================================================================
# Shared helpers (formerly _helpers.py)
# =========================================================================

SCALE = 10


def _quantize(value: Decimal) -> Decimal:
    """Quantize *value* to *SCALE* decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP)


# =========================================================================
# Moving average calculations (formerly moving_average_calculator.py)
# =========================================================================


def sma(data: Sequence[PriceData], end_index: int, period: int) -> Decimal:
    """Calculate the Simple Moving Average of close prices.

    Args:
        data: List of ``PriceData`` objects.
        end_index: The last index (inclusive) of the window.
        period: Number of bars in the average.

    Returns:
        The SMA as a ``Decimal`` rounded to *SCALE* decimal places.

    Raises:
        ValueError: If *period* is less than 1.
    """
    if period < 1:
        raise ValueError(f"SMA period must be >= 1, got {period}")
    total = Decimal('0')
    for i in range(end_index - period + 1, end_index + 1):
        total += data[i].close
    return _quantize(total / Decimal(str(period)))


def ema(data: Sequence[PriceData], end_index: int, period: int) -> Decimal:
    """Calculate the Exponential Moving Average of close prices.

    The seed value is the SMA over the first *period* bars.  Subsequent bars
    apply the standard EMA smoothing multiplier ``2 / (period + 1)``.

    Args:
        data: List of ``PriceData`` objects.
        end_index: The last index (inclusive) of the window.
        period: Number of bars used for the EMA.

    Returns:
        The EMA as a ``Decimal`` rounded to *SCALE* decimal places.

    Raises:
        ValueError: If *period* is less than 1.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}")
    multiplier = Decimal(str(2.0 / (period + 1)))
    one_minus_multiplier = Decimal('1') - multiplier

    start_index = max(0, end_index - period - period)
    seed_end = min(start_index + period, end_index + 1)

    total = Decimal('0')
    for i in range(start_index, seed_end):
        total += data[i].close
    ema_val = _quantize(total / Decimal(str(period)))

    for i in range(start_index + period, end_index + 1):
        ema_val = _quantize(data[i].close * multiplier + ema_val * one_minus_multiplier)

    return ema_val


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


def standard_deviation(
    data: Sequence[PriceData], end_index: int, period: int
) -> Decimal:
    """Calculate standard deviation of close prices over a period."""
    if end_index < period - 1:
        return ZERO

    total = ZERO
    for i in range(end_index - period + 1, end_index + 1):
        total += data[i].close
    mean = _quantize(total / Decimal(str(period)))

    sum_sq_diff = ZERO
    for i in range(end_index - period + 1, end_index + 1):
        diff = data[i].close - mean
        sum_sq_diff += diff * diff

    variance = max(0.0, float(_quantize(sum_sq_diff / Decimal(str(period)))))
    return Decimal(str(math.sqrt(variance))).quantize(
        Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP
    )


def _period_midpoint(
    data: Sequence[PriceData], end_index: int, period: int
) -> Decimal:
    highest = ZERO
    lowest = Decimal("Infinity")

    for i in range(end_index - period + 1, end_index + 1):
        if data[i].high > highest:
            highest = data[i].high
        if data[i].low < lowest:
            lowest = data[i].low

    from stockdownloader.util.math import TWO
    return _quantize((highest + lowest) / TWO)


def _deduplicate_levels(
    levels: list[Decimal], reference: Decimal,
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
# Crossover detection
# =========================================================================


def crossed_above(
    current: Decimal,
    previous: Decimal,
    threshold: Decimal,
) -> bool:
    """Return ``True`` if *current* is above *threshold* while *previous* was at or below it.

    This detects the exact bar where an upward crossing occurs.
    """
    return current > threshold and previous <= threshold


def crossed_below(
    current: Decimal,
    previous: Decimal,
    threshold: Decimal,
) -> bool:
    """Return ``True`` if *current* is below *threshold* while *previous* was at or above it.

    This detects the exact bar where a downward crossing occurs.
    """
    return current < threshold and previous >= threshold


def crossed_above_series(
    current_a: Decimal,
    prev_a: Decimal,
    current_b: Decimal,
    prev_b: Decimal,
) -> bool:
    """Return ``True`` if series *A* crossed above series *B*.

    Useful for MACD-over-signal, SMA-fast-over-slow, and similar
    two-series crossover patterns.
    """
    return current_a > current_b and prev_a <= prev_b


def crossed_below_series(
    current_a: Decimal,
    prev_a: Decimal,
    current_b: Decimal,
    prev_b: Decimal,
) -> bool:
    """Return ``True`` if series *A* crossed below series *B*.

    Useful for MACD-under-signal, SMA-fast-under-slow, and similar
    two-series crossover patterns.
    """
    return current_a < current_b and prev_a >= prev_b


# =========================================================================
# Sub-module re-exports
# =========================================================================

from stockdownloader.util.technical.volatility import (
    BollingerBands,
    _bollinger_percent_b_from_bands,
    atr,
    bollinger_bands,
    bollinger_percent_b,
)
from stockdownloader.util.technical.momentum import (
    Stochastic,
    _macd_signal_from_lines,
    average_volume,
    cci,
    is_obv_rising,
    macd_histogram,
    macd_line,
    macd_signal,
    mfi,
    obv,
    roc,
    rsi,
    stochastic,
    williams_r,
)
from stockdownloader.util.technical.trend import (
    ADXResult,
    FibonacciLevels,
    IchimokuCloud,
    SessionVWAP,
    SupportResistance,
    _compute_session_vwap_core,
    _find_session_start,
    adx,
    fibonacci_retracement,
    ichimoku,
    is_sar_bullish,
    parabolic_sar,
    session_vwap,
    session_vwap_bands,
    support_resistance,
    vwap,
)

__all__ = [
    # helpers
    "SCALE",
    "_quantize",
    "true_range",
    "standard_deviation",
    "_period_midpoint",
    "_deduplicate_levels",
    # moving averages
    "sma",
    "ema",
    # crossover detection
    "crossed_above",
    "crossed_below",
    "crossed_above_series",
    "crossed_below_series",
    # volatility
    "BollingerBands",
    "bollinger_bands",
    "bollinger_percent_b",
    "_bollinger_percent_b_from_bands",
    "atr",
    # momentum
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
    # trend
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
    # volume
    "obv",
    "is_obv_rising",
    "average_volume",
]
