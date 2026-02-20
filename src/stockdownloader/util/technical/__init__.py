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

from stockdownloader.util.big_decimal_math import ZERO

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

    from stockdownloader.util.big_decimal_math import TWO
    return _quantize((highest + lowest) / TWO)


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
