"""Volatility indicators: Bollinger Bands, ATR, standard deviation."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.technical import (
    SCALE,
    _quantize,
    sma as _sma,
    standard_deviation,
    true_range,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

__all__ = [
    "BollingerBands",
    "bollinger_bands",
    "bollinger_percent_b",
    "_bollinger_percent_b_from_bands",
    "atr",
    "true_range",
    "standard_deviation",
]

# Re-export from _helpers so they remain accessible from this module
true_range = true_range
standard_deviation = standard_deviation


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass(frozen=True, slots=True)
class BollingerBands:
    """Bollinger Band values: upper, middle (SMA), lower, and width."""
    upper: Decimal
    middle: Decimal
    lower: Decimal
    width: Decimal


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
        return BollingerBands(ZERO, ZERO, ZERO, ZERO)

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
    if band_range == ZERO:
        return ZERO
    return _quantize((data[end_index].close - bb.lower) / band_range)

def _bollinger_percent_b_from_bands(
    close_price: Decimal,
    bb: BollingerBands,
) -> Decimal:
    """Compute Bollinger %B from a pre-computed :class:`BollingerBands`.

    Identical arithmetic to :func:`bollinger_percent_b` but avoids
    recomputing the bands.  Used by :class:`IndicatorHub` to compose
    through the cache.
    """
    band_range = bb.upper - bb.lower
    if band_range == ZERO:
        return ZERO
    return _quantize((close_price - bb.lower) / band_range)


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
        return ZERO

    start_index = max(1, end_index - period * 2)

    # Seed ATR as simple average of first *period* TRs
    atr_val = ZERO
    count = 0
    for i in range(start_index, min(start_index + period, end_index + 1)):
        atr_val += true_range(data, i)
        count += 1
    if count == 0:
        return ZERO
    atr_val = _quantize(atr_val / Decimal(str(count)))

    multiplier = _quantize(Decimal('1') / Decimal(str(period)))
    one_minus_mult = Decimal('1') - multiplier

    for i in range(start_index + count, end_index + 1):
        atr_val = _quantize(true_range(data, i) * multiplier + atr_val * one_minus_mult)

    return atr_val
