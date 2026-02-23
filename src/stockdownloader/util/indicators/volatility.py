"""Volatility indicators: Bollinger Bands, ATR, standard deviation.

Section 1 — Batch indicators
Section 2 — Streaming indicators (StreamingEMA, StreamingATR)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.math import ONE, ZERO, quantize
from stockdownloader.util.indicators._core import (
    sma as _sma,
    standard_deviation,
    true_range,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

# Re-export from _core so they remain accessible from this module
true_range = true_range
standard_deviation = standard_deviation

__all__ = [
    "BollingerBands",
    "bollinger_bands",
    "bollinger_percent_b",
    "_bollinger_percent_b_from_bands",
    "atr",
    "true_range",
    "standard_deviation",
    "StreamingEMA",
    "StreamingATR",
]


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
# BATCH — BOLLINGER BANDS
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
    return quantize((data[end_index].close - bb.lower) / band_range)

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
    return quantize((close_price - bb.lower) / band_range)


# =========================================================================
# BATCH — AVERAGE TRUE RANGE (ATR)
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
    atr_val = quantize(atr_val / Decimal(str(count)))

    multiplier = quantize(Decimal('1') / Decimal(str(period)))
    one_minus_mult = Decimal('1') - multiplier

    for i in range(start_index + count, end_index + 1):
        atr_val = quantize(true_range(data, i) * multiplier + atr_val * one_minus_mult)

    return atr_val


# =========================================================================
# STREAMING — EMA
# =========================================================================

class StreamingEMA:
    """Incremental Exponential Moving Average.

    After seeding with SMA over the first *period* bars, each subsequent
    bar is a single multiply-add (O(1)).

    Keyed by period so one instance handles one period.
    """

    __slots__ = ("_period", "_multiplier", "_one_minus", "_ema", "_last_index",
                 "_history", "_seed_computed")

    def __init__(self, period: int) -> None:
        self._period = period
        self._multiplier = Decimal(str(2.0 / (period + 1)))
        self._one_minus = ONE - self._multiplier
        self._ema: Decimal = ZERO
        self._last_index: int = -1
        self._history: list[Decimal] = []
        self._seed_computed: bool = False

    def reset(self) -> None:
        self._ema = ZERO
        self._last_index = -1
        self._history.clear()
        self._seed_computed = False

    def update(self, data: Sequence[PriceData], index: int) -> Decimal:
        """Update EMA to *index*."""
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return ZERO

        start = self._last_index + 1
        for i in range(start, index + 1):
            if not self._seed_computed:
                if i < self._period - 1:
                    # Not enough bars yet; store zero
                    self._history.append(ZERO)
                elif i == self._period - 1:
                    # Seed: SMA of first period bars
                    total = ZERO
                    for j in range(self._period):
                        total += data[j].close
                    self._ema = quantize(total / Decimal(str(self._period)))
                    self._history.append(self._ema)
                    self._seed_computed = True
                else:
                    self._history.append(ZERO)
            else:
                self._ema = quantize(
                    data[i].close * self._multiplier
                    + self._ema * self._one_minus
                )
                self._history.append(self._ema)

        self._last_index = index
        return self._ema

    def get(self, index: int) -> Decimal:
        if 0 <= index < len(self._history):
            return self._history[index]
        return ZERO


# =========================================================================
# STREAMING — ATR
# =========================================================================

class StreamingATR:
    """Incremental Average True Range with Wilder smoothing."""

    __slots__ = ("_period", "_atr", "_last_index", "_history",
                 "_seed_computed", "_seed_sum", "_seed_count", "_multiplier",
                 "_one_minus")

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._multiplier = quantize(ONE / Decimal(str(period)))
        self._one_minus = ONE - self._multiplier
        self._atr: Decimal = ZERO
        self._last_index: int = -1
        self._history: list[Decimal] = []
        self._seed_computed: bool = False
        self._seed_sum: Decimal = ZERO
        self._seed_count: int = 0

    def reset(self) -> None:
        self._atr = ZERO
        self._last_index = -1
        self._history.clear()
        self._seed_computed = False
        self._seed_sum = ZERO
        self._seed_count = 0

    def update(self, data: Sequence[PriceData], index: int) -> Decimal:
        """Update ATR to *index*."""
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return ZERO

        start = self._last_index + 1
        for i in range(start, index + 1):
            if i < 1:
                self._history.append(ZERO)
                continue

            tr = true_range(data, i)

            if not self._seed_computed:
                self._seed_sum += tr
                self._seed_count += 1
                if self._seed_count >= self._period:
                    self._atr = quantize(
                        self._seed_sum / Decimal(str(self._seed_count))
                    )
                    self._seed_computed = True
                    self._history.append(self._atr)
                else:
                    self._history.append(ZERO)
            else:
                self._atr = quantize(
                    tr * self._multiplier + self._atr * self._one_minus
                )
                self._history.append(self._atr)

        self._last_index = index
        return self._atr

    def get(self, index: int) -> Decimal:
        if 0 <= index < len(self._history):
            return self._history[index]
        return ZERO
