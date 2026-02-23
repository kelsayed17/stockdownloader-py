"""Shared helpers for the unified indicators package.

Contains moving-average calculations, true-range, standard deviation,
crossover detection, and VWAP core logic used by multiple indicator
modules.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.math import ZERO, TWO, quantize

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

# =========================================================================
# Constants
# =========================================================================

SCALE = 10

# =========================================================================
# Moving-average calculations
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
    return quantize(total / Decimal(str(period)))


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
    ema_val = quantize(total / Decimal(str(period)))

    for i in range(start_index + period, end_index + 1):
        ema_val = quantize(data[i].close * multiplier + ema_val * one_minus_multiplier)

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
    mean = quantize(total / Decimal(str(period)))

    sum_sq_diff = ZERO
    for i in range(end_index - period + 1, end_index + 1):
        diff = data[i].close - mean
        sum_sq_diff += diff * diff

    variance = max(0.0, float(quantize(sum_sq_diff / Decimal(str(period)))))
    return Decimal(str(math.sqrt(variance))).quantize(
        Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP
    )


# =========================================================================
# Period helpers
# =========================================================================


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

    return quantize((highest + lowest) / TWO)


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
# Session VWAP core (shared by volume.py and intraday.py)
# =========================================================================


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
        tp = quantize(
            (data[i].high + data[i].low + data[i].close) / Decimal('3')
        )
        vol = Decimal(str(data[i].volume))
        tps.append(tp)
        vols.append(vol)
        sum_tpv += tp * vol
        sum_vol += vol

    if sum_vol == ZERO:
        return ZERO, ZERO

    vwap_val = quantize(sum_tpv / sum_vol)

    # Weighted standard deviation
    sum_var = ZERO
    for tp, vol in zip(tps, vols, strict=True):
        diff = tp - vwap_val
        sum_var += diff * diff * vol

    variance = float(sum_var / sum_vol)
    std_float = math.sqrt(max(0.0, variance))
    std_val = quantize(Decimal(str(std_float)))

    return vwap_val, std_val
