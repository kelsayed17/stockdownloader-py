"""Shared calculations for Simple Moving Average (SMA) and Exponential Moving Average (EMA).

Used by multiple strategy implementations to avoid duplication.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

SCALE = 10


def sma(data: Sequence[PriceData], end_index: int, period: int) -> Decimal:
    """Calculate the Simple Moving Average of close prices.

    Args:
        data: List of ``PriceData`` objects.
        end_index: The last index (inclusive) of the window.
        period: Number of bars in the average.

    Returns:
        The SMA as a ``Decimal`` rounded to *SCALE* decimal places.
    """
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
    """
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


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _quantize(value: Decimal) -> Decimal:
    """Quantize *value* to *SCALE* decimal places."""
    return value.quantize(Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP)
