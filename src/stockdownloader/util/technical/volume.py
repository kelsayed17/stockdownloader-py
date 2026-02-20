"""Volume indicators: OBV, average volume."""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.technical._helpers import _quantize

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

__all__ = [
    "obv",
    "is_obv_rising",
    "average_volume",
]


# =========================================================================
# ON-BALANCE VOLUME (OBV)
# =========================================================================

def obv(data: Sequence[PriceData], end_index: int) -> Decimal:
    """Calculate On-Balance Volume.

    If close > prevClose: OBV += volume; if close < prevClose: OBV -= volume.
    """
    obv_val = ZERO
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
# AVERAGE VOLUME
# =========================================================================

def average_volume(
    data: Sequence[PriceData], end_index: int, period: int
) -> Decimal:
    """Calculate average volume over a period."""
    if end_index < period - 1:
        return ZERO
    total = ZERO
    for i in range(end_index - period + 1, end_index + 1):
        total += Decimal(str(data[i].volume))
    return _quantize(total / Decimal(str(period)))
