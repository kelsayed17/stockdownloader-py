"""Shared helpers for streaming (incremental) indicators."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

SCALE = 10


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP)


def _true_range(data: Sequence[PriceData], index: int) -> Decimal:
    """True Range for a single bar."""
    if index <= 0:
        return data[index].high - data[index].low
    high = data[index].high
    low = data[index].low
    prev_close = data[index - 1].close
    return max(high - low, abs(high - prev_close), abs(low - prev_close))
