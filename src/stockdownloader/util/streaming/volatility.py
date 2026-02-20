"""Streaming volatility indicators: ATR (Wilder smoothing)."""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import ONE, ZERO
from stockdownloader.util.streaming._base import _quantize, _true_range

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData


class StreamingATR:
    """Incremental Average True Range with Wilder smoothing."""

    __slots__ = ("_period", "_atr", "_last_index", "_history",
                 "_seed_computed", "_seed_sum", "_seed_count", "_multiplier",
                 "_one_minus")

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._multiplier = _quantize(ONE / Decimal(str(period)))
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

            tr = _true_range(data, i)

            if not self._seed_computed:
                self._seed_sum += tr
                self._seed_count += 1
                if self._seed_count >= self._period:
                    self._atr = _quantize(
                        self._seed_sum / Decimal(str(self._seed_count))
                    )
                    self._seed_computed = True
                    self._history.append(self._atr)
                else:
                    self._history.append(ZERO)
            else:
                self._atr = _quantize(
                    tr * self._multiplier + self._atr * self._one_minus
                )
                self._history.append(self._atr)

        self._last_index = index
        return self._atr

    def get(self, index: int) -> Decimal:
        if 0 <= index < len(self._history):
            return self._history[index]
        return ZERO
