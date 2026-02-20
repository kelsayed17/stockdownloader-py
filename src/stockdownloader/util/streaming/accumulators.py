"""Streaming accumulators: OBV and EMA."""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import ONE, ZERO
from stockdownloader.util.streaming import _quantize

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData


class StreamingOBV:
    """Incremental On-Balance Volume.

    Instead of scanning from bar 0 on every call, this maintains a running
    total and updates with each new bar in O(1).
    """

    __slots__ = ("_obv", "_last_index", "_history")

    def __init__(self) -> None:
        self._obv: Decimal = ZERO
        self._last_index: int = -1
        # Store OBV at each index for lookback queries (is_obv_rising)
        self._history: list[Decimal] = []

    def reset(self) -> None:
        self._obv = ZERO
        self._last_index = -1
        self._history.clear()

    def update(self, data: Sequence[PriceData], index: int) -> Decimal:
        """Update OBV to *index*, filling any gaps from last_index+1."""
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return ZERO

        # Fill forward from last computed index
        start = self._last_index + 1
        for i in range(start, index + 1):
            if i == 0:
                self._history.append(ZERO)
            else:
                if data[i].close > data[i - 1].close:
                    self._obv += Decimal(str(data[i].volume))
                elif data[i].close < data[i - 1].close:
                    self._obv -= Decimal(str(data[i].volume))
                self._history.append(self._obv)

        self._last_index = index
        return self._obv

    def get(self, index: int) -> Decimal:
        """Get OBV at a previously computed index."""
        if 0 <= index < len(self._history):
            return self._history[index]
        return ZERO

    def is_rising(self, index: int, lookback: int) -> bool:
        """Whether OBV is rising over *lookback* bars."""
        if index < lookback or index >= len(self._history):
            return False
        return self._history[index] > self._history[index - lookback]


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
                    self._ema = _quantize(total / Decimal(str(self._period)))
                    self._history.append(self._ema)
                    self._seed_computed = True
                else:
                    self._history.append(ZERO)
            else:
                self._ema = _quantize(
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
