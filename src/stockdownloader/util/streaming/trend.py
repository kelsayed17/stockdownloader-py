"""Streaming trend indicators: ADX/DMI and Parabolic SAR."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.math import HUNDRED, ZERO
from stockdownloader.util.streaming import _quantize, _true_range

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData


@dataclass(slots=True)
class ADXState:
    """Internal state for incremental ADX computation."""
    smooth_plus_dm: Decimal = ZERO
    smooth_minus_dm: Decimal = ZERO
    smooth_tr: Decimal = ZERO
    adx: Decimal = ZERO
    plus_di: Decimal = ZERO
    minus_di: Decimal = ZERO


class StreamingADX:
    """Incremental ADX with Wilder-smoothed +DI/-DI."""

    __slots__ = ("_period", "_state", "_last_index", "_history",
                 "_dx_buffer", "_seed_computed", "_seed_count")

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._state = ADXState()
        self._last_index: int = -1
        self._history: list[tuple[Decimal, Decimal, Decimal]] = []  # (adx, +di, -di)
        self._dx_buffer: deque[Decimal] = deque(maxlen=period)
        self._seed_computed: bool = False
        self._seed_count: int = 0

    def reset(self) -> None:
        self._state = ADXState()
        self._last_index = -1
        self._history.clear()
        self._dx_buffer = deque(maxlen=self._period)
        self._seed_computed = False
        self._seed_count = 0

    def update(
        self, data: Sequence[PriceData], index: int
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Update ADX to *index*. Returns ``(adx, plus_di, minus_di)``."""
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return (ZERO, ZERO, ZERO)

        start = self._last_index + 1
        period_bd = Decimal(str(self._period))

        for i in range(start, index + 1):
            if i < 1:
                self._history.append((ZERO, ZERO, ZERO))
                continue

            high = data[i].high
            low = data[i].low
            prev_high = data[i - 1].high
            prev_low = data[i - 1].low

            plus_dm = high - prev_high
            minus_dm = prev_low - low

            cur_plus = ZERO
            cur_minus = ZERO
            if plus_dm > ZERO and plus_dm > minus_dm:
                cur_plus = plus_dm
            if minus_dm > ZERO and minus_dm > plus_dm:
                cur_minus = minus_dm

            tr = _true_range(data, i)

            if not self._seed_computed:
                self._state.smooth_plus_dm += cur_plus
                self._state.smooth_minus_dm += cur_minus
                self._state.smooth_tr += tr
                self._seed_count += 1

                if self._seed_count >= self._period:
                    self._seed_computed = True
                    # Compute initial DI values
                    if self._state.smooth_tr != ZERO:
                        self._state.plus_di = _quantize(
                            self._state.smooth_plus_dm / self._state.smooth_tr
                        ) * HUNDRED
                        self._state.minus_di = _quantize(
                            self._state.smooth_minus_dm / self._state.smooth_tr
                        ) * HUNDRED
                    self._history.append((
                        ZERO, self._state.plus_di, self._state.minus_di
                    ))
                else:
                    self._history.append((ZERO, ZERO, ZERO))
            else:
                # Wilder smoothing
                s = self._state
                s.smooth_plus_dm = (
                    s.smooth_plus_dm - _quantize(s.smooth_plus_dm / period_bd)
                    + cur_plus
                )
                s.smooth_minus_dm = (
                    s.smooth_minus_dm - _quantize(s.smooth_minus_dm / period_bd)
                    + cur_minus
                )
                s.smooth_tr = (
                    s.smooth_tr - _quantize(s.smooth_tr / period_bd) + tr
                )

                if s.smooth_tr != ZERO:
                    s.plus_di = _quantize(s.smooth_plus_dm / s.smooth_tr) * HUNDRED
                    s.minus_di = _quantize(s.smooth_minus_dm / s.smooth_tr) * HUNDRED
                    di_sum = s.plus_di + s.minus_di
                    if di_sum != ZERO:
                        dx = _quantize(abs(s.plus_di - s.minus_di) / di_sum) * HUNDRED
                        self._dx_buffer.append(dx)

                # ADX = average of last `period` DX values
                if self._dx_buffer:
                    adx_period = min(self._period, len(self._dx_buffer))
                    total = ZERO
                    for j in range(
                        len(self._dx_buffer) - adx_period, len(self._dx_buffer)
                    ):
                        total += self._dx_buffer[j]
                    s.adx = _quantize(total / Decimal(str(adx_period)))

                self._history.append((s.adx, s.plus_di, s.minus_di))

        self._last_index = index
        if 0 <= index < len(self._history):
            return self._history[index]
        return (ZERO, ZERO, ZERO)

    def get(self, index: int) -> tuple[Decimal, Decimal, Decimal]:
        if 0 <= index < len(self._history):
            return self._history[index]
        return (ZERO, ZERO, ZERO)


class StreamingSAR:
    """Incremental Parabolic SAR tracking."""

    __slots__ = ("_af_start", "_af_step", "_af_max",
                 "_sar", "_ep", "_af", "_is_up",
                 "_last_index", "_history")

    def __init__(
        self,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.20,
    ) -> None:
        self._af_start = af_start
        self._af_step = af_step
        self._af_max = af_max
        self._sar: float = 0.0
        self._ep: float = 0.0
        self._af: float = af_start
        self._is_up: bool = True
        self._last_index: int = -1
        self._history: list[Decimal] = []

    def reset(self) -> None:
        self._sar = 0.0
        self._ep = 0.0
        self._af = self._af_start
        self._is_up = True
        self._last_index = -1
        self._history.clear()

    def update(self, data: Sequence[PriceData], index: int) -> Decimal:
        """Update SAR to *index*."""
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return data[0].low if data else ZERO

        start = self._last_index + 1
        for i in range(start, index + 1):
            if i < 2:
                if i == 0:
                    self._history.append(data[0].low)
                elif i == 1:
                    self._is_up = data[1].close > data[0].close
                    self._sar = float(data[0].low) if self._is_up else float(data[0].high)
                    self._ep = float(data[1].high) if self._is_up else float(data[1].low)
                    self._af = self._af_step
                    self._history.append(
                        Decimal(str(self._sar)).quantize(
                            Decimal("0.0001"), rounding=ROUND_HALF_UP
                        )
                    )
                continue

            high = float(data[i].high)
            low = float(data[i].low)

            self._sar = self._sar + self._af * (self._ep - self._sar)

            if self._is_up:
                self._sar = min(
                    self._sar, float(data[i - 1].low), float(data[i - 2].low)
                )
                if low < self._sar:
                    self._is_up = False
                    self._sar = self._ep
                    self._ep = low
                    self._af = self._af_step
                else:
                    if high > self._ep:
                        self._ep = high
                        self._af = min(self._af + self._af_step, self._af_max)
            else:
                self._sar = max(
                    self._sar, float(data[i - 1].high), float(data[i - 2].high)
                )
                if high > self._sar:
                    self._is_up = True
                    self._sar = self._ep
                    self._ep = high
                    self._af = self._af_step
                else:
                    if low < self._ep:
                        self._ep = low
                        self._af = min(self._af + self._af_step, self._af_max)

            self._history.append(
                Decimal(str(self._sar)).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
            )

        self._last_index = index
        if 0 <= index < len(self._history):
            return self._history[index]
        return data[0].low if data else ZERO

    def get(self, index: int) -> Decimal:
        if 0 <= index < len(self._history):
            return self._history[index]
        return ZERO

    def is_bullish(self, data: Sequence[PriceData], index: int) -> bool:
        """Whether SAR indicates uptrend (SAR below price) at *index*."""
        sar_val = self.get(index)
        if index < 0 or index >= len(data):
            return False
        return data[index].close > sar_val
