"""Trend indicators: ADX/DMI, Parabolic SAR, Ichimoku, VWAP (lookback), Fibonacci, Support/Resistance.

Section 1 — Batch indicators
Section 2 — Streaming indicators (StreamingADX, StreamingSAR)

Note: Session VWAP and related functions live in volume.py, not here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.core.math import HUNDRED, ONE, TWO, ZERO, quantize
from stockdownloader.indicators.core import (
    _deduplicate_levels,
    _period_midpoint,
    true_range,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.core.models.price import PriceData

__all__ = [
    # Batch
    "ADXResult",
    "IchimokuCloud",
    "FibonacciLevels",
    "SupportResistance",
    "adx",
    "parabolic_sar",
    "is_sar_bullish",
    "vwap",
    "fibonacci_retracement",
    "ichimoku",
    "support_resistance",
    # Streaming
    "ADXState",
    "StreamingADX",
    "StreamingSAR",
]


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass(frozen=True, slots=True)
class ADXResult:
    """Average Directional Index result: ADX value, +DI, and -DI."""
    adx: Decimal
    plus_di: Decimal
    minus_di: Decimal

@dataclass(frozen=True, slots=True)
class IchimokuCloud:
    """Ichimoku Cloud components."""
    tenkan_sen: Decimal       # Conversion Line (9-period)
    kijun_sen: Decimal        # Base Line (26-period)
    senkou_span_a: Decimal    # Leading Span A
    senkou_span_b: Decimal    # Leading Span B (52-period)
    chikou_span: Decimal      # Lagging Span
    price_above_cloud: bool

@dataclass(frozen=True, slots=True)
class FibonacciLevels:
    """Fibonacci retracement levels derived from a swing high/low."""
    high: Decimal
    low: Decimal
    level_236: Decimal
    level_382: Decimal
    level_500: Decimal
    level_618: Decimal
    level_786: Decimal

@dataclass(frozen=True, slots=True)
class SupportResistance:
    """Detected support and resistance price levels."""
    support_levels: list[Decimal]
    resistance_levels: list[Decimal]


# =========================================================================
# BATCH — AVERAGE DIRECTIONAL INDEX (ADX)
# =========================================================================

def adx(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> ADXResult:
    """Calculate ADX with +DI and -DI."""
    if end_index < period * 2:
        return ADXResult(ZERO, ZERO, ZERO)

    start_idx = max(1, end_index - period * 3)
    period_bd = Decimal(str(period))

    # Seed with sum of first *period* values
    smooth_plus_dm = ZERO
    smooth_minus_dm = ZERO
    smooth_tr = ZERO

    seed_end = min(start_idx + period, end_index + 1)
    for i in range(start_idx, seed_end):
        if i <= 0:
            continue
        high = data[i].high
        low = data[i].low
        prev_high = data[i - 1].high
        prev_low = data[i - 1].low

        plus_dm = high - prev_high
        minus_dm = prev_low - low

        if plus_dm > ZERO and plus_dm > minus_dm:
            smooth_plus_dm += plus_dm
        if minus_dm > ZERO and minus_dm > plus_dm:
            smooth_minus_dm += minus_dm
        smooth_tr += true_range(data, i)

    # Wilder smoothing for remaining bars
    dx_values: list[Decimal] = []

    for i in range(seed_end, end_index + 1):
        if i <= 0:
            continue
        high = data[i].high
        low = data[i].low
        prev_high = data[i - 1].high
        prev_low = data[i - 1].low

        plus_dm = high - prev_high
        minus_dm = prev_low - low

        cur_plus_dm = ZERO
        cur_minus_dm = ZERO

        if plus_dm > ZERO and plus_dm > minus_dm:
            cur_plus_dm = plus_dm
        if minus_dm > ZERO and minus_dm > plus_dm:
            cur_minus_dm = minus_dm

        smooth_plus_dm = smooth_plus_dm - quantize(smooth_plus_dm / period_bd) + cur_plus_dm
        smooth_minus_dm = smooth_minus_dm - quantize(smooth_minus_dm / period_bd) + cur_minus_dm
        smooth_tr = smooth_tr - quantize(smooth_tr / period_bd) + true_range(data, i)

        if smooth_tr != ZERO:
            p_di = quantize(smooth_plus_dm / smooth_tr) * HUNDRED
            m_di = quantize(smooth_minus_dm / smooth_tr) * HUNDRED
            di_sum = p_di + m_di
            if di_sum != ZERO:
                dx = quantize(abs(p_di - m_di) / di_sum) * HUNDRED
                dx_values.append(dx)

    # ADX = average of DX values
    adx_value = ZERO
    if dx_values:
        adx_period = min(period, len(dx_values))
        total = ZERO
        for i in range(len(dx_values) - adx_period, len(dx_values)):
            total += dx_values[i]
        adx_value = quantize(total / Decimal(str(adx_period)))

    # Current +DI and -DI
    plus_di = ZERO
    minus_di = ZERO
    if smooth_tr != ZERO:
        plus_di = quantize(smooth_plus_dm / smooth_tr) * HUNDRED
        minus_di = quantize(smooth_minus_dm / smooth_tr) * HUNDRED

    return ADXResult(adx_value, plus_di, minus_di)


# =========================================================================
# BATCH — PARABOLIC SAR
# =========================================================================

def parabolic_sar(
    data: Sequence[PriceData],
    end_index: int,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> Decimal:
    """Calculate Parabolic SAR at the given index.

    Uses the standard Wilder method with configurable acceleration factors.

    Parameters
    ----------
    af_start:
        Initial acceleration factor (default 0.02).
    af_step:
        Acceleration factor increment per new extreme point (default 0.02).
    af_max:
        Maximum acceleration factor (default 0.20).
    """
    if end_index < 2:
        return data[0].low

    af = af_start
    max_af = af_max

    is_up_trend = data[1].close > data[0].close
    sar = float(data[0].low) if is_up_trend else float(data[0].high)
    ep = float(data[1].high) if is_up_trend else float(data[1].low)

    for i in range(2, end_index + 1):
        high = float(data[i].high)
        low = float(data[i].low)

        sar = sar + af * (ep - sar)

        if is_up_trend:
            sar = min(sar, float(data[i - 1].low), float(data[i - 2].low))
            if low < sar:
                # Flip to downtrend
                is_up_trend = False
                sar = ep
                ep = low
                af = af_step
            else:
                if high > ep:
                    ep = high
                    af = min(af + af_step, max_af)
        else:
            sar = max(sar, float(data[i - 1].high), float(data[i - 2].high))
            if high > sar:
                # Flip to uptrend
                is_up_trend = True
                sar = ep
                ep = high
                af = af_step
            else:
                if low < ep:
                    ep = low
                    af = min(af + af_step, max_af)

    return Decimal(str(sar)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

def is_sar_bullish(
    data: Sequence[PriceData],
    end_index: int,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> bool:
    """Return ``True`` if SAR indicates uptrend (SAR below price)."""
    sar_val = parabolic_sar(data, end_index, af_start, af_step, af_max)
    return data[end_index].close > sar_val


# =========================================================================
# BATCH — VWAP (Volume-Weighted Average Price, lookback-based)
# =========================================================================

def vwap(
    data: Sequence[PriceData], end_index: int, lookback: int = 20
) -> Decimal:
    """Calculate VWAP over a lookback period.

    VWAP = Sum(TP * Volume) / Sum(Volume) where TP = (High + Low + Close) / 3.
    """
    start_idx = max(0, end_index - lookback + 1)

    sum_tpv = ZERO
    sum_vol = ZERO

    for i in range(start_idx, end_index + 1):
        tp = (data[i].high + data[i].low + data[i].close) / Decimal('3')
        vol = Decimal(str(data[i].volume))
        sum_tpv += tp * vol
        sum_vol += vol

    if sum_vol == ZERO:
        return ZERO
    return quantize(sum_tpv / sum_vol)


# =========================================================================
# BATCH — FIBONACCI RETRACEMENT
# =========================================================================

def fibonacci_retracement(
    data: Sequence[PriceData], end_index: int, lookback: int = 50
) -> FibonacciLevels:
    """Calculate Fibonacci retracement levels from the swing high/low within a lookback period."""
    start_idx = max(0, end_index - lookback + 1)

    highest = ZERO
    lowest = Decimal("Infinity")

    for i in range(start_idx, end_index + 1):
        if data[i].high > highest:
            highest = data[i].high
        if data[i].low < lowest:
            lowest = data[i].low

    fib_range = highest - lowest

    def _level(ratio: str) -> Decimal:
        return (highest - fib_range * Decimal(ratio)).quantize(
            Decimal('0.0001'), rounding=ROUND_HALF_UP
        )

    return FibonacciLevels(
        high=highest,
        low=lowest,
        level_236=_level('0.236'),
        level_382=_level('0.382'),
        level_500=_level('0.500'),
        level_618=_level('0.618'),
        level_786=_level('0.786'),
    )


# =========================================================================
# BATCH — ICHIMOKU CLOUD
# =========================================================================

def ichimoku(data: Sequence[PriceData], end_index: int) -> IchimokuCloud:
    """Calculate Ichimoku Cloud components."""
    if end_index < 52:
        return IchimokuCloud(ZERO, ZERO, ZERO, ZERO, ZERO, False)

    tenkan = _period_midpoint(data, end_index, 9)
    kijun = _period_midpoint(data, end_index, 26)
    senkou_a = quantize((tenkan + kijun) / TWO)

    # Senkou Span B uses 52-period midpoint
    senkou_b = _period_midpoint(data, end_index, 52)

    # Chikou Span = current close (plotted 26 periods back)
    chikou = data[end_index].close

    price = data[end_index].close
    above_cloud = price > max(senkou_a, senkou_b)

    return IchimokuCloud(tenkan, kijun, senkou_a, senkou_b, chikou, above_cloud)


# =========================================================================
# BATCH — SUPPORT & RESISTANCE
# =========================================================================

def support_resistance(
    data: Sequence[PriceData],
    end_index: int,
    lookback: int,
    window: int,
) -> SupportResistance:
    """Detect support and resistance levels using swing points.

    A swing high requires *window* bars on each side to be lower.
    A swing low requires *window* bars on each side to be higher.
    """
    start_idx = max(window, end_index - lookback)
    end_limit = min(end_index - window, end_index)

    supports: list[Decimal] = []
    resistances: list[Decimal] = []

    current_price = data[end_index].close

    for i in range(start_idx, end_limit + 1):
        is_swing_high = True
        is_swing_low = True

        for j in range(1, window + 1):
            if i - j < 0 or i + j > end_index:
                is_swing_high = False
                is_swing_low = False
                break
            if data[i].high <= data[i - j].high or data[i].high <= data[i + j].high:
                is_swing_high = False
            if data[i].low >= data[i - j].low or data[i].low >= data[i + j].low:
                is_swing_low = False

        if is_swing_high:
            level = data[i].high
            if level > current_price:
                resistances.append(level)
        if is_swing_low:
            level = data[i].low
            if level < current_price:
                supports.append(level)

    # Sort: supports descending (nearest first), resistances ascending (nearest first)
    supports.sort(reverse=True)
    resistances.sort()

    # Deduplicate levels within 1% of each other
    supports = _deduplicate_levels(supports, current_price)
    resistances = _deduplicate_levels(resistances, current_price)

    return SupportResistance(supports, resistances)


# =========================================================================
# STREAMING — ADX
# =========================================================================

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

            tr = true_range(data, i)

            if not self._seed_computed:
                self._state.smooth_plus_dm += cur_plus
                self._state.smooth_minus_dm += cur_minus
                self._state.smooth_tr += tr
                self._seed_count += 1

                if self._seed_count >= self._period:
                    self._seed_computed = True
                    # Compute initial DI values
                    if self._state.smooth_tr != ZERO:
                        self._state.plus_di = quantize(
                            self._state.smooth_plus_dm / self._state.smooth_tr
                        ) * HUNDRED
                        self._state.minus_di = quantize(
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
                    s.smooth_plus_dm - quantize(s.smooth_plus_dm / period_bd)
                    + cur_plus
                )
                s.smooth_minus_dm = (
                    s.smooth_minus_dm - quantize(s.smooth_minus_dm / period_bd)
                    + cur_minus
                )
                s.smooth_tr = (
                    s.smooth_tr - quantize(s.smooth_tr / period_bd) + tr
                )

                dx = ZERO
                if s.smooth_tr != ZERO:
                    s.plus_di = quantize(s.smooth_plus_dm / s.smooth_tr) * HUNDRED
                    s.minus_di = quantize(s.smooth_minus_dm / s.smooth_tr) * HUNDRED
                    di_sum = s.plus_di + s.minus_di
                    if di_sum != ZERO:
                        dx = quantize(abs(s.plus_di - s.minus_di) / di_sum) * HUNDRED
                        self._dx_buffer.append(dx)

                # ADX: Wilder's smoothing (matches PineScript ta.dmi)
                # Seed: SMA of first `period` DX values
                # Then: ADX = (prev_ADX * (period-1) + DX) / period
                if self._dx_buffer:
                    if len(self._dx_buffer) < self._period:
                        # Still seeding: use SMA of available DX values
                        total = ZERO
                        for v in self._dx_buffer:
                            total += v
                        s.adx = quantize(total / Decimal(str(len(self._dx_buffer))))
                    elif s.adx == ZERO:
                        # Seed complete: initial ADX = SMA of first period DX values
                        total = ZERO
                        for v in self._dx_buffer:
                            total += v
                        s.adx = quantize(total / period_bd)
                    else:
                        # Wilder's smoothing for subsequent bars
                        s.adx = quantize(
                            (s.adx * (period_bd - ONE) + dx) / period_bd
                        )

                self._history.append((s.adx, s.plus_di, s.minus_di))

        self._last_index = index
        if 0 <= index < len(self._history):
            return self._history[index]
        return (ZERO, ZERO, ZERO)

    def get(self, index: int) -> tuple[Decimal, Decimal, Decimal]:
        if 0 <= index < len(self._history):
            return self._history[index]
        return (ZERO, ZERO, ZERO)


# =========================================================================
# STREAMING — PARABOLIC SAR
# =========================================================================

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
