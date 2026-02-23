"""Momentum and volume indicators.

Section 1 — Batch: RSI, MACD, Stochastic, MFI, CCI, Williams %R, ROC, OBV, average volume
Section 2 — Streaming: StreamingRSI, StreamingMACD, StreamingOBV
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.util.math import HUNDRED, ONE, ZERO, quantize
from stockdownloader.util.indicators._core import ema as _ema
from stockdownloader.util.indicators.volatility import StreamingEMA

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from stockdownloader.model.price_data import PriceData

__all__ = [
    # Batch
    "Stochastic",
    "stochastic",
    "rsi",
    "macd_line",
    "macd_signal",
    "_macd_signal_from_lines",
    "macd_histogram",
    "roc",
    "mfi",
    "williams_r",
    "cci",
    "obv",
    "is_obv_rising",
    "average_volume",
    # Streaming
    "StreamingRSI",
    "StreamingMACD",
    "StreamingOBV",
]


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass(frozen=True, slots=True)
class Stochastic:
    """Stochastic Oscillator values: %K and %D."""
    percent_k: Decimal
    percent_d: Decimal


# =========================================================================
# BATCH — STOCHASTIC OSCILLATOR
# =========================================================================

def stochastic(
    data: Sequence[PriceData],
    end_index: int,
    k_period: int = 14,
    d_period: int = 3,
) -> Stochastic:
    """Calculate Stochastic Oscillator (%K and %D)."""
    if end_index < k_period - 1:
        return Stochastic(ZERO, ZERO)

    percent_k = _calculate_percent_k(data, end_index, k_period)

    # %D = SMA of recent %K values
    sum_k = ZERO
    count = 0
    for i in range(max(k_period - 1, end_index - d_period + 1), end_index + 1):
        sum_k += _calculate_percent_k(data, i, k_period)
        count += 1

    percent_d = quantize(sum_k / Decimal(str(count))) if count > 0 else ZERO

    return Stochastic(percent_k, percent_d)

def _calculate_percent_k(
    data: Sequence[PriceData], end_index: int, period: int
) -> Decimal:
    highest_high = ZERO
    lowest_low = Decimal("Infinity")

    for i in range(end_index - period + 1, end_index + 1):
        high = data[i].high
        low = data[i].low
        if high > highest_high:
            highest_high = high
        if low < lowest_low:
            lowest_low = low

    hl_range = highest_high - lowest_low
    if hl_range == ZERO:
        return ZERO

    return quantize((data[end_index].close - lowest_low) / hl_range) * HUNDRED


# =========================================================================
# BATCH — RSI
# =========================================================================

def rsi(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> Decimal:
    """Calculate Relative Strength Index at the given index."""
    if end_index < period + 1:
        return Decimal('50')

    avg_gain = ZERO
    avg_loss = ZERO

    for i in range(end_index - period + 1, end_index + 1):
        change = data[i].close - data[i - 1].close
        if change > ZERO:
            avg_gain += change
        else:
            avg_loss += abs(change)

    period_bd = Decimal(str(period))
    avg_gain = quantize(avg_gain / period_bd)
    avg_loss = quantize(avg_loss / period_bd)

    if avg_loss == ZERO:
        return HUNDRED

    rs = quantize(avg_gain / avg_loss)
    return HUNDRED - (HUNDRED / (Decimal('1') + rs)).quantize(
        Decimal('0.000001'), rounding=ROUND_HALF_UP
    )


# =========================================================================
# BATCH — MACD
# =========================================================================

def macd_line(
    data: Sequence[PriceData],
    end_index: int,
    fast: int = 12,
    slow: int = 26,
) -> Decimal:
    """Calculate MACD line value (fast EMA - slow EMA)."""
    if end_index < slow:
        return ZERO
    return _ema(data, end_index, fast) - _ema(data, end_index, slow)

def macd_signal(
    data: Sequence[PriceData],
    end_index: int,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Decimal:
    """Calculate MACD signal line."""
    if end_index < slow + signal:
        return ZERO

    multiplier = Decimal(str(2.0 / (signal + 1)))
    one_minus_mult = Decimal('1') - multiplier

    start_idx = max(slow, end_index - signal + 1)

    total = ZERO
    count = 0
    for i in range(start_idx, min(start_idx + signal, end_index + 1)):
        total += macd_line(data, i, fast, slow)
        count += 1
    if count == 0:
        return ZERO

    signal_ema = quantize(total / Decimal(str(count)))

    for i in range(start_idx + count, end_index + 1):
        macd_val = macd_line(data, i, fast, slow)
        signal_ema = quantize(macd_val * multiplier + signal_ema * one_minus_mult)

    return signal_ema

def _macd_signal_from_lines(
    data: Sequence[PriceData],
    end_index: int,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    *,
    line_fn: Callable[[Sequence[PriceData], int, int, int], Decimal] | None = None,
) -> Decimal:
    """MACD signal line with injectable MACD-line lookup.

    Identical to :func:`macd_signal` but calls *line_fn* instead of
    :func:`macd_line` for each intermediate MACD-line value.  When
    *line_fn* is ``None`` (default), falls back to :func:`macd_line`.

    The :class:`IndicatorHub` uses this to route sub-lookups through
    its cache, avoiding redundant EMA recomputation.
    """
    _line = line_fn or macd_line

    if end_index < slow + signal:
        return ZERO

    multiplier = Decimal(str(2.0 / (signal + 1)))
    one_minus_mult = Decimal('1') - multiplier

    start_idx = max(slow, end_index - signal + 1)

    total = ZERO
    count = 0
    for i in range(start_idx, min(start_idx + signal, end_index + 1)):
        total += _line(data, i, fast, slow)
        count += 1
    if count == 0:
        return ZERO

    signal_ema = quantize(total / Decimal(str(count)))

    for i in range(start_idx + count, end_index + 1):
        macd_val = _line(data, i, fast, slow)
        signal_ema = quantize(macd_val * multiplier + signal_ema * one_minus_mult)

    return signal_ema

def macd_histogram(
    data: Sequence[PriceData],
    end_index: int,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Decimal:
    """Calculate MACD histogram (MACD line - signal line)."""
    return macd_line(data, end_index, fast, slow) - macd_signal(
        data, end_index, fast, slow, signal
    )


# =========================================================================
# BATCH — RATE OF CHANGE (ROC)
# =========================================================================

def roc(
    data: Sequence[PriceData], end_index: int, period: int = 12
) -> Decimal:
    """Calculate Rate of Change.

    ROC = ((Close - Close_n) / Close_n) * 100
    """
    if end_index < period:
        return ZERO

    current_close = data[end_index].close
    past_close = data[end_index - period].close

    if past_close == ZERO:
        return ZERO

    return quantize((current_close - past_close) / past_close) * HUNDRED


# =========================================================================
# BATCH — MONEY FLOW INDEX (MFI)
# =========================================================================

def mfi(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> Decimal:
    """Calculate Money Flow Index (volume-weighted RSI).

    MFI = 100 - (100 / (1 + Money Ratio))
    where Money Ratio = Positive Money Flow / Negative Money Flow.
    """
    if end_index < period:
        return ZERO

    positive_flow = ZERO
    negative_flow = ZERO

    for i in range(end_index - period + 1, end_index + 1):
        tp = quantize(
            (data[i].high + data[i].low + data[i].close) / Decimal('3')
        )
        prev_tp = quantize(
            (data[i - 1].high + data[i - 1].low + data[i - 1].close) / Decimal('3')
        )

        money_flow = tp * Decimal(str(data[i].volume))

        if tp > prev_tp:
            positive_flow += money_flow
        elif tp < prev_tp:
            negative_flow += money_flow

    if negative_flow == ZERO:
        return HUNDRED if positive_flow > ZERO else ZERO

    money_ratio = quantize(positive_flow / negative_flow)
    return HUNDRED - quantize(HUNDRED / (Decimal('1') + money_ratio))


# =========================================================================
# BATCH — WILLIAMS %R
# =========================================================================

def williams_r(
    data: Sequence[PriceData], end_index: int, period: int = 14
) -> Decimal:
    """Calculate Williams %R.

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    """
    if end_index < period - 1:
        return ZERO

    highest_high = ZERO
    lowest_low = Decimal("Infinity")

    for i in range(end_index - period + 1, end_index + 1):
        if data[i].high > highest_high:
            highest_high = data[i].high
        if data[i].low < lowest_low:
            lowest_low = data[i].low

    hl_range = highest_high - lowest_low
    if hl_range == ZERO:
        return ZERO

    return quantize((highest_high - data[end_index].close) / hl_range) * Decimal('-100')


# =========================================================================
# BATCH — COMMODITY CHANNEL INDEX (CCI)
# =========================================================================

def cci(
    data: Sequence[PriceData], end_index: int, period: int = 20
) -> Decimal:
    """Calculate CCI.

    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    where TP = (High + Low + Close) / 3
    """
    if end_index < period - 1:
        return ZERO

    tp_values: list[Decimal] = []
    sum_tp = ZERO

    for i in range(period):
        idx = end_index - period + 1 + i
        tp = quantize(
            (data[idx].high + data[idx].low + data[idx].close) / Decimal('3')
        )
        tp_values.append(tp)
        sum_tp += tp

    sma_tp = quantize(sum_tp / Decimal(str(period)))

    # Mean deviation
    sum_dev = ZERO
    for tp in tp_values:
        sum_dev += abs(tp - sma_tp)
    mean_dev = quantize(sum_dev / Decimal(str(period)))

    constant = Decimal('0.015')
    divisor = constant * mean_dev

    if divisor == ZERO:
        return ZERO

    current_tp = tp_values[-1]
    return quantize((current_tp - sma_tp) / divisor)


# =========================================================================
# BATCH — ON-BALANCE VOLUME (OBV)
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
# BATCH — AVERAGE VOLUME
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
    return quantize(total / Decimal(str(period)))


# =========================================================================
# STREAMING — RSI
# =========================================================================

class StreamingRSI:
    """Incremental RSI using Wilder smoothing of average gains/losses."""

    __slots__ = ("_period", "_avg_gain", "_avg_loss", "_last_index",
                 "_history", "_seed_computed")

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._avg_gain: Decimal = ZERO
        self._avg_loss: Decimal = ZERO
        self._last_index: int = -1
        self._history: list[Decimal] = []
        self._seed_computed: bool = False

    def reset(self) -> None:
        self._avg_gain = ZERO
        self._avg_loss = ZERO
        self._last_index = -1
        self._history.clear()
        self._seed_computed = False

    def update(self, data: Sequence[PriceData], index: int) -> Decimal:
        """Update RSI to *index*."""
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return Decimal("50")

        start = self._last_index + 1
        period_bd = Decimal(str(self._period))

        for i in range(start, index + 1):
            if i < self._period + 1:
                # Not enough data; return 50
                self._history.append(Decimal("50"))
                continue

            if not self._seed_computed:
                # Seed: simple average of gains/losses over period
                total_gain = ZERO
                total_loss = ZERO
                for j in range(i - self._period + 1, i + 1):
                    change = data[j].close - data[j - 1].close
                    if change > ZERO:
                        total_gain += change
                    else:
                        total_loss += abs(change)
                self._avg_gain = quantize(total_gain / period_bd)
                self._avg_loss = quantize(total_loss / period_bd)
                self._seed_computed = True
            else:
                # Wilder smoothing
                change = data[i].close - data[i - 1].close
                gain = change if change > ZERO else ZERO
                loss = abs(change) if change < ZERO else ZERO
                self._avg_gain = quantize(
                    (self._avg_gain * (period_bd - ONE) + gain) / period_bd
                )
                self._avg_loss = quantize(
                    (self._avg_loss * (period_bd - ONE) + loss) / period_bd
                )

            if self._avg_loss == ZERO:
                rsi_val = HUNDRED
            else:
                rs = quantize(self._avg_gain / self._avg_loss)
                rsi_val = HUNDRED - (HUNDRED / (ONE + rs)).quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )
            self._history.append(rsi_val)

        self._last_index = index
        return self._history[index] if index < len(self._history) else Decimal("50")

    def get(self, index: int) -> Decimal:
        if 0 <= index < len(self._history):
            return self._history[index]
        return Decimal("50")


# =========================================================================
# STREAMING — MACD
# =========================================================================

class StreamingMACD:
    """Incremental MACD using streaming EMA for fast and slow lines.

    Also maintains a streaming EMA of the MACD line as the signal line.
    """

    __slots__ = ("_fast_ema", "_slow_ema", "_fast_period", "_slow_period",
                 "_signal_period", "_signal_mult", "_signal_one_minus",
                 "_signal_ema", "_signal_seeded", "_signal_seed_sum",
                 "_signal_seed_count", "_last_index",
                 "_line_history", "_signal_history")

    def __init__(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> None:
        self._fast_period = fast
        self._slow_period = slow
        self._signal_period = signal
        self._fast_ema = StreamingEMA(fast)
        self._slow_ema = StreamingEMA(slow)
        self._signal_mult = Decimal(str(2.0 / (signal + 1)))
        self._signal_one_minus = ONE - self._signal_mult
        self._signal_ema: Decimal = ZERO
        self._signal_seeded: bool = False
        self._signal_seed_sum: Decimal = ZERO
        self._signal_seed_count: int = 0
        self._last_index: int = -1
        self._line_history: list[Decimal] = []
        self._signal_history: list[Decimal] = []

    def reset(self) -> None:
        self._fast_ema.reset()
        self._slow_ema.reset()
        self._signal_ema = ZERO
        self._signal_seeded = False
        self._signal_seed_sum = ZERO
        self._signal_seed_count = 0
        self._last_index = -1
        self._line_history.clear()
        self._signal_history.clear()

    def update(
        self, data: Sequence[PriceData], index: int
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Update MACD to *index*. Returns ``(line, signal, histogram)``."""
        if index <= self._last_index:
            if 0 <= index < len(self._line_history):
                line = self._line_history[index]
                sig = self._signal_history[index]
                return line, sig, line - sig
            return (ZERO, ZERO, ZERO)

        start = self._last_index + 1
        for i in range(start, index + 1):
            fast_val = self._fast_ema.update(data, i)
            slow_val = self._slow_ema.update(data, i)

            if i < self._slow_period - 1:
                self._line_history.append(ZERO)
                self._signal_history.append(ZERO)
                continue

            macd_line_val = fast_val - slow_val
            self._line_history.append(macd_line_val)

            # Signal line: EMA of MACD line, seeded from the first
            # ``signal_period`` valid MACD values.
            if not self._signal_seeded:
                self._signal_seed_sum += macd_line_val
                self._signal_seed_count += 1
                if self._signal_seed_count >= self._signal_period:
                    self._signal_ema = quantize(
                        self._signal_seed_sum
                        / Decimal(str(self._signal_seed_count))
                    )
                    self._signal_seeded = True
                    self._signal_history.append(self._signal_ema)
                else:
                    self._signal_history.append(ZERO)
            else:
                self._signal_ema = quantize(
                    macd_line_val * self._signal_mult
                    + self._signal_ema * self._signal_one_minus
                )
                self._signal_history.append(self._signal_ema)

        self._last_index = index
        if 0 <= index < len(self._line_history):
            line = self._line_history[index]
            sig = self._signal_history[index]
            return line, sig, line - sig
        return (ZERO, ZERO, ZERO)

    def get_line(self, index: int) -> Decimal:
        if 0 <= index < len(self._line_history):
            return self._line_history[index]
        return ZERO

    def get_signal(self, index: int) -> Decimal:
        if 0 <= index < len(self._signal_history):
            return self._signal_history[index]
        return ZERO

    def get_histogram(self, index: int) -> Decimal:
        return self.get_line(index) - self.get_signal(index)


# =========================================================================
# STREAMING — OBV
# =========================================================================

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
