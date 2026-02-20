"""Streaming (incremental) indicator computation for backtesting.

The raw indicator functions in :mod:`technical_indicators` recompute from
scratch on every call (O(n) per bar, yielding O(n^2) per backtest).  This
module provides an incremental layer that maintains running state and updates
in O(1) per bar for indicators that support it.

Designed to sit *inside* :class:`IndicatorHub` as an acceleration layer:
when a backtest requests indicators sequentially bar-by-bar, the hub
delegates to these streaming accumulators instead of recomputing from
scratch each time.

Supported streaming indicators
-------------------------------
- **OBV** — cumulative addition/subtraction of volume (O(1) per bar)
- **EMA** — standard smoothing update (O(1) per bar after seed)
- **ATR** — Wilder smoothing update (O(1) per bar after seed)
- **ADX** — Wilder smoothing of +DM/-DM/TR and DX averaging (O(1) per bar)
- **RSI** — Wilder smoothing of avg gain / avg loss (O(1) per bar)
- **Parabolic SAR** — stateful uptrend/downtrend tracking (O(1) per bar)
- **Session VWAP** — incremental session VWAP with weighted std dev (O(1) per bar)

All values are numerically identical to the raw functions when evaluated
sequentially from bar 0.  The streaming state auto-resets when the data
reference changes.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from stockdownloader.util.big_decimal_math import HUNDRED, ONE, ZERO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData

SCALE = 10

def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal(10) ** -SCALE, rounding=ROUND_HALF_UP)

# =========================================================================
# STREAMING OBV
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

# =========================================================================
# STREAMING EMA
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

# =========================================================================
# STREAMING ATR (Wilder smoothing)
# =========================================================================

def _true_range(data: Sequence[PriceData], index: int) -> Decimal:
    """True Range for a single bar."""
    if index <= 0:
        return data[index].high - data[index].low
    high = data[index].high
    low = data[index].low
    prev_close = data[index - 1].close
    return max(high - low, abs(high - prev_close), abs(low - prev_close))

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

# =========================================================================
# STREAMING RSI (Wilder smoothing)
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
                self._avg_gain = _quantize(total_gain / period_bd)
                self._avg_loss = _quantize(total_loss / period_bd)
                self._seed_computed = True
            else:
                # Wilder smoothing
                change = data[i].close - data[i - 1].close
                gain = change if change > ZERO else ZERO
                loss = abs(change) if change < ZERO else ZERO
                self._avg_gain = _quantize(
                    (self._avg_gain * (period_bd - ONE) + gain) / period_bd
                )
                self._avg_loss = _quantize(
                    (self._avg_loss * (period_bd - ONE) + loss) / period_bd
                )

            if self._avg_loss == ZERO:
                rsi_val = HUNDRED
            else:
                rs = _quantize(self._avg_gain / self._avg_loss)
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
# STREAMING ADX
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

# =========================================================================
# STREAMING PARABOLIC SAR
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

# =========================================================================
# STREAMING MACD (composed from streaming EMAs)
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

            macd_line = fast_val - slow_val
            self._line_history.append(macd_line)

            # Signal line: EMA of MACD line, seeded from the first
            # ``signal_period`` valid MACD values.
            if not self._signal_seeded:
                self._signal_seed_sum += macd_line
                self._signal_seed_count += 1
                if self._signal_seed_count >= self._signal_period:
                    self._signal_ema = _quantize(
                        self._signal_seed_sum
                        / Decimal(str(self._signal_seed_count))
                    )
                    self._signal_seeded = True
                    self._signal_history.append(self._signal_ema)
                else:
                    self._signal_history.append(ZERO)
            else:
                self._signal_ema = _quantize(
                    macd_line * self._signal_mult
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
# STREAMING SESSION VWAP
# =========================================================================


class StreamingSessionVWAP:
    """Incremental Session VWAP with weighted standard deviation.

    Instead of scanning all session bars from scratch at each index (O(k)
    per bar, O(k^2) per session), this maintains running sums and updates
    in O(1) per bar.

    Uses the identity for weighted variance:

        Var = (Sum(TP^2 * Vol) / Sum(Vol)) - VWAP^2

    so only three running accumulators are needed, no stored bar lists.

    Results are **bit-identical** to
    :func:`~stockdownloader.util.technical_indicators._compute_session_vwap_core`
    when evaluated sequentially.
    """

    __slots__ = (
        "_sum_tpv", "_sum_vol", "_sum_tp2v",
        "_current_session", "_last_index",
        "_history",  # list of (vwap, std_dev) tuples per index
    )

    def __init__(self) -> None:
        self._sum_tpv: Decimal = ZERO
        self._sum_vol: Decimal = ZERO
        self._sum_tp2v: Decimal = ZERO
        self._current_session: str = ""
        self._last_index: int = -1
        self._history: list[tuple[Decimal, Decimal]] = []

    def reset(self) -> None:
        self._sum_tpv = ZERO
        self._sum_vol = ZERO
        self._sum_tp2v = ZERO
        self._current_session = ""
        self._last_index = -1
        self._history.clear()

    def update(
        self, data: Sequence[PriceData], index: int,
    ) -> tuple[Decimal, Decimal]:
        """Return ``(vwap, std_dev)`` at *index*, updating incrementally.

        Fills forward from ``_last_index + 1`` to *index* if there are gaps.
        Resets accumulators on session boundary (new trading day).
        """
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return ZERO, ZERO

        start = self._last_index + 1
        for i in range(start, index + 1):
            bar = data[i]
            session = bar.date[:10]

            # Session boundary detection — reset accumulators
            if session != self._current_session:
                self._sum_tpv = ZERO
                self._sum_vol = ZERO
                self._sum_tp2v = ZERO
                self._current_session = session

            # Typical price
            tp = _quantize(
                (bar.high + bar.low + bar.close) / Decimal("3")
            )
            vol = Decimal(str(bar.volume))

            # Update running sums
            self._sum_tpv += tp * vol
            self._sum_vol += vol
            self._sum_tp2v += tp * tp * vol

            # Compute VWAP and std dev
            if self._sum_vol == ZERO:
                vwap_val = ZERO
                std_val = ZERO
            else:
                vwap_val = _quantize(self._sum_tpv / self._sum_vol)

                # Weighted variance: E[X^2] - (E[X])^2
                mean_tp2 = float(self._sum_tp2v / self._sum_vol)
                vwap_f = float(vwap_val)
                variance = max(0.0, mean_tp2 - vwap_f * vwap_f)
                std_val = _quantize(Decimal(str(math.sqrt(variance))))

            self._history.append((vwap_val, std_val))

        self._last_index = index
        return self._history[index]


# =========================================================================
# STREAMING ANCHORED VWAP
# =========================================================================


class StreamingAnchoredVWAP:
    """Incremental Anchored VWAP that resets on event dates, not sessions.

    Mirrors :class:`StreamingSessionVWAP` but resets accumulators when the
    most recent anchor date changes (e.g., a new FOMC meeting) rather than
    on every new trading day.

    The ``_valid`` flag stays ``False`` until the first anchor date is
    encountered in the data, at which point accumulation begins.

    Results are **bit-identical** to
    :func:`~stockdownloader.util.technical_indicators._compute_session_vwap_core`
    when the anchor start index corresponds to the core function's start.
    """

    __slots__ = (
        "_sum_tpv", "_sum_vol", "_sum_tp2v",
        "_current_anchor", "_last_index",
        "_last_date",      # cache for per-day anchor lookups
        "_history",        # list of (vwap, std_dev) tuples per index
        "_anchor_type",
        "_valid",          # False until the first anchor is hit
    )

    def __init__(self, anchor_type: str = "fomc") -> None:
        self._sum_tpv: Decimal = ZERO
        self._sum_vol: Decimal = ZERO
        self._sum_tp2v: Decimal = ZERO
        self._current_anchor: str = ""
        self._last_index: int = -1
        self._last_date: str = ""
        self._history: list[tuple[Decimal, Decimal]] = []
        self._anchor_type: str = anchor_type
        self._valid: bool = False

    def reset(self) -> None:
        self._sum_tpv = ZERO
        self._sum_vol = ZERO
        self._sum_tp2v = ZERO
        self._current_anchor = ""
        self._last_index = -1
        self._last_date = ""
        self._history.clear()
        self._valid = False

    @property
    def current_anchor(self) -> str:
        """The anchor date string currently in use (empty if none yet)."""
        return self._current_anchor

    @property
    def valid(self) -> bool:
        """Whether the accumulator has reached its first anchor."""
        return self._valid

    def update(
        self, data: Sequence[PriceData], index: int,
    ) -> tuple[Decimal, Decimal]:
        """Return ``(avwap, std_dev)`` at *index*, updating incrementally.

        Fills forward from ``_last_index + 1`` to *index* if there are gaps.
        Resets accumulators when the anchor date changes (new event).
        Returns ``(ZERO, ZERO)`` while ``_valid`` is ``False``.
        """
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return ZERO, ZERO

        # Lazy import to avoid circular dependency
        from stockdownloader.util.event_calendar import get_anchor_date

        start = self._last_index + 1
        for i in range(start, index + 1):
            bar = data[i]
            trading_date = bar.date[:10]

            # Only re-lookup anchor when the trading date changes
            if trading_date != self._last_date:
                anchor = get_anchor_date(trading_date, self._anchor_type)
                if anchor is not None and anchor != self._current_anchor:
                    # New anchor event — reset accumulators
                    self._sum_tpv = ZERO
                    self._sum_vol = ZERO
                    self._sum_tp2v = ZERO
                    self._current_anchor = anchor
                    self._valid = True
                self._last_date = trading_date

            if not self._valid:
                self._history.append((ZERO, ZERO))
                continue

            # Typical price
            tp = _quantize(
                (bar.high + bar.low + bar.close) / Decimal("3")
            )
            vol = Decimal(str(bar.volume))

            # Update running sums
            self._sum_tpv += tp * vol
            self._sum_vol += vol
            self._sum_tp2v += tp * tp * vol

            # Compute VWAP and std dev
            if self._sum_vol == ZERO:
                vwap_val = ZERO
                std_val = ZERO
            else:
                vwap_val = _quantize(self._sum_tpv / self._sum_vol)

                # Weighted variance: E[X^2] - (E[X])^2
                mean_tp2 = float(self._sum_tp2v / self._sum_vol)
                vwap_f = float(vwap_val)
                variance = max(0.0, mean_tp2 - vwap_f * vwap_f)
                std_val = _quantize(Decimal(str(math.sqrt(variance))))

            self._history.append((vwap_val, std_val))

        self._last_index = index
        return self._history[index]


# =========================================================================
# STREAMING SESSION CVD
# =========================================================================


class StreamingCVD:
    """Incremental cumulative volume delta for intraday sessions.

    Replaces the O(k)-per-bar :func:`cvd_session` with an O(1) running
    sum that resets at each session boundary.
    """

    __slots__ = ("_cum", "_current_session", "_last_index", "_history")

    def __init__(self) -> None:
        self._cum: Decimal = ZERO
        self._current_session: str = ""
        self._last_index: int = -1
        self._history: list[Decimal] = []

    def reset(self) -> None:
        self._cum = ZERO
        self._current_session = ""
        self._last_index = -1
        self._history.clear()

    def update(self, data: Sequence[PriceData], index: int) -> Decimal:
        """Return session CVD at *index*, updating incrementally."""
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return ZERO

        _half = Decimal("0.5")
        start = self._last_index + 1
        for i in range(start, index + 1):
            bar = data[i]
            session = bar.date[:10]

            if session != self._current_session:
                self._cum = ZERO
                self._current_session = session

            bar_range = bar.high - bar.low
            if bar_range > ZERO:
                v_delta = (bar.close - bar.low) / bar_range
            else:
                v_delta = _half
            self._cum += (v_delta - _half) * Decimal(str(bar.volume))

            self._history.append(_quantize(self._cum))

        self._last_index = index
        return self._history[index]


# =========================================================================
# STREAMING HTF RESAMPLING (5m → 15m)
# =========================================================================


class StreamingHTFResample:
    """Incremental higher-timeframe bar resampling.

    Instead of rebuilding all HTF candles from session start on every bar
    (O(k) per bar), this maintains the current HTF candle in-progress and
    only emits a new complete candle when *factor* bars accumulate.

    Returns the complete HTF bars (as PriceData) seen so far within the
    current session, compatible with :func:`_ema` and other functions that
    access ``data[i].close``.
    """

    __slots__ = (
        "_factor", "_current_session", "_last_index",
        "_session_bars",  # complete HTF PriceData bars in current session
        "_pending_bars",  # bars accumulated for current incomplete group
        "_history",       # list of complete-bar-count per index
    )

    def __init__(self, factor: int = 3) -> None:
        self._factor = factor
        self._current_session: str = ""
        self._last_index: int = -1
        self._session_bars: list = []
        self._pending_bars: list = []
        self._history: list[int] = []

    def reset(self) -> None:
        self._current_session = ""
        self._last_index = -1
        self._session_bars.clear()
        self._pending_bars.clear()
        self._history.clear()

    def update(
        self, data: Sequence[PriceData], index: int,
    ) -> list:
        """Return list of complete HTF PriceData bars at *index*."""
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                count = self._history[index]
                return self._session_bars[:count]
            return []

        # Lazy import to avoid circular dependency
        from stockdownloader.model.price_data import PriceData as PD

        start = self._last_index + 1
        for i in range(start, index + 1):
            bar = data[i]
            session = bar.date[:10]

            if session != self._current_session:
                self._current_session = session
                self._session_bars.clear()
                self._pending_bars.clear()

            self._pending_bars.append(bar)

            if len(self._pending_bars) == self._factor:
                # Emit complete HTF bar as PriceData
                pb = self._pending_bars
                o = pb[0].open
                h = pb[0].high
                lo = pb[0].low
                c = pb[-1].close
                vol = 0
                for b in pb:
                    h = max(h, b.high)
                    lo = min(lo, b.low)
                    vol += b.volume
                self._session_bars.append(PD(
                    date=pb[0].date, open=o, high=h, low=lo,
                    close=c, adj_close=c, volume=vol,
                ))
                self._pending_bars = []

            self._history.append(len(self._session_bars))

        self._last_index = index
        count = self._history[index]
        return self._session_bars[:count]
