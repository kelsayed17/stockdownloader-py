"""Streaming accumulators: OBV, EMA, ATR, and HTF resampling."""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import ONE, ZERO
from stockdownloader.util.streaming import _quantize, _true_range

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
