"""Streaming higher-timeframe resampling (e.g., 5m to 15m)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData


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
