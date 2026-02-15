"""Multi-timeframe bar aggregation from 5-minute intraday data.

Derives higher-timeframe OHLCV bars (15m, 30m, 1h, 4h, daily) from
5-minute bars.  All aggregation is session-aware: partial candles at
session boundaries are excluded.

Usage::

    from stockdownloader.util.timeframe_aggregator import TimeframeAggregator

    agg = TimeframeAggregator(data)  # data = list[IntradayPriceData]
    bars_15m = agg.get_bars(Timeframe.M15)
    bars_1h  = agg.get_bars(Timeframe.H1)
    daily    = agg.get_bars(Timeframe.DAILY)

    # Get the most recent complete bar at or before an index:
    bar = agg.get_latest_bar(Timeframe.H1, current_index=500)

    # Get all complete bars through an index (for indicator computation):
    bars = agg.get_bars_through(Timeframe.M30, current_index=500)

The aggregator caches results so repeated calls with the same timeframe
are free.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData


class Timeframe(Enum):
    """Supported aggregation timeframes.

    The ``factor`` is the number of 5-minute bars per candle.
    ``DAILY`` uses session-based grouping rather than a fixed factor.
    """

    M5 = ("5m", 1)
    M15 = ("15m", 3)
    M30 = ("30m", 6)
    H1 = ("1h", 12)
    H4 = ("4h", 48)
    DAILY = ("1d", 0)  # factor=0 signals session-based grouping

    def __init__(self, label: str, factor: int) -> None:
        self.label = label
        self.factor = factor


@dataclass(frozen=True, slots=True)
class AggregatedBar:
    """A higher-timeframe OHLCV bar with metadata.

    Attributes
    ----------
    date:
        ISO timestamp of the first 5m bar in this candle.
    trading_date:
        The ``YYYY-MM-DD`` trading date (first 10 chars of date).
    open, high, low, close, volume:
        Standard OHLCV values aggregated from constituent 5m bars.
    source_start:
        Index of the first 5m bar in the source data.
    source_end:
        Index of the last 5m bar in the source data (inclusive).
    bar_count:
        Number of 5m bars aggregated into this candle.
    """

    date: str
    trading_date: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    source_start: int
    source_end: int
    bar_count: int


def _find_session_boundaries(
    data: Sequence[PriceData],
) -> list[tuple[int, int, str]]:
    """Find session boundaries as ``(start_idx, end_idx, trading_date)`` tuples.

    A new session starts when ``date[:10]`` changes.
    """
    if not data:
        return []

    boundaries: list[tuple[int, int, str]] = []
    cur_date = data[0].date[:10]
    start = 0

    for i in range(1, len(data)):
        d = data[i].date[:10]
        if d != cur_date:
            boundaries.append((start, i - 1, cur_date))
            cur_date = d
            start = i

    boundaries.append((start, len(data) - 1, cur_date))
    return boundaries


class TimeframeAggregator:
    """Aggregates 5-minute bars into higher timeframes.

    Thread-safety: NOT thread-safe.  Each strategy or analysis pipeline
    should hold its own aggregator instance.

    Parameters
    ----------
    data:
        The full list of 5-minute bars (``IntradayPriceData`` or ``PriceData``).
        The aggregator stores a reference — mutations to the list after
        construction lead to undefined behavior.
    """

    __slots__ = ("_data", "_cache", "_sessions", "_source_ends", "_pd_cache")

    def __init__(self, data: Sequence[PriceData]) -> None:
        self._data = data
        self._cache: dict[Timeframe, list[AggregatedBar]] = {}
        self._sessions: list[tuple[int, int, str]] | None = None
        # Sorted source_end indices per timeframe for bisect lookups
        self._source_ends: dict[Timeframe, list[int]] = {}
        # Cached PriceData lists per timeframe (grow-only, reused across calls)
        self._pd_cache: dict[Timeframe, list[PriceData]] = {}

    @property
    def sessions(self) -> list[tuple[int, int, str]]:
        """Lazily computed session boundaries."""
        if self._sessions is None:
            self._sessions = _find_session_boundaries(self._data)
        return self._sessions

    def get_bars(self, tf: Timeframe) -> list[AggregatedBar]:
        """Return all aggregated bars for the given timeframe.

        Results are cached — repeated calls return the same list.
        """
        if tf in self._cache:
            return self._cache[tf]

        if tf == Timeframe.M5:
            bars = self._identity_bars()
        elif tf == Timeframe.DAILY:
            bars = self._aggregate_daily()
        else:
            bars = self._aggregate_intraday(tf.factor)

        self._cache[tf] = bars
        return bars

    def _get_source_ends(self, tf: Timeframe) -> list[int]:
        """Return the sorted list of ``source_end`` values (cached)."""
        if tf not in self._source_ends:
            self._source_ends[tf] = [b.source_end for b in self.get_bars(tf)]
        return self._source_ends[tf]

    def get_bars_through(
        self, tf: Timeframe, current_index: int
    ) -> list[AggregatedBar]:
        """Return all complete aggregated bars whose source data ends
        at or before *current_index*.

        Useful for computing indicators on higher-timeframe data up to
        the current point in a backtest without look-ahead bias.

        Uses bisect for O(log n) lookup instead of linear scan.
        """
        all_bars = self.get_bars(tf)
        ends = self._get_source_ends(tf)
        # bisect_right gives the insertion point for current_index + 1,
        # which equals the count of bars with source_end <= current_index.
        n = bisect.bisect_right(ends, current_index)
        return all_bars[:n]

    def get_latest_bar(
        self, tf: Timeframe, current_index: int
    ) -> AggregatedBar | None:
        """Return the most recent complete aggregated bar at or before
        *current_index*, or ``None`` if none exists yet.
        """
        bars = self.get_bars_through(tf, current_index)
        return bars[-1] if bars else None

    def as_price_data(self, tf: Timeframe) -> list[PriceData]:
        """Convert aggregated bars to ``PriceData`` objects for use with
        existing indicator functions that accept ``Sequence[PriceData]``.
        """
        from stockdownloader.model.price_data import PriceData as PD

        return [
            PD(
                date=bar.date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                adj_close=bar.close,
                volume=bar.volume,
            )
            for bar in self.get_bars(tf)
        ]

    def as_price_data_through(
        self, tf: Timeframe, current_index: int
    ) -> list[PriceData]:
        """Convert aggregated bars through *current_index* to ``PriceData``.

        Results are cached and extended incrementally: only new bars
        are converted to PriceData on each call, making repeated calls
        O(1) amortized instead of O(n).

        .. code-block:: python

            htf_bars = agg.as_price_data_through(Timeframe.H1, current_index)
            if len(htf_bars) >= 14:
                atr_1h = hub.atr(htf_bars, len(htf_bars) - 1, period=14)
        """
        from stockdownloader.model.price_data import PriceData as PD

        htf_bars = self.get_bars_through(tf, current_index)
        n = len(htf_bars)

        cached = self._pd_cache.get(tf)
        if cached is not None:
            cached_len = len(cached)
            if cached_len == n:
                return cached
            # Extend: only convert newly-added bars
            if cached_len < n:
                for bar in htf_bars[cached_len:]:
                    cached.append(PD(
                        date=bar.date,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        adj_close=bar.close,
                        volume=bar.volume,
                    ))
                return cached

        # First call: full conversion
        result = [
            PD(
                date=bar.date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                adj_close=bar.close,
                volume=bar.volume,
            )
            for bar in htf_bars
        ]
        self._pd_cache[tf] = result
        return result

    def as_intraday_price_data(self, tf: Timeframe) -> list:
        """Convert aggregated bars to ``IntradayPriceData`` for use with
        :class:`IntradayBacktestEngine` (which requires the ``trading_date``
        property that plain ``PriceData`` lacks).
        """
        from stockdownloader.model.intraday_price_data import (
            IntradayPriceData as IPD,
        )

        return [
            IPD(
                date=bar.date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                adj_close=bar.close,
                volume=bar.volume,
            )
            for bar in self.get_bars(tf)
        ]

    # ------------------------------------------------------------------
    # Internal aggregation methods
    # ------------------------------------------------------------------

    def _identity_bars(self) -> list[AggregatedBar]:
        """5-minute bars as AggregatedBar (identity transformation)."""
        return [
            AggregatedBar(
                date=bar.date,
                trading_date=bar.date[:10],
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                source_start=i,
                source_end=i,
                bar_count=1,
            )
            for i, bar in enumerate(self._data)
        ]

    def _aggregate_daily(self) -> list[AggregatedBar]:
        """Aggregate by session (trading date)."""
        result: list[AggregatedBar] = []
        for start, end, trading_date in self.sessions:
            o = self._data[start].open
            h = self._data[start].high
            lo = self._data[start].low
            c = self._data[end].close
            vol = 0
            for i in range(start, end + 1):
                bar = self._data[i]
                h = max(h, bar.high)
                lo = min(lo, bar.low)
                vol += bar.volume

            result.append(AggregatedBar(
                date=self._data[start].date,
                trading_date=trading_date,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=vol,
                source_start=start,
                source_end=end,
                bar_count=end - start + 1,
            ))
        return result

    def _aggregate_intraday(self, factor: int) -> list[AggregatedBar]:
        """Aggregate 5m bars into *factor*-bar candles within each session.

        Only complete groups are included (partial candles at session
        end are excluded).
        """
        result: list[AggregatedBar] = []

        for sess_start, sess_end, trading_date in self.sessions:
            count = sess_end - sess_start + 1
            complete = count // factor

            for g in range(complete):
                base = sess_start + g * factor
                end_idx = base + factor - 1

                o = self._data[base].open
                h = self._data[base].high
                lo = self._data[base].low
                c = self._data[end_idx].close
                vol = 0

                for k in range(factor):
                    bar = self._data[base + k]
                    h = max(h, bar.high)
                    lo = min(lo, bar.low)
                    vol += bar.volume

                result.append(AggregatedBar(
                    date=self._data[base].date,
                    trading_date=trading_date,
                    open=o,
                    high=h,
                    low=lo,
                    close=c,
                    volume=vol,
                    source_start=base,
                    source_end=end_idx,
                    bar_count=factor,
                ))

        return result
