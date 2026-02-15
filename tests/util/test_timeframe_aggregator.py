"""Tests for multi-timeframe bar aggregation."""

from decimal import Decimal

import pytest

from stockdownloader.model.price_data import PriceData
from stockdownloader.util.timeframe_aggregator import (
    AggregatedBar,
    Timeframe,
    TimeframeAggregator,
)


def _bar(
    date: str,
    o: float = 100.0,
    h: float = 101.0,
    l: float = 99.0,
    c: float = 100.5,
    vol: int = 1000,
) -> PriceData:
    return PriceData(
        date=date,
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(l)),
        close=Decimal(str(c)),
        adj_close=Decimal(str(c)),
        volume=vol,
    )


def _session(trading_date: str, n_bars: int, base_price: float = 100.0) -> list[PriceData]:
    """Generate *n_bars* 5-minute bars for a session."""
    bars = []
    for i in range(n_bars):
        hour = 9 + (i * 5 + 30) // 60
        minute = (i * 5 + 30) % 60
        date = f"{trading_date} {hour:02d}:{minute:02d}:00-05:00"
        price = base_price + i * 0.1
        bars.append(_bar(
            date=date,
            o=price,
            h=price + 0.5,
            l=price - 0.5,
            c=price + 0.2,
            vol=1000 + i * 100,
        ))
    return bars


def _two_session_data() -> list[PriceData]:
    """Two sessions of 6 bars each (good for testing 15m = factor 3)."""
    return _session("2025-01-06", 6) + _session("2025-01-07", 6)


def _full_session_data() -> list[PriceData]:
    """One session of 78 bars (full RTH day at 5m intervals)."""
    return _session("2025-01-06", 78)


# =========================================================================
# Session boundary detection
# =========================================================================


class TestSessionBoundaries:
    def test_single_session(self):
        data = _session("2025-01-06", 10)
        agg = TimeframeAggregator(data)
        assert len(agg.sessions) == 1
        assert agg.sessions[0] == (0, 9, "2025-01-06")

    def test_two_sessions(self):
        data = _two_session_data()
        agg = TimeframeAggregator(data)
        assert len(agg.sessions) == 2
        assert agg.sessions[0] == (0, 5, "2025-01-06")
        assert agg.sessions[1] == (6, 11, "2025-01-07")

    def test_empty_data(self):
        agg = TimeframeAggregator([])
        assert len(agg.sessions) == 0


# =========================================================================
# M5 identity
# =========================================================================


class TestM5Identity:
    def test_m5_returns_all_bars(self):
        data = _session("2025-01-06", 5)
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.M5)
        assert len(bars) == 5

    def test_m5_preserves_values(self):
        data = _session("2025-01-06", 3)
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.M5)
        for i, bar in enumerate(bars):
            assert bar.open == data[i].open
            assert bar.high == data[i].high
            assert bar.low == data[i].low
            assert bar.close == data[i].close
            assert bar.volume == data[i].volume
            assert bar.source_start == i
            assert bar.source_end == i


# =========================================================================
# M15 aggregation (factor 3)
# =========================================================================


class TestM15:
    def test_correct_bar_count(self):
        data = _session("2025-01-06", 9)  # 9 bars -> 3 complete 15m candles
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.M15)
        assert len(bars) == 3

    def test_partial_candle_excluded(self):
        data = _session("2025-01-06", 8)  # 8 bars -> 2 complete, 2 partial dropped
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.M15)
        assert len(bars) == 2

    def test_ohlcv_aggregation(self):
        data = _two_session_data()
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.M15)

        # First 15m candle: bars 0-2 of session 1
        first = bars[0]
        assert first.open == data[0].open
        assert first.close == data[2].close
        assert first.high == max(data[0].high, data[1].high, data[2].high)
        assert first.low == min(data[0].low, data[1].low, data[2].low)
        assert first.volume == data[0].volume + data[1].volume + data[2].volume

    def test_source_indices(self):
        data = _session("2025-01-06", 6)
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.M15)
        assert bars[0].source_start == 0
        assert bars[0].source_end == 2
        assert bars[1].source_start == 3
        assert bars[1].source_end == 5

    def test_session_isolation(self):
        """15m candles don't span session boundaries."""
        data = _two_session_data()
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.M15)
        # 6 bars per session / 3 = 2 candles per session = 4 total
        assert len(bars) == 4
        # Session 1: bars 0-5
        assert bars[0].trading_date == "2025-01-06"
        assert bars[1].trading_date == "2025-01-06"
        # Session 2: bars 6-11
        assert bars[2].trading_date == "2025-01-07"
        assert bars[3].trading_date == "2025-01-07"


# =========================================================================
# M30 (factor 6) and H1 (factor 12)
# =========================================================================


class TestM30:
    def test_30m_bar_count(self):
        data = _full_session_data()  # 78 bars
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.M30)
        assert len(bars) == 78 // 6  # 13 complete 30m candles

    def test_30m_bar_count_property(self):
        data = _full_session_data()
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.M30)
        for bar in bars:
            assert bar.bar_count == 6


class TestH1:
    def test_1h_bar_count(self):
        data = _full_session_data()  # 78 bars
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.H1)
        assert len(bars) == 78 // 12  # 6 complete 1h candles

    def test_1h_bar_count_property(self):
        data = _full_session_data()
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.H1)
        for bar in bars:
            assert bar.bar_count == 12


class TestH4:
    def test_4h_bar_count(self):
        data = _full_session_data()  # 78 bars
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.H4)
        assert len(bars) == 78 // 48  # 1 complete 4h candle


# =========================================================================
# Daily aggregation
# =========================================================================


class TestDaily:
    def test_one_bar_per_session(self):
        data = _two_session_data()
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.DAILY)
        assert len(bars) == 2

    def test_daily_ohlcv(self):
        data = _session("2025-01-06", 6)
        agg = TimeframeAggregator(data)
        bars = agg.get_bars(Timeframe.DAILY)
        daily = bars[0]
        assert daily.open == data[0].open
        assert daily.close == data[-1].close
        assert daily.high == max(b.high for b in data)
        assert daily.low == min(b.low for b in data)
        assert daily.volume == sum(b.volume for b in data)
        assert daily.trading_date == "2025-01-06"
        assert daily.source_start == 0
        assert daily.source_end == 5


# =========================================================================
# Temporal filtering (get_bars_through, get_latest_bar)
# =========================================================================


class TestTemporalFiltering:
    def test_get_bars_through_excludes_future(self):
        data = _session("2025-01-06", 12)
        agg = TimeframeAggregator(data)
        # 15m bars: [0-2], [3-5], [6-8], [9-11]
        bars = agg.get_bars_through(Timeframe.M15, current_index=7)
        # Only bars ending at index <= 7: [0-2] (end=2), [3-5] (end=5)
        assert len(bars) == 2

    def test_get_bars_through_includes_exact_end(self):
        data = _session("2025-01-06", 12)
        agg = TimeframeAggregator(data)
        bars = agg.get_bars_through(Timeframe.M15, current_index=8)
        # Bars ending at 2, 5, 8 — all <= 8
        assert len(bars) == 3

    def test_get_latest_bar(self):
        data = _session("2025-01-06", 12)
        agg = TimeframeAggregator(data)
        bar = agg.get_latest_bar(Timeframe.M15, current_index=7)
        assert bar is not None
        assert bar.source_end == 5  # Last complete bar before index 7

    def test_get_latest_bar_returns_none_before_first(self):
        data = _session("2025-01-06", 12)
        agg = TimeframeAggregator(data)
        bar = agg.get_latest_bar(Timeframe.M15, current_index=1)
        assert bar is None  # First 15m bar ends at index 2


# =========================================================================
# Conversion to PriceData
# =========================================================================


class TestAsPriceData:
    def test_as_price_data_count(self):
        data = _session("2025-01-06", 9)
        agg = TimeframeAggregator(data)
        pd_bars = agg.as_price_data(Timeframe.M15)
        assert len(pd_bars) == 3

    def test_as_price_data_values_match(self):
        data = _session("2025-01-06", 6)
        agg = TimeframeAggregator(data)
        agg_bars = agg.get_bars(Timeframe.M15)
        pd_bars = agg.as_price_data(Timeframe.M15)
        for agg_bar, pd_bar in zip(agg_bars, pd_bars):
            assert agg_bar.open == pd_bar.open
            assert agg_bar.high == pd_bar.high
            assert agg_bar.low == pd_bar.low
            assert agg_bar.close == pd_bar.close
            assert agg_bar.volume == pd_bar.volume

    def test_as_price_data_through(self):
        data = _session("2025-01-06", 12)
        agg = TimeframeAggregator(data)
        pd_bars = agg.as_price_data_through(Timeframe.M15, current_index=7)
        assert len(pd_bars) == 2


# =========================================================================
# Caching
# =========================================================================


class TestCaching:
    def test_repeated_calls_return_same_list(self):
        data = _session("2025-01-06", 9)
        agg = TimeframeAggregator(data)
        bars1 = agg.get_bars(Timeframe.M15)
        bars2 = agg.get_bars(Timeframe.M15)
        assert bars1 is bars2

    def test_different_timeframes_cached_independently(self):
        data = _full_session_data()
        agg = TimeframeAggregator(data)
        m15 = agg.get_bars(Timeframe.M15)
        m30 = agg.get_bars(Timeframe.M30)
        assert len(m15) != len(m30)
        assert m15 is not m30


# =========================================================================
# Timeframe enum
# =========================================================================


class TestTimeframeEnum:
    def test_factors(self):
        assert Timeframe.M5.factor == 1
        assert Timeframe.M15.factor == 3
        assert Timeframe.M30.factor == 6
        assert Timeframe.H1.factor == 12
        assert Timeframe.H4.factor == 48
        assert Timeframe.DAILY.factor == 0

    def test_labels(self):
        assert Timeframe.M15.label == "15m"
        assert Timeframe.H1.label == "1h"
        assert Timeframe.DAILY.label == "1d"
