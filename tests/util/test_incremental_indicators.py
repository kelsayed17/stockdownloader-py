"""Tests for streaming (incremental) indicator implementations.

Verifies parity between streaming accumulators and the raw indicator
functions.  Only OBV and Parabolic SAR are currently wired into the
IndicatorHub, but all streaming classes are tested for correctness.
"""

import random
from decimal import Decimal

from stockdownloader.model.price_data import PriceData
from stockdownloader.util import indicators as ti
from stockdownloader.util.indicators.momentum import StreamingOBV
from stockdownloader.util.indicators.trend import StreamingSAR


def _generate_test_data(days: int, seed: int = 42) -> list[PriceData]:
    """Generate synthetic price data with mild trend and noise."""
    rng = random.Random(seed)
    data: list[PriceData] = []
    price = 100.0

    for i in range(days):
        change = (rng.random() - 0.48) * 3
        price = max(50, price + change)

        open_ = price + (rng.random() - 0.5) * 2
        high = max(open_, price) + rng.random() * 2
        low = min(open_, price) - rng.random() * 2
        close = price
        volume = int(1_000_000 + rng.random() * 5_000_000)

        data.append(
            PriceData(
                date=f"2020-01-{min(i + 1, 28):02d}",
                open=Decimal(str(round(open_, 4))),
                high=Decimal(str(round(high, 4))),
                low=Decimal(str(round(low, 4))),
                close=Decimal(str(round(close, 4))),
                adj_close=Decimal(str(round(close, 4))),
                volume=volume,
            )
        )
    return data


DATA = _generate_test_data(200)


# =========================================================================
# Streaming OBV parity
# =========================================================================


class TestStreamingOBV:
    def test_parity_with_raw_obv(self):
        """Streaming OBV matches raw ti.obv() at every index."""
        s = StreamingOBV()
        for i in range(len(DATA)):
            streaming_val = s.update(DATA, i)
            raw_val = ti.obv(DATA, i)
            assert streaming_val == raw_val, (
                f"OBV mismatch at index {i}: streaming={streaming_val}, raw={raw_val}"
            )

    def test_is_obv_rising_matches_raw(self):
        """Streaming is_rising matches raw ti.is_obv_rising()."""
        s = StreamingOBV()
        lookback = 5
        # Must update all bars first
        for i in range(len(DATA)):
            s.update(DATA, i)

        for i in range(lookback, len(DATA)):
            streaming_val = s.is_rising(i, lookback)
            raw_val = ti.is_obv_rising(DATA, i, lookback)
            assert streaming_val == raw_val, (
                f"is_obv_rising mismatch at index {i}: "
                f"streaming={streaming_val}, raw={raw_val}"
            )

    def test_backward_lookup(self):
        """After computing to index N, looking back at index < N returns cached value."""
        s = StreamingOBV()
        s.update(DATA, 50)
        val_at_10 = s.update(DATA, 10)
        raw_at_10 = ti.obv(DATA, 10)
        assert val_at_10 == raw_at_10

    def test_reset(self):
        """Reset clears all state."""
        s = StreamingOBV()
        s.update(DATA, 50)
        s.reset()
        assert s._last_index == -1
        assert len(s._history) == 0

    def test_gap_fill(self):
        """Skipping indices fills the gap correctly."""
        s1 = StreamingOBV()
        s2 = StreamingOBV()

        # s1: update sequentially 0..20
        for i in range(21):
            s1.update(DATA, i)

        # s2: skip directly to 20 — should fill gaps
        s2.update(DATA, 20)

        assert s1.get(20) == s2.get(20)
        assert s1.get(10) == s2.get(10)
        assert s1.get(0) == s2.get(0)

    def test_empty_data(self):
        """OBV of empty data returns zero."""
        s = StreamingOBV()
        # No data to process, so update with index -1 case
        assert s.get(0) == Decimal("0")


# =========================================================================
# Streaming Parabolic SAR parity
# =========================================================================


class TestStreamingSAR:
    def test_parity_with_raw_sar(self):
        """Streaming SAR matches raw ti.parabolic_sar() at every index."""
        s = StreamingSAR()
        for i in range(len(DATA)):
            streaming_val = s.update(DATA, i)
            raw_val = ti.parabolic_sar(DATA, i)
            assert streaming_val == raw_val, (
                f"SAR mismatch at index {i}: streaming={streaming_val}, raw={raw_val}"
            )

    def test_is_bullish_matches_raw(self):
        """Streaming is_bullish matches raw ti.is_sar_bullish()."""
        s = StreamingSAR()
        for i in range(len(DATA)):
            s.update(DATA, i)

        for i in range(2, len(DATA)):
            streaming_val = s.is_bullish(DATA, i)
            raw_val = ti.is_sar_bullish(DATA, i)
            assert streaming_val == raw_val, (
                f"is_sar_bullish mismatch at index {i}: "
                f"streaming={streaming_val}, raw={raw_val}"
            )

    def test_backward_lookup(self):
        """After computing to index N, looking back returns cached value."""
        s = StreamingSAR()
        s.update(DATA, 50)
        val_at_10 = s.update(DATA, 10)
        raw_at_10 = ti.parabolic_sar(DATA, 10)
        assert val_at_10 == raw_at_10

    def test_reset(self):
        """Reset clears all state."""
        s = StreamingSAR()
        s.update(DATA, 50)
        s.reset()
        assert s._last_index == -1
        assert len(s._history) == 0

    def test_gap_fill(self):
        """Skipping indices fills the gap correctly."""
        s1 = StreamingSAR()
        s2 = StreamingSAR()

        for i in range(51):
            s1.update(DATA, i)
        s2.update(DATA, 50)

        assert s1.get(50) == s2.get(50)
        assert s1.get(25) == s2.get(25)
        assert s1.get(0) == s2.get(0)

    def test_different_datasets_produce_different_results(self):
        """Different price data produces different SAR values."""
        data2 = _generate_test_data(200, seed=99)
        s1 = StreamingSAR()
        s2 = StreamingSAR()

        s1.update(DATA, 100)
        s2.update(data2, 100)

        # Very unlikely to be identical with different random data
        assert s1.get(100) != s2.get(100)


# =========================================================================
# Hub integration: OBV and SAR via IndicatorHub use streaming
# =========================================================================


class TestHubStreamingIntegration:
    """Verify that IndicatorHub's OBV and SAR methods use streaming
    and produce results matching the raw functions."""

    def test_hub_obv_matches_raw(self):
        from stockdownloader.util.indicators.hub import IndicatorHub

        hub = IndicatorHub()
        for i in [10, 50, 100, 150]:
            hub_val = hub.obv(DATA, i)
            raw_val = ti.obv(DATA, i)
            assert hub_val == raw_val, (
                f"Hub OBV mismatch at {i}: hub={hub_val}, raw={raw_val}"
            )

    def test_hub_is_obv_rising_matches_raw(self):
        from stockdownloader.util.indicators.hub import IndicatorHub

        hub = IndicatorHub()
        # Need to compute OBV first for all bars up to 100
        hub.obv(DATA, 100)
        for i in [20, 50, 80, 100]:
            hub_val = hub.is_obv_rising(DATA, i, lookback=5)
            raw_val = ti.is_obv_rising(DATA, i, 5)
            assert hub_val == raw_val, (
                f"Hub is_obv_rising mismatch at {i}"
            )

    def test_hub_parabolic_sar_matches_raw(self):
        from stockdownloader.util.indicators.hub import IndicatorHub

        hub = IndicatorHub()
        for i in [10, 50, 100, 150]:
            hub_val = hub.parabolic_sar(DATA, i)
            raw_val = ti.parabolic_sar(DATA, i)
            assert hub_val == raw_val, (
                f"Hub SAR mismatch at {i}: hub={hub_val}, raw={raw_val}"
            )

    def test_hub_is_sar_bullish_matches_raw(self):
        from stockdownloader.util.indicators.hub import IndicatorHub

        hub = IndicatorHub()
        # SAR needs to be computed up to the index
        hub.parabolic_sar(DATA, 150)
        for i in [10, 50, 100, 150]:
            hub_val = hub.is_sar_bullish(DATA, i)
            raw_val = ti.is_sar_bullish(DATA, i)
            assert hub_val == raw_val, (
                f"Hub is_sar_bullish mismatch at {i}"
            )
