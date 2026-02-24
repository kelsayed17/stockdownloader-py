"""Tests for MultiTimeframeAligner, TimeframeSignalSpec, and AlignedSignal."""
from __future__ import annotations

import pytest
from decimal import Decimal
from typing import Any
from collections.abc import Sequence

from stockdownloader.core.models.price import PriceData
from stockdownloader.strategy.signals.signal_generator import (
    AtomicSignalGenerator,
    SignalDirection,
    SignalResult,
)
from stockdownloader.strategy.signals.multi_timeframe_aligner import (
    AlignedSignal,
    MultiTimeframeAligner,
    TimeframeSignalSpec,
)
from stockdownloader.util.indicators.hub import IndicatorHub
from stockdownloader.core.timeframe import Timeframe, TimeframeAggregator


# ======================================================================
# Helpers
# ======================================================================


def _make_price(
    close: str,
    date: str = "2025-01-02",
    volume: int = 10000,
) -> PriceData:
    c = Decimal(close)
    return PriceData(
        date=date,
        open=c - Decimal("0.50"),
        high=c + Decimal("1"),
        low=c - Decimal("1"),
        close=c,
        adj_close=c,
        volume=volume,
    )


def _make_5m_series(
    n: int,
    base_price: float = 100.0,
    date: str = "2025-01-02",
) -> list[PriceData]:
    """Generate n 5-minute bars for a single trading day.

    Prices step up slightly each bar to create a mild uptrend.
    Uses datetime-like date strings (e.g., '2025-01-02 09:30:00').
    """
    bars = []
    for i in range(n):
        hour = 9 + (i * 5 + 30) // 60
        minute = (i * 5 + 30) % 60
        timestamp = f"{date} {hour:02d}:{minute:02d}:00"
        c = Decimal(str(base_price + i * 0.10))
        bars.append(PriceData(
            date=timestamp,
            open=c - Decimal("0.05"),
            high=c + Decimal("0.50"),
            low=c - Decimal("0.50"),
            close=c,
            adj_close=c,
            volume=10000 + i * 100,
        ))
    return bars


class _ControlledGenerator(AtomicSignalGenerator):
    """Generator that returns a predetermined result for testing."""

    def __init__(
        self,
        name: str = "ctrl",
        category: str = "momentum",
        warmup: int = 2,
        fixed_score: float = 0.5,
        fixed_fired: bool = True,
    ):
        self._name = name
        self._category = category
        self._warmup = warmup
        self._fixed_score = fixed_score
        self._fixed_fired = fixed_fired
        self.call_count = 0
        self.last_index = -1
        self.last_data_len = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return f"Ctrl({self._name})"

    @property
    def category(self) -> str:
        return self._category

    @property
    def warmup_period(self) -> int:
        return self._warmup

    def evaluate(
        self,
        data: Sequence[PriceData],
        index: int,
        hub: IndicatorHub,
    ) -> SignalResult:
        self.call_count += 1
        self.last_index = index
        self.last_data_len = len(data)

        if index < self._warmup:
            return SignalResult.neutral()

        direction = (
            SignalDirection.BULLISH if self._fixed_score > 0.1
            else SignalDirection.BEARISH if self._fixed_score < -0.1
            else SignalDirection.NEUTRAL
        )
        return SignalResult(
            score=self._fixed_score,
            direction=direction,
            fired=self._fixed_fired,
            metadata={"index": index, "data_len": len(data)},
        )

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {}


# ======================================================================
# TimeframeSignalSpec tests
# ======================================================================


class TestTimeframeSignalSpec:
    def test_construction(self):
        gen = _ControlledGenerator()
        spec = TimeframeSignalSpec(
            generator=gen,
            timeframe=Timeframe.M5,
            weight=1.5,
        )
        assert spec.generator is gen
        assert spec.timeframe == Timeframe.M5
        assert spec.weight == 1.5

    def test_default_weight(self):
        gen = _ControlledGenerator()
        spec = TimeframeSignalSpec(generator=gen, timeframe=Timeframe.H1)
        assert spec.weight == 1.0

    def test_frozen(self):
        gen = _ControlledGenerator()
        spec = TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5)
        with pytest.raises(AttributeError):
            spec.weight = 2.0  # type: ignore[misc]


# ======================================================================
# AlignedSignal tests
# ======================================================================


class TestAlignedSignal:
    def test_construction(self):
        result = SignalResult(
            score=0.7, direction=SignalDirection.BULLISH, fired=True,
        )
        aligned = AlignedSignal(
            result=result,
            timeframe=Timeframe.H1,
            generator_name="rsi_14",
            category="momentum",
            weight=1.5,
            htf_bar_index=10,
            htf_bar_count=20,
        )
        assert aligned.result.score == 0.7
        assert aligned.timeframe == Timeframe.H1
        assert aligned.generator_name == "rsi_14"
        assert aligned.category == "momentum"
        assert aligned.weight == 1.5
        assert aligned.htf_bar_index == 10
        assert aligned.htf_bar_count == 20

    def test_default_htf_fields(self):
        result = SignalResult.neutral()
        aligned = AlignedSignal(
            result=result,
            timeframe=Timeframe.M5,
            generator_name="test",
            category="trend",
            weight=1.0,
        )
        assert aligned.htf_bar_index == -1
        assert aligned.htf_bar_count == 0


# ======================================================================
# MultiTimeframeAligner tests
# ======================================================================


class TestMultiTimeframeAligner:
    def test_m5_evaluation(self):
        """M5 specs evaluate directly on raw data."""
        data = _make_5m_series(30)
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        gen = _ControlledGenerator(warmup=2, fixed_score=0.6)
        spec = TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5, weight=1.0)

        aligned = aligner.get_aligned_signals([spec], current_5m_index=20, hub_m5=hub)

        assert len(aligned) == 1
        sig = aligned[0]
        assert sig.timeframe == Timeframe.M5
        assert sig.generator_name == "ctrl"
        assert sig.category == "momentum"
        assert sig.weight == 1.0
        assert sig.result.score == 0.6

    def test_m5_evaluation_uses_data_through_index(self):
        """M5 evaluation uses as_price_data_through to slice data."""
        data = _make_5m_series(50)
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        gen = _ControlledGenerator(warmup=0, fixed_score=0.3)
        spec = TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5, weight=1.0)

        aligned = aligner.get_aligned_signals([spec], current_5m_index=25, hub_m5=hub)

        assert len(aligned) == 1
        # Generator should have been called with data up to index 25
        assert gen.call_count == 1

    def test_htf_evaluation_h1(self):
        """H1 specs evaluate on aggregated hourly bars."""
        # Need at least 12 x 5-min bars per H1 bar, plus warmup
        data = _make_5m_series(78)  # one full session
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        gen = _ControlledGenerator(
            name="htf_test", warmup=2, fixed_score=0.8,
        )
        spec = TimeframeSignalSpec(
            generator=gen, timeframe=Timeframe.H1, weight=2.0,
        )

        # Evaluate at the end of data
        aligned = aligner.get_aligned_signals(
            [spec], current_5m_index=77, hub_m5=hub,
        )

        assert len(aligned) == 1
        sig = aligned[0]
        assert sig.timeframe == Timeframe.H1
        assert sig.weight == 2.0
        assert sig.generator_name == "htf_test"

    def test_htf_caching(self):
        """HTF results are cached when the same bar is still current."""
        data = _make_5m_series(78)
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        gen = _ControlledGenerator(warmup=2, fixed_score=0.4)
        spec = TimeframeSignalSpec(
            generator=gen, timeframe=Timeframe.H1, weight=1.0,
        )

        # Two calls with different 5m indices that map to the same H1 bar
        # should use cached result
        aligner.get_aligned_signals([spec], current_5m_index=30, hub_m5=hub)
        call_count_after_first = gen.call_count

        # Bars 30 and 31 are within the same H1 bar (bars 24-35),
        # so the HTF bar count is the same
        aligner.get_aligned_signals([spec], current_5m_index=31, hub_m5=hub)

        # If cached, the generator should not have been called again
        assert gen.call_count == call_count_after_first

    def test_multiple_specs(self):
        """Multiple specs produce one aligned signal each."""
        data = _make_5m_series(78)
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        gen1 = _ControlledGenerator(name="gen_m5", warmup=2, fixed_score=0.5)
        gen2 = _ControlledGenerator(name="gen_h1", warmup=2, fixed_score=-0.3)

        specs = [
            TimeframeSignalSpec(generator=gen1, timeframe=Timeframe.M5, weight=1.0),
            TimeframeSignalSpec(generator=gen2, timeframe=Timeframe.H1, weight=1.5),
        ]

        aligned = aligner.get_aligned_signals(specs, current_5m_index=50, hub_m5=hub)

        assert len(aligned) == 2
        assert aligned[0].generator_name == "gen_m5"
        assert aligned[0].timeframe == Timeframe.M5
        assert aligned[1].generator_name == "gen_h1"
        assert aligned[1].timeframe == Timeframe.H1

    def test_insufficient_warmup_returns_neutral(self):
        """HTF evaluation returns neutral if not enough HTF bars for warmup."""
        data = _make_5m_series(10)
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        # H1 needs 12 bars to form one candle; with 10 bars we get 0 H1 bars
        gen = _ControlledGenerator(warmup=2, fixed_score=0.9)
        spec = TimeframeSignalSpec(
            generator=gen, timeframe=Timeframe.H1, weight=1.0,
        )

        aligned = aligner.get_aligned_signals([spec], current_5m_index=9, hub_m5=hub)

        assert len(aligned) == 1
        assert aligned[0].result.score == 0.0
        assert aligned[0].result.direction == SignalDirection.NEUTRAL
        assert aligned[0].result.fired is False

    def test_empty_specs(self):
        """Empty specs list produces empty aligned signal list."""
        data = _make_5m_series(30)
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        aligned = aligner.get_aligned_signals([], current_5m_index=20, hub_m5=hub)
        assert aligned == []

    def test_clear_cache(self):
        """clear_cache resets HTF caches."""
        data = _make_5m_series(78)
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        gen = _ControlledGenerator(warmup=2, fixed_score=0.4)
        spec = TimeframeSignalSpec(
            generator=gen, timeframe=Timeframe.H1, weight=1.0,
        )

        aligner.get_aligned_signals([spec], current_5m_index=50, hub_m5=hub)
        initial_count = gen.call_count

        aligner.clear_cache()

        # After clearing, next call should re-evaluate
        aligner.get_aligned_signals([spec], current_5m_index=50, hub_m5=hub)
        assert gen.call_count > initial_count

    def test_m15_evaluation(self):
        """M15 specs aggregate into 15-minute bars (3 x 5m)."""
        data = _make_5m_series(30)
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        gen = _ControlledGenerator(warmup=2, fixed_score=0.7)
        spec = TimeframeSignalSpec(
            generator=gen, timeframe=Timeframe.M15, weight=1.0,
        )

        aligned = aligner.get_aligned_signals(
            [spec], current_5m_index=29, hub_m5=hub,
        )

        assert len(aligned) == 1
        assert aligned[0].timeframe == Timeframe.M15
        # With 30 bars and factor=3, we get 10 M15 bars.
        # The generator warmup is 2, so it should produce a real signal.
        assert aligned[0].result.score == 0.7

    def test_neutral_result_preserves_metadata(self):
        """Neutral aligned signals preserve spec metadata (name, category, weight)."""
        data = _make_5m_series(5)
        aggregator = TimeframeAggregator(data)
        aligner = MultiTimeframeAligner(aggregator)
        hub = IndicatorHub()

        gen = _ControlledGenerator(
            name="myname", category="volatility", warmup=100, fixed_score=0.5,
        )
        spec = TimeframeSignalSpec(
            generator=gen, timeframe=Timeframe.H1, weight=3.0,
        )

        aligned = aligner.get_aligned_signals([spec], current_5m_index=4, hub_m5=hub)

        assert len(aligned) == 1
        sig = aligned[0]
        assert sig.generator_name == "myname"
        assert sig.category == "volatility"
        assert sig.weight == 3.0
        assert sig.result == SignalResult.neutral()
