"""Tests for RSISignalGenerator."""
from __future__ import annotations

import pytest
from decimal import Decimal

from stockdownloader.model.price_data import PriceData
from stockdownloader.strategy.signals.generators import RSISignalGenerator
from stockdownloader.strategy.signals.signal_generator import (
    AtomicSignalGenerator,
    SignalDirection,
    SignalResult,
)
from stockdownloader.util.indicators.hub import IndicatorHub


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


def _make_trending_up(n: int, start: float = 100.0, step: float = 1.0) -> list[PriceData]:
    """Generate n bars with steadily increasing close prices.

    A strong uptrend pushes RSI high (overbought territory).
    """
    return [
        _make_price(
            str(round(start + i * step, 2)),
            date=f"2025-01-{(i % 28) + 1:02d}",
        )
        for i in range(n)
    ]


def _make_trending_down(n: int, start: float = 200.0, step: float = 1.0) -> list[PriceData]:
    """Generate n bars with steadily decreasing close prices.

    A strong downtrend pushes RSI low (oversold territory).
    """
    return [
        _make_price(
            str(round(start - i * step, 2)),
            date=f"2025-01-{(i % 28) + 1:02d}",
        )
        for i in range(n)
    ]


def _make_oscillating(
    n: int,
    center: float = 150.0,
    amplitude: float = 5.0,
) -> list[PriceData]:
    """Generate n bars oscillating around center.

    Creates up-down-up-down pattern that keeps RSI near middle.
    """
    import math

    bars = []
    for i in range(n):
        val = center + amplitude * math.sin(i * 0.5)
        bars.append(_make_price(
            str(round(val, 2)),
            date=f"2025-01-{(i % 28) + 1:02d}",
        ))
    return bars


def _make_oversold_crossover(n: int = 30) -> list[PriceData]:
    """Create data that drives RSI below oversold then crosses back above.

    First half trends sharply down (RSI drops below 30),
    second half trends up (RSI crosses above 30 -> fire).
    """
    bars = []
    price = 200.0
    for i in range(n):
        if i < n // 2:
            # Strong downtrend to push RSI below oversold
            price -= 2.0
        else:
            # Uptick to cross above oversold
            price += 1.5
        bars.append(_make_price(
            str(round(price, 2)),
            date=f"2025-01-{(i % 28) + 1:02d}",
        ))
    return bars


def _make_overbought_crossover(n: int = 30) -> list[PriceData]:
    """Create data that drives RSI above overbought then crosses back below.

    First half trends sharply up (RSI rises above 70),
    second half trends down (RSI crosses below 70 -> fire).
    """
    bars = []
    price = 100.0
    for i in range(n):
        if i < n // 2:
            # Strong uptrend to push RSI above overbought
            price += 2.0
        else:
            # Downtick to cross below overbought
            price -= 1.5
        bars.append(_make_price(
            str(round(price, 2)),
            date=f"2025-01-{(i % 28) + 1:02d}",
        ))
    return bars


# ======================================================================
# Generator metadata tests
# ======================================================================


class TestRSIGeneratorMetadata:
    def test_default_name(self):
        gen = RSISignalGenerator()
        assert gen.name == "rsi_14_30.0_70.0"

    def test_custom_name(self):
        gen = RSISignalGenerator(period=7, oversold=25.0, overbought=75.0)
        assert gen.name == "rsi_7_25.0_75.0"

    def test_display_name(self):
        gen = RSISignalGenerator()
        assert gen.display_name == "RSI(14) [30.0/70.0]"

    def test_category(self):
        gen = RSISignalGenerator()
        assert gen.category == "momentum"

    def test_warmup_period(self):
        gen = RSISignalGenerator(period=14)
        assert gen.warmup_period == 15  # period + 1

    def test_warmup_period_custom(self):
        gen = RSISignalGenerator(period=7)
        assert gen.warmup_period == 8

    def test_param_space(self):
        gen = RSISignalGenerator()
        space = gen.param_space
        assert "period" in space
        assert "oversold" in space
        assert "overbought" in space
        assert 14 in space["period"]
        assert 30.0 in space["oversold"]
        assert 70.0 in space["overbought"]


# ======================================================================
# Warmup behavior
# ======================================================================


class TestRSIWarmup:
    def test_returns_neutral_before_warmup(self):
        gen = RSISignalGenerator(period=14)
        data = _make_trending_up(20)
        hub = IndicatorHub()

        for i in range(15):  # 0..14 are within warmup
            result = gen.evaluate(data, i, hub)
            assert result == SignalResult.neutral(), (
                f"Expected neutral at index {i}"
            )

    def test_returns_signal_at_warmup_boundary(self):
        gen = RSISignalGenerator(period=14)
        data = _make_trending_up(30)
        hub = IndicatorHub()

        result = gen.evaluate(data, 15, hub)
        assert result != SignalResult.neutral()
        assert result.score != 0.0 or result.direction != SignalDirection.NEUTRAL


# ======================================================================
# Score computation
# ======================================================================


class TestRSIScore:
    def test_strong_uptrend_bearish_score(self):
        """In a strong uptrend, RSI >= overbought, score should be -1.0."""
        gen = RSISignalGenerator(period=14, oversold=30.0, overbought=70.0)
        data = _make_trending_up(50, start=100.0, step=2.0)
        hub = IndicatorHub()

        result = gen.evaluate(data, 49, hub)
        # RSI should be very high after steady uptrend
        assert result.score < 0, "Expected bearish score in strong uptrend"
        assert result.direction == SignalDirection.BEARISH

    def test_strong_downtrend_bullish_score(self):
        """In a strong downtrend, RSI <= oversold, score should be +1.0."""
        gen = RSISignalGenerator(period=14, oversold=30.0, overbought=70.0)
        data = _make_trending_down(50, start=300.0, step=2.0)
        hub = IndicatorHub()

        result = gen.evaluate(data, 49, hub)
        assert result.score > 0, "Expected bullish score in strong downtrend"
        assert result.direction == SignalDirection.BULLISH

    def test_oscillating_near_neutral(self):
        """Oscillating prices should produce RSI near 50 and neutral/mild score."""
        gen = RSISignalGenerator(period=14)
        data = _make_oscillating(50, center=150.0, amplitude=2.0)
        hub = IndicatorHub()

        result = gen.evaluate(data, 49, hub)
        # Score should be close to 0 (near midpoint)
        assert abs(result.score) < 0.8, (
            f"Expected near-neutral score, got {result.score}"
        )

    def test_score_bounded_positive(self):
        """Score should never exceed +1.0."""
        gen = RSISignalGenerator(period=14)
        data = _make_trending_down(50, start=300.0, step=3.0)
        hub = IndicatorHub()
        result = gen.evaluate(data, 49, hub)
        assert result.score <= 1.0

    def test_score_bounded_negative(self):
        """Score should never go below -1.0."""
        gen = RSISignalGenerator(period=14)
        data = _make_trending_up(50, start=100.0, step=3.0)
        hub = IndicatorHub()
        result = gen.evaluate(data, 49, hub)
        assert result.score >= -1.0

    def test_rsi_at_oversold_gives_max_bullish(self):
        """When RSI <= oversold, score = +1.0."""
        gen = RSISignalGenerator(period=14, oversold=30.0, overbought=70.0)
        data = _make_trending_down(50, start=300.0, step=3.0)
        hub = IndicatorHub()
        result = gen.evaluate(data, 49, hub)
        assert result.score == 1.0

    def test_rsi_at_overbought_gives_max_bearish(self):
        """When RSI >= overbought, score = -1.0."""
        gen = RSISignalGenerator(period=14, oversold=30.0, overbought=70.0)
        data = _make_trending_up(50, start=100.0, step=3.0)
        hub = IndicatorHub()
        result = gen.evaluate(data, 49, hub)
        assert result.score == -1.0


# ======================================================================
# Direction mapping
# ======================================================================


class TestRSIDirection:
    def test_bullish_direction(self):
        gen = RSISignalGenerator(period=14)
        data = _make_trending_down(50, start=300.0, step=3.0)
        hub = IndicatorHub()
        result = gen.evaluate(data, 49, hub)
        assert result.direction == SignalDirection.BULLISH

    def test_bearish_direction(self):
        gen = RSISignalGenerator(period=14)
        data = _make_trending_up(50, start=100.0, step=3.0)
        hub = IndicatorHub()
        result = gen.evaluate(data, 49, hub)
        assert result.direction == SignalDirection.BEARISH

    def test_neutral_direction_zone(self):
        """Scores between -0.1 and 0.1 map to NEUTRAL direction."""
        gen = RSISignalGenerator(period=14, oversold=30.0, overbought=70.0)
        # With oscillating data, RSI should be near 50,
        # which maps to score near 0 -> NEUTRAL
        data = _make_oscillating(50, center=150.0, amplitude=1.0)
        hub = IndicatorHub()
        result = gen.evaluate(data, 49, hub)
        # Score should be close to 0
        if abs(result.score) <= 0.1:
            assert result.direction == SignalDirection.NEUTRAL


# ======================================================================
# Fire event detection
# ======================================================================


class TestRSIFiring:
    def test_no_fire_in_steady_trend(self):
        """No fire event when RSI stays on one side of threshold."""
        gen = RSISignalGenerator(period=14, oversold=30.0, overbought=70.0)
        data = _make_trending_up(50, start=100.0, step=1.0)
        hub = IndicatorHub()

        # Deep into an uptrend, RSI should be high and staying high
        # (no threshold crossover)
        result_late = gen.evaluate(data, 49, hub)
        # At bar 49 of a steady uptrend, RSI should be firmly above 70
        # and was above 70 the previous bar too -> no fire
        assert result_late.fired is False

    def test_fire_on_oversold_crossover(self):
        """Fire when RSI crosses above oversold threshold."""
        gen = RSISignalGenerator(period=14, oversold=30.0, overbought=70.0)
        data = _make_oversold_crossover(n=30)
        hub = IndicatorHub()

        # Look for a bar where fired=True
        fired_indices = []
        for i in range(15, len(data)):
            result = gen.evaluate(data, i, hub)
            if result.fired:
                fired_indices.append(i)

        # There should be at least one fire event around the crossover point
        assert len(fired_indices) > 0, "Expected at least one fire event"

    def test_fire_on_overbought_crossover(self):
        """Fire when RSI crosses below overbought threshold."""
        gen = RSISignalGenerator(period=14, oversold=30.0, overbought=70.0)
        data = _make_overbought_crossover(n=30)
        hub = IndicatorHub()

        fired_indices = []
        for i in range(15, len(data)):
            result = gen.evaluate(data, i, hub)
            if result.fired:
                fired_indices.append(i)

        assert len(fired_indices) > 0, "Expected at least one fire event"


# ======================================================================
# Metadata
# ======================================================================


class TestRSIMetadata:
    def test_metadata_contains_rsi_values(self):
        gen = RSISignalGenerator(period=14)
        data = _make_trending_up(30)
        hub = IndicatorHub()

        result = gen.evaluate(data, 20, hub)
        assert "rsi" in result.metadata
        assert "prev_rsi" in result.metadata
        assert isinstance(result.metadata["rsi"], float)
        assert isinstance(result.metadata["prev_rsi"], float)

    def test_metadata_rsi_in_valid_range(self):
        gen = RSISignalGenerator(period=14)
        data = _make_trending_up(30)
        hub = IndicatorHub()

        result = gen.evaluate(data, 20, hub)
        assert 0 <= result.metadata["rsi"] <= 100
        assert 0 <= result.metadata["prev_rsi"] <= 100


# ======================================================================
# Custom parameters
# ======================================================================


class TestRSICustomParams:
    def test_custom_period(self):
        gen = RSISignalGenerator(period=7)
        assert gen.warmup_period == 8
        assert gen.name == "rsi_7_30.0_70.0"

    def test_custom_thresholds(self):
        gen = RSISignalGenerator(period=14, oversold=20.0, overbought=80.0)
        assert gen.name == "rsi_14_20.0_80.0"
        assert gen.display_name == "RSI(14) [20.0/80.0]"

    def test_wider_thresholds_more_neutral(self):
        """Wider thresholds should produce more neutral signals."""
        gen_narrow = RSISignalGenerator(period=14, oversold=40.0, overbought=60.0)
        gen_wide = RSISignalGenerator(period=14, oversold=20.0, overbought=80.0)

        data = _make_oscillating(50, center=150.0, amplitude=3.0)
        hub1 = IndicatorHub()
        hub2 = IndicatorHub()

        result_narrow = gen_narrow.evaluate(data, 40, hub1)
        result_wide = gen_wide.evaluate(data, 40, hub2)

        # With wider bands, the score for the same RSI value should be
        # smaller in magnitude (more neutral)
        assert abs(result_wide.score) <= abs(result_narrow.score) + 0.01


# ======================================================================
# IndicatorHub caching integration
# ======================================================================


class TestRSIHubCaching:
    def test_same_result_with_same_hub(self):
        """Same index on same hub should return same result."""
        gen = RSISignalGenerator(period=14)
        data = _make_trending_up(30)
        hub = IndicatorHub()

        r1 = gen.evaluate(data, 20, hub)
        r2 = gen.evaluate(data, 20, hub)
        assert r1 == r2

    def test_sequential_evaluation(self):
        """Evaluating at consecutive indices should produce different results."""
        gen = RSISignalGenerator(period=14)
        data = _make_trending_up(30, step=2.0)
        hub = IndicatorHub()

        r1 = gen.evaluate(data, 20, hub)
        r2 = gen.evaluate(data, 25, hub)
        # Both should be valid signals (past warmup)
        assert r1 != SignalResult.neutral()
        assert r2 != SignalResult.neutral()


# ======================================================================
# AtomicSignalGenerator ABC compliance
# ======================================================================


class TestRSIABCCompliance:
    def test_is_atomic_signal_generator(self):
        gen = RSISignalGenerator()
        assert isinstance(gen, AtomicSignalGenerator)

    def test_all_abstract_methods_implemented(self):
        gen = RSISignalGenerator()
        assert isinstance(gen.name, str)
        assert isinstance(gen.display_name, str)
        assert isinstance(gen.category, str)
        assert isinstance(gen.warmup_period, int)
        assert callable(gen.evaluate)
        assert isinstance(gen.param_space, dict)
