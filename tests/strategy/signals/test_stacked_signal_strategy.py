"""Tests for StackedDailyStrategy."""
from __future__ import annotations

import pytest
from decimal import Decimal
from typing import Any
from collections.abc import Sequence

from stockdownloader.model.price_data import PriceData
from stockdownloader.strategy.signals.signal_generator import (
    AtomicSignalGenerator,
    SignalDirection,
    SignalResult,
)
from stockdownloader.strategy.signals.multi_timeframe_aligner import (
    TimeframeSignalSpec,
)
from stockdownloader.strategy.signals.stacked_signal_engine import (
    AggregationMode,
    StackConfig,
)
from stockdownloader.strategy.signals.stacked_daily_strategy import (
    StackedDailyStrategy,
)
from stockdownloader.strategy.trading_strategy import Signal
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.timeframe import Timeframe


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


def _make_trending_up(n: int, start: float = 100.0) -> list[PriceData]:
    """Generate n bars trending up."""
    return [
        _make_price(str(round(start + i * 0.5, 2)), date=f"2025-01-{(i % 28) + 1:02d}")
        for i in range(n)
    ]


def _make_trending_down(n: int, start: float = 200.0) -> list[PriceData]:
    """Generate n bars trending down."""
    return [
        _make_price(str(round(start - i * 0.5, 2)), date=f"2025-01-{(i % 28) + 1:02d}")
        for i in range(n)
    ]


def _make_flat(n: int, price: float = 150.0) -> list[PriceData]:
    """Generate n bars with constant price."""
    return [
        _make_price(str(price), date=f"2025-01-{(i % 28) + 1:02d}")
        for i in range(n)
    ]


class _FixedGenerator(AtomicSignalGenerator):
    """Generator returning a fixed score after warmup."""

    def __init__(
        self,
        name: str = "fixed",
        category: str = "momentum",
        warmup: int = 5,
        score: float = 0.0,
        fired: bool = False,
    ):
        self._name = name
        self._category = category
        self._warmup = warmup
        self._score = score
        self._fired = fired

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return f"Fixed({self._name})"

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
        if index < self._warmup:
            return SignalResult.neutral()

        direction = (
            SignalDirection.BULLISH if self._score > 0.1
            else SignalDirection.BEARISH if self._score < -0.1
            else SignalDirection.NEUTRAL
        )
        return SignalResult(
            score=self._score,
            direction=direction,
            fired=self._fired,
        )

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {}


class _IndexDependentGenerator(AtomicSignalGenerator):
    """Generator whose score depends on the index (for sequence testing)."""

    def __init__(
        self,
        name: str = "idx_dep",
        warmup: int = 5,
        buy_from: int = 10,
        sell_from: int = 20,
    ):
        self._name = name
        self._warmup = warmup
        self._buy_from = buy_from
        self._sell_from = sell_from

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def warmup_period(self) -> int:
        return self._warmup

    def evaluate(
        self,
        data: Sequence[PriceData],
        index: int,
        hub: IndicatorHub,
    ) -> SignalResult:
        if index < self._warmup:
            return SignalResult.neutral()
        if index >= self._sell_from:
            return SignalResult(
                score=-0.8, direction=SignalDirection.BEARISH, fired=True,
            )
        if index >= self._buy_from:
            return SignalResult(
                score=0.8, direction=SignalDirection.BULLISH, fired=True,
            )
        return SignalResult.neutral()

    @property
    def param_space(self) -> dict[str, list[Any]]:
        return {}


# ======================================================================
# StackedDailyStrategy tests
# ======================================================================


class TestStackedDailyStrategy:
    def test_name(self):
        strategy = StackedDailyStrategy(
            name="TestStack",
            config=StackConfig(),
            specs=[],
        )
        assert strategy.name == "TestStack"

    def test_warmup_empty_specs(self):
        strategy = StackedDailyStrategy(
            name="Empty",
            config=StackConfig(),
            specs=[],
        )
        assert strategy.warmup_period == 0

    def test_warmup_single_spec(self):
        gen = _FixedGenerator(warmup=10)
        strategy = StackedDailyStrategy(
            name="Single",
            config=StackConfig(),
            specs=[TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5)],
        )
        assert strategy.warmup_period == 10

    def test_warmup_max_of_specs(self):
        gen1 = _FixedGenerator(name="a", warmup=5)
        gen2 = _FixedGenerator(name="b", warmup=15)
        gen3 = _FixedGenerator(name="c", warmup=10)
        strategy = StackedDailyStrategy(
            name="Multi",
            config=StackConfig(),
            specs=[
                TimeframeSignalSpec(generator=gen1, timeframe=Timeframe.M5),
                TimeframeSignalSpec(generator=gen2, timeframe=Timeframe.M5),
                TimeframeSignalSpec(generator=gen3, timeframe=Timeframe.M5),
            ],
        )
        assert strategy.warmup_period == 15

    def test_hold_during_warmup(self):
        gen = _FixedGenerator(warmup=10, score=0.9, fired=True)
        strategy = StackedDailyStrategy(
            name="Test",
            config=StackConfig(buy_threshold=0.3, require_fire=True),
            specs=[TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5)],
        )
        data = _make_trending_up(30)
        signal = strategy.evaluate(data, 5)  # index < warmup
        assert signal == Signal.HOLD

    def test_buy_signal(self):
        gen = _FixedGenerator(warmup=5, score=0.8, fired=True)
        strategy = StackedDailyStrategy(
            name="BuyTest",
            config=StackConfig(
                buy_threshold=0.3,
                require_fire=True,
                min_fire_count=1,
            ),
            specs=[TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5)],
        )
        data = _make_trending_up(20)
        signal = strategy.evaluate(data, 10)
        assert signal == Signal.BUY

    def test_sell_signal(self):
        gen = _FixedGenerator(warmup=5, score=-0.8, fired=True)
        strategy = StackedDailyStrategy(
            name="SellTest",
            config=StackConfig(
                sell_threshold=0.3,
                require_fire=True,
                min_fire_count=1,
            ),
            specs=[TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5)],
        )
        data = _make_trending_down(20)
        signal = strategy.evaluate(data, 10)
        assert signal == Signal.SELL

    def test_hold_neutral_score(self):
        gen = _FixedGenerator(warmup=5, score=0.0, fired=True)
        strategy = StackedDailyStrategy(
            name="HoldTest",
            config=StackConfig(buy_threshold=0.3, sell_threshold=0.3),
            specs=[TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5)],
        )
        data = _make_flat(20)
        signal = strategy.evaluate(data, 10)
        assert signal == Signal.HOLD

    def test_hold_below_threshold(self):
        gen = _FixedGenerator(warmup=5, score=0.2, fired=True)
        strategy = StackedDailyStrategy(
            name="BelowThreshold",
            config=StackConfig(buy_threshold=0.5),
            specs=[TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5)],
        )
        data = _make_trending_up(20)
        signal = strategy.evaluate(data, 10)
        assert signal == Signal.HOLD

    def test_hold_when_fire_required_but_not_fired(self):
        gen = _FixedGenerator(warmup=5, score=0.8, fired=False)
        strategy = StackedDailyStrategy(
            name="NoFire",
            config=StackConfig(
                buy_threshold=0.3,
                require_fire=True,
                min_fire_count=1,
            ),
            specs=[TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5)],
        )
        data = _make_trending_up(20)
        signal = strategy.evaluate(data, 10)
        assert signal == Signal.HOLD

    def test_multiple_generators_buy(self):
        """Multiple concordant bullish generators produce BUY."""
        gen1 = _FixedGenerator(name="a", warmup=5, score=0.7, fired=True)
        gen2 = _FixedGenerator(name="b", warmup=5, score=0.5, fired=True, category="trend")
        strategy = StackedDailyStrategy(
            name="MultiGen",
            config=StackConfig(
                buy_threshold=0.3,
                require_fire=True,
                min_fire_count=1,
            ),
            specs=[
                TimeframeSignalSpec(generator=gen1, timeframe=Timeframe.M5),
                TimeframeSignalSpec(generator=gen2, timeframe=Timeframe.M5),
            ],
        )
        data = _make_trending_up(20)
        signal = strategy.evaluate(data, 10)
        assert signal == Signal.BUY

    def test_conflicting_generators_hold(self):
        """Conflicting generators with high threshold produce HOLD."""
        gen1 = _FixedGenerator(name="bull", warmup=5, score=0.5, fired=True)
        gen2 = _FixedGenerator(name="bear", warmup=5, score=-0.5, fired=True)
        strategy = StackedDailyStrategy(
            name="Conflict",
            config=StackConfig(
                buy_threshold=0.3,
                sell_threshold=0.3,
                require_fire=False,
            ),
            specs=[
                TimeframeSignalSpec(generator=gen1, timeframe=Timeframe.M5),
                TimeframeSignalSpec(generator=gen2, timeframe=Timeframe.M5),
            ],
        )
        data = _make_flat(20)
        signal = strategy.evaluate(data, 10)
        # Weighted average of 0.5 and -0.5 = 0.0
        assert signal == Signal.HOLD

    def test_sequential_signals(self):
        """Strategy transitions from HOLD to BUY to SELL across indices."""
        gen = _IndexDependentGenerator(warmup=5, buy_from=10, sell_from=20)
        strategy = StackedDailyStrategy(
            name="Sequential",
            config=StackConfig(
                buy_threshold=0.3,
                sell_threshold=0.3,
                require_fire=True,
                min_fire_count=1,
            ),
            specs=[TimeframeSignalSpec(generator=gen, timeframe=Timeframe.M5)],
        )
        data = _make_trending_up(30)

        # During warmup
        assert strategy.evaluate(data, 3) == Signal.HOLD

        # After warmup, before buy_from
        assert strategy.evaluate(data, 7) == Signal.HOLD

        # In buy zone
        assert strategy.evaluate(data, 15) == Signal.BUY

        # In sell zone
        assert strategy.evaluate(data, 25) == Signal.SELL

    def test_weighted_specs(self):
        """Spec weights influence the composite score."""
        # Strong bearish with low weight, weak bullish with high weight
        gen_bear = _FixedGenerator(name="bear", warmup=5, score=-0.8, fired=True)
        gen_bull = _FixedGenerator(name="bull", warmup=5, score=0.3, fired=True, category="trend")
        strategy = StackedDailyStrategy(
            name="Weighted",
            config=StackConfig(
                buy_threshold=0.1,
                sell_threshold=0.1,
                require_fire=False,
            ),
            specs=[
                TimeframeSignalSpec(generator=gen_bear, timeframe=Timeframe.M5, weight=0.5),
                TimeframeSignalSpec(generator=gen_bull, timeframe=Timeframe.M5, weight=3.0),
            ],
        )
        data = _make_flat(20)
        signal = strategy.evaluate(data, 10)
        # Weighted avg: (-0.8*0.5 + 0.3*3.0) / (0.5+3.0) = 0.5/3.5 = ~0.143
        assert signal == Signal.BUY

    def test_unanimous_mode_blocks_conflicting(self):
        """UNANIMOUS mode blocks signal when generators disagree."""
        gen1 = _FixedGenerator(name="bull", warmup=5, score=0.6, fired=True)
        gen2 = _FixedGenerator(name="bear", warmup=5, score=-0.3, fired=True, category="trend")
        strategy = StackedDailyStrategy(
            name="Unanimous",
            config=StackConfig(
                buy_threshold=0.3,
                mode=AggregationMode.UNANIMOUS,
                require_fire=False,
            ),
            specs=[
                TimeframeSignalSpec(generator=gen1, timeframe=Timeframe.M5),
                TimeframeSignalSpec(generator=gen2, timeframe=Timeframe.M5),
            ],
        )
        data = _make_flat(20)
        signal = strategy.evaluate(data, 10)
        assert signal == Signal.HOLD

    def test_category_weights_in_strategy(self):
        """Category weights from StackConfig apply to the strategy."""
        gen_mom = _FixedGenerator(
            name="mom", category="momentum", warmup=5, score=0.5, fired=True,
        )
        gen_trend = _FixedGenerator(
            name="trend", category="trend", warmup=5, score=0.5, fired=True,
        )
        strategy = StackedDailyStrategy(
            name="CatWeights",
            config=StackConfig(
                buy_threshold=0.3,
                require_fire=False,
                category_weights={"momentum": 3.0, "trend": 1.0},
            ),
            specs=[
                TimeframeSignalSpec(generator=gen_mom, timeframe=Timeframe.M5, weight=1.0),
                TimeframeSignalSpec(generator=gen_trend, timeframe=Timeframe.M5, weight=1.0),
            ],
        )
        data = _make_flat(20)
        signal = strategy.evaluate(data, 10)
        # Both generators produce 0.5, weighted avg is 0.5 regardless of cat weights
        # (since all scores are the same)
        assert signal == Signal.BUY

    def test_is_trading_strategy(self):
        """StackedDailyStrategy implements TradingStrategy interface."""
        from stockdownloader.strategy.trading_strategy import TradingStrategy

        strategy = StackedDailyStrategy(
            name="Test",
            config=StackConfig(),
            specs=[],
        )
        assert isinstance(strategy, TradingStrategy)

    def test_empty_specs_always_hold(self):
        """No specs means no signals, always HOLD."""
        strategy = StackedDailyStrategy(
            name="Empty",
            config=StackConfig(buy_threshold=0.0),
            specs=[],
        )
        data = _make_flat(10)
        # warmup=0 for empty specs, so index 0 is past warmup
        signal = strategy.evaluate(data, 5)
        assert signal == Signal.HOLD
