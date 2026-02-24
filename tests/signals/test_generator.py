"""Tests for SignalResult, SignalDirection, and AtomicSignalGenerator ABC."""
from __future__ import annotations

import pytest
from decimal import Decimal

from stockdownloader.signals.generator import (
    AtomicSignalGenerator,
    SignalDirection,
    SignalResult,
)
from stockdownloader.core.models.price import PriceData
from stockdownloader.indicators.hub import IndicatorHub


# ======================================================================
# Helpers
# ======================================================================


def _make_price(close: str, date: str = "2025-01-01") -> PriceData:
    c = Decimal(close)
    return PriceData(
        date=date,
        open=c,
        high=c + Decimal("1"),
        low=c - Decimal("1"),
        close=c,
        adj_close=c,
        volume=1000,
    )


class _DummyGenerator(AtomicSignalGenerator):
    """Minimal concrete generator for testing the ABC contract."""

    def __init__(self, name: str = "dummy", category: str = "momentum"):
        self._name = name
        self._category = category

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return f"Dummy({self._name})"

    @property
    def category(self) -> str:
        return self._category

    @property
    def warmup_period(self) -> int:
        return 5

    def evaluate(self, data, index, hub) -> SignalResult:
        if index < self.warmup_period:
            return SignalResult.neutral()
        return SignalResult(
            score=0.5,
            direction=SignalDirection.BULLISH,
            fired=True,
            confidence=0.9,
            metadata={"test": True},
        )

    @property
    def param_space(self) -> dict:
        return {"period": [7, 14]}


# ======================================================================
# SignalDirection tests
# ======================================================================


class TestSignalDirection:
    def test_enum_values(self):
        assert SignalDirection.BULLISH.value == "bullish"
        assert SignalDirection.BEARISH.value == "bearish"
        assert SignalDirection.NEUTRAL.value == "neutral"

    def test_enum_members_count(self):
        assert len(SignalDirection) == 3

    def test_enum_identity(self):
        assert SignalDirection.BULLISH is SignalDirection.BULLISH
        assert SignalDirection.BULLISH != SignalDirection.BEARISH

    def test_enum_string_representation(self):
        assert "BULLISH" in repr(SignalDirection.BULLISH)


# ======================================================================
# SignalResult tests
# ======================================================================


class TestSignalResult:
    def test_basic_construction(self):
        result = SignalResult(
            score=0.7,
            direction=SignalDirection.BULLISH,
            fired=True,
        )
        assert result.score == 0.7
        assert result.direction == SignalDirection.BULLISH
        assert result.fired is True
        assert result.confidence == 1.0  # default
        assert result.metadata == {}  # default

    def test_construction_with_all_fields(self):
        meta = {"rsi": 25.0, "threshold": 30.0}
        result = SignalResult(
            score=-0.8,
            direction=SignalDirection.BEARISH,
            fired=False,
            confidence=0.6,
            metadata=meta,
        )
        assert result.score == -0.8
        assert result.direction == SignalDirection.BEARISH
        assert result.fired is False
        assert result.confidence == 0.6
        assert result.metadata == meta

    def test_neutral_factory(self):
        n = SignalResult.neutral()
        assert n.score == 0.0
        assert n.direction == SignalDirection.NEUTRAL
        assert n.fired is False
        assert n.confidence == 1.0
        assert n.metadata == {}

    def test_neutral_is_consistent(self):
        n1 = SignalResult.neutral()
        n2 = SignalResult.neutral()
        assert n1 == n2

    def test_frozen_dataclass(self):
        result = SignalResult(
            score=0.5,
            direction=SignalDirection.BULLISH,
            fired=True,
        )
        with pytest.raises(AttributeError):
            result.score = 0.9  # type: ignore[misc]

    def test_score_boundary_positive(self):
        result = SignalResult(
            score=1.0,
            direction=SignalDirection.BULLISH,
            fired=True,
        )
        assert result.score == 1.0

    def test_score_boundary_negative(self):
        result = SignalResult(
            score=-1.0,
            direction=SignalDirection.BEARISH,
            fired=True,
        )
        assert result.score == -1.0

    def test_score_zero(self):
        result = SignalResult(
            score=0.0,
            direction=SignalDirection.NEUTRAL,
            fired=False,
        )
        assert result.score == 0.0

    def test_equality(self):
        r1 = SignalResult(score=0.5, direction=SignalDirection.BULLISH, fired=True)
        r2 = SignalResult(score=0.5, direction=SignalDirection.BULLISH, fired=True)
        assert r1 == r2

    def test_inequality_on_score(self):
        r1 = SignalResult(score=0.5, direction=SignalDirection.BULLISH, fired=True)
        r2 = SignalResult(score=0.6, direction=SignalDirection.BULLISH, fired=True)
        assert r1 != r2

    def test_inequality_on_direction(self):
        r1 = SignalResult(score=0.5, direction=SignalDirection.BULLISH, fired=True)
        r2 = SignalResult(score=0.5, direction=SignalDirection.BEARISH, fired=True)
        assert r1 != r2

    def test_inequality_on_fired(self):
        r1 = SignalResult(score=0.5, direction=SignalDirection.BULLISH, fired=True)
        r2 = SignalResult(score=0.5, direction=SignalDirection.BULLISH, fired=False)
        assert r1 != r2

    def test_metadata_default_is_independent(self):
        """Each instance should get its own empty dict."""
        r1 = SignalResult(score=0.0, direction=SignalDirection.NEUTRAL, fired=False)
        r2 = SignalResult(score=0.0, direction=SignalDirection.NEUTRAL, fired=False)
        assert r1.metadata is not r2.metadata


# ======================================================================
# AtomicSignalGenerator ABC tests
# ======================================================================


class TestAtomicSignalGenerator:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            AtomicSignalGenerator()  # type: ignore[abstract]

    def test_concrete_subclass_methods(self):
        gen = _DummyGenerator()
        assert gen.name == "dummy"
        assert gen.display_name == "Dummy(dummy)"
        assert gen.category == "momentum"
        assert gen.warmup_period == 5

    def test_evaluate_returns_neutral_during_warmup(self):
        gen = _DummyGenerator()
        data = [_make_price("100") for _ in range(10)]
        hub = IndicatorHub()
        result = gen.evaluate(data, 2, hub)  # index < warmup=5
        assert result == SignalResult.neutral()

    def test_evaluate_returns_signal_after_warmup(self):
        gen = _DummyGenerator()
        data = [_make_price("100") for _ in range(10)]
        hub = IndicatorHub()
        result = gen.evaluate(data, 5, hub)
        assert result.score == 0.5
        assert result.direction == SignalDirection.BULLISH
        assert result.fired is True
        assert result.confidence == 0.9
        assert result.metadata == {"test": True}

    def test_param_space(self):
        gen = _DummyGenerator()
        space = gen.param_space
        assert "period" in space
        assert space["period"] == [7, 14]

    def test_different_categories(self):
        gen_momentum = _DummyGenerator(category="momentum")
        gen_trend = _DummyGenerator(category="trend")
        gen_vol = _DummyGenerator(category="volatility")
        gen_volume = _DummyGenerator(category="volume")

        assert gen_momentum.category == "momentum"
        assert gen_trend.category == "trend"
        assert gen_vol.category == "volatility"
        assert gen_volume.category == "volume"
