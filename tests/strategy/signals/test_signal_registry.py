"""Tests for SignalGeneratorRegistry and SignalGeneratorEntry."""
from __future__ import annotations

import pytest

from stockdownloader.strategy.signals.signal_generator import (
    AtomicSignalGenerator,
    SignalDirection,
    SignalResult,
)
from stockdownloader.strategy.signals.signal_registry import (
    SignalGeneratorEntry,
    SignalGeneratorRegistry,
)


# ======================================================================
# Helpers
# ======================================================================


class _StubGenerator(AtomicSignalGenerator):
    """Minimal generator for registry tests."""

    def __init__(self, period: int = 14, threshold: float = 30.0):
        self._period = period
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"stub_{self._period}"

    @property
    def display_name(self) -> str:
        return f"Stub({self._period})"

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def warmup_period(self) -> int:
        return self._period

    def evaluate(self, data, index, hub) -> SignalResult:
        return SignalResult.neutral()

    @property
    def param_space(self) -> dict:
        return {"period": [7, 14, 21]}


class _TrendGenerator(AtomicSignalGenerator):
    """Second generator to test category filtering."""

    def __init__(self, lookback: int = 50):
        self._lookback = lookback

    @property
    def name(self) -> str:
        return f"trend_{self._lookback}"

    @property
    def display_name(self) -> str:
        return f"Trend({self._lookback})"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def warmup_period(self) -> int:
        return self._lookback

    def evaluate(self, data, index, hub) -> SignalResult:
        return SignalResult.neutral()

    @property
    def param_space(self) -> dict:
        return {"lookback": [20, 50]}


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(SignalGeneratorRegistry._entries)
    SignalGeneratorRegistry.clear()
    yield
    SignalGeneratorRegistry._entries = saved


# ======================================================================
# SignalGeneratorEntry tests
# ======================================================================


class TestSignalGeneratorEntry:
    def test_construction(self):
        entry = SignalGeneratorEntry(
            name="test",
            display_name="Test Generator",
            category="momentum",
            factory=_StubGenerator,
            default_kwargs={"period": 14},
            param_space={"period": [7, 14, 21]},
        )
        assert entry.name == "test"
        assert entry.display_name == "Test Generator"
        assert entry.category == "momentum"
        assert entry.factory is _StubGenerator
        assert entry.default_kwargs == {"period": 14}
        assert entry.param_space == {"period": [7, 14, 21]}

    def test_frozen(self):
        entry = SignalGeneratorEntry(
            name="test",
            display_name="Test",
            category="momentum",
            factory=_StubGenerator,
        )
        with pytest.raises(AttributeError):
            entry.name = "changed"  # type: ignore[misc]

    def test_default_kwargs_empty(self):
        entry = SignalGeneratorEntry(
            name="test",
            display_name="Test",
            category="momentum",
            factory=_StubGenerator,
        )
        assert entry.default_kwargs == {}
        assert entry.param_space == {}


# ======================================================================
# SignalGeneratorRegistry tests
# ======================================================================


class TestSignalGeneratorRegistry:
    def test_register_and_get(self):
        SignalGeneratorRegistry.register(
            name="stub",
            display_name="Stub",
            category="momentum",
            factory=_StubGenerator,
            default_kwargs={"period": 14},
        )
        entry = SignalGeneratorRegistry.get("stub")
        assert entry.name == "stub"
        assert entry.display_name == "Stub"
        assert entry.category == "momentum"

    def test_register_case_insensitive(self):
        SignalGeneratorRegistry.register(
            name="MyGen",
            display_name="My Gen",
            category="trend",
            factory=_TrendGenerator,
        )
        entry = SignalGeneratorRegistry.get("mygen")
        assert entry.name == "mygen"

        entry2 = SignalGeneratorRegistry.get("MYGEN")
        assert entry2.name == "mygen"

    def test_get_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown signal generator"):
            SignalGeneratorRegistry.get("nonexistent")

    def test_get_unknown_lists_available(self):
        SignalGeneratorRegistry.register(
            name="alpha",
            display_name="Alpha",
            category="momentum",
            factory=_StubGenerator,
        )
        SignalGeneratorRegistry.register(
            name="beta",
            display_name="Beta",
            category="trend",
            factory=_TrendGenerator,
        )
        with pytest.raises(ValueError, match="alpha, beta"):
            SignalGeneratorRegistry.get("gamma")

    def test_create_with_defaults(self):
        SignalGeneratorRegistry.register(
            name="stub",
            display_name="Stub",
            category="momentum",
            factory=_StubGenerator,
            default_kwargs={"period": 14, "threshold": 30.0},
        )
        gen = SignalGeneratorRegistry.create("stub")
        assert isinstance(gen, _StubGenerator)
        assert gen._period == 14
        assert gen._threshold == 30.0

    def test_create_with_overrides(self):
        SignalGeneratorRegistry.register(
            name="stub",
            display_name="Stub",
            category="momentum",
            factory=_StubGenerator,
            default_kwargs={"period": 14, "threshold": 30.0},
        )
        gen = SignalGeneratorRegistry.create("stub", period=7, threshold=25.0)
        assert isinstance(gen, _StubGenerator)
        assert gen._period == 7
        assert gen._threshold == 25.0

    def test_create_partial_override(self):
        SignalGeneratorRegistry.register(
            name="stub",
            display_name="Stub",
            category="momentum",
            factory=_StubGenerator,
            default_kwargs={"period": 14, "threshold": 30.0},
        )
        gen = SignalGeneratorRegistry.create("stub", period=21)
        assert gen._period == 21
        assert gen._threshold == 30.0  # kept from defaults

    def test_all_entries_unfiltered(self):
        SignalGeneratorRegistry.register(
            name="alpha",
            display_name="Alpha",
            category="momentum",
            factory=_StubGenerator,
        )
        SignalGeneratorRegistry.register(
            name="beta",
            display_name="Beta",
            category="trend",
            factory=_TrendGenerator,
        )
        entries = SignalGeneratorRegistry.all_entries()
        assert len(entries) == 2
        names = [e.name for e in entries]
        assert names == ["alpha", "beta"]  # sorted

    def test_all_entries_filtered_by_category(self):
        SignalGeneratorRegistry.register(
            name="alpha",
            display_name="Alpha",
            category="momentum",
            factory=_StubGenerator,
        )
        SignalGeneratorRegistry.register(
            name="beta",
            display_name="Beta",
            category="trend",
            factory=_TrendGenerator,
        )
        SignalGeneratorRegistry.register(
            name="gamma",
            display_name="Gamma",
            category="momentum",
            factory=_StubGenerator,
        )
        momentum = SignalGeneratorRegistry.all_entries(category="momentum")
        assert len(momentum) == 2
        assert all(e.category == "momentum" for e in momentum)

        trend = SignalGeneratorRegistry.all_entries(category="trend")
        assert len(trend) == 1
        assert trend[0].name == "beta"

    def test_all_entries_empty_category_returns_nothing(self):
        SignalGeneratorRegistry.register(
            name="alpha",
            display_name="Alpha",
            category="momentum",
            factory=_StubGenerator,
        )
        result = SignalGeneratorRegistry.all_entries(category="volume")
        assert result == []

    def test_list_names(self):
        SignalGeneratorRegistry.register(
            name="charlie",
            display_name="Charlie",
            category="momentum",
            factory=_StubGenerator,
        )
        SignalGeneratorRegistry.register(
            name="alpha",
            display_name="Alpha",
            category="trend",
            factory=_TrendGenerator,
        )
        names = SignalGeneratorRegistry.list_names()
        assert names == ["alpha", "charlie"]  # sorted

    def test_list_names_with_category(self):
        SignalGeneratorRegistry.register(
            name="alpha",
            display_name="Alpha",
            category="momentum",
            factory=_StubGenerator,
        )
        SignalGeneratorRegistry.register(
            name="beta",
            display_name="Beta",
            category="trend",
            factory=_TrendGenerator,
        )
        assert SignalGeneratorRegistry.list_names(category="momentum") == ["alpha"]
        assert SignalGeneratorRegistry.list_names(category="trend") == ["beta"]
        assert SignalGeneratorRegistry.list_names(category="volatility") == []

    def test_clear(self):
        SignalGeneratorRegistry.register(
            name="test",
            display_name="Test",
            category="momentum",
            factory=_StubGenerator,
        )
        assert len(SignalGeneratorRegistry.all_entries()) == 1
        SignalGeneratorRegistry.clear()
        assert len(SignalGeneratorRegistry.all_entries()) == 0

    def test_overwrite_registration(self):
        """Re-registering same name replaces the previous entry."""
        SignalGeneratorRegistry.register(
            name="gen",
            display_name="Version 1",
            category="momentum",
            factory=_StubGenerator,
        )
        SignalGeneratorRegistry.register(
            name="gen",
            display_name="Version 2",
            category="trend",
            factory=_TrendGenerator,
        )
        entry = SignalGeneratorRegistry.get("gen")
        assert entry.display_name == "Version 2"
        assert entry.category == "trend"

    def test_register_with_param_space(self):
        space = {"period": [7, 14, 21], "threshold": [20.0, 30.0]}
        SignalGeneratorRegistry.register(
            name="paramtest",
            display_name="Param Test",
            category="momentum",
            factory=_StubGenerator,
            param_space=space,
        )
        entry = SignalGeneratorRegistry.get("paramtest")
        assert entry.param_space == space

    def test_all_entries_sorted_alphabetically(self):
        for name in ["zeta", "alpha", "mu", "beta"]:
            SignalGeneratorRegistry.register(
                name=name,
                display_name=name.title(),
                category="momentum",
                factory=_StubGenerator,
            )
        entries = SignalGeneratorRegistry.all_entries()
        names = [e.name for e in entries]
        assert names == sorted(names)
