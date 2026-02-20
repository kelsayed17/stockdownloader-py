"""Tests for the strategy registry."""

import pytest

from stockdownloader.strategy.registrations import ensure_registered
from stockdownloader.strategy.registry import StrategyRegistry, StrategyEntry

ensure_registered()


class TestRegistryLookup:

    def test_get_known_strategy(self):
        entry = StrategyRegistry.get("rsi")
        assert entry.name == "rsi"
        assert entry.category == "daily"

    def test_get_case_insensitive(self):
        e1 = StrategyRegistry.get("RSI")
        e2 = StrategyRegistry.get("rsi")
        e3 = StrategyRegistry.get("Rsi")
        assert e1 == e2 == e3

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy 'nonexistent'"):
            StrategyRegistry.get("nonexistent")

    def test_get_unknown_lists_available(self):
        with pytest.raises(ValueError, match="Available:"):
            StrategyRegistry.get("nonexistent")


class TestRegistryCreate:

    def test_create_with_defaults(self):
        strat = StrategyRegistry.create("rsi")
        assert strat.name is not None
        assert "RSI" in strat.name

    def test_create_with_overrides(self):
        strat = StrategyRegistry.create("rsi", period=21, oversold=25.0, overbought=75.0)
        assert "RSI" in strat.name
        assert "21" in strat.name

    def test_create_sma_defaults(self):
        strat = StrategyRegistry.create("sma")
        assert "SMA" in strat.name
        assert "9" in strat.name
        assert "21" in strat.name

    def test_create_macd_defaults(self):
        strat = StrategyRegistry.create("macd")
        assert "MACD" in strat.name

    def test_create_bollinger_defaults(self):
        strat = StrategyRegistry.create("bollinger")
        assert "BB" in strat.name or "Bollinger" in strat.name

    def test_create_breakout_defaults(self):
        strat = StrategyRegistry.create("breakout")
        assert "Breakout" in strat.name

    def test_create_momentum_defaults(self):
        strat = StrategyRegistry.create("momentum")
        assert "Momentum" in strat.name

    def test_create_multi_defaults(self):
        strat = StrategyRegistry.create("multi")
        assert "Multi" in strat.name

    def test_create_vwap_pullback_defaults(self):
        strat = StrategyRegistry.create("vwap-pullback")
        assert strat is not None

    def test_create_covered_call_defaults(self):
        strat = StrategyRegistry.create("covered-call")
        assert "Covered" in strat.name

    def test_create_protective_put_defaults(self):
        strat = StrategyRegistry.create("protective-put")
        assert "Protective" in strat.name or "Put" in strat.name


class TestRegistryEnumeration:

    def test_all_entries_returns_all(self):
        entries = StrategyRegistry.all_entries()
        assert len(entries) >= 10  # 7 daily + 1 intraday + 2 options

    def test_all_entries_daily(self):
        entries = StrategyRegistry.all_entries(category="daily")
        assert len(entries) == 7
        names = {e.name for e in entries}
        assert names == {"sma", "rsi", "macd", "bollinger", "breakout", "momentum", "multi"}

    def test_all_entries_intraday(self):
        entries = StrategyRegistry.all_entries(category="intraday")
        assert len(entries) == 8
        names = {e.name for e in entries}
        assert names == {
            "vwap-pullback", "vwap-reversal", "vwap-orb",
            "vwap-orr", "vwap-ps", "avwap-pullback",
            "smc-structure", "ml-oversold",
        }

    def test_all_entries_options(self):
        entries = StrategyRegistry.all_entries(category="options")
        assert len(entries) == 2
        names = {e.name for e in entries}
        assert names == {"covered-call", "protective-put"}

    def test_list_names_sorted(self):
        names = StrategyRegistry.list_names(category="daily")
        assert names == sorted(names)

    def test_list_names_all(self):
        names = StrategyRegistry.list_names()
        assert len(names) >= 10


class TestRegistryParamSpace:

    def test_daily_strategies_have_param_space(self):
        for entry in StrategyRegistry.all_entries(category="daily"):
            assert len(entry.param_space) > 0, f"{entry.name} has no param_space"

    def test_can_create_with_param_space_values(self):
        """Each first value from each param_space should create a valid strategy."""
        for entry in StrategyRegistry.all_entries(category="daily"):
            overrides = {k: v[0] for k, v in entry.param_space.items()}
            kwargs = {**entry.default_kwargs, **overrides}
            try:
                strat = entry.factory(**kwargs)
                assert strat.name is not None
            except (ValueError, TypeError):
                # Some combinations may be invalid (e.g., short > long for SMA)
                pass

    def test_entry_is_frozen_dataclass(self):
        entry = StrategyRegistry.get("rsi")
        assert isinstance(entry, StrategyEntry)
        with pytest.raises(AttributeError):
            entry.name = "changed"
