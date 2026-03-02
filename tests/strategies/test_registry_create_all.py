"""Test that every registered strategy can be created and accepts overrides.

Validates the unified constructor pattern: all strategies should be
constructible via ``StrategyRegistry.create(name)`` with no arguments,
and the ``save_optimized`` / ``load_and_create`` round-trip should work.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from stockdownloader.strategies.loader import ensure_registered
from stockdownloader.strategies.registry import StrategyRegistry


@pytest.fixture(autouse=True)
def _register():
    ensure_registered()


# -- Create all strategies with defaults ------------------------------------


class TestCreateAllDefaults:
    """Every registered strategy must be creatable with no overrides."""

    def test_create_all_with_defaults(self):
        entries = StrategyRegistry.all_entries()
        assert len(entries) > 0, "No strategies registered"

        for entry in entries:
            strat = StrategyRegistry.create(entry.name)
            assert strat.name, f"{entry.name} returned empty name"


# -- Override support -------------------------------------------------------


class TestOverrides:
    """Strategies should accept keyword overrides via the registry."""

    def test_intraday_overrides_via_registry(self):
        strat = StrategyRegistry.create("spy-macd-obv", macd_fast=10)
        assert strat._c.macd_fast == 10

    def test_daily_overrides_via_registry(self):
        strat = StrategyRegistry.create(
            "macd", fast_period=8, slow_period=21, signal_period=5,
        )
        assert strat._fast_period == 8

    def test_daily_rsi_overrides(self):
        strat = StrategyRegistry.create("rsi", period=10, oversold=25.0)
        assert strat._period == 10

    def test_daily_sma_overrides(self):
        strat = StrategyRegistry.create("sma", short_period=15, long_period=50)
        assert strat._short_period == 15


# -- Save / load round-trip ------------------------------------------------


class TestSaveLoadRoundTrip:
    """save_optimized → load_and_create should produce an equivalent strategy."""

    def test_save_and_load_daily(self, tmp_path):
        path = tmp_path / "optimized.json"
        params = {"fast_period": 8, "slow_period": 21, "signal_period": 5}
        StrategyRegistry.save_optimized("macd", params, str(path))

        strat = StrategyRegistry.load_and_create(str(path))
        assert strat._fast_period == 8
        assert strat._slow_period == 21
        assert strat._signal_period == 5

    def test_save_and_load_intraday_with_decimal(self, tmp_path):
        path = tmp_path / "optimized.json"
        params = {"macd_fast": 10, "rr_ratio": Decimal("2.0")}
        StrategyRegistry.save_optimized("spy-macd-obv", params, str(path))

        strat = StrategyRegistry.load_and_create(str(path))
        assert strat._c.macd_fast == 10
        assert strat._c.rr_ratio == Decimal("2.0")

    def test_saved_json_is_valid(self, tmp_path):
        path = tmp_path / "cfg.json"
        StrategyRegistry.save_optimized("rsi", {"period": 7}, str(path))
        data = json.loads(path.read_text())
        assert data["strategy"] == "rsi"
        assert data["params"]["period"] == 7

    def test_load_with_extra_overrides(self, tmp_path):
        path = tmp_path / "cfg.json"
        StrategyRegistry.save_optimized("rsi", {"period": 7}, str(path))
        strat = StrategyRegistry.load_and_create(str(path), oversold=20.0)
        assert strat._period == 7


# -- Config serialization on intraday configs ------------------------------


class TestIntradayConfigSerialization:
    """InfraExitConfig subclasses should serialize/deserialize via the mixin."""

    def test_macd_obv_config_round_trip(self):
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVConfig

        c = MACDOBVConfig(macd_fast=10, rr_ratio=Decimal("2.0"))
        j = c.to_json()
        c2 = MACDOBVConfig.from_json(j)
        assert c == c2

    def test_or_breakout_config_round_trip(self):
        from stockdownloader.strategies.intraday.or_breakout import (
            ORBreakoutStrategyConfig,
        )

        c = ORBreakoutStrategyConfig(or_bars=5)
        j = c.to_json()
        c2 = ORBreakoutStrategyConfig.from_json(j)
        assert c == c2


# -- Daily config dataclasses ---------------------------------------------


class TestDailyConfigs:
    """Daily config dataclasses should serialize/deserialize correctly."""

    def test_macd_config_round_trip(self):
        from stockdownloader.strategies.daily.configs import MACDConfig

        c = MACDConfig(fast_period=8, slow_period=21, signal_period=5)
        c2 = MACDConfig.from_json(c.to_json())
        assert c == c2

    def test_rsi_config_round_trip(self):
        from stockdownloader.strategies.daily.configs import RSIConfig

        c = RSIConfig(period=10, oversold=25.0, overbought=75.0)
        c2 = RSIConfig.from_json(c.to_json())
        assert c == c2

    def test_all_daily_configs_have_defaults(self):
        from stockdownloader.strategies.daily.configs import (
            MACDConfig, SMACrossoverConfig, RSIConfig,
            BollingerRSIConfig, BreakoutConfig, MomentumConfig,
            MultiIndicatorConfig,
        )

        for cls in [MACDConfig, SMACrossoverConfig, RSIConfig,
                    BollingerRSIConfig, BreakoutConfig, MomentumConfig,
                    MultiIndicatorConfig]:
            c = cls()
            j = c.to_json()
            c2 = cls.from_json(j)
            assert c == c2, f"{cls.__name__} round-trip failed"
