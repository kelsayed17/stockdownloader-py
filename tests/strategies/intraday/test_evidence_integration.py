"""Integration tests for evidence-based strategies."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.strategies.intraday.gao_momentum import GaoMomentumStrategy
from stockdownloader.strategies.intraday.noise_boundary import NoiseBoundaryStrategy
from stockdownloader.strategies.intraday.connors_rsi2 import ConnorsRSI2Strategy
from stockdownloader.strategies.intraday.fomc_drift import FOMCDriftStrategy
from stockdownloader.strategies.intraday.vix_regime_filter import VixFilteredStrategy


class TestAllStrategiesConstructible:
    """Every new strategy must be constructible with defaults."""

    def test_gao_momentum_default(self):
        s = GaoMomentumStrategy()
        assert s.name
        assert s.warmup_period > 0

    def test_noise_boundary_default(self):
        s = NoiseBoundaryStrategy()
        assert s.name
        assert s.warmup_period > 0

    def test_connors_rsi2_default(self):
        s = ConnorsRSI2Strategy()
        assert s.name
        assert s.warmup_period > 0

    def test_fomc_drift_default(self):
        s = FOMCDriftStrategy()
        assert s.name
        assert s.warmup_period > 0

    def test_vix_filtered_wrapper(self):
        inner = GaoMomentumStrategy()
        wrapped = VixFilteredStrategy(inner=inner, allowed_regimes=["mid", "high"])
        assert "VIX" in wrapped.name
        assert wrapped.warmup_period > 0


class TestOverridesWork:
    """New strategies must accept **overrides via unified constructor."""

    def test_gao_overrides(self):
        s = GaoMomentumStrategy(entry_start_bar=70, require_dual_signal=False)
        assert s._c.entry_start_bar == 70
        assert s._c.require_dual_signal is False

    def test_noise_overrides(self):
        s = NoiseBoundaryStrategy(lookback_days=10, vol_multiplier=Decimal("1.5"))
        assert s._c.lookback_days == 10

    def test_connors_overrides(self):
        s = ConnorsRSI2Strategy(rsi_threshold=Decimal("10"))
        assert s._c.rsi_threshold == Decimal("10")

    def test_fomc_overrides(self):
        s = FOMCDriftStrategy(require_high_vix=False)
        assert s._c.require_high_vix is False


class TestConfigSerialization:
    """Config round-trip via StrategyConfigMixin."""

    def test_gao_config_round_trip(self):
        from stockdownloader.strategies.intraday.gao_momentum import GaoMomentumConfig
        c = GaoMomentumConfig(min_r1_magnitude=Decimal("0.001"))
        c2 = GaoMomentumConfig.from_json(c.to_json())
        assert c == c2

    def test_noise_config_round_trip(self):
        from stockdownloader.strategies.intraday.noise_boundary import NoiseBoundaryConfig
        c = NoiseBoundaryConfig(vol_multiplier=Decimal("1.5"))
        c2 = NoiseBoundaryConfig.from_json(c.to_json())
        assert c == c2

    def test_connors_config_round_trip(self):
        from stockdownloader.strategies.intraday.connors_rsi2 import ConnorsRSI2Config
        c = ConnorsRSI2Config(rsi_threshold=Decimal("3"))
        c2 = ConnorsRSI2Config.from_json(c.to_json())
        assert c == c2

    def test_fomc_config_round_trip(self):
        from stockdownloader.strategies.intraday.fomc_drift import FOMCDriftConfig
        c = FOMCDriftConfig(vix_threshold=Decimal("25"))
        c2 = FOMCDriftConfig.from_json(c.to_json())
        assert c == c2


class TestRegistryCreation:
    """Verify new strategies work via StrategyRegistry.create()."""

    @pytest.fixture(autouse=True)
    def _register(self):
        from stockdownloader.strategies.loader import ensure_registered
        ensure_registered()

    def test_gao_via_registry(self):
        from stockdownloader.strategies.registry import StrategyRegistry
        s = StrategyRegistry.create("spy-gao-momentum")
        assert s.name

    def test_noise_via_registry(self):
        from stockdownloader.strategies.registry import StrategyRegistry
        s = StrategyRegistry.create("spy-noise-boundary")
        assert s.name

    def test_connors_via_registry(self):
        from stockdownloader.strategies.registry import StrategyRegistry
        s = StrategyRegistry.create("spy-connors-rsi2")
        assert s.name

    def test_fomc_via_registry(self):
        from stockdownloader.strategies.registry import StrategyRegistry
        s = StrategyRegistry.create("spy-fomc-drift")
        assert s.name
