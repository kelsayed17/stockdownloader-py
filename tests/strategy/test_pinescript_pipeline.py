"""Integration tests for the Python→PineScript auto-generation pipeline.

Verifies:
- Round-trip: strategy.to_pinescript() → PineScriptGenerator.generate() succeeds
- STRATEGY_CATALOG entries all produce valid Pine Script
- Pipeline wrappers produce the same structural output as direct calls
"""

from __future__ import annotations

import pytest

from stockdownloader.util.pinescript_generator import PineScriptGenerator
from stockdownloader.util.pinescript_strategies import STRATEGY_CATALOG


# ======================================================================
# Round-trip generation tests
# ======================================================================


class TestDailyRoundTrip:
    """Each daily strategy's to_pinescript() → generate() works end-to-end."""

    def setup_method(self):
        self.gen = PineScriptGenerator()

    def _validate_pine(self, pine: str, label: str) -> None:
        """Basic structural checks on generated Pine Script."""
        assert pine, f"{label}: empty output"
        assert "indicator(" in pine, f"{label}: missing indicator()"
        assert "alertcondition(" in pine, f"{label}: missing alerts"
        assert len(pine) > 200, f"{label}: output too short"
        assert len(pine.splitlines()) > 20, f"{label}: too few lines"

    def test_sma_crossover(self):
        from stockdownloader.strategy.daily import SMACrossoverStrategy
        defn = SMACrossoverStrategy(9, 21).to_pinescript()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "SMA")
        assert "ta.sma" in pine
        assert "ta.crossover" in pine

    def test_rsi(self):
        from stockdownloader.strategy.daily import RSIStrategy
        defn = RSIStrategy(14, 30, 70).to_pinescript()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "RSI")
        assert "ta.rsi" in pine

    def test_macd(self):
        from stockdownloader.strategy.daily import MACDStrategy
        defn = MACDStrategy(12, 26, 9).to_pinescript()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "MACD")
        assert "ta.macd" in pine

    def test_bollinger_rsi(self):
        from stockdownloader.strategy.daily import BollingerBandRSIStrategy
        defn = BollingerBandRSIStrategy().to_pinescript()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "BB-RSI")
        assert "ta.bb" in pine
        assert "ta.rsi" in pine
        assert "stochK" in pine

    def test_breakout(self):
        from stockdownloader.strategy.daily import BreakoutStrategy
        defn = BreakoutStrategy().to_pinescript()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "Breakout")
        assert "isSqueeze" in pine
        assert "volSpike" in pine

    def test_momentum_confluence(self):
        from stockdownloader.strategy.daily import MomentumConfluenceStrategy
        defn = MomentumConfluenceStrategy().to_pinescript()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "MomConf")
        assert "ta.macd" in pine
        assert "ta.ema" in pine
        assert "ta.obv" in pine

    def test_multi_indicator(self):
        from stockdownloader.strategy.daily import MultiIndicatorStrategy
        defn = MultiIndicatorStrategy().to_pinescript()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "MultiInd")
        assert "buyScore" in pine
        assert "sellScore" in pine

    def test_dmi_vwap(self):
        from stockdownloader.strategy.dmi_vwap_strategy import DmiVwapStrategy
        defn = DmiVwapStrategy().to_pinescript()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "DMI+VWAP")
        assert "ta.dmi" in pine
        assert "isNewSession" in pine
        assert "Buy Call" in pine
        assert "Buy Put" in pine


# ======================================================================
# STRATEGY_CATALOG tests
# ======================================================================


class TestStrategyCatalog:
    """All STRATEGY_CATALOG entries produce valid Pine Script."""

    def setup_method(self):
        self.gen = PineScriptGenerator()

    def test_catalog_count(self):
        """Catalog has the expected number of strategies."""
        assert len(STRATEGY_CATALOG) == 13

    def test_all_catalog_entries_generate(self):
        """Every factory in the catalog produces valid Pine Script."""
        for name, factory in STRATEGY_CATALOG.items():
            defn = factory()
            pine = self.gen.generate(defn)
            assert pine, f"{name}: empty output"
            assert "indicator(" in pine, f"{name}: missing indicator()"
            assert len(pine) > 200, f"{name}: output too short ({len(pine)})"
            assert len(pine.splitlines()) > 20, f"{name}: too few lines"

    def test_catalog_keys(self):
        """Catalog contains all expected strategy keys."""
        expected = {
            "sma_crossover", "rsi", "macd", "macd_obv",
            "bollinger_rsi", "dmi_vwap", "momentum_confluence",
            "breakout", "vwap_pullback", "vwap_reversal",
            "vwap_or_breakout", "vwap_or_reversal", "vwap_pattern_scalp",
        }
        assert set(STRATEGY_CATALOG.keys()) == expected


# ======================================================================
# Pipeline wrapper consistency
# ======================================================================


class TestPipelineWrapperConsistency:
    """Verify pipeline wrappers produce structurally sound output."""

    def test_sma_wrapper_params_match_python(self):
        """SMA wrapper defaults match Python strategy defaults."""
        from stockdownloader.util.pinescript_strategies import sma_crossover_strategy
        defn = sma_crossover_strategy()
        defaults = {i.name: i.default for i in defn.inputs}
        # These match SMACrossoverStrategy(9, 21) defaults
        assert defaults["fastPeriod"] == 9
        assert defaults["slowPeriod"] == 21

    def test_sma_wrapper_custom_params(self):
        """SMA wrapper passes custom params through to Python strategy."""
        from stockdownloader.util.pinescript_strategies import sma_crossover_strategy
        defn = sma_crossover_strategy(short_period=20, long_period=50)
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["fastPeriod"] == 20
        assert defaults["slowPeriod"] == 50

    def test_rsi_wrapper_params(self):
        from stockdownloader.util.pinescript_strategies import rsi_strategy
        defn = rsi_strategy(period=7, oversold=35.0, overbought=65.0)
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["rsiPeriod"] == 7
        assert defaults["oversold"] == 35.0
        assert defaults["overbought"] == 65.0

    def test_macd_wrapper_params(self):
        from stockdownloader.util.pinescript_strategies import macd_strategy
        defn = macd_strategy(fast=8, slow=21, signal=5)
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["macdFast"] == 8
        assert defaults["macdSlow"] == 21
        assert defaults["macdSignal"] == 5

    def test_bollinger_wrapper_no_params(self):
        from stockdownloader.util.pinescript_strategies import bollinger_rsi_strategy
        defn = bollinger_rsi_strategy()
        assert defn.short_name == "BB-RSI"

    def test_dmi_vwap_wrapper(self):
        from stockdownloader.util.pinescript_strategies import dmi_vwap_strategy
        defn = dmi_vwap_strategy()
        assert defn.use_session_filter is True
        assert defn.long_label == "Buy Call"
        assert defn.short_label == "Buy Put"

    def test_momentum_wrapper(self):
        from stockdownloader.util.pinescript_strategies import (
            momentum_confluence_strategy,
        )
        defn = momentum_confluence_strategy()
        assert defn.short_name == "MOM-CONF"

    def test_breakout_wrapper(self):
        from stockdownloader.util.pinescript_strategies import breakout_strategy
        defn = breakout_strategy()
        assert defn.short_name == "BRKOUT"

    def test_macd_obv_stays_handwritten(self):
        """MACD+OBV has no Python counterpart — stays hand-written."""
        from stockdownloader.util.pinescript_strategies import macd_obv_strategy
        defn = macd_obv_strategy()
        assert defn.name == "MACD + OBV"
        assert "Walk-forward" in defn.description


# ======================================================================
# Intraday mode round-trip
# ======================================================================


class TestIntradayModeRoundTrip:
    """Intraday pinescript_mode() output can be converted and generated."""

    def setup_method(self):
        self.gen = PineScriptGenerator()

    def _mode_to_pine(self, mode: "ModeDefinition") -> str:
        from stockdownloader.util.pinescript_generator import mode_to_strategy
        from stockdownloader.util.pinescript_strategies import (
            _vwap_shared_infrastructure,
        )
        defn = mode_to_strategy(
            mode, shared=_vwap_shared_infrastructure(),
            name_override=f"Test {mode.name}",
            short_name_override=f"T-{mode.short_name.upper()}",
        )
        return self.gen.generate(defn)

    def test_pullback_round_trip(self):
        from stockdownloader.strategy.intraday import PullbackStrategy
        pine = self._mode_to_pine(PullbackStrategy.pinescript_mode())
        assert "indicator(" in pine

    def test_reversal_round_trip(self):
        from stockdownloader.strategy.intraday import ReversalStrategy
        pine = self._mode_to_pine(ReversalStrategy.pinescript_mode())
        assert "indicator(" in pine

    def test_or_breakout_round_trip(self):
        from stockdownloader.strategy.intraday import ORBreakoutStrategy
        pine = self._mode_to_pine(ORBreakoutStrategy.pinescript_mode())
        assert "indicator(" in pine

    def test_or_reversal_round_trip(self):
        from stockdownloader.strategy.intraday import ORReversalStrategy
        pine = self._mode_to_pine(ORReversalStrategy.pinescript_mode())
        assert "indicator(" in pine

    def test_pattern_scalp_round_trip(self):
        from stockdownloader.strategy.intraday import PatternScalpStrategy
        pine = self._mode_to_pine(PatternScalpStrategy.pinescript_mode())
        assert "indicator(" in pine
