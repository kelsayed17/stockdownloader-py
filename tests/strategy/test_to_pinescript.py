"""Tests for to_pinescript() / pinescript_mode() on strategy classes.

Verifies that each Python strategy can auto-generate its PineScript
definition and that parameters propagate correctly.
"""

from __future__ import annotations

import pytest

from stockdownloader.util.pinescript_models import (
    ModeDefinition,
    StrategyDefinition,
)


# ======================================================================
# Daily strategy to_pinescript()
# ======================================================================


class TestSMACrossoverToPine:
    """SMACrossoverStrategy.to_pinescript()."""

    def test_returns_strategy_definition(self):
        from stockdownloader.strategy.daily import SMACrossoverStrategy
        defn = SMACrossoverStrategy(9, 21).to_pinescript()
        assert isinstance(defn, StrategyDefinition)
        assert defn.name == "SMA Crossover"
        assert defn.short_name == "SMA-X"

    def test_default_params(self):
        from stockdownloader.strategy.daily import SMACrossoverStrategy
        defn = SMACrossoverStrategy(9, 21).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["fastPeriod"] == 9
        assert defaults["slowPeriod"] == 21

    def test_custom_params_propagate(self):
        from stockdownloader.strategy.daily import SMACrossoverStrategy
        defn = SMACrossoverStrategy(15, 50).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["fastPeriod"] == 15
        assert defaults["slowPeriod"] == 50

    def test_has_indicators(self):
        from stockdownloader.strategy.daily import SMACrossoverStrategy
        defn = SMACrossoverStrategy(9, 21).to_pinescript()
        names = [ind.var_name for ind in defn.indicators]
        assert "smaFast" in names
        assert "smaSlow" in names

    def test_long_only(self):
        from stockdownloader.strategy.daily import SMACrossoverStrategy
        defn = SMACrossoverStrategy(9, 21).to_pinescript()
        assert defn.long_entry is not None
        assert defn.long_exit is not None
        assert defn.short_entry is None

    def test_conditions_contain_crossover(self):
        from stockdownloader.strategy.daily import SMACrossoverStrategy
        defn = SMACrossoverStrategy(9, 21).to_pinescript()
        assert "ta.crossover" in defn.long_entry.expr
        assert "ta.crossunder" in defn.long_exit.expr


class TestRSIToPine:
    """RSIStrategy.to_pinescript()."""

    def test_returns_strategy_definition(self):
        from stockdownloader.strategy.daily import RSIStrategy
        defn = RSIStrategy(14, 30, 70).to_pinescript()
        assert isinstance(defn, StrategyDefinition)
        assert defn.short_name == "RSI"

    def test_default_params(self):
        from stockdownloader.strategy.daily import RSIStrategy
        defn = RSIStrategy(14, 30, 70).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["rsiPeriod"] == 14
        assert defaults["oversold"] == 30.0
        assert defaults["overbought"] == 70.0

    def test_custom_params_propagate(self):
        from stockdownloader.strategy.daily import RSIStrategy
        defn = RSIStrategy(7, 25.0, 75.0).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["rsiPeriod"] == 7
        assert defaults["oversold"] == 25.0
        assert defaults["overbought"] == 75.0

    def test_has_rsi_indicator(self):
        from stockdownloader.strategy.daily import RSIStrategy
        defn = RSIStrategy(14, 30, 70).to_pinescript()
        exprs = [ind.code for ind in defn.indicators]
        assert any("ta.rsi" in e for e in exprs)

    def test_has_entry_and_exit(self):
        from stockdownloader.strategy.daily import RSIStrategy
        defn = RSIStrategy(14, 30, 70).to_pinescript()
        assert defn.long_entry is not None
        assert defn.short_entry is not None


class TestMACDToPine:
    """MACDStrategy.to_pinescript()."""

    def test_returns_strategy_definition(self):
        from stockdownloader.strategy.daily import MACDStrategy
        defn = MACDStrategy(12, 26, 9).to_pinescript()
        assert isinstance(defn, StrategyDefinition)
        assert defn.short_name == "MACD"

    def test_default_params(self):
        from stockdownloader.strategy.daily import MACDStrategy
        defn = MACDStrategy(12, 26, 9).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["macdFast"] == 12
        assert defaults["macdSlow"] == 26
        assert defaults["macdSignal"] == 9

    def test_custom_params(self):
        from stockdownloader.strategy.daily import MACDStrategy
        defn = MACDStrategy(8, 21, 5).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["macdFast"] == 8
        assert defaults["macdSlow"] == 21
        assert defaults["macdSignal"] == 5

    def test_has_macd_indicators(self):
        from stockdownloader.strategy.daily import MACDStrategy
        defn = MACDStrategy(12, 26, 9).to_pinescript()
        all_names = " ".join(ind.var_name for ind in defn.indicators)
        assert "macdLine" in all_names
        assert "macdSignal" in all_names

    def test_conditions(self):
        from stockdownloader.strategy.daily import MACDStrategy
        defn = MACDStrategy(12, 26, 9).to_pinescript()
        assert "ta.crossover" in defn.long_entry.expr
        assert "ta.crossunder" in defn.short_entry.expr


class TestBollingerRSIToPine:
    """BollingerBandRSIStrategy.to_pinescript()."""

    def test_returns_strategy_definition(self):
        from stockdownloader.strategy.daily import BollingerBandRSIStrategy
        defn = BollingerBandRSIStrategy().to_pinescript()
        assert isinstance(defn, StrategyDefinition)
        assert defn.short_name == "BB-RSI"

    def test_has_all_indicators(self):
        from stockdownloader.strategy.daily import BollingerBandRSIStrategy
        defn = BollingerBandRSIStrategy().to_pinescript()
        exprs = " ".join(ind.code for ind in defn.indicators)
        assert "ta.bb" in exprs
        assert "ta.rsi" in exprs
        assert "ta.stoch" in exprs
        assert "ta.dmi" in exprs

    def test_conditions_reference_stoch(self):
        from stockdownloader.strategy.daily import BollingerBandRSIStrategy
        defn = BollingerBandRSIStrategy().to_pinescript()
        assert "stochK" in defn.long_entry.expr
        assert "adxValue" in defn.long_entry.expr

    def test_custom_params(self):
        from stockdownloader.strategy.daily import BollingerBandRSIStrategy
        defn = BollingerBandRSIStrategy(
            bb_period=30, rsi_period=21, adx_threshold=30,
        ).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["bbPeriod"] == 30
        assert defaults["rsiPeriod"] == 21
        assert defaults["adxMax"] == 30


class TestBreakoutToPine:
    """BreakoutStrategy.to_pinescript()."""

    def test_returns_strategy_definition(self):
        from stockdownloader.strategy.daily import BreakoutStrategy
        defn = BreakoutStrategy().to_pinescript()
        assert isinstance(defn, StrategyDefinition)
        assert defn.short_name == "BRKOUT"

    def test_has_squeeze_indicators(self):
        from stockdownloader.strategy.daily import BreakoutStrategy
        defn = BreakoutStrategy().to_pinescript()
        names = [ind.var_name for ind in defn.indicators]
        assert "isSqueeze" in names
        assert "volSpike" in names
        assert "atrExpanding" in names

    def test_conditions(self):
        from stockdownloader.strategy.daily import BreakoutStrategy
        defn = BreakoutStrategy().to_pinescript()
        long_expr = defn.long_entry.expr
        assert "isSqueeze" in long_expr
        assert "wasSqueeze" in long_expr
        assert "volSpike" in long_expr

    def test_custom_params(self):
        from stockdownloader.strategy.daily import BreakoutStrategy
        defn = BreakoutStrategy(bb_period=30, volume_multiplier=2.0).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["bbPeriod"] == 30
        assert defaults["volMult"] == 2.0


class TestMomentumConfluenceToPine:
    """MomentumConfluenceStrategy.to_pinescript()."""

    def test_returns_strategy_definition(self):
        from stockdownloader.strategy.daily import MomentumConfluenceStrategy
        defn = MomentumConfluenceStrategy().to_pinescript()
        assert isinstance(defn, StrategyDefinition)
        assert defn.short_name == "MOM-CONF"

    def test_has_all_indicators(self):
        from stockdownloader.strategy.daily import MomentumConfluenceStrategy
        defn = MomentumConfluenceStrategy().to_pinescript()
        exprs = " ".join(ind.code for ind in defn.indicators)
        assert "ta.macd" in exprs
        assert "ta.ema" in exprs
        assert "ta.dmi" in exprs
        assert "ta.obv" in exprs

    def test_custom_params(self):
        from stockdownloader.strategy.daily import MomentumConfluenceStrategy
        defn = MomentumConfluenceStrategy(
            fast_ema=8, slow_ema=21, ema_trend_filter=100,
        ).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["mcFast"] == 8
        assert defaults["mcSlow"] == 21
        assert defaults["mcTrendEma"] == 100


class TestMultiIndicatorToPine:
    """MultiIndicatorStrategy.to_pinescript()."""

    def test_returns_strategy_definition(self):
        from stockdownloader.strategy.daily import MultiIndicatorStrategy
        defn = MultiIndicatorStrategy().to_pinescript()
        assert isinstance(defn, StrategyDefinition)
        assert "Multi" in defn.name

    def test_has_extra_code(self):
        from stockdownloader.strategy.daily import MultiIndicatorStrategy
        defn = MultiIndicatorStrategy().to_pinescript()
        assert defn.extra_code is not None
        code = "\n".join(defn.extra_code)
        assert "buyScore" in code
        assert "sellScore" in code

    def test_has_ichimoku(self):
        from stockdownloader.strategy.daily import MultiIndicatorStrategy
        defn = MultiIndicatorStrategy().to_pinescript()
        names = [ind.var_name for ind in defn.indicators]
        assert "tenkanSen" in names
        assert "kijunSen" in names

    def test_custom_thresholds(self):
        from stockdownloader.strategy.daily import MultiIndicatorStrategy
        defn = MultiIndicatorStrategy(buy_threshold=6, sell_threshold=5).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["buyThreshold"] == 6
        assert defaults["sellThreshold"] == 5

    def test_threshold_in_conditions(self):
        from stockdownloader.strategy.daily import MultiIndicatorStrategy
        defn = MultiIndicatorStrategy().to_pinescript()
        assert "buyThreshold" in defn.long_entry.expr
        assert "sellThreshold" in defn.short_entry.expr


# ======================================================================
# DmiVwapStrategy to_pinescript()
# ======================================================================


class TestDmiVwapToPine:
    """DmiVwapStrategy.to_pinescript()."""

    def test_returns_strategy_definition(self):
        from stockdownloader.strategy.dmi_vwap_strategy import DmiVwapStrategy
        defn = DmiVwapStrategy().to_pinescript()
        assert isinstance(defn, StrategyDefinition)
        assert defn.name == "DMI + VWAP"

    def test_session_filter_enabled(self):
        from stockdownloader.strategy.dmi_vwap_strategy import DmiVwapStrategy
        defn = DmiVwapStrategy().to_pinescript()
        assert defn.use_session_filter is True

    def test_option_labels(self):
        from stockdownloader.strategy.dmi_vwap_strategy import DmiVwapStrategy
        defn = DmiVwapStrategy().to_pinescript()
        assert defn.long_label == "Buy Call"
        assert defn.short_label == "Buy Put"

    def test_has_session_vwap(self):
        from stockdownloader.strategy.dmi_vwap_strategy import DmiVwapStrategy
        defn = DmiVwapStrategy().to_pinescript()
        exprs = [ind.code for ind in defn.indicators]
        assert any("cumTPV" in e or "session_vwap" in e.lower()
                    for e in exprs) or any(
            ind.var_name == "vwapValue" for ind in defn.indicators
        )

    def test_has_dmi(self):
        from stockdownloader.strategy.dmi_vwap_strategy import DmiVwapStrategy
        defn = DmiVwapStrategy().to_pinescript()
        exprs = " ".join(ind.code for ind in defn.indicators)
        assert "ta.dmi" in exprs

    def test_custom_config(self):
        from stockdownloader.strategy.dmi_vwap_strategy import (
            DmiVwapConfig, DmiVwapStrategy,
        )
        from decimal import Decimal
        cfg = DmiVwapConfig(dmi_period=10, adx_threshold=Decimal("30"))
        defn = DmiVwapStrategy(cfg).to_pinescript()
        defaults = {i.name: i.default for i in defn.inputs}
        assert defaults["dmiPeriod"] == 10
        assert defaults["adxThreshold"] == 30.0


# ======================================================================
# Base class default
# ======================================================================


class TestTradingStrategyBase:
    """TradingStrategy.to_pinescript() raises by default."""

    def test_raises_not_implemented(self):
        from stockdownloader.strategy.trading_strategy import TradingStrategy

        # Create a minimal concrete subclass
        class Dummy(TradingStrategy):
            @property

            def name(self): return "dummy"
            @property
            def warmup_period(self): return 0
            def evaluate(self, data, idx): pass

        with pytest.raises(NotImplementedError, match="Dummy"):
            Dummy().to_pinescript()


# ======================================================================
# Intraday strategy pinescript_mode()
# ======================================================================


class TestPullbackPineMode:
    def test_returns_mode_definition(self):
        from stockdownloader.strategy.intraday import PullbackStrategy
        mode = PullbackStrategy.pinescript_mode()
        assert isinstance(mode, ModeDefinition)
        assert mode.short_name == "pb"

    def test_has_entries(self):
        from stockdownloader.strategy.intraday import PullbackStrategy
        mode = PullbackStrategy.pinescript_mode()
        assert mode.long_entry is not None
        assert mode.short_entry is not None


class TestReversalPineMode:
    def test_returns_mode_definition(self):
        from stockdownloader.strategy.intraday import ReversalStrategy
        mode = ReversalStrategy.pinescript_mode()
        assert isinstance(mode, ModeDefinition)
        assert mode.short_name == "rev"


class TestORBreakoutPineMode:
    def test_returns_mode_definition(self):
        from stockdownloader.strategy.intraday import ORBreakoutStrategy
        mode = ORBreakoutStrategy.pinescript_mode()
        assert isinstance(mode, ModeDefinition)
        assert mode.short_name == "orb"


class TestORReversalPineMode:
    def test_returns_mode_definition(self):
        from stockdownloader.strategy.intraday import ORReversalStrategy
        mode = ORReversalStrategy.pinescript_mode()
        assert isinstance(mode, ModeDefinition)
        assert mode.short_name == "orr"


class TestPatternScalpPineMode:
    def test_returns_mode_definition(self):
        from stockdownloader.strategy.intraday import PatternScalpStrategy
        mode = PatternScalpStrategy.pinescript_mode()
        assert isinstance(mode, ModeDefinition)
        assert mode.short_name == "ps"
        assert mode.enabled_default is False
