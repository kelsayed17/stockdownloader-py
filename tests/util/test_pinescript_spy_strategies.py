"""Tests for enhanced SPY PineScript strategies with TP/SL overlays."""

from __future__ import annotations

import pytest

from stockdownloader.app.pinescript_catalog.spy_strategies import (
    _tp_sl_extra_plots,
    _tp_sl_indicators,
    _tp_sl_inputs,
    spy_macd_obv_strategy,
    spy_macd_optimized_strategy,
    spy_sma_crossover_strategy,
)
from stockdownloader.app.pinescript_catalog.strategies import STRATEGY_CATALOG
from stockdownloader.util.pinescript_generator import PineScriptGenerator


# ======================================================================
# Shared TP/SL infrastructure
# ======================================================================


class TestTPSLInputs:
    """Test the shared TP/SL input definitions."""

    def test_returns_four_inputs(self):
        inputs = _tp_sl_inputs()
        assert len(inputs) == 4

    def test_input_names(self):
        names = {i.name for i in _tp_sl_inputs()}
        assert names == {"atrLen", "slMult", "rrRatio", "slCapDollars"}

    def test_default_values(self):
        defaults = {i.name: i.default for i in _tp_sl_inputs()}
        assert defaults["atrLen"] == 14
        assert defaults["slMult"] == 1.5
        assert defaults["rrRatio"] == 1.5
        assert defaults["slCapDollars"] == 2.0

    def test_custom_group(self):
        inputs = _tp_sl_inputs(group="Custom Group")
        assert all(i.group == "Custom Group" for i in inputs)


class TestTPSLIndicators:
    """Test the shared ATR indicator."""

    def test_returns_one_indicator(self):
        indicators = _tp_sl_indicators()
        assert len(indicators) == 1

    def test_atr_indicator(self):
        ind = _tp_sl_indicators()[0]
        assert ind.var_name == "atrVal"
        assert "ta.atr" in ind.code


class TestTPSLExtraPlots:
    """Test the unified TP/SL extra_plots code."""

    def test_with_shorts(self):
        lines = _tp_sl_extra_plots("Test Strategy", has_short=True)
        code = "\n".join(lines)
        assert "entryPrice" in code
        assert "stopPrice" in code
        assert "targetPrice" in code
        assert "buySignal" in code
        assert "sellSignal" in code
        assert "longTPHit" in code
        assert "shortTPHit" in code
        assert "posState" in code

    def test_without_shorts(self):
        lines = _tp_sl_extra_plots("Test Strategy", has_short=False)
        code = "\n".join(lines)
        assert "entryPrice" in code
        assert "buySignal" in code
        # sellSignal should NOT appear in the tracking code
        assert "if sellSignal" not in code
        # Short TP/SL should be set to false
        assert "bool shortTPHit = false" in code
        assert "bool shortSLHit = false" in code

    def test_contains_entry_price_tracking(self):
        lines = _tp_sl_extra_plots("Test", has_short=True)
        code = "\n".join(lines)
        assert "var float entryPrice" in code
        assert "entryPrice  := close" in code

    def test_contains_tp_sl_computation(self):
        lines = _tp_sl_extra_plots("Test", has_short=True)
        code = "\n".join(lines)
        assert "stopPrice" in code
        assert "targetPrice" in code
        assert "slMult" in code
        assert "rrRatio" in code
        assert "slCapDollars" in code

    def test_contains_hit_detection(self):
        lines = _tp_sl_extra_plots("Test", has_short=True)
        code = "\n".join(lines)
        assert "longTPHit" in code
        assert "longSLHit" in code
        assert "shortTPHit" in code
        assert "shortSLHit" in code

    def test_contains_auto_exit(self):
        lines = _tp_sl_extra_plots("Test", has_short=True)
        code = "\n".join(lines)
        assert "posState    := 0" in code
        assert "if longTPHit or longSLHit" in code

    def test_contains_visual_plots(self):
        lines = _tp_sl_extra_plots("Test", has_short=True)
        code = "\n".join(lines)
        assert 'title="Entry Price"' in code
        assert 'title="Take Profit"' in code
        assert 'title="Stop Loss"' in code
        assert "plot.style_linebr" in code

    def test_contains_hit_markers(self):
        lines = _tp_sl_extra_plots("Test", has_short=True)
        code = "\n".join(lines)
        assert 'title="Long TP Hit"' in code
        assert 'title="Long SL Hit"' in code
        assert "plotshape" in code

    def test_contains_alert_conditions(self):
        lines = _tp_sl_extra_plots("My Strategy", has_short=True)
        code = "\n".join(lines)
        assert "alertcondition(longTPHit" in code
        assert "alertcondition(longSLHit" in code
        assert "alertcondition(shortTPHit" in code
        assert "alertcondition(shortSLHit" in code
        assert 'message="My Strategy: Long Take Profit hit!"' in code
        assert 'title="Any TP/SL Hit"' in code

    def test_contains_data_window_metrics(self):
        lines = _tp_sl_extra_plots("Test", has_short=True)
        code = "\n".join(lines)
        assert 'title="ATR"' in code
        assert 'title="Trade Risk ($)"' in code
        assert 'title="TP Distance ($)"' in code
        assert 'title="SL Distance ($)"' in code
        assert "display.data_window" in code

    def test_no_extra_code_dependencies(self):
        """TP/SL code only goes in extra_plots, never extra_code."""
        for factory in (spy_macd_obv_strategy,
                        spy_sma_crossover_strategy,
                        spy_macd_optimized_strategy):
            defn = factory()
            assert not defn.extra_code, (
                f"{defn.name} should not have extra_code — "
                "all TP/SL code belongs in extra_plots"
            )


# ======================================================================
# Strategy definitions
# ======================================================================


class TestSpyMacdObvStrategy:
    """Test the MACD+OBV tournament winner strategy."""

    def setup_method(self):
        self.defn = spy_macd_obv_strategy()

    def test_name(self):
        assert "MACD" in self.defn.name
        assert "OBV" in self.defn.name
        assert "Tournament Winner" in self.defn.name

    def test_short_name(self):
        assert self.defn.short_name == "SPY-MACD-OBV"

    def test_has_description(self):
        assert "74.03" in self.defn.description
        assert "Walk-forward" in self.defn.description

    def test_inputs(self):
        names = {i.name for i in self.defn.inputs}
        assert "macdFast" in names
        assert "macdSlow" in names
        assert "macdSignal" in names
        assert "obvSmoothing" in names
        # TP/SL inputs
        assert "atrLen" in names
        assert "slMult" in names
        assert "rrRatio" in names
        assert "slCapDollars" in names

    def test_macd_defaults(self):
        defaults = {i.name: i.default for i in self.defn.inputs}
        assert defaults["macdFast"] == 12
        assert defaults["macdSlow"] == 26
        assert defaults["macdSignal"] == 9

    def test_has_macd_indicators(self):
        codes = " ".join(ind.code for ind in self.defn.indicators)
        assert "ta.macd" in codes

    def test_has_obv_indicators(self):
        names = {ind.var_name for ind in self.defn.indicators}
        assert "obvRaw" in names
        assert "obvSmoothed" in names
        assert "obvRising" in names

    def test_long_entry(self):
        assert self.defn.long_entry is not None
        assert "crossover" in self.defn.long_entry.expr
        assert "obvRising" in self.defn.long_entry.expr

    def test_short_entry(self):
        assert self.defn.short_entry is not None
        assert "crossunder" in self.defn.short_entry.expr
        assert "obvFalling" in self.defn.short_entry.expr

    def test_exits_are_pure_strategy(self):
        """Exit conditions should NOT reference longTPHit/longSLHit."""
        assert "TPHit" not in self.defn.long_exit.expr
        assert "SLHit" not in self.defn.long_exit.expr
        assert "TPHit" not in self.defn.short_exit.expr
        assert "SLHit" not in self.defn.short_exit.expr

    def test_no_exit_on_reverse(self):
        assert self.defn.exit_on_reverse is False

    def test_has_extra_plots(self):
        assert self.defn.extra_plots
        code = "\n".join(self.defn.extra_plots)
        assert "entryPrice" in code
        assert "targetPrice" in code
        assert "stopPrice" in code

    def test_labels(self):
        assert self.defn.long_label == "Buy"
        assert self.defn.short_label == "Sell"


class TestSpySmaCrossoverStrategy:
    """Test the SMA 20/21 tournament #2 strategy."""

    def setup_method(self):
        self.defn = spy_sma_crossover_strategy()

    def test_name(self):
        assert "SMA" in self.defn.name
        assert "20/21" in self.defn.name

    def test_short_name(self):
        assert self.defn.short_name == "SPY-SMA-2021"

    def test_has_description(self):
        assert "65.29" in self.defn.description

    def test_inputs(self):
        names = {i.name for i in self.defn.inputs}
        assert "smaShort" in names
        assert "smaLong" in names
        # TP/SL inputs
        assert "atrLen" in names

    def test_sma_defaults(self):
        defaults = {i.name: i.default for i in self.defn.inputs}
        assert defaults["smaShort"] == 20
        assert defaults["smaLong"] == 21

    def test_has_sma_indicators(self):
        codes = " ".join(ind.code for ind in self.defn.indicators)
        assert "ta.sma" in codes

    def test_sma_plots_enabled(self):
        sma_inds = [ind for ind in self.defn.indicators
                    if "ta.sma" in ind.code]
        assert all(ind.plot for ind in sma_inds)

    def test_long_only(self):
        assert self.defn.long_entry is not None
        assert self.defn.short_entry is None

    def test_long_exit_pure(self):
        """Exit should not reference TP/SL hits."""
        assert "TPHit" not in self.defn.long_exit.expr
        assert "SLHit" not in self.defn.long_exit.expr

    def test_extra_plots_long_only(self):
        """Extra plots should not include sellSignal tracking."""
        code = "\n".join(self.defn.extra_plots)
        assert "if sellSignal" not in code
        assert "bool shortTPHit = false" in code


class TestSpyMacdOptimizedStrategy:
    """Test the MACD 8/35/5 tournament #3 strategy."""

    def setup_method(self):
        self.defn = spy_macd_optimized_strategy()

    def test_name(self):
        assert "MACD" in self.defn.name
        assert "8/35/5" in self.defn.name

    def test_short_name(self):
        assert self.defn.short_name == "SPY-MACD-OPT"

    def test_has_description(self):
        assert "62.80" in self.defn.description

    def test_optimized_macd_defaults(self):
        defaults = {i.name: i.default for i in self.defn.inputs}
        assert defaults["macdFast"] == 8
        assert defaults["macdSlow"] == 35
        assert defaults["macdSignal"] == 5

    def test_long_and_short_entries(self):
        assert self.defn.long_entry is not None
        assert self.defn.short_entry is not None
        assert "crossover" in self.defn.long_entry.expr
        assert "crossunder" in self.defn.short_entry.expr

    def test_exits_are_pure_strategy(self):
        assert "TPHit" not in self.defn.long_exit.expr
        assert "TPHit" not in self.defn.short_exit.expr


# ======================================================================
# Pine Script generation
# ======================================================================


class TestSpyStrategiesPineOutput:
    """Test that generated Pine Script output is structurally valid."""

    def setup_method(self):
        self.gen = PineScriptGenerator()

    @pytest.mark.parametrize("factory,short", [
        (spy_macd_obv_strategy, "SPY-MACD-OBV"),
        (spy_sma_crossover_strategy, "SPY-SMA-2021"),
        (spy_macd_optimized_strategy, "SPY-MACD-OPT"),
    ])
    def test_generates_valid_pine(self, factory, short):
        defn = factory()
        pine = self.gen.generate(defn)
        assert pine
        assert "indicator(" in pine
        assert f'shorttitle="{short}"' in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_version_6(self, factory):
        pine = self.gen.generate(factory())
        assert "//@version=6" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_buy_signal(self, factory):
        pine = self.gen.generate(factory())
        assert "buySignal" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_entry_price_tracking(self, factory):
        pine = self.gen.generate(factory())
        assert "var float entryPrice" in pine
        assert "entryPrice  := close" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_tp_sl_lines(self, factory):
        pine = self.gen.generate(factory())
        assert 'title="Entry Price"' in pine
        assert 'title="Take Profit"' in pine
        assert 'title="Stop Loss"' in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_tp_sl_hit_detection(self, factory):
        pine = self.gen.generate(factory())
        assert "longTPHit" in pine
        assert "longSLHit" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_tp_sl_alerts(self, factory):
        pine = self.gen.generate(factory())
        assert 'title="Long TP Hit"' in pine
        assert 'title="Long SL Hit"' in pine
        assert 'title="Any TP/SL Hit"' in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_buy_sell_alerts(self, factory):
        """Standard Buy/Sell alerts from the generator."""
        pine = self.gen.generate(factory())
        assert "alertcondition(buySignal" in pine
        assert 'title="Buy"' in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_labels(self, factory):
        pine = self.gen.generate(factory())
        assert "label.new" in pine
        assert "label.style_label_up" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_background_coloring(self, factory):
        pine = self.gen.generate(factory())
        assert "bgcolor" in pine
        assert "Position Zone" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_position_state_machine(self, factory):
        pine = self.gen.generate(factory())
        assert "var int posState = 0" in pine
        assert "posState := 1" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_atr_input(self, factory):
        pine = self.gen.generate(factory())
        assert "atrLen" in pine
        assert "ta.atr" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_data_window_metrics(self, factory):
        pine = self.gen.generate(factory())
        assert 'title="ATR"' in pine
        assert 'title="Trade Risk ($)"' in pine
        assert "display.data_window" in pine

    def test_macd_obv_has_histogram(self):
        pine = self.gen.generate(spy_macd_obv_strategy())
        assert 'title="MACD Histogram"' in pine

    def test_macd_optimized_has_histogram(self):
        pine = self.gen.generate(spy_macd_optimized_strategy())
        assert 'title="MACD Histogram"' in pine

    def test_sma_crossover_has_sma_plots(self):
        pine = self.gen.generate(spy_sma_crossover_strategy())
        assert "SMA(smaShort)" in pine
        assert "SMA(smaLong)" in pine


class TestSpyStrategiesCodeOrdering:
    """Verify the generated Pine Script has correct code ordering.

    The critical ordering requirement:
    1. INDICATORS (atrVal)
    2. SIGNAL LOGIC (buySignal, posState, exitLongSignal)
    3. ADDITIONAL PLOTS (entryPrice tracking, TP/SL hit detection)
    """

    def setup_method(self):
        self.gen = PineScriptGenerator()

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_signal_logic_before_tp_sl(self, factory):
        """buySignal/posState must be defined BEFORE TP/SL tracking."""
        pine = self.gen.generate(factory())
        lines = pine.split("\n")

        # Find key line positions
        buy_signal_pos = None
        pos_state_pos = None
        entry_price_pos = None
        tp_hit_pos = None

        for i, line in enumerate(lines):
            if "bool buySignal" in line and buy_signal_pos is None:
                buy_signal_pos = i
            if "var int posState = 0" in line and pos_state_pos is None:
                pos_state_pos = i
            if "var float entryPrice" in line and entry_price_pos is None:
                entry_price_pos = i
            if "bool longTPHit" in line and tp_hit_pos is None:
                tp_hit_pos = i

        assert buy_signal_pos is not None, "buySignal not found"
        assert pos_state_pos is not None, "posState not found"
        assert entry_price_pos is not None, "entryPrice not found"
        assert tp_hit_pos is not None, "longTPHit not found"

        assert pos_state_pos < entry_price_pos, (
            f"posState (line {pos_state_pos}) must come before "
            f"entryPrice (line {entry_price_pos})"
        )
        assert buy_signal_pos < entry_price_pos, (
            f"buySignal (line {buy_signal_pos}) must come before "
            f"entryPrice (line {entry_price_pos})"
        )
        assert entry_price_pos < tp_hit_pos, (
            f"entryPrice (line {entry_price_pos}) must come before "
            f"longTPHit (line {tp_hit_pos})"
        )

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_no_extra_code(self, factory):
        """All TP/SL code must be in extra_plots, not extra_code."""
        defn = factory()
        assert not defn.extra_code


# ======================================================================
# Catalog integration
# ======================================================================


class TestSpyStrategiesInCatalog:
    """Verify the SPY strategies are registered in the catalog."""

    def test_spy_macd_obv_in_catalog(self):
        assert "spy_macd_obv" in STRATEGY_CATALOG

    def test_spy_sma_crossover_in_catalog(self):
        assert "spy_sma_crossover" in STRATEGY_CATALOG

    def test_spy_macd_optimized_in_catalog(self):
        assert "spy_macd_optimized" in STRATEGY_CATALOG

    def test_catalog_factories_produce_output(self):
        gen = PineScriptGenerator()
        for key in ("spy_macd_obv", "spy_sma_crossover", "spy_macd_optimized"):
            defn = STRATEGY_CATALOG[key]()
            pine = gen.generate(defn)
            assert pine, f"{key}: empty output"
            assert len(pine) > 200, f"{key}: output too short"


# ======================================================================
# Output line count sanity check
# ======================================================================


class TestOutputSize:
    """Ensure generated output is a reasonable size."""

    def setup_method(self):
        self.gen = PineScriptGenerator()

    @pytest.mark.parametrize("factory,min_lines", [
        (spy_macd_obv_strategy, 100),
        (spy_sma_crossover_strategy, 100),
        (spy_macd_optimized_strategy, 100),
    ])
    def test_minimum_output_lines(self, factory, min_lines):
        pine = self.gen.generate(factory())
        actual = len(pine.splitlines())
        assert actual >= min_lines, (
            f"Expected >= {min_lines} lines, got {actual}"
        )
