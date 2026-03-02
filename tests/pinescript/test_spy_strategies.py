"""Tests for SPY PineScript strategies — strategy mode.

All SPY strategies now use strategy() mode with strategy.entry(),
strategy.exit(), ATR-based position sizing, breakeven management,
circuit breaker, and EOD close.
"""

from __future__ import annotations

import pytest

from stockdownloader.app.pinescript_catalog.spy_strategies import (
    spy_macd_obv_strategy,
    spy_macd_optimized_strategy,
    spy_sma_crossover_strategy,
)
from stockdownloader.app.pinescript_catalog.catalogs import STRATEGY_CATALOG
from stockdownloader.pinescript import PineScriptGenerator


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

    def test_strategy_mode_enabled(self):
        assert self.defn.strategy_mode is True

    def test_inputs(self):
        names = {i.name for i in self.defn.inputs}
        assert "macdFast" in names
        assert "macdSlow" in names
        assert "macdSignal" in names
        assert "obvSmoothing" in names

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

    def test_exits_defined(self):
        """Exit conditions reference MACD crossover/crossunder."""
        assert self.defn.long_exit is not None
        assert self.defn.short_exit is not None

    def test_no_exit_on_reverse(self):
        assert self.defn.exit_on_reverse is False

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

    def test_strategy_mode_enabled(self):
        assert self.defn.strategy_mode is True

    def test_inputs(self):
        names = {i.name for i in self.defn.inputs}
        assert "smaShort" in names
        assert "smaLong" in names

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

    def test_long_exit_defined(self):
        """Exit should be SMA crossunder."""
        assert self.defn.long_exit is not None
        assert "crossunder" in self.defn.long_exit.expr


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

    def test_strategy_mode_enabled(self):
        assert self.defn.strategy_mode is True

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

    def test_exits_defined(self):
        assert self.defn.long_exit is not None
        assert self.defn.short_exit is not None


# ======================================================================
# Pine Script generation — strategy mode output
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
    def test_generates_strategy_mode(self, factory, short):
        defn = factory()
        pine = self.gen.generate(defn)
        assert pine
        assert "strategy(" in pine
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
    def test_contains_strategy_entry(self, factory):
        """Strategy mode uses strategy.entry() for trade execution."""
        pine = self.gen.generate(factory())
        assert "strategy.entry(" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_strategy_exit(self, factory):
        """Strategy mode uses strategy.exit() for SL/TP management."""
        pine = self.gen.generate(factory())
        assert "strategy.exit(" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_entry_labels(self, factory):
        """Strategy mode draws entry labels with E/S/T prices."""
        pine = self.gen.generate(factory())
        assert "label.new" in pine
        assert "label.style_label_up" in pine
        assert "fillPrice" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_level_plots(self, factory):
        """Strategy mode plots Entry/TP/SL level lines."""
        pine = self.gen.generate(factory())
        assert 'title="Entry Price"' in pine
        assert 'title="Take Profit"' in pine
        assert 'title="Stop Loss"' in pine
        assert "plot.style_linebr" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_background(self, factory):
        pine = self.gen.generate(factory())
        assert "bgcolor" in pine
        assert "Position Zone" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_uses_strategy_position_size(self, factory):
        """Strategy mode uses strategy.position_size for state."""
        pine = self.gen.generate(factory())
        assert "strategy.position_size" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_atr(self, factory):
        pine = self.gen.generate(factory())
        assert "ta.atr" in pine
        assert "atrVal" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_position_sizing(self, factory):
        """Strategy mode includes f_qty position sizing function."""
        pine = self.gen.generate(factory())
        assert "f_qty" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_risk_management(self, factory):
        """Strategy mode includes circuit breaker and day trade limits."""
        pine = self.gen.generate(factory())
        assert "dayTrades" in pine
        assert "consecLoss" in pine
        assert "tripped" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_breakeven(self, factory):
        """Strategy mode includes breakeven management."""
        pine = self.gen.generate(factory())
        assert "movedBE" in pine
        assert "beBuf" in pine

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_contains_eod_close(self, factory):
        """Strategy mode includes end-of-day close."""
        pine = self.gen.generate(factory())
        assert "EOD" in pine
        assert "strategy.close_all" in pine

    def test_macd_obv_has_macd_indicator(self):
        pine = self.gen.generate(spy_macd_obv_strategy())
        assert "ta.macd" in pine

    def test_macd_optimized_has_macd_indicator(self):
        pine = self.gen.generate(spy_macd_optimized_strategy())
        assert "ta.macd" in pine

    def test_sma_crossover_has_sma_plots(self):
        pine = self.gen.generate(spy_sma_crossover_strategy())
        assert "SMA(smaShort)" in pine
        assert "SMA(smaLong)" in pine


class TestSpyStrategiesCodeOrdering:
    """Verify the generated Pine Script has correct code ordering.

    Strategy mode ordering:
    1. INDICATORS
    2. SIGNAL LOGIC (longCondition/shortCondition)
    3. SL/TP CALCULATION
    4. EXECUTION (strategy.entry/exit)
    5. TRADE LABELS
    """

    def setup_method(self):
        self.gen = PineScriptGenerator()

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_indicators_before_execution(self, factory):
        """Indicators must be computed before strategy execution."""
        pine = self.gen.generate(factory())
        lines = pine.split("\n")

        indicator_pos = None
        entry_pos = None

        for i, line in enumerate(lines):
            if "INDICATORS" in line and indicator_pos is None:
                indicator_pos = i
            if "strategy.entry(" in line and entry_pos is None:
                entry_pos = i

        assert indicator_pos is not None, "INDICATORS section not found"
        assert entry_pos is not None, "strategy.entry not found"
        assert indicator_pos < entry_pos, (
            f"INDICATORS (line {indicator_pos}) must come before "
            f"strategy.entry (line {entry_pos})"
        )

    @pytest.mark.parametrize("factory", [
        spy_macd_obv_strategy,
        spy_sma_crossover_strategy,
        spy_macd_optimized_strategy,
    ])
    def test_no_extra_code(self, factory):
        """SPY strategies should not have extra_code."""
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
