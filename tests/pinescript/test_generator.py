"""Tests for the Pine Script v6 generator."""

from __future__ import annotations

import pytest

from stockdownloader.app.pinescript_catalog.catalogs import (
    STRATEGY_CATALOG,
    bollinger_rsi_strategy,
    breakout_strategy,
    dmi_vwap_strategy,
    macd_obv_strategy,
    macd_strategy,
    momentum_confluence_strategy,
    rsi_strategy,
    sma_crossover_strategy,
    vwap_pullback_strategy,
    vwap_reversal_strategy,
    vwap_or_breakout_strategy,
    vwap_or_reversal_strategy,
    vwap_pattern_scalp_strategy,
)
from stockdownloader.pinescript.models import (
    Condition,
    Indicator,
    Input,
    InputType,
    StrategyDefinition,
)
from stockdownloader.pinescript import PineScriptGenerator


class TestInput:
    def test_int_input(self):
        inp = Input.int_("period", 14, "RSI Period")
        assert inp.name == "period"
        assert inp.input_type == InputType.INT
        assert inp.default == 14
        assert inp.title == "RSI Period"

    def test_float_input(self):
        inp = Input.float_("threshold", 25.0, "ADX Threshold", step=0.5)
        assert inp.input_type == InputType.FLOAT
        assert inp.step == 0.5

    def test_bool_input(self):
        inp = Input.bool_("showLabels", True, "Show Labels")
        assert inp.input_type == InputType.BOOL
        assert inp.default is True


class TestIndicator:
    def test_sma(self):
        ind = Indicator.sma("smaFast", "close", "fastPeriod")
        assert ind.var_name == "smaFast"
        assert "ta.sma" in ind.code
        assert ind.plot is True

    def test_ema(self):
        ind = Indicator.ema("emaFast", "close", "fastPeriod")
        assert "ta.ema" in ind.code

    def test_rsi(self):
        ind = Indicator.rsi("rsiVal", "close", "14")
        assert "ta.rsi" in ind.code
        assert ind.display == "data_window"

    def test_macd_returns_list(self):
        inds = Indicator.macd("12", "26", "9")
        assert len(inds) == 1
        assert "ta.macd" in inds[0].code
        assert "macdLine" in inds[0].var_name

    def test_bbands(self):
        ind = Indicator.bbands("mid", "upper", "lower", "close", "20", "2.0")
        assert "ta.bb" in ind.code
        assert ind.var_name == "[mid, upper, lower]"

    def test_dmi(self):
        ind = Indicator.dmi("14", "14")
        assert "ta.dmi" in ind.code
        assert "plusDI" in ind.var_name

    def test_stochastic_returns_list(self):
        inds = Indicator.stochastic()
        assert len(inds) == 2
        assert "ta.stoch" in inds[0].code

    def test_obv(self):
        ind = Indicator.obv()
        assert ind.code == "ta.obv"

    def test_session_vwap(self):
        ind = Indicator.session_vwap()
        assert ind.code == "__SESSION_VWAP__"

    def test_raw(self):
        ind = Indicator.raw("myVar", "close > open")
        assert ind.code == "close > open"
        assert ind.plot is False

    def test_atr(self):
        ind = Indicator.atr("atrVal", "14")
        assert "ta.atr" in ind.code

    def test_ichimoku_returns_five(self):
        inds = Indicator.ichimoku()
        assert len(inds) == 5

    def test_ichimoku_default_names(self):
        inds = Indicator.ichimoku()
        names = [ind.var_name for ind in inds]
        assert "tenkanSen" in names
        assert "kijunSen" in names
        assert "senkouA" in names
        assert "senkouB" in names
        assert "aboveCloud" in names

    def test_ichimoku_custom_names(self):
        inds = Indicator.ichimoku(var_tenkan="tk", var_kijun="kj")
        names = [ind.var_name for ind in inds]
        assert "tk" in names
        assert "kj" in names

    def test_ichimoku_expressions(self):
        inds = Indicator.ichimoku()
        codes = {ind.var_name: ind.code for ind in inds}
        assert "ta.highest" in codes["tenkanSen"]
        assert "ta.lowest" in codes["tenkanSen"]
        assert "9" in codes["tenkanSen"]
        assert "26" in codes["kijunSen"]
        assert "52" in codes["senkouB"]
        assert "math.max" in codes["aboveCloud"]

    def test_ichimoku_span_a_references_components(self):
        inds = Indicator.ichimoku()
        span_a = next(i for i in inds if i.var_name == "senkouA")
        assert "tenkanSen" in span_a.code
        assert "kijunSen" in span_a.code


class TestCondition:
    def test_basic(self):
        c = Condition("close > open")
        assert c.expr == "close > open"

    def test_and(self):
        c1 = Condition("a > b")
        c2 = Condition("c > d")
        combined = c1 & c2
        assert "a > b" in combined.expr
        assert "c > d" in combined.expr
        assert "and" in combined.expr

    def test_or(self):
        c1 = Condition("a > b")
        c2 = Condition("c > d")
        combined = c1 | c2
        assert "or" in combined.expr

    def test_negate(self):
        c = Condition("a > b")
        neg = c.negate()
        assert "not" in neg.expr
        assert "a > b" in neg.expr


class TestPineScriptGenerator:
    def setup_method(self):
        self.gen = PineScriptGenerator()

    def test_minimal_strategy(self):
        """Simplest possible strategy — just long entry."""
        defn = StrategyDefinition(
            name="Test Strategy",
            short_name="TEST",
            indicators=[
                Indicator.sma("sma20", "close", "20"),
            ],
            long_entry=Condition("close > sma20"),
        )
        pine = self.gen.generate(defn)
        assert '//@version=6' in pine
        assert 'indicator("Test Strategy"' in pine
        assert 'ta.sma(close, 20)' in pine
        assert 'close > sma20' in pine
        assert 'buySignal' in pine
        assert 'exitLongSignal' in pine
        assert 'alertcondition' in pine

    def test_long_short_strategy(self):
        """Strategy with both long and short signals."""
        defn = StrategyDefinition(
            name="Both Directions",
            short_name="BOTH",
            long_entry=Condition("close > open"),
            short_entry=Condition("close < open"),
        )
        pine = self.gen.generate(defn)
        assert 'buySignal' in pine
        assert 'sellSignal' in pine
        assert 'exitLongSignal' in pine
        assert 'exitShortSignal' in pine
        assert 'posState := 1' in pine
        assert 'posState := -1' in pine
        assert 'posState := 0' in pine

    def test_inputs_generated(self):
        defn = StrategyDefinition(
            name="With Inputs",
            short_name="INP",
            inputs=[
                Input.int_("period", 14, "Period"),
                Input.float_("threshold", 25.0, "Threshold"),
                Input.bool_("showLabels", True, "Labels"),
            ],
            long_entry=Condition("true"),
        )
        pine = self.gen.generate(defn)
        assert 'input.int(14' in pine
        assert 'input.float(25.0' in pine
        assert 'input.bool(true' in pine

    def test_session_filter(self):
        defn = StrategyDefinition(
            name="Session Filtered",
            short_name="SESS",
            use_session_filter=True,
            session="0930-1600",
            timezone="America/New_York",
            long_entry=Condition("close > open"),
        )
        pine = self.gen.generate(defn)
        assert 'SESSION' in pine
        assert '"0930-1600"' in pine
        assert '"America/New_York"' in pine
        assert 'isNewSession' in pine
        assert 'inSession' in pine
        assert 'posState := 0' in pine  # reset on new session

    def test_session_vwap_indicator(self):
        defn = StrategyDefinition(
            name="VWAP Test",
            short_name="VWAP",
            use_session_filter=True,
            indicators=[Indicator.session_vwap("vwapVal")],
            long_entry=Condition("close > vwapVal"),
        )
        pine = self.gen.generate(defn)
        assert 'cumTPV' in pine
        assert 'cumVol' in pine
        assert 'isNewSession' in pine
        assert 'vwapVal' in pine

    def test_labels_generated(self):
        defn = StrategyDefinition(
            name="Labels Test",
            short_name="LBL",
            long_entry=Condition("true"),
            short_entry=Condition("false"),
            long_label="Buy Call",
            short_label="Buy Put",
        )
        pine = self.gen.generate(defn)
        assert '"Buy Call"' in pine
        assert '"Buy Put"' in pine
        assert '"Exit Long"' in pine
        assert '"Exit Short"' in pine
        assert 'label.new' in pine
        assert 'label.style_label_up' in pine
        assert 'label.style_label_down' in pine

    def test_background_coloring(self):
        defn = StrategyDefinition(
            name="BG Test",
            short_name="BG",
            long_entry=Condition("true"),
            short_entry=Condition("false"),
        )
        pine = self.gen.generate(defn)
        assert 'bgcolor' in pine
        assert 'color.green' in pine
        assert 'color.red' in pine

    def test_alerts_generated(self):
        defn = StrategyDefinition(
            name="Alert Test",
            short_name="ALT",
            long_entry=Condition("true"),
            short_entry=Condition("false"),
        )
        pine = self.gen.generate(defn)
        assert pine.count('alertcondition') >= 4  # buy, sell, exit_long, exit_short, any

    def test_extra_code(self):
        defn = StrategyDefinition(
            name="Extra Code",
            short_name="EXT",
            long_entry=Condition("true"),
            extra_code=["// Custom code here", "float myVar = 42.0"],
        )
        pine = self.gen.generate(defn)
        assert "// Custom code here" in pine
        assert "float myVar = 42.0" in pine

    def test_extra_plots(self):
        defn = StrategyDefinition(
            name="Extra Plot",
            short_name="PLT",
            long_entry=Condition("true"),
            extra_plots=['plot(close, title="Close", color=color.red)'],
        )
        pine = self.gen.generate(defn)
        assert 'plot(close, title="Close"' in pine

    def test_exit_on_reverse_default(self):
        """By default, long exits when short condition is met."""
        defn = StrategyDefinition(
            name="Reverse Exit",
            short_name="REV",
            long_entry=Condition("close > open"),
            short_entry=Condition("close < open"),
            exit_on_reverse=True,
        )
        pine = self.gen.generate(defn)
        assert "shortCondition" in pine
        # exitLong fires when short condition met
        assert "posState == 1 and (shortCondition)" in pine

    def test_custom_exit_conditions(self):
        defn = StrategyDefinition(
            name="Custom Exit",
            short_name="CX",
            long_entry=Condition("close > open"),
            long_exit=Condition("ta.crossunder(close, sma50)"),
        )
        pine = self.gen.generate(defn)
        assert "ta.crossunder(close, sma50)" in pine

    def test_long_only_strategy(self):
        """Long-only strategy should not have sell signals."""
        defn = StrategyDefinition(
            name="Long Only",
            short_name="LONG",
            long_entry=Condition("close > open"),
            long_exit=Condition("close < open"),
            short_entry=None,
        )
        pine = self.gen.generate(defn)
        assert "buySignal" in pine
        assert "sellSignal" not in pine
        assert "exitShortSignal" not in pine

    def test_description_in_header(self):
        defn = StrategyDefinition(
            name="Desc Test",
            short_name="DESC",
            description="Line one\nLine two",
            long_entry=Condition("true", "My long entry description"),
        )
        pine = self.gen.generate(defn)
        assert "// Line one" in pine
        assert "// Line two" in pine
        assert "My long entry description" in pine

    def test_indicator_plots(self):
        defn = StrategyDefinition(
            name="Plot Test",
            short_name="PLT",
            indicators=[
                Indicator.sma("sma20", "close", "20", plot=True),
            ],
            long_entry=Condition("true"),
        )
        pine = self.gen.generate(defn)
        assert "plot(sma20" in pine

    def test_tuple_assignment(self):
        """MACD-style [a, b, c] = ta.func() should not have type prefix."""
        defn = StrategyDefinition(
            name="Tuple Test",
            short_name="TUP",
            indicators=Indicator.macd("12", "26", "9"),
            long_entry=Condition("true"),
        )
        pine = self.gen.generate(defn)
        assert "[macdLine, macdSignal, macdHist] = ta.macd" in pine
        # Should NOT have "float [macdLine..."
        assert "float [" not in pine

    def test_group_input_comments(self):
        defn = StrategyDefinition(
            name="Group Test",
            short_name="GRP",
            inputs=[
                Input.int_("a", 1, "Param A", group="Group 1"),
                Input.int_("b", 2, "Param B", group="Group 2"),
            ],
            long_entry=Condition("true"),
        )
        pine = self.gen.generate(defn)
        assert "// --- Group 1 ---" in pine
        assert "// --- Group 2 ---" in pine


# ------------------------------------------------------------------
# Pre-built strategy smoke tests
# ------------------------------------------------------------------


class TestPrebuiltStrategies:
    """Verify all pre-built strategy factories produce valid Pine Script."""

    def setup_method(self):
        self.gen = PineScriptGenerator()

    def _validate_pine(self, pine: str, name: str):
        """Basic validation that generated Pine Script is well-formed."""
        assert '//@version=6' in pine, f"{name}: missing version"
        # Strategy mode uses strategy(), indicator mode uses indicator()
        assert ('indicator(' in pine or 'strategy(' in pine), (
            f"{name}: missing indicator() or strategy()"
        )
        # Strategy mode uses inline alert(), indicator mode uses alertcondition()
        assert ('alertcondition(' in pine or 'alert(' in pine), (
            f"{name}: missing alerts"
        )
        # Strategy mode uses strategy.position_size, indicator mode uses posState
        assert ('posState' in pine or 'strategy.position_size' in pine), (
            f"{name}: missing position state"
        )
        # Check balanced braces aren't wildly off
        # (Pine uses indentation, not braces, so just check for common errors)
        assert pine.count('if ') >= 1, f"{name}: no if statements"

    def test_sma_crossover(self):
        defn = sma_crossover_strategy()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "SMA Crossover")
        assert "ta.sma" in pine
        assert "ta.crossover" in pine
        assert "ta.crossunder" in pine
        # Long-only
        assert "sellSignal" not in pine

    def test_sma_crossover_custom_params(self):
        defn = sma_crossover_strategy(short_period=20, long_period=21)
        pine = self.gen.generate(defn)
        assert "20" in pine
        assert "21" in pine

    def test_rsi_strategy(self):
        defn = rsi_strategy()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "RSI")
        assert "ta.rsi" in pine
        assert "ta.crossover" in pine
        assert "oversold" in pine
        assert "overbought" in pine

    def test_rsi_custom_params(self):
        defn = rsi_strategy(period=7, oversold=35.0, overbought=65.0)
        pine = self.gen.generate(defn)
        assert "7" in pine

    def test_macd_strategy(self):
        defn = macd_strategy()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "MACD")
        assert "ta.macd" in pine
        assert "macdLine" in pine
        assert "macdSignal" in pine

    def test_macd_obv_strategy(self):
        defn = macd_obv_strategy()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "MACD+OBV")
        assert "ta.macd" in pine
        assert "ta.obv" in pine
        assert "obvRising" in pine or "obvSmoothed" in pine
        assert "Walk-forward validated winner" in pine

    def test_bollinger_rsi_strategy(self):
        defn = bollinger_rsi_strategy()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "BB+RSI")
        assert "ta.bb" in pine
        assert "ta.rsi" in pine
        assert "ta.stoch" in pine
        assert "ta.dmi" in pine
        assert "adxValue" in pine
        assert "stochK" in pine

    def test_dmi_vwap_strategy(self):
        defn = dmi_vwap_strategy()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "DMI+VWAP")
        assert "ta.dmi" in pine
        assert "cumTPV" in pine  # session VWAP
        assert "isNewSession" in pine
        assert "Buy Call" in pine
        assert "Buy Put" in pine
        assert "0930-1600" in pine

    def test_momentum_confluence_strategy(self):
        defn = momentum_confluence_strategy()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "Momentum Confluence")
        assert "ta.macd" in pine
        assert "ta.ema" in pine
        assert "ta.dmi" in pine
        assert "ta.obv" in pine

    def test_breakout_strategy(self):
        defn = breakout_strategy()
        pine = self.gen.generate(defn)
        self._validate_pine(pine, "Breakout")
        assert "ta.bb" in pine
        assert "isSqueeze" in pine
        assert "volSpike" in pine
        assert "atrExpanding" in pine

    def test_all_catalog_strategies(self):
        """Every strategy in the catalog should generate valid Pine Script."""
        for name, factory in STRATEGY_CATALOG.items():
            defn = factory()
            pine = self.gen.generate(defn)
            self._validate_pine(pine, name)
            # Should be a reasonable size
            assert len(pine) > 200, f"{name}: output too short"
            assert len(pine.splitlines()) > 20, f"{name}: too few lines"


# ------------------------------------------------------------------
# CLI smoke test
# ------------------------------------------------------------------


class TestSectionDividers:
    """Verify section dividers use unicode box-drawing character."""

    def setup_method(self):
        self.gen = PineScriptGenerator()

    def test_divider_uses_box_drawing(self):
        defn = StrategyDefinition(
            name="Divider Test",
            short_name="DIV",
            long_entry=Condition("true"),
        )
        pine = self.gen.generate(defn)
        assert "\u2500" * 77 in pine
        assert "=" * 76 not in pine


# ------------------------------------------------------------------
# Standalone VWAP strategy tests
# ------------------------------------------------------------------


class TestStandaloneVwapStrategies:
    """Verify standalone VWAP strategies generate valid Pine Script."""

    def setup_method(self):
        self.gen = PineScriptGenerator()

    def _validate_vwap_pine(self, pine: str, name: str):
        """Validate common VWAP infrastructure is present."""
        assert '//@version=6' in pine, f"{name}: missing version"
        assert 'indicator(' in pine, f"{name}: missing indicator()"
        assert 'alertcondition(' in pine, f"{name}: missing alerts"
        assert 'posState' in pine, f"{name}: missing position state"
        # Session detection
        assert 'isNewSession' in pine, f"{name}: missing session detection"
        assert 'inSession' in pine, f"{name}: missing inSession"
        assert '0930-1600' in pine, f"{name}: missing session time"
        # VWAP infrastructure
        assert 'cumTPV' in pine, f"{name}: missing VWAP cumTPV"
        assert 'vwapValue' in pine, f"{name}: missing vwapValue"
        # Shared code (OR tracking, candle helpers)
        assert 'orHigh' in pine, f"{name}: missing orHigh"
        assert 'bullCandle' in pine, f"{name}: missing bullCandle"
        assert 'relVol' in pine, f"{name}: missing relVol"
        # No enable toggles (composite-only)
        assert 'enable' not in pine.lower() or 'enableA' not in pine, \
            f"{name}: standalone should not have enable toggles"

    def test_vwap_pullback(self):
        defn = vwap_pullback_strategy()
        pine = self.gen.generate(defn)
        self._validate_vwap_pine(pine, "VWAP Pullback")
        assert "pbZone" in pine
        assert "pbAdxMin" in pine
        assert "pbInZone" in pine

    def test_vwap_reversal(self):
        defn = vwap_reversal_strategy()
        pine = self.gen.generate(defn)
        self._validate_vwap_pine(pine, "VWAP Reversal")
        assert "revAdxMax" in pine
        assert "revBandMult" in pine
        assert "revUpper" in pine
        assert "revLower" in pine

    def test_vwap_or_breakout(self):
        defn = vwap_or_breakout_strategy()
        pine = self.gen.generate(defn)
        self._validate_vwap_pine(pine, "VWAP OR Breakout")
        assert "orbWindow" in pine
        assert "orbRvol" in pine
        assert "orbFiredToday" in pine
        assert "orDone" in pine

    def test_vwap_or_reversal(self):
        defn = vwap_or_reversal_strategy()
        pine = self.gen.generate(defn)
        self._validate_vwap_pine(pine, "VWAP OR Reversal")
        assert "orrWindow" in pine
        assert "orrFiredToday" in pine
        assert "isHammer" in pine

    def test_vwap_pattern_scalp(self):
        defn = vwap_pattern_scalp_strategy()
        pine = self.gen.generate(defn)
        self._validate_vwap_pine(pine, "VWAP Pattern Scalp")
        assert "psAtrPct" in pine
        assert "isManip" in pine
        assert "psFiredToday" in pine

    def test_all_five_in_catalog(self):
        expected = {
            "vwap_pullback", "vwap_reversal", "vwap_or_breakout",
            "vwap_or_reversal", "vwap_pattern_scalp",
        }
        assert expected.issubset(set(STRATEGY_CATALOG.keys()))

    def test_no_enable_toggles(self):
        """Standalone strategies should NOT have enable toggles."""
        for name in ("vwap_pullback", "vwap_reversal", "vwap_or_breakout",
                      "vwap_or_reversal", "vwap_pattern_scalp"):
            defn = STRATEGY_CATALOG[name]()
            pine = self.gen.generate(defn)
            assert 'enablePB' not in pine
            assert 'enableREV' not in pine
            assert 'enableORB' not in pine
            assert 'enableORR' not in pine
            assert 'enablePS' not in pine

    def test_catalog_count(self):
        """Catalog should have 21 strategies (9 + 5 VWAP + GME + 3 SPY + 3 SPY v2)."""
        assert len(STRATEGY_CATALOG) == 21


# ------------------------------------------------------------------
# CLI smoke test
# ------------------------------------------------------------------


class TestGeneratePinescriptCLI:
    def test_list_works(self):
        """--list should print all strategies without error."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "stockdownloader.app.pinescript", "--list"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "sma_crossover" in result.stdout
        assert "macd_obv" in result.stdout
        assert "vwap_pullback" in result.stdout
