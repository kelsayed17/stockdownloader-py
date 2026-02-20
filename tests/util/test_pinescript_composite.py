"""Tests for composite PineScript generation (multi-mode, toggleable)."""

from __future__ import annotations

import pytest

from stockdownloader.app.pinescript_catalog.composites import (
    COMPOSITE_STRATEGY_CATALOG,
    full_composite_strategy,
    signal_stack_composite_strategy,
    vwap_composite_strategy,
)
from stockdownloader.app.pinescript_catalog.strategies import (
    macd_strategy,
    rsi_strategy,
    sma_crossover_strategy,
)
from stockdownloader.util.pinescript_generator import (
    CompositeStrategyDefinition,
    Condition,
    Indicator,
    Input,
    ModeDefinition,
    PineScriptGenerator,
    SharedInfrastructure,
    mode_to_strategy,
    strategy_to_mode,
)
from stockdownloader.util.pinescript_modes import (
    orb_mode as _orb_mode,
    orr_mode as _orr_mode,
    pb_mode as _pb_mode,
    ps_mode as _ps_mode,
    rev_mode as _rev_mode,
    vwap_shared_infrastructure as _vwap_shared_infrastructure,
)


@pytest.fixture
def gen() -> PineScriptGenerator:
    return PineScriptGenerator()


# ======================================================================
# ModeDefinition dataclass
# ======================================================================


class TestModeDefinition:
    def test_basic_mode(self) -> None:
        mode = ModeDefinition(
            name="Test Mode",
            short_name="tm",
            group="Test",
            long_entry=Condition("close > open"),
            short_entry=Condition("close < open"),
        )
        assert mode.name == "Test Mode"
        assert mode.short_name == "tm"
        assert mode.enabled_default is True
        assert mode.exit_on_reverse is True

    def test_mode_defaults(self) -> None:
        mode = ModeDefinition(
            name="M", short_name="m", group="G",
            enabled_default=False,
        )
        assert mode.enabled_default is False
        assert mode.long_entry is None
        assert mode.short_entry is None
        assert mode.inputs == []
        assert mode.indicators == []
        assert mode.extra_code == []


# ======================================================================
# CompositeStrategyDefinition dataclass
# ======================================================================


class TestCompositeStrategyDefinition:
    def test_basic_composite(self) -> None:
        defn = CompositeStrategyDefinition(
            name="Test Composite",
            short_name="TC",
            modes=[
                ModeDefinition(
                    name="Alpha", short_name="a", group="Alpha",
                    long_entry=Condition("close > open"),
                ),
                ModeDefinition(
                    name="Beta", short_name="b", group="Beta",
                    long_entry=Condition("volume > 1000"),
                ),
            ],
        )
        assert len(defn.modes) == 2
        assert defn.aggregation == "first_to_fire"
        assert defn.use_session_filter is True

    def test_any_aggregation(self) -> None:
        defn = CompositeStrategyDefinition(
            name="Any", short_name="A",
            modes=[], aggregation="any",
        )
        assert defn.aggregation == "any"


# ======================================================================
# strategy_to_mode converter
# ======================================================================


class TestStrategyToMode:
    def test_convert_macd(self) -> None:
        strategy = macd_strategy()
        mode = strategy_to_mode(strategy, "macd", "MACD Crossover")
        assert mode.short_name == "macd"
        assert mode.group == "MACD Crossover"
        assert mode.long_entry is not None
        assert mode.short_entry is not None
        assert mode.inputs == strategy.inputs
        assert mode.indicators == strategy.indicators

    def test_convert_rsi(self) -> None:
        mode = strategy_to_mode(rsi_strategy(), "rsi", "RSI")
        assert mode.short_name == "rsi"
        assert mode.enabled_default is True

    def test_convert_sma(self) -> None:
        mode = strategy_to_mode(
            sma_crossover_strategy(), "sma", "SMA",
            enabled_default=False,
        )
        assert mode.enabled_default is False
        assert mode.short_entry is None  # SMA is long-only

    def test_custom_label_colors(self) -> None:
        mode = strategy_to_mode(
            macd_strategy(), "macd", "MACD",
            label_color_long="color.blue",
            label_color_short="color.purple",
        )
        assert mode.label_color_long == "color.blue"
        assert mode.label_color_short == "color.purple"


# ======================================================================
# VWAP Mode Definitions
# ======================================================================


class TestVwapModes:
    def test_pb_mode(self) -> None:
        mode = _pb_mode()
        assert mode.short_name == "pb"
        assert mode.group == "Pullback"
        assert mode.long_entry is not None
        assert mode.short_entry is not None
        assert "adxValue >= pbAdxMin" in mode.long_entry.expr
        assert any(i.name == "pbZone" for i in mode.inputs)

    def test_rev_mode(self) -> None:
        mode = _rev_mode()
        assert mode.short_name == "rev"
        assert mode.group == "Reversal"
        assert "adxValue < revAdxMax" in mode.long_entry.expr

    def test_orb_mode(self) -> None:
        mode = _orb_mode()
        assert mode.short_name == "orb"
        assert mode.group == "OR Breakout"
        assert "orDone" in mode.long_entry.expr
        assert "orbFiredToday" in mode.long_entry.expr
        assert any("orbFiredToday" in c for c in mode.extra_code)

    def test_orr_mode(self) -> None:
        mode = _orr_mode()
        assert mode.short_name == "orr"
        assert mode.group == "OR Reversal"
        assert "orrFiredToday" in mode.long_entry.expr

    def test_ps_mode(self) -> None:
        mode = _ps_mode()
        assert mode.short_name == "ps"
        assert mode.enabled_default is False  # PS disabled by default
        assert "isManip" in mode.long_entry.expr


# ======================================================================
# generate_composite() — Core functionality
# ======================================================================


class TestGenerateComposite:
    def test_minimal_composite(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="Minimal",
            short_name="MIN",
            modes=[
                ModeDefinition(
                    name="Alpha", short_name="a", group="Alpha",
                    long_entry=Condition("close > open"),
                    short_entry=Condition("close < open"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert '//@version=6' in pine
        assert 'indicator("Minimal"' in pine
        assert 'enableA' in pine
        assert 'bool aLong = enableA and (close > open)' in pine
        assert 'bool aShort = enableA and (close < open)' in pine

    def test_session_filter(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="Session", short_name="S",
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    long_entry=Condition("true"),
                ),
            ],
            use_session_filter=True,
        )
        pine = gen.generate_composite(defn)
        assert 'string SESSION  = "0930-1600"' in pine
        assert "isNewSession" in pine
        assert "posState := 0" in pine  # Reset on new session

    def test_no_session_filter(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="NoSession", short_name="NS",
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    long_entry=Condition("true"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert "SESSION" not in pine

    def test_first_to_fire_aggregation(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = CompositeStrategyDefinition(
            name="FTF", short_name="FTF",
            modes=[
                ModeDefinition(
                    name="A", short_name="a", group="A",
                    long_entry=Condition("condA"),
                ),
                ModeDefinition(
                    name="B", short_name="b", group="B",
                    long_entry=Condition("condB"),
                ),
            ],
            aggregation="first_to_fire",
            priority_order=["a", "b"],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        # Check priority chain: a before b
        a_pos = pine.index("if aLong")
        b_pos = pine.index("else if bLong")
        assert a_pos < b_pos

    def test_any_aggregation(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="Any", short_name="ANY",
            modes=[
                ModeDefinition(
                    name="A", short_name="a", group="A",
                    long_entry=Condition("condA"),
                ),
                ModeDefinition(
                    name="B", short_name="b", group="B",
                    long_entry=Condition("condB"),
                ),
            ],
            aggregation="any",
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert "aLong or bLong" in pine
        assert "aShort or bShort" in pine

    def test_shared_inputs(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="Shared", short_name="SH",
            shared_inputs=[
                Input.int_("atrLen", 14, "ATR Length"),
            ],
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    inputs=[
                        Input.float_("mZone", 0.5, "Zone"),
                    ],
                    long_entry=Condition("true"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert 'input.int(14' in pine
        assert 'input.float(0.5' in pine

    def test_shared_indicators(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="SharedInd", short_name="SI",
            shared_indicators=[
                Indicator.atr("atrVal", "14"),
            ],
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    long_entry=Condition("true"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert "float atrVal = ta.atr(14)" in pine

    def test_mode_indicators(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="ModeInd", short_name="MI",
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    indicators=[
                        Indicator.rsi("mRsi", "close", "14"),
                    ],
                    long_entry=Condition("mRsi < 30"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert "float mRsi = ta.rsi(close, 14)" in pine

    def test_mode_extra_code(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="Extra", short_name="EX",
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    extra_code=[
                        "var bool flag = false",
                        "if isNewSession",
                        "    flag := false",
                    ],
                    long_entry=Condition("true"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert "var bool flag = false" in pine

    def test_labels_include_mode_name(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = CompositeStrategyDefinition(
            name="Labels", short_name="LBL",
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    long_entry=Condition("true"),
                    short_entry=Condition("false"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert 'activeMode + " Buy"' in pine
        assert 'activeMode + " Sell"' in pine

    def test_background(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="BG", short_name="BG",
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    long_entry=Condition("true"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert "bgcolor(bgColor" in pine
        assert 'color.new(color.green, 90)' in pine

    def test_alerts(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="Alerts Test", short_name="AT",
            modes=[
                ModeDefinition(
                    name="Alpha", short_name="a", group="A",
                    long_entry=Condition("true"),
                    short_entry=Condition("false"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert 'title="Buy"' in pine
        assert 'title="Sell"' in pine
        assert 'title="Exit Long"' in pine
        assert 'title="Exit Short"' in pine
        assert 'title="Alpha Signal"' in pine
        assert 'title="Any Signal"' in pine

    def test_per_mode_alerts_parenthesized(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = CompositeStrategyDefinition(
            name="Parens", short_name="P",
            modes=[
                ModeDefinition(
                    name="X", short_name="x", group="X",
                    long_entry=Condition("true"),
                    short_entry=Condition("false"),
                ),
            ],
            use_session_filter=True,
        )
        pine = gen.generate_composite(defn)
        # Ensure OR is parenthesized before session guard
        assert "(xLong or xShort) and inSession" in pine

    def test_position_state_machine(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = CompositeStrategyDefinition(
            name="State", short_name="ST",
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    long_entry=Condition("true"),
                    short_entry=Condition("false"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert "var int posState = 0" in pine
        assert "buySignal      = longCondition and posState != 1" in pine
        assert "sellSignal     = shortCondition and posState != -1" in pine

    def test_vwap_in_shared_indicators(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = CompositeStrategyDefinition(
            name="VWAP", short_name="V",
            shared_indicators=[
                Indicator.session_vwap("vwapValue"),
            ],
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    long_entry=Condition("close > vwapValue"),
                ),
            ],
            use_session_filter=True,
        )
        pine = gen.generate_composite(defn)
        assert "cumTPV" in pine
        assert "cumVol" in pine
        assert "float vwapValue" in pine

    def test_mode_disabled_by_default(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = CompositeStrategyDefinition(
            name="Disabled", short_name="D",
            modes=[
                ModeDefinition(
                    name="Off", short_name="off", group="Off",
                    enabled_default=False,
                    long_entry=Condition("true"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert 'input.bool(false, title="Enable Off"' in pine

    def test_shared_code(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="Shared Code", short_name="SC",
            shared_code=["float x = 42.0"],
            modes=[
                ModeDefinition(
                    name="M", short_name="m", group="M",
                    long_entry=Condition("true"),
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert "float x = 42.0" in pine
        assert "SHARED COMPUTATIONS" in pine

    def test_mode_long_only(self, gen: PineScriptGenerator) -> None:
        defn = CompositeStrategyDefinition(
            name="Long Only", short_name="LO",
            modes=[
                ModeDefinition(
                    name="L", short_name="l", group="L",
                    long_entry=Condition("close > open"),
                    short_entry=None,
                ),
            ],
            use_session_filter=False,
        )
        pine = gen.generate_composite(defn)
        assert "bool lLong = enableL and (close > open)" in pine
        assert "bool lShort = false" in pine


# ======================================================================
# Pre-built composite strategies
# ======================================================================


class TestPrebuiltComposites:
    def test_vwap_composite_structure(self) -> None:
        defn = vwap_composite_strategy()
        assert defn.short_name == "VWAP-ALL"
        assert len(defn.modes) == 5
        mode_names = {m.short_name for m in defn.modes}
        assert mode_names == {"ps", "orb", "orr", "pb", "rev"}
        assert defn.aggregation == "first_to_fire"
        assert defn.priority_order == ["ps", "orb", "orr", "pb", "rev"]
        assert defn.use_session_filter is True

    def test_vwap_composite_generates(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = vwap_composite_strategy()
        pine = gen.generate_composite(defn)
        assert '//@version=6' in pine
        assert 'enablePB' in pine
        assert 'enableORB' in pine
        assert 'enableREV' in pine
        assert 'enableORR' in pine
        assert 'enablePS' in pine
        # VWAP code
        assert 'cumTPV' in pine
        # OR tracking
        assert 'orHigh' in pine
        assert 'orDone' in pine

    def test_vwap_composite_priority_order(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = vwap_composite_strategy()
        pine = gen.generate_composite(defn)
        # PS should be first in priority chain
        ps_pos = pine.index("if psLong")
        orb_pos = pine.index("else if orbLong")
        pb_pos = pine.index("else if pbLong")
        assert ps_pos < orb_pos < pb_pos

    def test_signal_stack_composite_structure(self) -> None:
        defn = signal_stack_composite_strategy()
        assert defn.short_name == "SIG-STACK"
        assert len(defn.modes) == 5
        mode_names = {m.short_name for m in defn.modes}
        assert mode_names == {"macd", "rsi", "sma", "obv", "adx"}
        assert defn.aggregation == "any"
        assert defn.use_session_filter is False

    def test_signal_stack_composite_generates(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = signal_stack_composite_strategy()
        pine = gen.generate_composite(defn)
        assert '//@version=6' in pine
        assert 'enableMACD' in pine
        assert 'enableRSI' in pine
        assert 'enableSMA' in pine
        assert 'enableOBV' in pine
        assert 'enableADX' in pine
        # "any" aggregation
        assert "macdLong or rsiLong" in pine

    def test_full_composite_structure(self) -> None:
        defn = full_composite_strategy()
        assert defn.short_name == "FULL-COMP"
        assert len(defn.modes) == 6
        mode_names = {m.short_name for m in defn.modes}
        assert mode_names == {
            "dmivwap", "macdobv", "mom", "bbrsi", "rsi", "sma",
        }
        assert defn.aggregation == "first_to_fire"

    def test_full_composite_generates(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = full_composite_strategy()
        pine = gen.generate_composite(defn)
        assert '//@version=6' in pine
        assert 'enableDMIVWAP' in pine
        assert 'enableMACDOBV' in pine
        assert 'enableMOM' in pine
        assert 'enableBBRSI' in pine
        # All variable names are unique (prefixed)
        assert 'dvDmiPeriod' in pine
        assert 'moFast' in pine
        assert 'mcFast' in pine
        assert 'brBbPeriod' in pine
        assert 'rsRsiPeriod' in pine
        assert 'smFast' in pine

    def test_full_composite_no_var_conflicts(
        self, gen: PineScriptGenerator
    ) -> None:
        """Verify unique variable names don't collide."""
        defn = full_composite_strategy()
        pine = gen.generate_composite(defn)
        lines = pine.split("\n")
        # Check that no input variable is declared twice
        declarations: dict[str, int] = {}
        for line in lines:
            stripped = line.strip()
            for prefix in ("int ", "float ", "bool ", "string "):
                if stripped.startswith(prefix) and "= input." in stripped:
                    var_name = stripped.split("=")[0].strip().split()[-1]
                    declarations[var_name] = (
                        declarations.get(var_name, 0) + 1
                    )
        duplicates = {k: v for k, v in declarations.items() if v > 1}
        assert not duplicates, f"Duplicate declarations: {duplicates}"


# ======================================================================
# Catalog
# ======================================================================


class TestCompositeCatalog:
    def test_catalog_has_all(self) -> None:
        assert "vwap_composite" in COMPOSITE_STRATEGY_CATALOG
        assert "signal_stack" in COMPOSITE_STRATEGY_CATALOG
        assert "full_composite" in COMPOSITE_STRATEGY_CATALOG

    def test_all_catalog_entries_generate(
        self, gen: PineScriptGenerator
    ) -> None:
        for name, factory in COMPOSITE_STRATEGY_CATALOG.items():
            defn = factory()
            pine = gen.generate_composite(defn)
            assert '//@version=6' in pine, f"{name} missing version"
            assert 'indicator(' in pine, f"{name} missing indicator"

    def test_all_composites_have_alerts(
        self, gen: PineScriptGenerator
    ) -> None:
        for name, factory in COMPOSITE_STRATEGY_CATALOG.items():
            defn = factory()
            pine = gen.generate_composite(defn)
            assert 'alertcondition(' in pine, f"{name} missing alerts"
            assert '"Any Signal"' in pine, f"{name} missing combined alert"


# ======================================================================
# Section dividers
# ======================================================================


class TestCompositeDividers:
    def test_divider_uses_box_drawing(
        self, gen: PineScriptGenerator
    ) -> None:
        defn = vwap_composite_strategy()
        pine = gen.generate_composite(defn)
        assert "\u2500" * 77 in pine
        assert "=" * 76 not in pine


# ======================================================================
# SharedInfrastructure & mode_to_strategy
# ======================================================================


class TestSharedInfrastructure:
    def test_empty_defaults(self) -> None:
        infra = SharedInfrastructure()
        assert infra.inputs == []
        assert infra.indicators == []
        assert infra.code == []

    def test_vwap_shared_infra_has_content(self) -> None:
        infra = _vwap_shared_infrastructure()
        assert len(infra.inputs) > 0
        assert len(infra.indicators) > 0
        assert len(infra.code) > 0

    def test_vwap_shared_infra_has_atr_input(self) -> None:
        infra = _vwap_shared_infrastructure()
        names = {inp.name for inp in infra.inputs}
        assert "atrLen" in names
        assert "adxLen" in names

    def test_vwap_shared_infra_has_vwap_indicator(self) -> None:
        infra = _vwap_shared_infrastructure()
        var_names = {ind.var_name for ind in infra.indicators}
        assert "vwapValue" in var_names

    def test_vwap_shared_infra_has_or_tracking_code(self) -> None:
        infra = _vwap_shared_infrastructure()
        code_str = "\n".join(infra.code)
        assert "orHigh" in code_str
        assert "barOfDay" in code_str


class TestModeToStrategy:
    def test_basic_conversion(self, gen: PineScriptGenerator) -> None:
        mode = ModeDefinition(
            name="Test Mode",
            short_name="tm",
            group="Test",
            long_entry=Condition("close > open"),
            short_entry=Condition("close < open"),
            description="A test mode.",
        )
        strat = mode_to_strategy(mode, use_session_filter=False)
        assert strat.name == "Test Mode"
        assert strat.short_name == "TM"
        assert strat.description == "A test mode."
        pine = gen.generate(strat)
        assert '//@version=6' in pine
        assert "close > open" in pine

    def test_name_override(self) -> None:
        mode = _pb_mode()
        strat = mode_to_strategy(
            mode,
            name_override="My Pullback",
            short_name_override="MY-PB",
        )
        assert strat.name == "My Pullback"
        assert strat.short_name == "MY-PB"

    def test_shared_infrastructure_merged(
        self, gen: PineScriptGenerator,
    ) -> None:
        mode = _pb_mode()
        infra = _vwap_shared_infrastructure()
        strat = mode_to_strategy(
            mode, shared=infra,
            name_override="VWAP Pullback",
            short_name_override="VWAP-PB",
        )
        # Shared inputs + mode inputs
        input_names = {inp.name for inp in strat.inputs}
        assert "atrLen" in input_names  # shared
        assert "pbZone" in input_names  # mode-specific
        # Shared indicators
        ind_names = {ind.var_name for ind in strat.indicators}
        assert "vwapValue" in ind_names
        # Mode indicators in extra_code
        code = "\n".join(strat.extra_code)
        assert "vwapStd" in code  # PB mode indicator
        assert "orHigh" in code   # shared code
        # Generates valid Pine
        pine = gen.generate(strat)
        assert '//@version=6' in pine
        assert "VWAP Pullback" in pine

    def test_round_trip_strategy_mode_strategy(
        self, gen: PineScriptGenerator,
    ) -> None:
        """strategy → mode → strategy should produce equivalent PineScript."""
        original = sma_crossover_strategy()
        mode = strategy_to_mode(original, "sma", "SMA Group")
        reconverted = mode_to_strategy(
            mode, use_session_filter=False,
            name_override=original.name,
            short_name_override=original.short_name,
        )
        assert reconverted.name == original.name
        assert reconverted.short_name == original.short_name
        # Both should generate valid Pine
        pine_orig = gen.generate(original)
        pine_round = gen.generate(reconverted)
        assert '//@version=6' in pine_orig
        assert '//@version=6' in pine_round
        # Entry conditions preserved
        assert original.long_entry.expr == reconverted.long_entry.expr

    def test_all_vwap_modes_convertible(
        self, gen: PineScriptGenerator,
    ) -> None:
        """All 5 VWAP modes convert to standalone strategies."""
        infra = _vwap_shared_infrastructure()
        for factory in [_pb_mode, _rev_mode, _orb_mode, _orr_mode, _ps_mode]:
            mode = factory()
            strat = mode_to_strategy(mode, shared=infra)
            pine = gen.generate(strat)
            assert '//@version=6' in pine
            assert mode.name in pine


# ======================================================================
# CLI smoke tests
# ======================================================================


class TestCompositeCLI:
    def test_composite_list(self) -> None:
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "stockdownloader.app.generate_pinescript",
             "--composite", "--list"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "vwap_composite" in result.stdout
        assert "signal_stack" in result.stdout
        assert "full_composite" in result.stdout

    def test_standard_list_shows_composites(self) -> None:
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "stockdownloader.app.generate_pinescript",
             "--list"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "composite" in result.stdout.lower()
