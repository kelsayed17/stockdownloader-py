"""Tests for PineScript strategy mode (v2) generation.

Validates that strategy-mode SPY factories produce correct ``strategy()``
scripts with:
- ``strategy()`` header (not ``indicator()``)
- ``strategy.entry()`` / ``strategy.exit()`` calls
- ``f_qty()`` position sizing function
- Breakeven management
- Circuit breaker & daily loss limit
- EOD close via ``strategy.close_all()``
- Rich ``alert()`` calls (not ``alertcondition()``)
- ``strategy.position_size`` for background (not ``posState``)
- Visual level plots (Entry Price, Take Profit, Stop Loss)

Also verifies backwards compatibility: existing indicator-mode tests unaffected.
"""

from __future__ import annotations

import pytest

from stockdownloader.app.pinescript_catalog.spy_strategies import (
    spy_macd_obv_strategy,
    spy_macd_obv_strategy_v2,
    spy_macd_optimized_strategy,
    spy_macd_optimized_strategy_v2,
    spy_sma_crossover_strategy,
    spy_sma_crossover_strategy_v2,
)
from stockdownloader.app.pinescript_catalog.catalogs import STRATEGY_CATALOG
from stockdownloader.pinescript import PineScriptGenerator


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def gen():
    return PineScriptGenerator()


@pytest.fixture
def macd_obv_v2(gen):
    return gen.generate(spy_macd_obv_strategy_v2())


@pytest.fixture
def sma_cross_v2(gen):
    return gen.generate(spy_sma_crossover_strategy_v2())


@pytest.fixture
def macd_opt_v2(gen):
    return gen.generate(spy_macd_optimized_strategy_v2())


# ======================================================================
# Test: Strategy header (not indicator)
# ======================================================================


class TestStrategyHeader:
    """Strategy mode scripts use strategy() not indicator()."""

    def test_macd_obv_has_strategy_header(self, macd_obv_v2):
        assert 'strategy("SPY MACD+OBV (Tournament Winner)"' in macd_obv_v2

    def test_sma_cross_has_strategy_header(self, sma_cross_v2):
        assert 'strategy("SPY SMA Cross 20/21 (Tournament #2)"' in sma_cross_v2

    def test_macd_opt_has_strategy_header(self, macd_opt_v2):
        assert 'strategy("SPY MACD Optimized 8/35/5 (Tournament #3)"' in macd_opt_v2

    def test_no_indicator_keyword(self, macd_obv_v2, sma_cross_v2, macd_opt_v2):
        for pine in [macd_obv_v2, sma_cross_v2, macd_opt_v2]:
            assert 'indicator(' not in pine

    def test_initial_capital(self, macd_obv_v2):
        assert "initial_capital=100000" in macd_obv_v2

    def test_commission(self, macd_obv_v2):
        assert "commission_type=strategy.commission.cash_per_order" in macd_obv_v2
        assert "commission_value=1.0" in macd_obv_v2

    def test_slippage(self, macd_obv_v2):
        assert "slippage=1" in macd_obv_v2

    def test_version_6(self, macd_obv_v2):
        assert "//@version=6" in macd_obv_v2


# ======================================================================
# Test: strategy.entry() and strategy.exit() calls
# ======================================================================


class TestStrategyEntryExit:
    """Strategy mode uses strategy.entry/exit instead of manual posState."""

    def test_has_strategy_entry_long(self, macd_obv_v2):
        assert 'strategy.entry("L", strategy.long' in macd_obv_v2

    def test_has_strategy_entry_short(self, macd_obv_v2):
        assert 'strategy.entry("S", strategy.short' in macd_obv_v2

    def test_has_strategy_exit_long(self, macd_obv_v2):
        assert 'strategy.exit("LX", "L"' in macd_obv_v2

    def test_has_strategy_exit_short(self, macd_obv_v2):
        assert 'strategy.exit("SX", "S"' in macd_obv_v2

    def test_exit_has_stop_and_limit(self, macd_obv_v2):
        assert "stop=longSL, limit=longTP" in macd_obv_v2
        assert "stop=shortSL, limit=shortTP" in macd_obv_v2

    def test_strategy_close_for_condition_exit(self, macd_obv_v2):
        assert 'strategy.close("L", comment="ExitLong")' in macd_obv_v2
        assert 'strategy.close("S", comment="ExitShort")' in macd_obv_v2

    def test_flip_close_before_entry(self, macd_obv_v2):
        assert 'strategy.close("S", comment="Flip")' in macd_obv_v2
        assert 'strategy.close("L", comment="Flip")' in macd_obv_v2

    def test_qty_parameter(self, macd_obv_v2):
        assert "qty=qty" in macd_obv_v2

    def test_comment_parameter(self, macd_obv_v2):
        assert 'comment="SPY-MACD-OBV|L"' in macd_obv_v2
        assert 'comment="SPY-MACD-OBV|S"' in macd_obv_v2

    def test_no_manual_pos_state(self, macd_obv_v2, sma_cross_v2, macd_opt_v2):
        """Strategy mode must NOT use manual posState tracking."""
        for pine in [macd_obv_v2, sma_cross_v2, macd_opt_v2]:
            assert "var int posState" not in pine
            assert "posState :=" not in pine

    # SMA crossover is long-only — no short entry/exit
    def test_sma_no_short_entry(self, sma_cross_v2):
        assert 'strategy.entry("S"' not in sma_cross_v2

    def test_sma_no_short_exit(self, sma_cross_v2):
        assert 'strategy.exit("SX"' not in sma_cross_v2

    def test_sma_has_long_entry(self, sma_cross_v2):
        assert 'strategy.entry("L", strategy.long' in sma_cross_v2

    def test_sma_has_long_exit(self, sma_cross_v2):
        assert 'strategy.exit("LX", "L"' in sma_cross_v2


# ======================================================================
# Test: Position sizing (f_qty)
# ======================================================================


class TestPositionSizing:
    """f_qty() position sizing function is generated."""

    def test_f_qty_function(self, macd_obv_v2):
        assert "f_qty(float entry, float stop)" in macd_obv_v2

    def test_f_qty_uses_capital(self, macd_obv_v2):
        assert "100000.0" in macd_obv_v2

    def test_f_qty_risk_calculation(self, macd_obv_v2):
        assert "risk / perSh" in macd_obv_v2

    def test_f_qty_max_qty_cap(self, macd_obv_v2):
        assert "cap / entry * 0.95" in macd_obv_v2

    def test_f_qty_min_qty(self, macd_obv_v2):
        assert "math.max(math.min(raw, maxQty), 1.0)" in macd_obv_v2


# ======================================================================
# Test: SL / TP calculation
# ======================================================================


class TestSlTpCalculation:
    """ATR-based SL/TP with dollar cap."""

    def test_sl_risk_calculation(self, macd_obv_v2):
        assert "slRisk" in macd_obv_v2
        assert "atrVal * 1.5" in macd_obv_v2

    def test_sl_dollar_cap(self, macd_obv_v2):
        assert "math.min(atrVal * 1.5, 2.0)" in macd_obv_v2

    def test_long_sl_tp(self, macd_obv_v2):
        assert "longSL   = close - slRisk" in macd_obv_v2
        assert "longTP   = close + slRisk * 1.5" in macd_obv_v2

    def test_short_sl_tp(self, macd_obv_v2):
        assert "shortSL  = close + slRisk" in macd_obv_v2
        assert "shortTP  = close - slRisk * 1.5" in macd_obv_v2


# ======================================================================
# Test: Breakeven management
# ======================================================================


class TestBreakevenManagement:
    """BE trigger moves SL to entry + buffer."""

    def test_be_trigger_code(self, macd_obv_v2):
        assert "Breakeven trigger" in macd_obv_v2

    def test_be_uses_entry_price(self, macd_obv_v2):
        assert "entryPrice + beBuf" in macd_obv_v2

    def test_be_moved_flag(self, macd_obv_v2):
        assert "movedBE := true" in macd_obv_v2

    def test_be_comment(self, macd_obv_v2):
        assert 'comment="BE"' in macd_obv_v2

    def test_fill_alignment(self, macd_obv_v2):
        assert "strategy.position_avg_price" in macd_obv_v2

    def test_be_alert(self, macd_obv_v2):
        assert "BE LONG" in macd_obv_v2
        assert "BE SHORT" in macd_obv_v2


# ======================================================================
# Test: Risk management (circuit breaker, daily loss limit)
# ======================================================================


class TestRiskManagement:
    """Circuit breaker and daily loss limit."""

    def test_day_trades_counter(self, macd_obv_v2):
        assert "dayTrades" in macd_obv_v2
        assert "dayTrades < 4" in macd_obv_v2

    def test_circuit_breaker(self, macd_obv_v2):
        assert "consecLoss" in macd_obv_v2
        assert "tripped" in macd_obv_v2
        assert "CIRCUIT BREAKER" in macd_obv_v2
        assert "consecLoss >= 3" in macd_obv_v2

    def test_daily_loss_limit(self, macd_obv_v2):
        assert "dayLimited" in macd_obv_v2
        assert "-3.0" in macd_obv_v2

    def test_ready_guard(self, macd_obv_v2):
        assert "bool ready" in macd_obv_v2
        assert "not tripped" in macd_obv_v2
        assert "not dayLimited" in macd_obv_v2

    def test_bar_spacing(self, macd_obv_v2):
        assert "bool spaced" in macd_obv_v2
        assert "lastBarEntry" in macd_obv_v2

    def test_daily_reset(self, macd_obv_v2):
        assert "isNewDay" in macd_obv_v2
        assert "dayTrades  := 0" in macd_obv_v2

    def test_barstate_confirmed(self, macd_obv_v2):
        assert "barstate.isconfirmed" in macd_obv_v2


# ======================================================================
# Test: EOD close
# ======================================================================


class TestEodClose:
    """End-of-day close via strategy.close_all()."""

    def test_close_all(self, macd_obv_v2):
        assert 'strategy.close_all(comment="EOD")' in macd_obv_v2

    def test_eod_condition(self, macd_obv_v2):
        assert "not isMarket and isMarket[1]" in macd_obv_v2

    def test_eod_alert(self, macd_obv_v2):
        assert "EOD CLOSE" in macd_obv_v2


# ======================================================================
# Test: Alerts (rich alert(), not alertcondition())
# ======================================================================


class TestAlerts:
    """Strategy mode uses inline alert() not alertcondition()."""

    def test_no_alertcondition(self, macd_obv_v2, sma_cross_v2, macd_opt_v2):
        for pine in [macd_obv_v2, sma_cross_v2, macd_opt_v2]:
            assert "alertcondition(" not in pine

    def test_has_alert_calls(self, macd_obv_v2):
        assert "alert(" in macd_obv_v2

    def test_long_alert_content(self, macd_obv_v2):
        assert "LONG | E " in macd_obv_v2
        assert "SL " in macd_obv_v2
        assert "TP " in macd_obv_v2
        assert "Qty " in macd_obv_v2

    def test_short_alert_content(self, macd_obv_v2):
        assert "SHORT | E " in macd_obv_v2

    def test_alert_frequency(self, macd_obv_v2):
        assert "alert.freq_once_per_bar_close" in macd_obv_v2


# ======================================================================
# Test: Background uses strategy.position_size
# ======================================================================


class TestBackground:
    """Background coloring uses strategy.position_size."""

    def test_uses_position_size(self, macd_obv_v2):
        assert "strategy.position_size > 0" in macd_obv_v2

    def test_no_pos_state_in_bg(self, macd_obv_v2):
        assert "posState ==" not in macd_obv_v2

    def test_bgcolor_present(self, macd_obv_v2):
        assert 'bgcolor(bgColor, title="Position Zone")' in macd_obv_v2

    def test_green_for_long(self, macd_obv_v2):
        assert "color.new(color.green, 90)" in macd_obv_v2

    def test_red_for_short(self, macd_obv_v2):
        assert "color.new(color.red, 90)" in macd_obv_v2


# ======================================================================
# Test: Visual level plots
# ======================================================================


class TestLevelPlots:
    """Entry Price, Take Profit, Stop Loss visual lines."""

    def test_entry_price_plot(self, macd_obv_v2):
        assert 'title="Entry Price"' in macd_obv_v2

    def test_take_profit_plot(self, macd_obv_v2):
        assert 'title="Take Profit"' in macd_obv_v2

    def test_stop_loss_plot(self, macd_obv_v2):
        assert 'title="Stop Loss"' in macd_obv_v2

    def test_plot_style_linebr(self, macd_obv_v2):
        assert "plot.style_linebr" in macd_obv_v2


# ======================================================================
# Test: Labels emitted inline (not empty)
# ======================================================================


class TestLabels:
    """Strategy mode includes labels inline with execution."""

    def test_buy_label(self, macd_obv_v2):
        assert 'label.new(bar_index, low, "Buy"' in macd_obv_v2

    def test_sell_label(self, macd_obv_v2):
        assert 'label.new(bar_index, high, "Sell"' in macd_obv_v2

    def test_exit_long_label(self, macd_obv_v2):
        assert 'label.new(bar_index, high, "Exit Long"' in macd_obv_v2

    def test_exit_short_label(self, macd_obv_v2):
        assert 'label.new(bar_index, low, "Exit Short"' in macd_obv_v2


# ======================================================================
# Test: Strategy definitions have correct model fields
# ======================================================================


class TestStrategyDefinitionFields:
    """v2 factories set strategy_mode=True and correct defaults."""

    def test_macd_obv_strategy_mode(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.strategy_mode is True

    def test_sma_cross_strategy_mode(self):
        defn = spy_sma_crossover_strategy_v2()
        assert defn.strategy_mode is True

    def test_macd_opt_strategy_mode(self):
        defn = spy_macd_optimized_strategy_v2()
        assert defn.strategy_mode is True

    def test_default_capital(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.initial_capital == 100000.0

    def test_default_commission(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.commission_per_order == 1.0

    def test_default_slippage(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.slippage == 1

    def test_default_risk_pct(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.risk_per_trade_pct == 1.0

    def test_default_sl_atr_mult(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.sl_atr_mult == 1.5

    def test_default_sl_cap(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.sl_cap_dollars == 2.0

    def test_default_rr_ratio(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.rr_ratio == 1.5

    def test_default_be_trigger(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.be_trigger == 0.5

    def test_default_max_trades(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.max_trades_per_day == 4

    def test_default_circuit_breaker(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.circuit_breaker_losses == 3

    def test_default_daily_loss_limit(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.daily_loss_limit_pct == 3.0

    def test_close_eod_default(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.close_eod is True

    def test_barstate_confirmed_default(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.barstate_confirmed is True

    def test_sma_is_long_only(self):
        defn = spy_sma_crossover_strategy_v2()
        assert defn.long_entry is not None
        assert defn.short_entry is None

    def test_macd_obv_has_both_entries(self):
        defn = spy_macd_obv_strategy_v2()
        assert defn.long_entry is not None
        assert defn.short_entry is not None

    def test_indicators_include_atr(self):
        """Strategy mode needs ATR for SL/TP calculation."""
        for factory in [
            spy_macd_obv_strategy_v2,
            spy_sma_crossover_strategy_v2,
            spy_macd_optimized_strategy_v2,
        ]:
            defn = factory()
            atr_vars = [i.var_name for i in defn.indicators if "atr" in i.code.lower()]
            assert len(atr_vars) > 0, f"{defn.name}: missing ATR indicator"

    def test_no_extra_plots(self):
        """Strategy mode handles TP/SL natively — no extra_plots needed."""
        for factory in [
            spy_macd_obv_strategy_v2,
            spy_sma_crossover_strategy_v2,
            spy_macd_optimized_strategy_v2,
        ]:
            defn = factory()
            assert len(defn.extra_plots) == 0, f"{defn.name}: unexpected extra_plots"


# ======================================================================
# Test: Backwards compatibility (indicator mode unchanged)
# ======================================================================


class TestBackwardsCompatibility:
    """Existing indicator-mode factories still produce indicator() scripts."""

    def test_v1_macd_obv_is_indicator(self, gen):
        pine = gen.generate(spy_macd_obv_strategy())
        assert 'indicator(' in pine
        assert 'strategy(' not in pine

    def test_v1_sma_crossover_is_indicator(self, gen):
        pine = gen.generate(spy_sma_crossover_strategy())
        assert 'indicator(' in pine
        assert 'strategy(' not in pine

    def test_v1_macd_optimized_is_indicator(self, gen):
        pine = gen.generate(spy_macd_optimized_strategy())
        assert 'indicator(' in pine
        assert 'strategy(' not in pine

    def test_v1_has_posstate(self, gen):
        pine = gen.generate(spy_macd_obv_strategy())
        assert "var int posState" in pine

    def test_v1_has_alertcondition(self, gen):
        pine = gen.generate(spy_macd_obv_strategy())
        assert "alertcondition(" in pine

    def test_v1_has_extra_plots(self, gen):
        pine = gen.generate(spy_macd_obv_strategy())
        assert 'title="Entry Price"' in pine
        assert 'title="Take Profit"' in pine
        assert 'title="Stop Loss"' in pine


# ======================================================================
# Test: Catalog registration
# ======================================================================


class TestCatalogRegistration:
    """v2 strategies registered in STRATEGY_CATALOG."""

    def test_macd_obv_v2_in_catalog(self):
        assert "spy_macd_obv_v2" in STRATEGY_CATALOG

    def test_sma_crossover_v2_in_catalog(self):
        assert "spy_sma_crossover_v2" in STRATEGY_CATALOG

    def test_macd_optimized_v2_in_catalog(self):
        assert "spy_macd_optimized_v2" in STRATEGY_CATALOG

    def test_catalog_count(self):
        assert len(STRATEGY_CATALOG) == 21

    def test_v2_catalog_entries_generate(self, gen):
        for key in ["spy_macd_obv_v2", "spy_sma_crossover_v2", "spy_macd_optimized_v2"]:
            factory = STRATEGY_CATALOG[key]
            defn = factory()
            pine = gen.generate(defn)
            assert pine, f"{key}: empty output"
            assert "strategy(" in pine, f"{key}: missing strategy()"
            assert len(pine) > 200, f"{key}: output too short"


# ======================================================================
# Test: Long-only SMA crossover v2 specifics
# ======================================================================


class TestSmaLongOnly:
    """SMA crossover v2 is long-only — no short-related code."""

    def test_no_short_sl(self, sma_cross_v2):
        assert "shortSL" not in sma_cross_v2

    def test_no_short_tp(self, sma_cross_v2):
        assert "shortTP" not in sma_cross_v2

    def test_no_go_short(self, sma_cross_v2):
        assert "goShort" not in sma_cross_v2

    def test_no_short_condition(self, sma_cross_v2):
        assert "shortCondition" not in sma_cross_v2

    def test_no_exit_short_cond(self, sma_cross_v2):
        assert "exitShortCond" not in sma_cross_v2

    def test_has_long_sl(self, sma_cross_v2):
        assert "longSL" in sma_cross_v2

    def test_has_long_tp(self, sma_cross_v2):
        assert "longTP" in sma_cross_v2

    def test_has_exit_long(self, sma_cross_v2):
        assert "exitLongCond" in sma_cross_v2

    def test_no_short_be_trigger(self, sma_cross_v2):
        assert "BE SHORT" not in sma_cross_v2

    def test_has_long_be_trigger(self, sma_cross_v2):
        assert "BE LONG" in sma_cross_v2

    def test_bg_no_red_zone(self, sma_cross_v2):
        """Long-only: no red background zone for shorts."""
        assert "strategy.position_size < 0" not in sma_cross_v2


# ======================================================================
# Test: Output structure matches reference (vwap_v11_2.pine)
# ======================================================================


class TestOutputStructure:
    """Verify output structure matches reference strategy format."""

    def test_section_order(self, macd_obv_v2):
        """Sections appear in correct order."""
        sections = [
            "INPUTS",
            "INDICATORS",
            "SESSION",
            "POSITION SIZING",
            "RISK MANAGEMENT",
            "SIGNAL LOGIC",
            "SL / TP CALCULATION",
            "EXECUTION",
            "EXIT MANAGEMENT",
            "EOD CLOSE",
            "LOSS TRACKING",
            "LEVEL PLOTS",
            "BACKGROUND",
        ]
        last_pos = -1
        for section in sections:
            pos = macd_obv_v2.find(section)
            assert pos > last_pos, (
                f"Section '{section}' not found after previous section "
                f"(pos={pos}, last={last_pos})"
            )
            last_pos = pos

    def test_script_starts_with_license(self, macd_obv_v2):
        assert macd_obv_v2.startswith(
            "// This Pine Script(tm) v6 strategy"
        )

    def test_reasonable_line_count(self, macd_obv_v2, sma_cross_v2, macd_opt_v2):
        for pine in [macd_obv_v2, sma_cross_v2, macd_opt_v2]:
            lines = pine.strip().splitlines()
            assert len(lines) > 150, f"Too few lines: {len(lines)}"
            assert len(lines) < 400, f"Too many lines: {len(lines)}"
