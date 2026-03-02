"""Tests for PineScript strategy mode generation.

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
- Rich entry labels with direction icon, E/S/T prices
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
# Fixtures
# ======================================================================


@pytest.fixture
def gen():
    return PineScriptGenerator()


@pytest.fixture
def macd_obv_pine(gen):
    return gen.generate(spy_macd_obv_strategy())


@pytest.fixture
def sma_cross_pine(gen):
    return gen.generate(spy_sma_crossover_strategy())


@pytest.fixture
def macd_opt_pine(gen):
    return gen.generate(spy_macd_optimized_strategy())


# ======================================================================
# Test: Strategy header (not indicator)
# ======================================================================


class TestStrategyHeader:
    """Strategy mode scripts use strategy() not indicator()."""

    def test_macd_obv_has_strategy_header(self, macd_obv_pine):
        assert 'strategy("SPY MACD+OBV (Tournament Winner)"' in macd_obv_pine

    def test_sma_cross_has_strategy_header(self, sma_cross_pine):
        assert 'strategy("SPY SMA Cross 20/21 (Tournament #2)"' in sma_cross_pine

    def test_macd_opt_has_strategy_header(self, macd_opt_pine):
        assert 'strategy("SPY MACD Optimized 8/35/5 (Tournament #3)"' in macd_opt_pine

    def test_no_indicator_keyword(self, macd_obv_pine, sma_cross_pine, macd_opt_pine):
        for pine in [macd_obv_pine, sma_cross_pine, macd_opt_pine]:
            assert 'indicator(' not in pine

    def test_initial_capital(self, macd_obv_pine):
        assert "initial_capital=100000" in macd_obv_pine

    def test_commission(self, macd_obv_pine):
        assert "commission_type=strategy.commission.cash_per_order" in macd_obv_pine
        assert "commission_value=1.0" in macd_obv_pine

    def test_slippage(self, macd_obv_pine):
        assert "slippage=1" in macd_obv_pine

    def test_version_6(self, macd_obv_pine):
        assert "//@version=6" in macd_obv_pine


# ======================================================================
# Test: strategy.entry() and strategy.exit() calls
# ======================================================================


class TestStrategyEntryExit:
    """Strategy mode uses strategy.entry/exit instead of manual posState."""

    def test_has_strategy_entry_long(self, macd_obv_pine):
        assert 'strategy.entry("L", strategy.long' in macd_obv_pine

    def test_has_strategy_entry_short(self, macd_obv_pine):
        assert 'strategy.entry("S", strategy.short' in macd_obv_pine

    def test_has_strategy_exit_long(self, macd_obv_pine):
        assert 'strategy.exit("LX", "L"' in macd_obv_pine

    def test_has_strategy_exit_short(self, macd_obv_pine):
        assert 'strategy.exit("SX", "S"' in macd_obv_pine

    def test_exit_has_stop_and_limit(self, macd_obv_pine):
        assert "stop=longSL, limit=longTP" in macd_obv_pine
        assert "stop=shortSL, limit=shortTP" in macd_obv_pine

    def test_strategy_close_for_condition_exit(self, macd_obv_pine):
        assert 'strategy.close("L", comment="ExitLong")' in macd_obv_pine
        assert 'strategy.close("S", comment="ExitShort")' in macd_obv_pine

    def test_flip_close_before_entry(self, macd_obv_pine):
        assert 'strategy.close("S", comment="Flip")' in macd_obv_pine
        assert 'strategy.close("L", comment="Flip")' in macd_obv_pine

    def test_qty_parameter(self, macd_obv_pine):
        assert "qty=qty" in macd_obv_pine

    def test_comment_parameter(self, macd_obv_pine):
        assert 'comment="SPY-MACD-OBV|L"' in macd_obv_pine
        assert 'comment="SPY-MACD-OBV|S"' in macd_obv_pine

    def test_no_manual_pos_state(self, macd_obv_pine, sma_cross_pine, macd_opt_pine):
        """Strategy mode must NOT use manual posState tracking."""
        for pine in [macd_obv_pine, sma_cross_pine, macd_opt_pine]:
            assert "var int posState" not in pine
            assert "posState :=" not in pine

    # SMA crossover is long-only — no short entry/exit
    def test_sma_no_short_entry(self, sma_cross_pine):
        assert 'strategy.entry("S"' not in sma_cross_pine

    def test_sma_no_short_exit(self, sma_cross_pine):
        assert 'strategy.exit("SX"' not in sma_cross_pine

    def test_sma_has_long_entry(self, sma_cross_pine):
        assert 'strategy.entry("L", strategy.long' in sma_cross_pine

    def test_sma_has_long_exit(self, sma_cross_pine):
        assert 'strategy.exit("LX", "L"' in sma_cross_pine


# ======================================================================
# Test: Position sizing (f_qty)
# ======================================================================


class TestPositionSizing:
    """f_qty() position sizing function is generated."""

    def test_f_qty_function(self, macd_obv_pine):
        assert "f_qty(float entry, float stop)" in macd_obv_pine

    def test_f_qty_uses_capital(self, macd_obv_pine):
        assert "100000.0" in macd_obv_pine

    def test_f_qty_risk_calculation(self, macd_obv_pine):
        assert "risk / perSh" in macd_obv_pine

    def test_f_qty_max_qty_cap(self, macd_obv_pine):
        assert "cap / entry * 0.95" in macd_obv_pine

    def test_f_qty_min_qty(self, macd_obv_pine):
        assert "math.max(math.min(raw, maxQty), 1.0)" in macd_obv_pine


# ======================================================================
# Test: SL / TP calculation
# ======================================================================


class TestSlTpCalculation:
    """ATR-based SL/TP with dollar cap."""

    def test_sl_risk_calculation(self, macd_obv_pine):
        assert "slRisk" in macd_obv_pine
        assert "atrVal * 1.5" in macd_obv_pine

    def test_sl_dollar_cap(self, macd_obv_pine):
        assert "math.min(atrVal * 1.5, 2.0)" in macd_obv_pine

    def test_long_sl_tp(self, macd_obv_pine):
        assert "longSL   = close - slRisk" in macd_obv_pine
        assert "longTP   = close + slRisk * 1.5" in macd_obv_pine

    def test_short_sl_tp(self, macd_obv_pine):
        assert "shortSL  = close + slRisk" in macd_obv_pine
        assert "shortTP  = close - slRisk * 1.5" in macd_obv_pine


# ======================================================================
# Test: Breakeven management
# ======================================================================


class TestBreakevenManagement:
    """BE trigger moves SL to entry + buffer."""

    def test_be_trigger_code(self, macd_obv_pine):
        assert "Breakeven trigger" in macd_obv_pine

    def test_be_uses_entry_price(self, macd_obv_pine):
        assert "entryPrice + beBuf" in macd_obv_pine

    def test_be_moved_flag(self, macd_obv_pine):
        assert "movedBE := true" in macd_obv_pine

    def test_be_comment(self, macd_obv_pine):
        assert 'comment="BE"' in macd_obv_pine

    def test_fill_alignment(self, macd_obv_pine):
        assert "strategy.position_avg_price" in macd_obv_pine

    def test_be_alert(self, macd_obv_pine):
        assert "BE LONG" in macd_obv_pine
        assert "BE SHORT" in macd_obv_pine


# ======================================================================
# Test: Risk management (circuit breaker, daily loss limit)
# ======================================================================


class TestRiskManagement:
    """Circuit breaker and daily loss limit."""

    def test_day_trades_counter(self, macd_obv_pine):
        assert "dayTrades" in macd_obv_pine
        assert "dayTrades < 4" in macd_obv_pine

    def test_circuit_breaker(self, macd_obv_pine):
        assert "consecLoss" in macd_obv_pine
        assert "tripped" in macd_obv_pine
        assert "CIRCUIT BREAKER" in macd_obv_pine
        assert "consecLoss >= 3" in macd_obv_pine

    def test_daily_loss_limit(self, macd_obv_pine):
        assert "dayLimited" in macd_obv_pine
        assert "-3.0" in macd_obv_pine

    def test_ready_guard(self, macd_obv_pine):
        assert "bool ready" in macd_obv_pine
        assert "not tripped" in macd_obv_pine
        assert "not dayLimited" in macd_obv_pine

    def test_bar_spacing(self, macd_obv_pine):
        assert "bool spaced" in macd_obv_pine
        assert "lastBarEntry" in macd_obv_pine

    def test_daily_reset(self, macd_obv_pine):
        assert "isNewDay" in macd_obv_pine
        assert "dayTrades  := 0" in macd_obv_pine

    def test_barstate_confirmed(self, macd_obv_pine):
        assert "barstate.isconfirmed" in macd_obv_pine


# ======================================================================
# Test: EOD close
# ======================================================================


class TestEodClose:
    """End-of-day close via strategy.close_all()."""

    def test_close_all(self, macd_obv_pine):
        assert 'strategy.close_all(comment="EOD")' in macd_obv_pine

    def test_eod_condition(self, macd_obv_pine):
        assert "not isMarket and isMarket[1]" in macd_obv_pine

    def test_eod_alert(self, macd_obv_pine):
        assert "EOD CLOSE" in macd_obv_pine


# ======================================================================
# Test: Alerts (rich alert(), not alertcondition())
# ======================================================================


class TestAlerts:
    """Strategy mode uses inline alert() not alertcondition()."""

    def test_no_alertcondition(self, macd_obv_pine, sma_cross_pine, macd_opt_pine):
        for pine in [macd_obv_pine, sma_cross_pine, macd_opt_pine]:
            assert "alertcondition(" not in pine

    def test_has_alert_calls(self, macd_obv_pine):
        assert "alert(" in macd_obv_pine

    def test_long_alert_content(self, macd_obv_pine):
        assert "LONG | E " in macd_obv_pine
        assert "SL " in macd_obv_pine
        assert "TP " in macd_obv_pine
        assert "Qty " in macd_obv_pine

    def test_short_alert_content(self, macd_obv_pine):
        assert "SHORT | E " in macd_obv_pine

    def test_alert_frequency(self, macd_obv_pine):
        assert "alert.freq_once_per_bar_close" in macd_obv_pine


# ======================================================================
# Test: Background uses strategy.position_size
# ======================================================================


class TestBackground:
    """Background coloring uses strategy.position_size."""

    def test_uses_position_size(self, macd_obv_pine):
        assert "strategy.position_size > 0" in macd_obv_pine

    def test_no_pos_state_in_bg(self, macd_obv_pine):
        assert "posState ==" not in macd_obv_pine

    def test_bgcolor_present(self, macd_obv_pine):
        assert 'bgcolor(bgColor, title="Position Zone")' in macd_obv_pine

    def test_green_for_long(self, macd_obv_pine):
        assert "color.new(color.green, 90)" in macd_obv_pine

    def test_red_for_short(self, macd_obv_pine):
        assert "color.new(color.red, 90)" in macd_obv_pine


# ======================================================================
# Test: Visual level plots
# ======================================================================


class TestLevelPlots:
    """Entry Price, Take Profit, Stop Loss visual lines."""

    def test_entry_price_plot(self, macd_obv_pine):
        assert 'title="Entry Price"' in macd_obv_pine

    def test_take_profit_plot(self, macd_obv_pine):
        assert 'title="Take Profit"' in macd_obv_pine

    def test_stop_loss_plot(self, macd_obv_pine):
        assert 'title="Stop Loss"' in macd_obv_pine

    def test_plot_style_linebr(self, macd_obv_pine):
        assert "plot.style_linebr" in macd_obv_pine


# ======================================================================
# Test: Rich entry labels with direction icon and E/S/T prices
# ======================================================================


class TestEntryLabels:
    """Strategy mode draws rich entry labels after fill detection."""

    def test_long_label_icon(self, macd_obv_pine):
        assert "LONG" in macd_obv_pine
        assert "label.style_label_up" in macd_obv_pine

    def test_short_label_icon(self, macd_obv_pine):
        assert "SHORT" in macd_obv_pine
        assert "label.style_label_down" in macd_obv_pine

    def test_label_shows_fill_price(self, macd_obv_pine):
        assert "fillPrice" in macd_obv_pine
        assert "strategy.position_avg_price" in macd_obv_pine

    def test_label_shows_sl(self, macd_obv_pine):
        assert "pendingSL" in macd_obv_pine

    def test_label_shows_tp(self, macd_obv_pine):
        assert "pendingTP_" in macd_obv_pine

    def test_label_shows_strategy_name(self, macd_obv_pine):
        assert "SPY MACD+OBV (Tournament Winner)" in macd_obv_pine

    def test_label_uses_new_position_detection(self, macd_obv_pine):
        """Labels drawn on new position, not on signal."""
        assert "newLong" in macd_obv_pine
        assert "newShort" in macd_obv_pine
        assert "strategy.position_size > 0 and strategy.position_size[1] <= 0" in macd_obv_pine


# ======================================================================
# Test: Strategy definitions have correct model fields
# ======================================================================


class TestStrategyDefinitionFields:
    """Strategy factories set strategy_mode=True and correct defaults."""

    def test_macd_obv_strategy_mode(self):
        defn = spy_macd_obv_strategy()
        assert defn.strategy_mode is True

    def test_sma_cross_strategy_mode(self):
        defn = spy_sma_crossover_strategy()
        assert defn.strategy_mode is True

    def test_macd_opt_strategy_mode(self):
        defn = spy_macd_optimized_strategy()
        assert defn.strategy_mode is True

    def test_default_capital(self):
        defn = spy_macd_obv_strategy()
        assert defn.initial_capital == 100000.0

    def test_default_commission(self):
        defn = spy_macd_obv_strategy()
        assert defn.commission_per_order == 1.0

    def test_default_slippage(self):
        defn = spy_macd_obv_strategy()
        assert defn.slippage == 1

    def test_default_risk_pct(self):
        defn = spy_macd_obv_strategy()
        assert defn.risk_per_trade_pct == 1.0

    def test_default_sl_atr_mult(self):
        defn = spy_macd_obv_strategy()
        assert defn.sl_atr_mult == 1.5

    def test_default_sl_cap(self):
        defn = spy_macd_obv_strategy()
        assert defn.sl_cap_dollars == 2.0

    def test_default_rr_ratio(self):
        defn = spy_macd_obv_strategy()
        assert defn.rr_ratio == 1.5

    def test_default_be_trigger(self):
        defn = spy_macd_obv_strategy()
        assert defn.be_trigger == 0.5

    def test_default_max_trades(self):
        defn = spy_macd_obv_strategy()
        assert defn.max_trades_per_day == 4

    def test_default_circuit_breaker(self):
        defn = spy_macd_obv_strategy()
        assert defn.circuit_breaker_losses == 3

    def test_default_daily_loss_limit(self):
        defn = spy_macd_obv_strategy()
        assert defn.daily_loss_limit_pct == 3.0

    def test_close_eod_default(self):
        defn = spy_macd_obv_strategy()
        assert defn.close_eod is True

    def test_barstate_confirmed_default(self):
        defn = spy_macd_obv_strategy()
        assert defn.barstate_confirmed is True

    def test_sma_is_long_only(self):
        defn = spy_sma_crossover_strategy()
        assert defn.long_entry is not None
        assert defn.short_entry is None

    def test_macd_obv_has_both_entries(self):
        defn = spy_macd_obv_strategy()
        assert defn.long_entry is not None
        assert defn.short_entry is not None

    def test_no_extra_plots(self):
        """Strategy mode handles TP/SL natively — no extra_plots needed."""
        for factory in [
            spy_macd_obv_strategy,
            spy_sma_crossover_strategy,
            spy_macd_optimized_strategy,
        ]:
            defn = factory()
            assert len(defn.extra_plots) == 0, f"{defn.name}: unexpected extra_plots"


# ======================================================================
# Test: Catalog registration
# ======================================================================


class TestCatalogRegistration:
    """All SPY strategies registered in STRATEGY_CATALOG."""

    def test_macd_obv_in_catalog(self):
        assert "spy_macd_obv" in STRATEGY_CATALOG

    def test_sma_crossover_in_catalog(self):
        assert "spy_sma_crossover" in STRATEGY_CATALOG

    def test_macd_optimized_in_catalog(self):
        assert "spy_macd_optimized" in STRATEGY_CATALOG

    def test_catalog_count(self):
        assert len(STRATEGY_CATALOG) == 18

    def test_catalog_entries_generate_strategy(self, gen):
        for key in ["spy_macd_obv", "spy_sma_crossover", "spy_macd_optimized"]:
            factory = STRATEGY_CATALOG[key]
            defn = factory()
            pine = gen.generate(defn)
            assert pine, f"{key}: empty output"
            assert "strategy(" in pine, f"{key}: missing strategy()"
            assert len(pine) > 200, f"{key}: output too short"


# ======================================================================
# Test: Long-only SMA crossover specifics
# ======================================================================


class TestSmaLongOnly:
    """SMA crossover is long-only — no short-related code."""

    def test_no_short_sl(self, sma_cross_pine):
        assert "shortSL" not in sma_cross_pine

    def test_no_short_tp(self, sma_cross_pine):
        assert "shortTP" not in sma_cross_pine

    def test_no_go_short(self, sma_cross_pine):
        assert "goShort" not in sma_cross_pine

    def test_no_short_condition(self, sma_cross_pine):
        assert "shortCondition" not in sma_cross_pine

    def test_no_exit_short_cond(self, sma_cross_pine):
        assert "exitShortCond" not in sma_cross_pine

    def test_has_long_sl(self, sma_cross_pine):
        assert "longSL" in sma_cross_pine

    def test_has_long_tp(self, sma_cross_pine):
        assert "longTP" in sma_cross_pine

    def test_has_exit_long(self, sma_cross_pine):
        assert "exitLongCond" in sma_cross_pine

    def test_no_short_be_trigger(self, sma_cross_pine):
        assert "BE SHORT" not in sma_cross_pine

    def test_has_long_be_trigger(self, sma_cross_pine):
        assert "BE LONG" in sma_cross_pine

    def test_bg_no_red_zone(self, sma_cross_pine):
        """Long-only: no red background zone for shorts in BACKGROUND."""
        # newShort detection exists but short background coloring should not
        assert "else if strategy.position_size < 0" not in sma_cross_pine


# ======================================================================
# Test: Output structure
# ======================================================================


class TestOutputStructure:
    """Verify output structure matches reference strategy format."""

    def test_section_order(self, macd_obv_pine):
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
            pos = macd_obv_pine.find(section)
            assert pos > last_pos, (
                f"Section '{section}' not found after previous section "
                f"(pos={pos}, last={last_pos})"
            )
            last_pos = pos

    def test_script_starts_with_license(self, macd_obv_pine):
        assert macd_obv_pine.startswith(
            "// This Pine Script(tm) v6 strategy"
        )

    def test_reasonable_line_count(self, macd_obv_pine, sma_cross_pine, macd_opt_pine):
        for pine in [macd_obv_pine, sma_cross_pine, macd_opt_pine]:
            lines = pine.strip().splitlines()
            assert len(lines) > 150, f"Too few lines: {len(lines)}"
            assert len(lines) < 400, f"Too many lines: {len(lines)}"
