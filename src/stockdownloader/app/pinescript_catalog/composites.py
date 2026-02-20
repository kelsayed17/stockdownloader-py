"""Pre-built composite (multi-mode) strategy definitions for Pine Script.

Contains composite strategy factories that combine multiple toggleable
modes into single indicators, plus the ``COMPOSITE_STRATEGY_CATALOG``.
"""

from __future__ import annotations

from stockdownloader.util.pinescript_models import (
    CompositeStrategyDefinition,
    Condition,
    Indicator,
    Input,
    ModeDefinition,
)
from stockdownloader.util.pinescript import strategy_to_mode
from stockdownloader.util.pinescript_modes import (
    orb_mode as _orb_mode,
    orr_mode as _orr_mode,
    pb_mode as _pb_mode,
    ps_mode as _ps_mode,
    rev_mode as _rev_mode,
    vwap_shared_code as _vwap_shared_code,
    vwap_shared_indicators as _vwap_shared_indicators,
    vwap_shared_inputs as _vwap_shared_inputs,
)
from stockdownloader.app.pinescript_catalog.strategies import (
    macd_strategy,
    rsi_strategy,
    sma_crossover_strategy,
)


# ======================================================================
# Composite strategy factories
# ======================================================================


def vwap_composite_strategy() -> CompositeStrategyDefinition:
    """All 5 VWAP modes as a single toggleable indicator."""
    return CompositeStrategyDefinition(
        name="VWAP Composite Strategy",
        short_name="VWAP-ALL",
        description=(
            "Comprehensive VWAP intraday strategy with 5 entry modes.\n"
            "Each mode can be independently enabled/disabled.\n"
            "Modes: PB (Pullback), REV (Reversal), ORB (OR Breakout),\n"
            "ORR (OR Reversal), PS (Pattern Scalp).\n"
            "Default aggregation: first-to-fire with priority ordering."
        ),
        shared_inputs=_vwap_shared_inputs(),
        shared_indicators=_vwap_shared_indicators(),
        shared_code=_vwap_shared_code(),
        modes=[
            _ps_mode(),
            _orb_mode(),
            _orr_mode(),
            _pb_mode(),
            _rev_mode(),
        ],
        aggregation="first_to_fire",
        priority_order=["ps", "orb", "orr", "pb", "rev"],
        use_session_filter=True,
    )


def signal_stack_composite_strategy() -> CompositeStrategyDefinition:
    """Top signal generators as toggleable modes."""
    macd_mode = strategy_to_mode(
        macd_strategy(), "macd", "MACD Crossover",
        label_color_long="color.blue",
        label_color_short="color.blue",
    )
    rsi_mode = strategy_to_mode(
        rsi_strategy(), "rsi", "RSI Extremes",
        label_color_long="color.purple",
        label_color_short="color.purple",
    )
    sma_mode = strategy_to_mode(
        sma_crossover_strategy(), "sma", "SMA Crossover",
        label_color_long="color.orange",
        label_color_short="color.orange",
    )
    # OBV as a standalone mode
    obv_mode = ModeDefinition(
        name="OBV Trend",
        short_name="obv",
        group="OBV Trend",
        enabled_default=True,
        inputs=[
            Input.int_("obvSmooth", 5, "OBV Smoothing"),
        ],
        indicators=[
            Indicator.obv("obvRaw"),
            Indicator.raw("obvSmoothed", "ta.ema(obvRaw, obvSmooth)"),
            Indicator.raw("obvRising", "obvSmoothed > obvSmoothed[1]"),
            Indicator.raw("obvFalling", "obvSmoothed < obvSmoothed[1]"),
            Indicator.raw("obvCrossUp",
                          "obvRising and not obvRising[1]"),
            Indicator.raw("obvCrossDown",
                          "obvFalling and not obvFalling[1]"),
        ],
        long_entry=Condition(
            "obvCrossUp",
            "OBV smoothed turns bullish",
        ),
        short_entry=Condition(
            "obvCrossDown",
            "OBV smoothed turns bearish",
        ),
        label_color_long="color.teal",
        label_color_short="color.teal",
    )
    # ADX trend mode
    adx_mode = ModeDefinition(
        name="ADX Trend",
        short_name="adx",
        group="ADX Trend",
        enabled_default=True,
        inputs=[
            Input.int_("adxPeriod", 14, "ADX Period"),
            Input.float_("adxThreshold", 25.0, "ADX Threshold",
                          step=1.0),
        ],
        indicators=[
            Indicator.dmi("adxPeriod", "adxPeriod",
                          var_plus="adxPlusDI", var_minus="adxMinusDI",
                          var_adx="adxVal"),
        ],
        long_entry=Condition(
            "adxVal > adxThreshold "
            "and ta.crossover(adxPlusDI, adxMinusDI)",
            "ADX strong trend with +DI crossing above -DI",
        ),
        short_entry=Condition(
            "adxVal > adxThreshold "
            "and ta.crossover(adxMinusDI, adxPlusDI)",
            "ADX strong trend with -DI crossing above +DI",
        ),
        label_color_long="color.yellow",
        label_color_short="color.yellow",
    )
    return CompositeStrategyDefinition(
        name="Signal Stack Composite",
        short_name="SIG-STACK",
        description=(
            "Top signal generators as independently toggleable modes.\n"
            "Combines MACD crossover, RSI extremes, SMA crossover,\n"
            "OBV trend, and ADX trend into one indicator.\n"
            "Default aggregation: any mode that fires triggers a signal."
        ),
        modes=[macd_mode, rsi_mode, sma_mode, obv_mode, adx_mode],
        aggregation="any",
        use_session_filter=False,
    )


def full_composite_strategy() -> CompositeStrategyDefinition:
    """All top-performing strategies as toggleable modes.

    Each mode uses uniquely prefixed variable names to avoid collisions.
    """
    dmi_vwap_mode = ModeDefinition(
        name="DMI + VWAP",
        short_name="dmivwap",
        group="DMI + VWAP",
        enabled_default=True,
        description="DMI crossover filtered by session VWAP.",
        inputs=[
            Input.int_("dvDmiPeriod", 14, "DMI Period"),
            Input.float_("dvAdxThresh", 25.0, "ADX Threshold", step=0.5),
        ],
        indicators=[
            Indicator.session_vwap("dvVwap"),
            Indicator.dmi("dvDmiPeriod", "dvDmiPeriod",
                          var_plus="dvPlusDI", var_minus="dvMinusDI",
                          var_adx="dvAdx"),
        ],
        long_entry=Condition(
            "close > dvVwap and dvPlusDI > dvMinusDI "
            "and dvAdx > dvAdxThresh",
        ),
        short_entry=Condition(
            "close < dvVwap and dvMinusDI > dvPlusDI "
            "and dvAdx > dvAdxThresh",
        ),
        label_color_long="color.blue",
        label_color_short="color.blue",
    )
    macd_obv_mode = ModeDefinition(
        name="MACD + OBV",
        short_name="macdobv",
        group="MACD + OBV",
        enabled_default=True,
        description="MACD crossover confirmed by OBV trend.",
        inputs=[
            Input.int_("moFast", 12, "MACD Fast"),
            Input.int_("moSlow", 26, "MACD Slow"),
            Input.int_("moSignal", 9, "Signal Length"),
            Input.int_("moObvSmooth", 5, "OBV Smoothing"),
        ],
        indicators=[
            *Indicator.macd("moFast", "moSlow", "moSignal",
                            var_line="moMacdLine",
                            var_signal="moMacdSignal",
                            var_hist="moMacdHist"),
            Indicator.obv("moObvRaw"),
            Indicator.raw("moObvSmoothed",
                          "ta.ema(moObvRaw, moObvSmooth)"),
            Indicator.raw("moObvRising",
                          "moObvSmoothed > moObvSmoothed[1]"),
            Indicator.raw("moObvFalling",
                          "moObvSmoothed < moObvSmoothed[1]"),
        ],
        long_entry=Condition(
            "ta.crossover(moMacdLine, moMacdSignal) and moObvRising",
        ),
        short_entry=Condition(
            "ta.crossunder(moMacdLine, moMacdSignal) and moObvFalling",
        ),
        label_color_long="color.teal",
        label_color_short="color.teal",
    )
    mom_mode = ModeDefinition(
        name="Momentum Confluence",
        short_name="mom",
        group="Momentum Confluence",
        enabled_default=True,
        description="MACD + ADX + EMA trend + OBV confluence.",
        inputs=[
            Input.int_("mcFast", 12, "Fast EMA"),
            Input.int_("mcSlow", 26, "Slow EMA"),
            Input.int_("mcSignal", 9, "Signal Period"),
            Input.int_("mcTrendEma", 200, "Trend EMA Period"),
            Input.float_("mcAdxThresh", 25.0, "ADX Threshold", step=1.0),
        ],
        indicators=[
            *Indicator.macd("mcFast", "mcSlow", "mcSignal",
                            var_line="mcMacdLine",
                            var_signal="mcMacdSignal",
                            var_hist="mcMacdHist"),
            Indicator.ema("mcEmaTrend", "close", "mcTrendEma",
                          plot=True, color="color.gray"),
            Indicator.dmi("14", "14",
                          var_plus="mcPlusDI", var_minus="mcMinusDI",
                          var_adx="mcAdx"),
            Indicator.obv("mcObvRaw"),
            Indicator.raw("mcObvEma", "ta.ema(mcObvRaw, 5)"),
            Indicator.raw("mcObvUp", "mcObvEma > mcObvEma[1]"),
        ],
        long_entry=Condition(
            "ta.crossover(mcMacdLine, mcMacdSignal) "
            "and mcAdx > mcAdxThresh "
            "and mcPlusDI > mcMinusDI "
            "and close > mcEmaTrend and mcObvUp",
        ),
        short_entry=Condition(
            "ta.crossunder(mcMacdLine, mcMacdSignal) "
            "and mcAdx > mcAdxThresh "
            "and mcMinusDI > mcPlusDI and close < mcEmaTrend",
        ),
        label_color_long="color.lime",
        label_color_short="color.lime",
    )
    bb_rsi_mode = ModeDefinition(
        name="BB + RSI",
        short_name="bbrsi",
        group="BB + RSI",
        enabled_default=True,
        description="BB touch + RSI extremes in ranging market.",
        inputs=[
            Input.int_("brBbPeriod", 20, "BB Period"),
            Input.float_("brBbMult", 2.0, "BB Std Dev",
                          min_val=0.5, step=0.1),
            Input.int_("brRsiPeriod", 14, "RSI Period"),
            Input.float_("brRsiOS", 30.0, "RSI Oversold", step=1.0),
            Input.float_("brRsiOB", 70.0, "RSI Overbought", step=1.0),
            Input.float_("brAdxMax", 25.0, "ADX Max (ranging)", step=1.0),
        ],
        indicators=[
            Indicator.bbands("brMid", "brUpper", "brLower",
                             "close", "brBbPeriod", "brBbMult"),
            Indicator.rsi("brRsi", "close", "brRsiPeriod"),
            Indicator.dmi("14", "14",
                          var_plus="brPlusDI", var_minus="brMinusDI",
                          var_adx="brAdx"),
        ],
        long_entry=Condition(
            "(close <= brLower or ta.crossover(brRsi, brRsiOS)) "
            "and brAdx < brAdxMax",
        ),
        short_entry=Condition(
            "(close >= brUpper or ta.crossunder(brRsi, brRsiOB)) "
            "and brAdx < brAdxMax",
        ),
        label_color_long="color.purple",
        label_color_short="color.purple",
    )
    rsi_mode = ModeDefinition(
        name="RSI Strategy",
        short_name="rsi",
        group="RSI Strategy",
        enabled_default=True,
        description="RSI oversold/overbought crossover.",
        inputs=[
            Input.int_("rsRsiPeriod", 14, "RSI Period"),
            Input.float_("rsOversold", 30.0, "Oversold", step=1.0),
            Input.float_("rsOverbought", 70.0, "Overbought", step=1.0),
        ],
        indicators=[
            Indicator.rsi("rsRsi", "close", "rsRsiPeriod"),
        ],
        long_entry=Condition("ta.crossover(rsRsi, rsOversold)"),
        short_entry=Condition("ta.crossunder(rsRsi, rsOverbought)"),
        label_color_long="color.orange",
        label_color_short="color.orange",
    )
    sma_mode = ModeDefinition(
        name="SMA Crossover",
        short_name="sma",
        group="SMA Crossover",
        enabled_default=True,
        description="Golden/Death Cross SMA crossover.",
        inputs=[
            Input.int_("smFast", 10, "Fast SMA Period"),
            Input.int_("smSlow", 20, "Slow SMA Period"),
        ],
        indicators=[
            Indicator.sma("smFastSma", "close", "smFast",
                          plot=True, color="color.blue"),
            Indicator.sma("smSlowSma", "close", "smSlow",
                          plot=True, color="color.red"),
        ],
        long_entry=Condition("ta.crossover(smFastSma, smSlowSma)"),
        short_entry=None,
        label_color_long="color.yellow",
        label_color_short="color.yellow",
    )

    return CompositeStrategyDefinition(
        name="Full Strategy Composite",
        short_name="FULL-COMP",
        description=(
            "All top-performing strategies as independently toggleable modes.\n"
            "Walk-forward validated winners plus optimizer tournament leaders.\n"
            "DMI+VWAP, MACD+OBV, Momentum Confluence, BB+RSI, RSI, SMA."
        ),
        modes=[
            dmi_vwap_mode,
            macd_obv_mode,
            mom_mode,
            bb_rsi_mode,
            rsi_mode,
            sma_mode,
        ],
        aggregation="first_to_fire",
        priority_order=[
            "macdobv", "dmivwap", "mom", "bbrsi", "rsi", "sma",
        ],
        use_session_filter=False,
    )


# All composite strategies
COMPOSITE_STRATEGY_CATALOG: dict[str, callable] = {
    "vwap_composite": vwap_composite_strategy,
    "signal_stack": signal_stack_composite_strategy,
    "full_composite": full_composite_strategy,
}
