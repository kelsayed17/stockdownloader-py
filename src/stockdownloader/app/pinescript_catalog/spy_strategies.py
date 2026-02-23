"""Enhanced SPY PineScript strategies with TP/SL/Entry price overlays.

Generates production-ready TradingView scripts for the top 3
walk-forward validated strategies from the Grand Tournament:

1. **MACD+OBV** — OOS +74.03, degradation 0.96 (best robustness)
2. **SMA Crossover (20/21)** — OOS +65.29, degradation 2.52
3. **MACD (8/35/5)** — OOS +62.80, degradation 1.57

Two variants are provided for each strategy:

**Indicator mode** (v1):
    ``spy_macd_obv_strategy()``, ``spy_sma_crossover_strategy()``,
    ``spy_macd_optimized_strategy()``
    — Manual ``posState`` tracking with visual TP/SL overlays.

**Strategy mode** (v2):
    ``spy_macd_obv_strategy_v2()``, ``spy_sma_crossover_strategy_v2()``,
    ``spy_macd_optimized_strategy_v2()``
    — Full ``strategy()`` backtesting with ``strategy.entry()``,
    ``strategy.exit()``, ATR-based position sizing, breakeven management,
    circuit breaker, and EOD close.

Usage::

    from stockdownloader.app.pinescript_catalog.spy_strategies import (
        spy_macd_obv_strategy,
        spy_macd_obv_strategy_v2,
    )
    from stockdownloader.util.pinescript import PineScriptGenerator

    gen = PineScriptGenerator()
    # Indicator mode
    pine_v1 = gen.generate(spy_macd_obv_strategy())
    # Strategy mode
    pine_v2 = gen.generate(spy_macd_obv_strategy_v2())
"""

from __future__ import annotations

from stockdownloader.util.pinescript.models import (
    Condition,
    Indicator,
    Input,
    StrategyDefinition,
)


# ======================================================================
# Shared TP/SL infrastructure
# ======================================================================


def _tp_sl_inputs(group: str = "Risk Management") -> list[Input]:
    """Shared inputs for ATR-based TP/SL."""
    return [
        Input.int_("atrLen", 14, "ATR Length", min_val=1, group=group),
        Input.float_("slMult", 1.5, "Stop Loss (x ATR)",
                      min_val=0.5, step=0.25, group=group),
        Input.float_("rrRatio", 1.5, "Reward:Risk Ratio",
                      min_val=0.5, step=0.25, group=group),
        Input.float_("slCapDollars", 2.0, "SL Cap ($, 0=off)",
                      min_val=0.0, step=0.25, group=group),
    ]


def _tp_sl_indicators() -> list[Indicator]:
    """ATR indicator for TP/SL computation."""
    return [
        Indicator.atr("atrVal", "atrLen"),
    ]


def _tp_sl_extra_plots(name: str, *, has_short: bool = True) -> list[str]:
    """All TP/SL code for extra_plots (rendered AFTER signal logic).

    This includes:
    1. Entry price & TP/SL level tracking (uses buySignal, posState)
    2. TP/SL hit detection
    3. Auto-exit on TP/SL hit (resets posState)
    4. Visual plots (entry, TP, SL lines)
    5. Hit markers (plotshape)
    6. Alert conditions for TP/SL events
    7. Data window metrics

    Parameters
    ----------
    name:
        Strategy display name for alert messages.
    has_short:
        Whether the strategy supports short entries.
    """
    lines: list[str] = []

    # --- Entry price & TP/SL level tracking ---
    lines.extend([
        "// ═══════════════════════════════════════════════════════════════",
        "// ENTRY PRICE & TP/SL LEVEL TRACKING",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "var float entryPrice  = na",
        "var float stopPrice   = na",
        "var float targetPrice = na",
        "var float tradeRisk   = na",
        "",
        "if buySignal",
        "    entryPrice  := close",
        "    // SL = ATR-based, capped if slCapDollars > 0",
        "    float rawSL = atrVal * slMult",
        "    tradeRisk   := slCapDollars > 0"
        " ? math.min(rawSL, slCapDollars) : rawSL",
        "    stopPrice   := close - tradeRisk",
        "    targetPrice := close + tradeRisk * rrRatio",
    ])

    if has_short:
        lines.extend([
            "",
            "if sellSignal",
            "    entryPrice  := close",
            "    float rawSL = atrVal * slMult",
            "    tradeRisk   := slCapDollars > 0"
            " ? math.min(rawSL, slCapDollars) : rawSL",
            "    stopPrice   := close + tradeRisk",
            "    targetPrice := close - tradeRisk * rrRatio",
        ])

    # Clear on strategy exit
    exit_cond = "exitLongSignal"
    if has_short:
        exit_cond += " or exitShortSignal"
    lines.extend([
        "",
        "// Clear levels on strategy exit",
        f"if {exit_cond}",
        "    entryPrice  := na",
        "    stopPrice   := na",
        "    targetPrice := na",
        "    tradeRisk   := na",
    ])

    # --- TP/SL hit detection ---
    lines.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// TP/SL HIT DETECTION",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "bool longTPHit  = posState == 1 and not na(targetPrice)"
        " and high >= targetPrice",
        "bool longSLHit  = posState == 1 and not na(stopPrice)"
        " and low <= stopPrice",
    ])

    if has_short:
        lines.extend([
            "bool shortTPHit = posState == -1 and not na(targetPrice)"
            " and low <= targetPrice",
            "bool shortSLHit = posState == -1 and not na(stopPrice)"
            " and high >= stopPrice",
        ])
    else:
        lines.extend([
            "bool shortTPHit = false",
            "bool shortSLHit = false",
        ])

    # --- Auto-exit on TP/SL ---
    lines.extend([
        "",
        "// Auto-exit on TP or SL hit",
        "if longTPHit or longSLHit or shortTPHit or shortSLHit",
        "    posState    := 0",
        "    entryPrice  := na",
        "    stopPrice   := na",
        "    targetPrice := na",
        "    tradeRisk   := na",
    ])

    # --- Visual plots ---
    lines.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// ENTRY PRICE LINE",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "plot(not na(entryPrice) ? entryPrice : na,"
        ' title="Entry Price", color=color.white,'
        " style=plot.style_linebr, linewidth=2)",
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// TAKE PROFIT LINE (green)",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "plot(not na(targetPrice) ? targetPrice : na,"
        ' title="Take Profit", color=color.new(color.lime, 20),'
        " style=plot.style_linebr, linewidth=2)",
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// STOP LOSS LINE (red)",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "plot(not na(stopPrice) ? stopPrice : na,"
        ' title="Stop Loss", color=color.new(color.red, 20),'
        " style=plot.style_linebr, linewidth=2)",
    ])

    # --- TP/SL hit markers ---
    lines.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// TP/SL HIT MARKERS",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        'plotshape(longTPHit, title="Long TP Hit",'
        " style=shape.flag, location=location.abovebar,"
        " color=color.lime, size=size.small,"
        ' text="TP")',
        'plotshape(longSLHit, title="Long SL Hit",'
        " style=shape.xcross, location=location.belowbar,"
        " color=color.red, size=size.small,"
        ' text="SL")',
    ])

    if has_short:
        lines.extend([
            'plotshape(shortTPHit, title="Short TP Hit",'
            " style=shape.flag, location=location.belowbar,"
            " color=color.lime, size=size.small,"
            ' text="TP")',
            'plotshape(shortSLHit, title="Short SL Hit",'
            " style=shape.xcross, location=location.abovebar,"
            " color=color.red, size=size.small,"
            ' text="SL")',
        ])

    # --- TP/SL alert conditions ---
    lines.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// TP/SL ALERT CONDITIONS",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        f'alertcondition(longTPHit,\n'
        f'     title="Long TP Hit",\n'
        f'     message="{name}: Long Take Profit hit!")',
        f'alertcondition(longSLHit,\n'
        f'     title="Long SL Hit",\n'
        f'     message="{name}: Long Stop Loss hit!")',
    ])

    if has_short:
        lines.extend([
            f'alertcondition(shortTPHit,\n'
            f'     title="Short TP Hit",\n'
            f'     message="{name}: Short Take Profit hit!")',
            f'alertcondition(shortSLHit,\n'
            f'     title="Short SL Hit",\n'
            f'     message="{name}: Short Stop Loss hit!")',
        ])

    lines.extend([
        f'alertcondition(longTPHit or longSLHit or shortTPHit or shortSLHit,\n'
        f'     title="Any TP/SL Hit",\n'
        f'     message="{name}: TP or SL level hit. Check chart.")',
    ])

    # --- Data window metrics ---
    lines.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// DATA WINDOW METRICS",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        'plot(atrVal, title="ATR",'
        " color=color.new(color.gray, 100), display=display.data_window)",
        'plot(not na(tradeRisk) ? tradeRisk : na, title="Trade Risk ($)",'
        " color=color.new(color.gray, 100), display=display.data_window)",
        'plot(not na(entryPrice) and not na(targetPrice)'
        " ? math.abs(targetPrice - entryPrice) : na,"
        ' title="TP Distance ($)",'
        " color=color.new(color.gray, 100), display=display.data_window)",
        'plot(not na(entryPrice) and not na(stopPrice)'
        " ? math.abs(entryPrice - stopPrice) : na,"
        ' title="SL Distance ($)",'
        " color=color.new(color.gray, 100), display=display.data_window)",
    ])

    return lines


# ======================================================================
# Strategy 1: MACD + OBV (Grand Tournament Winner)
# OOS +74.03, Degradation 0.96
# ======================================================================


def _macd_obv_core_inputs() -> list[Input]:
    """MACD+OBV inputs shared by v1 (indicator) and v2 (strategy)."""
    return [
        Input.int_("macdFast", 12, "MACD Fast Length",
                    min_val=2, group="MACD"),
        Input.int_("macdSlow", 26, "MACD Slow Length",
                    min_val=5, group="MACD"),
        Input.int_("macdSignal", 9, "MACD Signal Length",
                    min_val=2, group="MACD"),
        Input.int_("obvSmoothing", 5, "OBV Smoothing Period",
                    min_val=2, group="OBV"),
    ]


def _macd_obv_core_indicators() -> list[Indicator]:
    """MACD+OBV indicators shared by v1 and v2."""
    return [
        *Indicator.macd("macdFast", "macdSlow", "macdSignal"),
        Indicator.obv("obvRaw"),
        Indicator.raw("obvSmoothed", "ta.ema(obvRaw, obvSmoothing)"),
        Indicator.raw("obvRising", "obvSmoothed > obvSmoothed[1]"),
        Indicator.raw("obvFalling", "obvSmoothed < obvSmoothed[1]"),
    ]


_MACD_OBV_LONG = Condition(
    "ta.crossover(macdLine, macdSignal) and obvRising",
    "MACD bullish crossover + rising OBV confirms buying pressure",
)
_MACD_OBV_SHORT = Condition(
    "ta.crossunder(macdLine, macdSignal) and obvFalling",
    "MACD bearish crossunder + falling OBV confirms selling pressure",
)
_MACD_OBV_EXIT_LONG = Condition("ta.crossunder(macdLine, macdSignal)")
_MACD_OBV_EXIT_SHORT = Condition("ta.crossover(macdLine, macdSignal)")

_MACD_HIST_PLOT = [
    "",
    "// MACD histogram in data window",
    'plot(macdHist, title="MACD Histogram",'
    " color=color.new(color.gray, 100), display=display.data_window)",
]


def spy_macd_obv_strategy() -> StrategyDefinition:
    """MACD + OBV signal stack — Grand Tournament winner (indicator mode).

    Walk-forward validated: OOS score +74.03, degradation 0.96
    (near-perfect IS/OOS consistency). Includes ATR-based TP/SL overlay.
    """
    name = "SPY MACD+OBV (Tournament Winner)"
    return StrategyDefinition(
        name=name,
        short_name="SPY-MACD-OBV",
        description=(
            "Grand Tournament Winner — Walk-forward validated.\n"
            "OOS Score: +74.03 | Degradation: 0.96 (near-perfect)\n"
            "\n"
            "MACD crossover confirmed by OBV trend alignment.\n"
            "ATR-based TP/SL with configurable R:R ratio.\n"
            "SL capped at $2.00 default (institutional risk control)."
        ),
        inputs=[*_macd_obv_core_inputs(), *_tp_sl_inputs()],
        indicators=[*_macd_obv_core_indicators(), *_tp_sl_indicators()],
        long_entry=_MACD_OBV_LONG,
        short_entry=_MACD_OBV_SHORT,
        long_exit=_MACD_OBV_EXIT_LONG,
        short_exit=_MACD_OBV_EXIT_SHORT,
        exit_on_reverse=False,
        extra_plots=[*_tp_sl_extra_plots(name, has_short=True), *_MACD_HIST_PLOT],
        long_label="Buy",
        short_label="Sell",
    )


def spy_macd_obv_strategy_v2() -> StrategyDefinition:
    """MACD + OBV — Strategy mode with full backtesting.

    Same entry/exit logic as :func:`spy_macd_obv_strategy` but uses
    ``strategy()`` with ATR-based position sizing, breakeven management,
    circuit breaker, and EOD close.
    """
    return StrategyDefinition(
        name="SPY MACD+OBV (Tournament Winner)",
        short_name="SPY-MACD-OBV",
        strategy_mode=True,
        description=(
            "Grand Tournament Winner — Walk-forward validated.\n"
            "OOS Score: +74.03 | Degradation: 0.96 (near-perfect)\n"
            "\n"
            "MACD crossover confirmed by OBV trend alignment.\n"
            "Strategy mode: ATR position sizing, breakeven, circuit breaker."
        ),
        inputs=_macd_obv_core_inputs(),
        indicators=[*_macd_obv_core_indicators(), Indicator.atr("atrVal", "14")],
        long_entry=_MACD_OBV_LONG,
        short_entry=_MACD_OBV_SHORT,
        long_exit=_MACD_OBV_EXIT_LONG,
        short_exit=_MACD_OBV_EXIT_SHORT,
        exit_on_reverse=False,
        long_label="Buy",
        short_label="Sell",
    )


# ======================================================================
# Strategy 2: SMA Crossover (20/21) — #2 Ranked
# OOS +65.29, Degradation 2.52 (GOOD — improves OOS)
# ======================================================================


def _sma_core_inputs() -> list[Input]:
    """SMA Crossover inputs shared by v1 and v2."""
    return [
        Input.int_("smaShort", 20, "SMA Short Period",
                    min_val=2, group="SMA"),
        Input.int_("smaLong", 21, "SMA Long Period",
                    min_val=3, group="SMA"),
    ]


def _sma_core_indicators() -> list[Indicator]:
    """SMA Crossover indicators shared by v1 and v2."""
    return [
        Indicator.sma("smaFast", "close", "smaShort",
                       plot=True, color="color.teal"),
        Indicator.sma("smaSlow", "close", "smaLong",
                       plot=True, color="color.orange"),
    ]


_SMA_LONG = Condition(
    "ta.crossover(smaFast, smaSlow)",
    "SMA 20 crosses above SMA 21 — golden cross",
)
_SMA_EXIT_LONG = Condition("ta.crossunder(smaFast, smaSlow)")


def spy_sma_crossover_strategy() -> StrategyDefinition:
    """SMA Crossover (20/21) — Grand Tournament #2 (indicator mode).

    Walk-forward validated: OOS score +65.29, degradation 2.52
    (GOOD). Tight golden cross. Long-only. ATR-based TP/SL overlay.
    """
    name = "SPY SMA Cross 20/21 (Tournament #2)"
    return StrategyDefinition(
        name=name,
        short_name="SPY-SMA-2021",
        description=(
            "Grand Tournament #2 — Walk-forward validated.\n"
            "OOS Score: +65.29 | Degradation: 2.52 (GOOD)\n"
            "\n"
            "Tight SMA 20/21 golden cross — fast response.\n"
            "Long-only. ATR-based TP/SL overlay."
        ),
        inputs=[*_sma_core_inputs(), *_tp_sl_inputs()],
        indicators=[*_sma_core_indicators(), *_tp_sl_indicators()],
        long_entry=_SMA_LONG,
        short_entry=None,
        long_exit=_SMA_EXIT_LONG,
        exit_on_reverse=False,
        extra_plots=_tp_sl_extra_plots(name, has_short=False),
        long_label="Buy",
        short_label="Sell",
    )


def spy_sma_crossover_strategy_v2() -> StrategyDefinition:
    """SMA Crossover (20/21) — Strategy mode with full backtesting.

    Same entry/exit logic as :func:`spy_sma_crossover_strategy` but uses
    ``strategy()`` with ATR-based position sizing, breakeven management,
    circuit breaker, and EOD close.  Long-only (no shorts).
    """
    return StrategyDefinition(
        name="SPY SMA Cross 20/21 (Tournament #2)",
        short_name="SPY-SMA-2021",
        strategy_mode=True,
        description=(
            "Grand Tournament #2 — Walk-forward validated.\n"
            "OOS Score: +65.29 | Degradation: 2.52 (GOOD)\n"
            "\n"
            "Tight SMA 20/21 golden cross — fast response.\n"
            "Long-only. Strategy mode: ATR sizing, breakeven, circuit breaker."
        ),
        inputs=_sma_core_inputs(),
        indicators=[*_sma_core_indicators(), Indicator.atr("atrVal", "14")],
        long_entry=_SMA_LONG,
        short_entry=None,
        long_exit=_SMA_EXIT_LONG,
        exit_on_reverse=False,
        long_label="Buy",
        short_label="Sell",
    )


# ======================================================================
# Strategy 3: MACD Optimized (8/35/5) — #3 Ranked
# OOS +62.80, Degradation 1.57 (GOOD)
# ======================================================================


def _macd_opt_core_inputs() -> list[Input]:
    """MACD Optimized 8/35/5 inputs shared by v1 and v2."""
    return [
        Input.int_("macdFast", 8, "MACD Fast Length",
                    min_val=2, group="MACD"),
        Input.int_("macdSlow", 35, "MACD Slow Length",
                    min_val=5, group="MACD"),
        Input.int_("macdSignal", 5, "MACD Signal Length",
                    min_val=2, group="MACD"),
    ]


_MACD_OPT_LONG = Condition(
    "ta.crossover(macdLine, macdSignal)",
    "MACD 8/35/5 bullish crossover",
)
_MACD_OPT_SHORT = Condition(
    "ta.crossunder(macdLine, macdSignal)",
    "MACD 8/35/5 bearish crossunder",
)
_MACD_OPT_EXIT_LONG = Condition("ta.crossunder(macdLine, macdSignal)")
_MACD_OPT_EXIT_SHORT = Condition("ta.crossover(macdLine, macdSignal)")


def spy_macd_optimized_strategy() -> StrategyDefinition:
    """MACD Optimized (8/35/5) — Grand Tournament #3 (indicator mode).

    Walk-forward validated: OOS score +62.80, degradation 1.57
    (GOOD). Wide slow period (35) filters noise. ATR-based TP/SL overlay.
    """
    name = "SPY MACD Optimized 8/35/5 (Tournament #3)"
    return StrategyDefinition(
        name=name,
        short_name="SPY-MACD-OPT",
        description=(
            "Grand Tournament #3 — Walk-forward validated.\n"
            "OOS Score: +62.80 | Degradation: 1.57 (GOOD)\n"
            "\n"
            "MACD 8/35/5 — wide slow period filters noise,\n"
            "fast signal provides quick entry timing.\n"
            "ATR-based TP/SL overlay."
        ),
        inputs=[*_macd_opt_core_inputs(), *_tp_sl_inputs()],
        indicators=[
            *Indicator.macd("macdFast", "macdSlow", "macdSignal"),
            *_tp_sl_indicators(),
        ],
        long_entry=_MACD_OPT_LONG,
        short_entry=_MACD_OPT_SHORT,
        long_exit=_MACD_OPT_EXIT_LONG,
        short_exit=_MACD_OPT_EXIT_SHORT,
        exit_on_reverse=False,
        extra_plots=[*_tp_sl_extra_plots(name, has_short=True), *_MACD_HIST_PLOT],
        long_label="Buy",
        short_label="Sell",
    )


def spy_macd_optimized_strategy_v2() -> StrategyDefinition:
    """MACD Optimized (8/35/5) — Strategy mode with full backtesting.

    Same entry/exit logic as :func:`spy_macd_optimized_strategy` but uses
    ``strategy()`` with ATR-based position sizing, breakeven management,
    circuit breaker, and EOD close.
    """
    return StrategyDefinition(
        name="SPY MACD Optimized 8/35/5 (Tournament #3)",
        short_name="SPY-MACD-OPT",
        strategy_mode=True,
        description=(
            "Grand Tournament #3 — Walk-forward validated.\n"
            "OOS Score: +62.80 | Degradation: 1.57 (GOOD)\n"
            "\n"
            "MACD 8/35/5 — wide slow period filters noise,\n"
            "fast signal provides quick entry timing.\n"
            "Strategy mode: ATR sizing, breakeven, circuit breaker."
        ),
        inputs=_macd_opt_core_inputs(),
        indicators=[
            *Indicator.macd("macdFast", "macdSlow", "macdSignal"),
            Indicator.atr("atrVal", "14"),
        ],
        long_entry=_MACD_OPT_LONG,
        short_entry=_MACD_OPT_SHORT,
        long_exit=_MACD_OPT_EXIT_LONG,
        short_exit=_MACD_OPT_EXIT_SHORT,
        exit_on_reverse=False,
        long_label="Buy",
        short_label="Sell",
    )
