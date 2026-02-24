"""SPY ML Ensemble PineScript strategy with ML-driven dynamic exits.

Takes a trained :class:`~stockdownloader.ml.deep_surrogate.DeepSurrogateExporter`
and produces a :class:`StrategyDefinition` for the
:class:`~stockdownloader.pinescript.generator.PineScriptGenerator`.

The strategy uses a decision-tree surrogate of the full ensemble model to
compute an ML probability score on each bar.  Dynamic TP/SL levels are
scaled by ML confidence (distance from 0.5), and optional 15-minute VWAP
confirmation reduces false signals.

Usage::

    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter
    from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
        spy_ml_ensemble_strategy,
    )
    from stockdownloader.pinescript import PineScriptGenerator

    exporter = DeepSurrogateExporter(max_depth=10)
    exporter.train_surrogate(dataset, importances, ensemble_probs=probs)

    strategy = spy_ml_ensemble_strategy(exporter)
    pine = PineScriptGenerator().generate(strategy)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stockdownloader.pinescript.models import (
    Condition,
    Input,
    StrategyDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter


def spy_ml_ensemble_strategy(
    exporter: DeepSurrogateExporter,
    buy_threshold: float = 0.65,
    sell_threshold: float = 0.35,
) -> StrategyDefinition:
    """Build a strategy definition for the SPY ML Ensemble strategy.

    The strategy injects a surrogate decision tree into Pine Script and
    uses the predicted probability to generate buy/sell signals with
    confidence-scaled dynamic TP/SL levels.

    Parameters
    ----------
    exporter:
        A trained :class:`DeepSurrogateExporter`.
    buy_threshold:
        Probability above which a buy signal fires (default 0.65).
    sell_threshold:
        Probability below which a sell signal fires (default 0.35).

    Returns
    -------
    StrategyDefinition
        A strategy definition ready for ``PineScriptGenerator().generate()``.

    Raises
    ------
    RuntimeError
        If *exporter* has not been trained.
    """
    if not exporter.is_trained:
        raise RuntimeError("DeepSurrogateExporter must be trained first")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    inputs = [
        # ML Signal
        Input.float_(
            "buyThresh", buy_threshold, "Buy Threshold",
            min_val=0.0, max_val=1.0, step=0.05, group="ML Signal",
        ),
        Input.float_(
            "sellThresh", sell_threshold, "Sell Threshold",
            min_val=0.0, max_val=1.0, step=0.05, group="ML Signal",
        ),
        # Risk
        Input.int_("atrLen", 14, "ATR Length", min_val=1, group="Risk"),
        Input.float_(
            "baseSL", 2.0, "Base SL (x ATR)",
            min_val=0.5, step=0.25, group="Risk",
        ),
        Input.float_(
            "baseTP", 2.0, "Base TP (x ATR)",
            min_val=0.5, step=0.25, group="Risk",
        ),
        Input.float_(
            "slCapDollars", 3.0, "SL Cap ($, 0=off)",
            min_val=0.0, step=0.25, group="Risk",
        ),
        Input.int_(
            "circuitMax", 3, "Circuit Breaker Losses",
            min_val=1, group="Risk",
        ),
        # Multi-Timeframe
        Input.bool_(
            "useVwapConfirm", True,
            "Use 15-min VWAP Confirmation", group="Multi-Timeframe",
        ),
        # Session
        Input.bool_(
            "closeEOD", True, "Close at End of Day", group="Session",
        ),
    ]

    # ------------------------------------------------------------------
    # No Indicator objects -- the strategy renderer auto-declares
    # float atrVal = ta.atr(14).  We avoid conflicts by NOT using
    # Indicator.atr here and instead computing our own atrDyn in
    # extra_code for dynamic SL/TP.
    # ------------------------------------------------------------------
    indicators = []

    # ------------------------------------------------------------------
    # extra_code -- injected BEFORE the signal logic section
    # ------------------------------------------------------------------
    extra_code: list[str] = []

    # 1. Tree dependency declarations
    dep_lines = exporter.get_pine_dependencies()
    extra_code.extend(dep_lines)

    # 2. Tree node computations
    tree_lines = exporter.tree_to_pine_lines()
    extra_code.extend(tree_lines)

    # 3. ML probability and confidence
    extra_code.extend([
        "float mlProb = tn0",
        "float confidence = math.abs(mlProb - 0.5) * 2.0",
    ])

    # 4. Multi-timeframe VWAP (15-min)
    extra_code.extend([
        "",
        "// 15-minute VWAP for multi-timeframe confirmation",
        "float vwap15 = request.security(syminfo.tickerid, '15', ta.vwap)",
        "bool vwapBullish = not useVwapConfirm or close > vwap15",
        "bool vwapBearish = not useVwapConfirm or close < vwap15",
    ])

    # 5. Buy/sell signal booleans (these are referenced by
    #    long_entry/short_entry Conditions via the renderer).
    #    Circuit breaker is handled by the renderer's own `ready`
    #    guard which checks `not tripped`.
    extra_code.extend([
        "",
        "// ML-driven entry signals",
        "bool buySignal  = mlProb > buyThresh and vwapBullish",
        "bool sellSignal = mlProb < sellThresh and vwapBearish",
    ])

    # ------------------------------------------------------------------
    # extra_plots -- injected AFTER the signal logic section
    # ------------------------------------------------------------------
    extra_plots: list[str] = []

    # 6. Dynamic TP/SL using confidence scaling.
    #    The renderer already declared `float atrVal = ta.atr(14)`.
    #    We use atrLen for our own dynamic ATR computation.
    extra_plots.extend([
        "// ═══════════════════════════════════════════════════════════════",
        "// DYNAMIC ML-DRIVEN TP/SL OVERRIDE",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "float atrDyn = ta.atr(atrLen)",
        "float dynamicSL = atrDyn * (baseSL - confidence * 0.5)",
        "float dynamicTP = atrDyn * (baseTP + confidence * 1.0)",
        "",
        "// Cap SL if slCapDollars > 0",
        "float cappedSL = slCapDollars > 0"
        " ? math.min(dynamicSL, slCapDollars) : dynamicSL",
        "",
        "// Override the renderer's fixed exits with dynamic ones",
        "if strategy.position_size > 0",
        "    float dynLongSL = strategy.position_avg_price - cappedSL",
        "    float dynLongTP = strategy.position_avg_price + dynamicTP",
        '    strategy.exit("LX", "L", stop=dynLongSL, limit=dynLongTP)',
        "if strategy.position_size < 0",
        "    float dynShortSL = strategy.position_avg_price + cappedSL",
        "    float dynShortTP = strategy.position_avg_price - dynamicTP",
        '    strategy.exit("SX", "S", stop=dynShortSL, limit=dynShortTP)',
    ])

    # 7. Custom EOD close (we handle it ourselves since the renderer
    #    uses the StrategyDefinition's close_eod flag which is also fine)
    extra_plots.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// CUSTOM EOD CLOSE (respects closeEOD input toggle)",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "if closeEOD and not session.ismarket and session.ismarket[1]"
        " and strategy.position_size != 0",
        '    strategy.close_all(comment="EOD")',
    ])

    # 8. Background coloring
    extra_plots.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// BACKGROUND COLORING",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "color mlBg = strategy.position_size > 0"
        " ? color.new(color.green, 90)"
        " : strategy.position_size < 0"
        " ? color.new(color.red, 90) : na",
        'bgcolor(mlBg, title="ML Position Zone")',
    ])

    # 9. Buy/Sell labels
    extra_plots.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// ENTRY LABELS",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "if strategy.position_size > 0 and strategy.position_size[1] <= 0",
        '    label.new(bar_index, low, "\\u25B2",'
        " style=label.style_label_up,"
        " color=color.green, textcolor=color.white, size=size.small)",
        "if strategy.position_size < 0 and strategy.position_size[1] >= 0",
        '    label.new(bar_index, high, "\\u25BC",'
        " style=label.style_label_down,"
        " color=color.red, textcolor=color.white, size=size.small)",
    ])

    # 10. TP/SL/Entry level plots
    extra_plots.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// TP/SL/ENTRY LEVEL PLOTS",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        "var float mlEntryPrice  = na",
        "var float mlStopPrice   = na",
        "var float mlTargetPrice = na",
        "",
        "if strategy.position_size != 0 and strategy.position_size[1] == 0",
        "    mlEntryPrice := strategy.position_avg_price",
        "    if strategy.position_size > 0",
        "        mlStopPrice   := mlEntryPrice - cappedSL",
        "        mlTargetPrice := mlEntryPrice + dynamicTP",
        "    else",
        "        mlStopPrice   := mlEntryPrice + cappedSL",
        "        mlTargetPrice := mlEntryPrice - dynamicTP",
        "if strategy.position_size == 0",
        "    mlEntryPrice  := na",
        "    mlStopPrice   := na",
        "    mlTargetPrice := na",
        "",
        'plot(mlEntryPrice, title="Entry Price",'
        " color=color.white, style=plot.style_linebr, linewidth=2)",
        'plot(mlTargetPrice, title="Take Profit",'
        " color=color.new(color.lime, 20),"
        " style=plot.style_linebr, linewidth=2)",
        'plot(mlStopPrice, title="Stop Loss",'
        " color=color.new(color.red, 20),"
        " style=plot.style_linebr, linewidth=2)",
    ])

    # 11. ML confidence in data window
    extra_plots.extend([
        "",
        "// ═══════════════════════════════════════════════════════════════",
        "// DATA WINDOW METRICS",
        "// ═══════════════════════════════════════════════════════════════",
        "",
        'plot(mlProb, title="ML Probability",'
        " color=color.new(color.gray, 100), display=display.data_window)",
        'plot(confidence, title="ML Confidence",'
        " color=color.new(color.gray, 100), display=display.data_window)",
        'plot(atrDyn, title="ATR (dynamic)",'
        " color=color.new(color.gray, 100), display=display.data_window)",
        'plot(dynamicSL, title="Dynamic SL ($)",'
        " color=color.new(color.gray, 100), display=display.data_window)",
        'plot(dynamicTP, title="Dynamic TP ($)",'
        " color=color.new(color.gray, 100), display=display.data_window)",
    ])

    # ------------------------------------------------------------------
    # Entry conditions
    # ------------------------------------------------------------------
    long_entry = Condition(
        "buySignal",
        "ML prob above threshold + VWAP confirmation",
    )
    short_entry = Condition(
        "sellSignal",
        "ML prob below threshold + VWAP confirmation",
    )

    # ------------------------------------------------------------------
    # Build the StrategyDefinition
    # ------------------------------------------------------------------
    features_desc = ", ".join(exporter.feature_names[:8])
    if len(exporter.feature_names) > 8:
        features_desc += f" (+{len(exporter.feature_names) - 8} more)"

    return StrategyDefinition(
        name="SPY ML Ensemble",
        short_name="SPY-ML-ENS",
        strategy_mode=True,
        description=(
            "ML Ensemble Strategy with dynamic confidence-scaled TP/SL.\n"
            "Surrogate decision tree approximates full ensemble probability.\n"
            f"Features: {features_desc}\n"
            "\n"
            "Dynamic exits: SL tightens and TP expands with higher confidence.\n"
            "Optional 15-minute VWAP multi-timeframe confirmation."
        ),
        inputs=inputs,
        indicators=indicators,
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=None,
        short_exit=None,
        exit_on_reverse=False,
        extra_code=extra_code,
        extra_plots=extra_plots,
        # Strategy infrastructure defaults -- the renderer uses these
        # for its built-in SL/TP (which we override with dynamic values).
        initial_capital=100000.0,
        sl_atr_mult=2.0,
        sl_cap_dollars=3.0,
        rr_ratio=2.0,
        circuit_breaker_losses=3,
        close_eod=True,
    )
