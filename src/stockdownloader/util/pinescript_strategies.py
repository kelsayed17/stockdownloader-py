"""Pre-built standalone strategy definitions for Pine Script generation.

Contains strategy factory functions for all signal-based and VWAP strategies
plus the ``STRATEGY_CATALOG`` registry.

**Pipeline wrappers:** The 7 daily strategy factories (SMA, RSI, MACD,
Bollinger+RSI, Breakout, Momentum Confluence, DMI+VWAP) now delegate to
their Python ``TradingStrategy.to_pinescript()`` methods, ensuring Pine
output always matches the Python backtesting logic.

Also exports VWAP mode helpers (``_pb_mode``, ``_rev_mode``, etc.) and shared
infrastructure, used by :mod:`pinescript_composites`.
"""

from __future__ import annotations

from stockdownloader.strategy.daily.bollinger_band_rsi_strategy import (
    BollingerBandRSIStrategy,
)
from stockdownloader.strategy.daily.breakout_strategy import BreakoutStrategy
from stockdownloader.strategy.daily.macd_strategy import MACDStrategy
from stockdownloader.strategy.daily.momentum_confluence_strategy import (
    MomentumConfluenceStrategy,
)
from stockdownloader.strategy.daily.rsi_strategy import RSIStrategy
from stockdownloader.strategy.daily.sma_crossover_strategy import (
    SMACrossoverStrategy,
)
from stockdownloader.strategy.dmi_vwap_strategy import DmiVwapStrategy
from stockdownloader.util.pinescript_generator import (
    Condition,
    Indicator,
    Input,
    ModeDefinition,
    SharedInfrastructure,
    StrategyDefinition,
    mode_to_strategy,
)


# ======================================================================
# Signal-based standalone strategies (pipeline wrappers)
# ======================================================================


def sma_crossover_strategy(
    short_period: int = 9, long_period: int = 21,
) -> StrategyDefinition:
    """SMA Golden Cross / Death Cross — delegates to Python strategy."""
    return SMACrossoverStrategy(short_period, long_period).to_pinescript()


def rsi_strategy(
    period: int = 14, oversold: float = 30.0, overbought: float = 70.0,
) -> StrategyDefinition:
    """RSI oversold/overbought — delegates to Python strategy."""
    return RSIStrategy(period, oversold, overbought).to_pinescript()


def macd_strategy(
    fast: int = 12, slow: int = 26, signal: int = 9,
) -> StrategyDefinition:
    """MACD crossover — delegates to Python strategy."""
    return MACDStrategy(fast, slow, signal).to_pinescript()


def bollinger_rsi_strategy() -> StrategyDefinition:
    """BB + RSI mean-reversion — delegates to Python strategy."""
    return BollingerBandRSIStrategy().to_pinescript()


def dmi_vwap_strategy() -> StrategyDefinition:
    """DMI + session VWAP — delegates to Python strategy."""
    return DmiVwapStrategy().to_pinescript()


def momentum_confluence_strategy() -> StrategyDefinition:
    """Momentum confluence — delegates to Python strategy."""
    return MomentumConfluenceStrategy().to_pinescript()


def breakout_strategy() -> StrategyDefinition:
    """BB squeeze breakout — delegates to Python strategy."""
    return BreakoutStrategy().to_pinescript()


# ======================================================================
# Pine-only strategies (no Python TradingStrategy counterpart)
# ======================================================================


def macd_obv_strategy(
    fast: int = 12, slow: int = 26, signal: int = 9,
    obv_smooth: int = 5,
) -> StrategyDefinition:
    """MACD crossover confirmed by OBV trend — walk-forward validated winner."""
    return StrategyDefinition(
        name="MACD + OBV",
        short_name="MACD-OBV",
        description="Walk-forward validated winner.\n"
                    "MACD crossover confirmed by OBV trend alignment.",
        inputs=[
            Input.int_("macdFast", fast, "MACD Fast Length"),
            Input.int_("macdSlow", slow, "MACD Slow Length"),
            Input.int_("macdSignal", signal, "MACD Signal Length"),
            Input.int_("obvSmoothing", obv_smooth, "OBV Smoothing Period"),
        ],
        indicators=[
            *Indicator.macd("macdFast", "macdSlow", "macdSignal"),
            Indicator.obv("obvRaw"),
            Indicator.raw("obvSmoothed", "ta.ema(obvRaw, obvSmoothing)"),
            Indicator.raw("obvRising",
                          "obvSmoothed > obvSmoothed[1]"),
            Indicator.raw("obvFalling",
                          "obvSmoothed < obvSmoothed[1]"),
        ],
        long_entry=Condition(
            "ta.crossover(macdLine, macdSignal) and obvRising",
            "MACD bullish crossover with rising OBV",
        ),
        short_entry=Condition(
            "ta.crossunder(macdLine, macdSignal) and obvFalling",
            "MACD bearish crossover with falling OBV",
        ),
    )


# ======================================================================
# VWAP shared infrastructure (used by modes + composites)
# ======================================================================


def _vwap_shared_infrastructure() -> SharedInfrastructure:
    """Bundle shared VWAP inputs, indicators, and code."""
    return SharedInfrastructure(
        inputs=_vwap_shared_inputs(),
        indicators=_vwap_shared_indicators(),
        code=_vwap_shared_code(),
    )


def _vwap_shared_indicators() -> list[Indicator]:
    """Shared indicators for VWAP composite modes."""
    return [
        Indicator.session_vwap("vwapValue"),
        Indicator.atr("atrVal", "atrLen"),
        Indicator.dmi("adxLen", "adxLen"),
        Indicator.ema("emaFast", "close", "emaFastLen",
                       plot=True, color="color.teal"),
        Indicator.ema("emaSlow", "close", "emaSlowLen",
                       plot=True, color="color.orange"),
    ]


def _vwap_shared_inputs() -> list[Input]:
    """Shared inputs for VWAP composite modes."""
    return [
        Input.int_("atrLen", 14, "ATR Length", group="Shared"),
        Input.int_("adxLen", 14, "ADX Length", group="Shared"),
        Input.int_("emaFastLen", 9, "EMA Fast", group="Shared"),
        Input.int_("emaSlowLen", 21, "EMA Slow", group="Shared"),
    ]


def _vwap_shared_code() -> list[str]:
    """Shared code for VWAP composite (OR tracking, candle helpers)."""
    return [
        "// --- Opening Range Tracking ---",
        "int    orBars        = 3",
        "var float orHigh     = na",
        "var float orLow      = na",
        "var bool  orDone     = false",
        "var int   barOfDay   = 0",
        "",
        "if isNewSession",
        "    orHigh  := high",
        "    orLow   := low",
        "    orDone  := false",
        "    barOfDay := 0",
        "",
        "barOfDay += 1",
        "",
        "if not orDone",
        "    orHigh := math.max(orHigh, high)",
        "    orLow  := math.min(orLow, low)",
        "    if barOfDay >= orBars",
        "        orDone := true",
        "",
        "float orRange = orDone ? orHigh - orLow : na",
        "",
        "// --- Candle Helpers ---",
        "float bodySize = math.abs(close - open)",
        "bool  bullCandle = close > open and bodySize >= atrVal * 0.15",
        "bool  bearCandle = close < open and bodySize >= atrVal * 0.15",
        "",
        "// --- VWAP Distance ---",
        "float vwapDist = math.abs(close - vwapValue)",
        "",
        "// --- Relative Volume ---",
        "float avgVol20 = ta.sma(volume, 20)",
        "float relVol   = avgVol20 > 0 ? volume / avgVol20 : 0",
    ]


# ======================================================================
# VWAP mode definitions
# ======================================================================


def _pb_mode() -> ModeDefinition:
    """Pullback mode: Trend bounce off VWAP zone."""
    return ModeDefinition(
        name="VWAP Pullback",
        short_name="pb",
        group="Pullback",
        enabled_default=True,
        description=(
            "Trend-following VWAP bounce. ADX confirms trending, "
            "EMA aligned, price pulls back to VWAP zone. "
            "Supports VWAP session bias filter, S/R confluence "
            "requirement, and VWAP-target TP mode."
        ),
        inputs=[
            Input.float_("pbZone", 0.5, "Zone Width (x band)",
                          min_val=0.1, step=0.1),
            Input.float_("pbAdxMin", 21.0, "Min ADX",
                          step=1.0),
            Input.int_("pbTrendBars", 3, "Trend Confirmation Bars"),
            Input.bool_("pbVwapBias", False, "VWAP Session Bias Filter"),
            Input.float_("pbVwapBiasPct", 0.7, "VWAP Bias Min %",
                          min_val=0.5, step=0.05),
            Input.bool_("pbRequireSr", False, "Require S/R Confluence"),
            Input.string_("pbTpMode", "rr", "TP Mode",
                           options=["rr", "vwap"]),
        ],
        indicators=[
            Indicator.raw("vwapStd",
                          "ta.stdev(close - vwapValue, 20)"),
            Indicator.raw("pbInZone",
                          "vwapDist <= vwapStd * pbZone"),
        ],
        long_entry=Condition(
            "adxValue >= pbAdxMin and emaFast > emaSlow "
            "and close > vwapValue - atrVal * 0.25 "
            "and pbInZone and bullCandle",
            "ADX trending + EMA bullish + price in VWAP zone + bull candle",
        ),
        short_entry=Condition(
            "adxValue >= pbAdxMin and emaFast < emaSlow "
            "and close < vwapValue + atrVal * 0.25 "
            "and pbInZone and bearCandle",
            "ADX trending + EMA bearish + price in VWAP zone + bear candle",
        ),
        label_color_long="color.teal",
        label_color_short="color.teal",
    )


def _rev_mode() -> ModeDefinition:
    """Reversal mode: Sideways market band fade."""
    return ModeDefinition(
        name="VWAP Reversal",
        short_name="rev",
        group="Reversal",
        enabled_default=True,
        description=(
            "Mean-reversion fade at VWAP band extremes. "
            "ADX low (sideways), VWAP flat, price at band. "
            "Supports separate trade window, band touch "
            "confirmation, configurable hug limit, RR-based "
            "TP, and S/R confluence requirement."
        ),
        inputs=[
            Input.float_("revAdxMax", 21.0, "Max ADX (sideways)",
                          step=1.0),
            Input.float_("revBandMult", 2.0, "Band Sigma",
                          min_val=1.0, step=0.5),
            Input.float_("revBody", 0.20, "Min Body (x ATR)",
                          step=0.05),
            Input.int_("revCanTradeBar", 0, "Earliest REV Bar (0=shared)"),
            Input.int_("revMinTouches", 0, "Min Band Touches (0=off)"),
            Input.int_("revHugLimit", 20, "Hugging Limit"),
            Input.string_("revTpMode", "vwap", "TP Mode",
                           options=["vwap", "rr"]),
            Input.float_("revRr", 1.0, "R:R Multiplier (rr mode)",
                          step=0.1),
            Input.bool_("revRequireSr", False, "Require S/R Confluence"),
        ],
        indicators=[
            Indicator.raw("revUpper",
                          "vwapValue + ta.stdev(close - vwapValue, 20)"
                          " * revBandMult"),
            Indicator.raw("revLower",
                          "vwapValue - ta.stdev(close - vwapValue, 20)"
                          " * revBandMult"),
            Indicator.raw("vwapFlat",
                          "math.abs(vwapValue - vwapValue[5])"
                          " <= atrVal * 0.05"),
        ],
        long_entry=Condition(
            "adxValue < revAdxMax and vwapFlat "
            "and low <= revLower + atrVal * 0.3 "
            "and bullCandle and bodySize >= atrVal * revBody",
            "Sideways + VWAP flat + price at lower band + bull candle",
        ),
        short_entry=Condition(
            "adxValue < revAdxMax and vwapFlat "
            "and high >= revUpper - atrVal * 0.3 "
            "and bearCandle and bodySize >= atrVal * revBody",
            "Sideways + VWAP flat + price at upper band + bear candle",
        ),
        label_color_long="color.purple",
        label_color_short="color.purple",
    )


def _orb_mode() -> ModeDefinition:
    """OR Breakout mode: Momentum continuation past opening range."""
    return ModeDefinition(
        name="OR Breakout",
        short_name="orb",
        group="OR Breakout",
        enabled_default=True,
        description=(
            "Momentum breakout beyond opening range with volume "
            "confirmation. Fire-once per session. Supports aggressive "
            "and retest entry modes, OR-range targets, gap/ADX filters."
        ),
        inputs=[
            Input.int_("orbWindow", 20, "Max Bar for ORB"),
            Input.float_("orbRvol", 2.0, "Min RVOL",
                          min_val=1.0, step=0.5),
            Input.float_("orbBodyMin", 0.2, "Min Body (x ATR)",
                          step=0.05),
            Input.string_("orbEntryMode", "aggressive",
                           "Entry Mode",
                           options=["aggressive", "retest"]),
            Input.string_("orbSlMode", "OR Opposite",
                           "SL Mode",
                           options=["OR Opposite", "OR Midpoint", "ATR-Based"]),
            Input.string_("orbTpMode", "trail_only",
                           "TP Mode",
                           options=["trail_only", "or_range", "2x_or_range"]),
            Input.bool_("orbReentryExit", False, "Exit on OR Re-entry"),
            Input.bool_("orbGapFilter", False, "Gap Alignment Filter"),
            Input.bool_("orbAdxFilter", False, "ADX Regime Filter"),
            Input.int_("orbTimeExit", 0, "Time Exit Bar (0=off)"),
        ],
        indicators=[],
        extra_code=[
            "var bool orbFiredToday = false",
            "if isNewSession",
            "    orbFiredToday := false",
        ],
        long_entry=Condition(
            "orDone and not orbFiredToday "
            "and barOfDay <= orbWindow "
            "and close > orHigh and close > open "
            "and bodySize >= atrVal * orbBodyMin "
            "and relVol >= orbRvol "
            "and close > vwapValue",
            "Price breaks above OR high with volume surge + VWAP alignment",
        ),
        short_entry=Condition(
            "orDone and not orbFiredToday "
            "and barOfDay <= orbWindow "
            "and close < orLow and close < open "
            "and bodySize >= atrVal * orbBodyMin "
            "and relVol >= orbRvol "
            "and close < vwapValue",
            "Price breaks below OR low with volume surge + VWAP alignment",
        ),
        label_color_long="color.lime",
        label_color_short="color.lime",
    )


def _orr_mode() -> ModeDefinition:
    """OR Reversal mode: Fade OR retest with pattern confirmation."""
    return ModeDefinition(
        name="OR Reversal",
        short_name="orr",
        group="OR Reversal",
        enabled_default=True,
        description=(
            "Fade opening range extreme on retest with reversal "
            "pattern. Supports VWAP disagreement filter, gap-fade "
            "context, low-ADX regime, breakout-then-reclaim, and "
            "OR-opposite targets. Fire-once per session."
        ),
        inputs=[
            Input.int_("orrWindow", 25, "Max Bar for ORR"),
            Input.float_("orrProx", 0.2, "OR Proximity (x ATR)",
                          step=0.05),
            Input.float_("orrRvol", 1.0, "Min RVOL",
                          min_val=0.5, step=0.5),
            Input.string_("orrTpMode", "VWAP",
                           "TP Mode",
                           options=["VWAP", "OR Mid", "OR Opposite"]),
            Input.bool_("orrVwapDisagree", False, "VWAP Disagreement Filter"),
            Input.bool_("orrGapFilter", False, "Gap Fade Filter"),
            Input.bool_("orrAdxFilter", False, "Low-ADX Regime Filter"),
            Input.bool_("orrRebreakExit", False, "Re-breakout Exit"),
            Input.bool_("orrRequireBreak", False, "Require Breakout+Reclaim"),
        ],
        indicators=[],
        extra_code=[
            "var bool orrFiredToday = false",
            "if isNewSession",
            "    orrFiredToday := false",
            "",
            "// Hammer: small body, long lower wick (bullish reversal)",
            "float upperWick = high - math.max(open, close)",
            "float lowerWick = math.min(open, close) - low",
            "float candleRange = high - low",
            "bool  isHammer = candleRange > 0 "
            "and lowerWick / candleRange >= 0.6 "
            "and bodySize / candleRange <= 0.3",
            "bool  isInvHammer = candleRange > 0 "
            "and upperWick / candleRange >= 0.6 "
            "and bodySize / candleRange <= 0.3",
        ],
        long_entry=Condition(
            "orDone and not orrFiredToday "
            "and barOfDay <= orrWindow "
            "and low <= orLow + atrVal * orrProx "
            "and (isHammer or bullCandle) "
            "and relVol >= orrRvol",
            "Price retests OR low with bullish reversal pattern",
        ),
        short_entry=Condition(
            "orDone and not orrFiredToday "
            "and barOfDay <= orrWindow "
            "and high >= orHigh - atrVal * orrProx "
            "and (isInvHammer or bearCandle) "
            "and relVol >= orrRvol",
            "Price retests OR high with bearish reversal pattern",
        ),
        label_color_long="color.aqua",
        label_color_short="color.aqua",
    )


def _ps_mode() -> ModeDefinition:
    """Pattern Scalp mode: Fade OR manipulation candles."""
    return ModeDefinition(
        name="Pattern Scalp",
        short_name="ps",
        group="Pattern Scalp",
        enabled_default=False,
        description=(
            "Fades opening range manipulation. Requires OR range \u2265 "
            "daily ATR pct, then takes the opposite direction."
        ),
        inputs=[
            Input.float_("psAtrPct", 30.0, "Min OR as % of ATR",
                          min_val=10.0, step=5.0),
            Input.int_("psWindow", 12, "Max Bar for PS"),
            Input.float_("psRvol", 1.0, "Min RVOL",
                          min_val=0.5, step=0.5),
        ],
        indicators=[
            Indicator.raw("dailyAtr", "ta.atr(14)"),
            Indicator.raw("isManip",
                          "orDone and orRange >= dailyAtr * psAtrPct / 100"),
            Indicator.raw("orBullish",
                          "orDone and close[barOfDay] > open[barOfDay]"),
        ],
        extra_code=[
            "var bool psFiredToday = false",
            "if isNewSession",
            "    psFiredToday := false",
        ],
        long_entry=Condition(
            "isManip and not psFiredToday "
            "and barOfDay >= 4 and barOfDay <= psWindow "
            "and not orBullish "
            "and bullCandle "
            "and relVol >= psRvol",
            "OR bearish manipulation detected, fade with bull pattern",
        ),
        short_entry=Condition(
            "isManip and not psFiredToday "
            "and barOfDay >= 4 and barOfDay <= psWindow "
            "and orBullish "
            "and bearCandle "
            "and relVol >= psRvol",
            "OR bullish manipulation detected, fade with bear pattern",
        ),
        label_color_long="color.yellow",
        label_color_short="color.yellow",
    )


# ======================================================================
# Standalone VWAP strategy factories
# ======================================================================


def vwap_pullback_strategy() -> StrategyDefinition:
    return mode_to_strategy(
        _pb_mode(), shared=_vwap_shared_infrastructure(),
        name_override="VWAP Pullback", short_name_override="VWAP-PB",
    )


def vwap_reversal_strategy() -> StrategyDefinition:
    return mode_to_strategy(
        _rev_mode(), shared=_vwap_shared_infrastructure(),
        name_override="VWAP Reversal", short_name_override="VWAP-REV",
    )


def vwap_or_breakout_strategy() -> StrategyDefinition:
    return mode_to_strategy(
        _orb_mode(), shared=_vwap_shared_infrastructure(),
        name_override="VWAP OR Breakout", short_name_override="VWAP-ORB",
    )


def vwap_or_reversal_strategy() -> StrategyDefinition:
    return mode_to_strategy(
        _orr_mode(), shared=_vwap_shared_infrastructure(),
        name_override="VWAP OR Reversal", short_name_override="VWAP-ORR",
    )


def vwap_pattern_scalp_strategy() -> StrategyDefinition:
    return mode_to_strategy(
        _ps_mode(), shared=_vwap_shared_infrastructure(),
        name_override="VWAP Pattern Scalp", short_name_override="VWAP-PS",
    )


# ======================================================================
# Strategy catalog
# ======================================================================


STRATEGY_CATALOG: dict[str, callable] = {
    "sma_crossover": sma_crossover_strategy,
    "rsi": rsi_strategy,
    "macd": macd_strategy,
    "macd_obv": macd_obv_strategy,
    "bollinger_rsi": bollinger_rsi_strategy,
    "dmi_vwap": dmi_vwap_strategy,
    "momentum_confluence": momentum_confluence_strategy,
    "breakout": breakout_strategy,
    "vwap_pullback": vwap_pullback_strategy,
    "vwap_reversal": vwap_reversal_strategy,
    "vwap_or_breakout": vwap_or_breakout_strategy,
    "vwap_or_reversal": vwap_or_reversal_strategy,
    "vwap_pattern_scalp": vwap_pattern_scalp_strategy,
}
