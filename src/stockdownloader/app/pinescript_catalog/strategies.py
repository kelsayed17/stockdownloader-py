"""Pre-built standalone strategy definitions for Pine Script generation.

Contains strategy factory functions for all signal-based and VWAP strategies
plus the ``STRATEGY_CATALOG`` registry.

**Pipeline wrappers:** The 8 daily strategy factories (SMA, RSI, MACD,
Bollinger+RSI, Breakout, Momentum Confluence, Multi-Indicator, DMI+VWAP)
delegate to their Python ``TradingStrategy.to_pinescript()`` methods,
ensuring Pine output always matches the Python backtesting logic.

VWAP mode helpers and shared infrastructure live in
:mod:`pinescript_modes` (no strategy-class imports) to avoid circular
dependencies between strategy modules and this generation layer.
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
from stockdownloader.strategy.daily.multi_indicator_strategy import (
    MultiIndicatorStrategy,
)
from stockdownloader.strategy.daily.rsi_strategy import RSIStrategy
from stockdownloader.strategy.daily.sma_crossover_strategy import (
    SMACrossoverStrategy,
)
from stockdownloader.strategy.intraday.dmi_vwap_strategy import DmiVwapStrategy
from stockdownloader.util.pinescript_models import (
    Condition,
    Indicator,
    Input,
    StrategyDefinition,
)
from stockdownloader.util.pinescript import mode_to_strategy
from stockdownloader.app.pinescript_catalog.gme_prediction import gme_prediction_strategy
from stockdownloader.util.pinescript_modes import (
    pb_mode as _pb_mode,
    rev_mode as _rev_mode,
    orb_mode as _orb_mode,
    orr_mode as _orr_mode,
    ps_mode as _ps_mode,
    vwap_shared_infrastructure as _vwap_shared_infrastructure,
)
from stockdownloader.app.pinescript_catalog.spy_strategies import (
    spy_macd_obv_strategy,
    spy_macd_obv_strategy_v2,
    spy_macd_optimized_strategy,
    spy_macd_optimized_strategy_v2,
    spy_sma_crossover_strategy,
    spy_sma_crossover_strategy_v2,
)


# ======================================================================
# Signal-based standalone strategies (pipeline wrappers — 8 daily)
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


def multi_indicator_strategy(
    buy_threshold: int = 4, sell_threshold: int = 4,
) -> StrategyDefinition:
    """Multi-indicator confluence scoring — delegates to Python strategy."""
    return MultiIndicatorStrategy(buy_threshold, sell_threshold).to_pinescript()


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
    "multi_indicator": multi_indicator_strategy,
    "vwap_pullback": vwap_pullback_strategy,
    "vwap_reversal": vwap_reversal_strategy,
    "vwap_or_breakout": vwap_or_breakout_strategy,
    "vwap_or_reversal": vwap_or_reversal_strategy,
    "vwap_pattern_scalp": vwap_pattern_scalp_strategy,
    "gme_prediction": gme_prediction_strategy,
    "spy_macd_obv": spy_macd_obv_strategy,
    "spy_sma_crossover": spy_sma_crossover_strategy,
    "spy_macd_optimized": spy_macd_optimized_strategy,
    "spy_macd_obv_v2": spy_macd_obv_strategy_v2,
    "spy_sma_crossover_v2": spy_sma_crossover_strategy_v2,
    "spy_macd_optimized_v2": spy_macd_optimized_strategy_v2,
}
