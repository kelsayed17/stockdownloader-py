"""Enhanced SPY PineScript strategies — strategy mode.

Generates production-ready TradingView strategy() scripts for the top 3
walk-forward validated strategies from the Grand Tournament:

1. **MACD+OBV** — OOS +74.03, degradation 0.96 (best robustness)
2. **SMA Crossover (20/21)** — OOS +65.29, degradation 2.52
3. **MACD (8/35/5)** — OOS +62.80, degradation 1.57

All strategies use ``strategy()`` mode with ``strategy.entry()``,
``strategy.exit()``, ATR-based position sizing, breakeven management,
circuit breaker, and EOD close.

Usage::

    from stockdownloader.app.pinescript_catalog.spy_strategies import (
        spy_macd_obv_strategy,
    )
    from stockdownloader.pinescript import PineScriptGenerator

    gen = PineScriptGenerator()
    pine = gen.generate(spy_macd_obv_strategy())
"""

from __future__ import annotations

from stockdownloader.pinescript.models import (
    Condition,
    Indicator,
    Input,
    StrategyDefinition,
)


# ======================================================================
# Strategy 1: MACD + OBV (Grand Tournament Winner)
# OOS +74.03, Degradation 0.96
# ======================================================================


def _macd_obv_core_inputs() -> list[Input]:
    """MACD+OBV inputs."""
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
    """MACD+OBV indicators."""
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


def spy_macd_obv_strategy() -> StrategyDefinition:
    """MACD + OBV — Grand Tournament winner (strategy mode).

    Walk-forward validated: OOS score +74.03, degradation 0.96
    (near-perfect IS/OOS consistency). Full backtesting with ATR-based
    position sizing, breakeven management, circuit breaker, and EOD close.
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
        indicators=_macd_obv_core_indicators(),
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
    """SMA Crossover inputs."""
    return [
        Input.int_("smaShort", 20, "SMA Short Period",
                    min_val=2, group="SMA"),
        Input.int_("smaLong", 21, "SMA Long Period",
                    min_val=3, group="SMA"),
    ]


def _sma_core_indicators() -> list[Indicator]:
    """SMA Crossover indicators."""
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
    """SMA Crossover (20/21) — Grand Tournament #2 (strategy mode).

    Walk-forward validated: OOS score +65.29, degradation 2.52
    (GOOD). Tight golden cross. Long-only. Full backtesting with
    ATR-based position sizing, breakeven management, circuit breaker,
    and EOD close.
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
        indicators=_sma_core_indicators(),
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
    """MACD Optimized 8/35/5 inputs."""
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
    """MACD Optimized (8/35/5) — Grand Tournament #3 (strategy mode).

    Walk-forward validated: OOS score +62.80, degradation 1.57
    (GOOD). Wide slow period (35) filters noise. Full backtesting with
    ATR-based position sizing, breakeven management, circuit breaker,
    and EOD close.
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
        ],
        long_entry=_MACD_OPT_LONG,
        short_entry=_MACD_OPT_SHORT,
        long_exit=_MACD_OPT_EXIT_LONG,
        short_exit=_MACD_OPT_EXIT_SHORT,
        exit_on_reverse=False,
        long_label="Buy",
        short_label="Sell",
    )
