"""Simple single-indicator daily strategies.

Each strategy extends :class:`TradingStrategy` and evaluates one technical
indicator to produce BUY/SELL/HOLD signals:

- :class:`RSIStrategy` -- Relative Strength Index oversold/overbought crossover
- :class:`MACDStrategy` -- MACD signal-line crossover
- :class:`SMACrossoverStrategy` -- Simple Moving Average golden/death cross
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategies.base import Signal, TradingStrategy
from stockdownloader.indicators import (
    crossed_above,
    crossed_below,
    crossed_above_series,
    crossed_below_series,
)
from stockdownloader.indicators.hub import IndicatorHub
from stockdownloader.pinescript.models import (
    Condition, Indicator, Input, StrategyDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.core.models.price import PriceData


# ── RSIStrategy ─────────────────────────────────────────────────────────


class RSIStrategy(TradingStrategy):
    """RSI-based strategy with configurable oversold/overbought thresholds.

    Generates BUY when RSI crosses above the oversold threshold
    and SELL when RSI crosses below the overbought threshold.
    """

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0, hub: IndicatorHub | None = None) -> None:
        if period <= 0:
            raise ValueError("Period must be positive")
        if oversold < 0 or overbought > 100 or oversold >= overbought:
            raise ValueError(
                "Invalid threshold values: oversold must be < overbought, within [0, 100]"
            )
        self._period = period
        self._oversold_threshold = Decimal(str(oversold))
        self._overbought_threshold = Decimal(str(overbought))
        self._hub = hub or IndicatorHub()

    @property
    def name(self) -> str:
        return f"RSI ({self._period}) [{self._oversold_threshold}/{self._overbought_threshold}]"

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self._period + 1:
            return Signal.HOLD

        current_rsi = self._hub.rsi(data, current_index, self._period)
        prev_rsi = self._hub.rsi(data, current_index - 1, self._period)

        if crossed_above(current_rsi, prev_rsi, self._oversold_threshold):
            return Signal.BUY
        if crossed_below(current_rsi, prev_rsi, self._overbought_threshold):
            return Signal.SELL
        return Signal.HOLD

    @property
    def warmup_period(self) -> int:
        return self._period + 1

    def to_pinescript(self) -> StrategyDefinition:
        return StrategyDefinition(
            name="RSI Strategy",
            short_name="RSI",
            description=(
                "Relative Strength Index oversold/overbought crossover.\n"
                "Buy when RSI crosses above oversold, sell when RSI crosses below overbought."
            ),
            inputs=[
                Input.int_("rsiPeriod", self._period, "RSI Period"),
                Input.float_("oversold", float(self._oversold_threshold),
                             "Oversold Threshold", step=1.0),
                Input.float_("overbought", float(self._overbought_threshold),
                             "Overbought Threshold", step=1.0),
            ],
            indicators=[
                Indicator.rsi("rsiVal", "close", "rsiPeriod"),
            ],
            long_entry=Condition(
                "ta.crossover(rsiVal, oversold)",
                "RSI crosses above oversold threshold",
            ),
            short_entry=Condition(
                "ta.crossunder(rsiVal, overbought)",
                "RSI crosses below overbought threshold",
            ),
        )


# ── MACDStrategy ────────────────────────────────────────────────────────


class MACDStrategy(TradingStrategy):
    """MACD signal-line crossover strategy.

    Generates BUY on bullish crossover (MACD crosses above signal line)
    and SELL on bearish crossover (MACD crosses below signal line).
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, hub: IndicatorHub | None = None) -> None:
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("All periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period
        self._hub = hub or IndicatorHub()

    @property
    def name(self) -> str:
        return f"MACD ({self._fast_period}/{self._slow_period}/{self._signal_period})"

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        min_required = self._slow_period + self._signal_period
        if current_index < min_required:
            return Signal.HOLD

        current_macd = self._hub.macd_line(data, current_index, self._fast_period, self._slow_period)
        current_signal = self._hub.macd_signal(
            data, current_index, self._fast_period, self._slow_period, self._signal_period
        )
        prev_macd = self._hub.macd_line(data, current_index - 1, self._fast_period, self._slow_period)
        prev_signal = self._hub.macd_signal(
            data, current_index - 1, self._fast_period, self._slow_period, self._signal_period
        )

        if crossed_above_series(current_macd, prev_macd,
                                current_signal, prev_signal):
            return Signal.BUY
        if crossed_below_series(current_macd, prev_macd,
                                current_signal, prev_signal):
            return Signal.SELL
        return Signal.HOLD

    @property
    def warmup_period(self) -> int:
        return self._slow_period + self._signal_period

    def to_pinescript(self) -> StrategyDefinition:
        return StrategyDefinition(
            name="MACD Crossover",
            short_name="MACD",
            description=(
                "MACD signal-line crossover strategy.\n"
                "Buy on bullish crossover, sell on bearish crossover."
            ),
            inputs=[
                Input.int_("macdFast", self._fast_period, "MACD Fast Period"),
                Input.int_("macdSlow", self._slow_period, "MACD Slow Period"),
                Input.int_("macdSignal", self._signal_period, "Signal Period"),
            ],
            indicators=[
                *Indicator.macd("macdFast", "macdSlow", "macdSignal"),
            ],
            long_entry=Condition(
                "ta.crossover(macdLine, macdSignal)",
                "MACD bullish crossover",
            ),
            short_entry=Condition(
                "ta.crossunder(macdLine, macdSignal)",
                "MACD bearish crossover",
            ),
        )


# ── SMACrossoverStrategy ───────────────────────────────────────────────


class SMACrossoverStrategy(TradingStrategy):
    """SMA crossover strategy using golden/death cross signals.

    Generates BUY on golden cross (short SMA crosses above long SMA)
    and SELL on death cross (short SMA crosses below long SMA).
    """

    def __init__(self, short_period: int = 9, long_period: int = 21, hub: IndicatorHub | None = None) -> None:
        if short_period <= 0 or long_period <= 0:
            raise ValueError("Periods must be positive")
        if short_period >= long_period:
            raise ValueError("Short period must be less than long period")
        self._short_period = short_period
        self._long_period = long_period
        self._hub = hub or IndicatorHub()

    @property
    def name(self) -> str:
        return f"SMA Crossover ({self._short_period}/{self._long_period})"

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self._long_period:
            return Signal.HOLD

        current_short_sma = self._hub.sma(data, current_index, self._short_period)
        current_long_sma = self._hub.sma(data, current_index, self._long_period)
        prev_short_sma = self._hub.sma(data, current_index - 1, self._short_period)
        prev_long_sma = self._hub.sma(data, current_index - 1, self._long_period)

        if crossed_above_series(current_short_sma, prev_short_sma,
                                current_long_sma, prev_long_sma):
            return Signal.BUY
        if crossed_below_series(current_short_sma, prev_short_sma,
                                current_long_sma, prev_long_sma):
            return Signal.SELL
        return Signal.HOLD

    @property
    def warmup_period(self) -> int:
        return self._long_period

    def to_pinescript(self) -> StrategyDefinition:
        return StrategyDefinition(
            name="SMA Crossover",
            short_name="SMA-X",
            description=(
                "Simple Moving Average crossover strategy.\n"
                "Golden Cross = Buy, Death Cross = Sell."
            ),
            inputs=[
                Input.int_("fastPeriod", self._short_period, "Fast SMA Period"),
                Input.int_("slowPeriod", self._long_period, "Slow SMA Period"),
            ],
            indicators=[
                Indicator.sma("smaFast", "close", "fastPeriod",
                              plot=True, color="color.blue"),
                Indicator.sma("smaSlow", "close", "slowPeriod",
                              plot=True, color="color.red"),
            ],
            long_entry=Condition(
                "ta.crossover(smaFast, smaSlow)",
                "Golden Cross: Fast SMA crosses above Slow SMA",
            ),
            long_exit=Condition(
                "ta.crossunder(smaFast, smaSlow)",
                "Death Cross: Fast SMA crosses below Slow SMA",
            ),
        )
