"""Moving Average Convergence Divergence strategy.

Generates BUY on bullish crossover (MACD crosses above signal line)
and SELL on bearish crossover (MACD crosses below signal line).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.util.crossover import crossed_above_series, crossed_below_series
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.pinescript_models import (
    Condition, Indicator, Input, StrategyDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


class MACDStrategy(TradingStrategy):
    """MACD signal-line crossover strategy."""

    def __init__(self, fast_period: int, slow_period: int, signal_period: int, hub: IndicatorHub | None = None) -> None:
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
