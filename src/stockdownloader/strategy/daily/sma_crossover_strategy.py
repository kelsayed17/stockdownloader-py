"""Simple Moving Average crossover strategy.

Generates BUY on golden cross (short SMA crosses above long SMA)
and SELL on death cross (short SMA crosses below long SMA).
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


class SMACrossoverStrategy(TradingStrategy):
    """SMA crossover strategy using golden/death cross signals."""

    def __init__(self, short_period: int, long_period: int) -> None:
        if short_period <= 0 or long_period <= 0:
            raise ValueError("Periods must be positive")
        if short_period >= long_period:
            raise ValueError("Short period must be less than long period")
        self._short_period = short_period
        self._long_period = long_period
        self._hub = IndicatorHub()

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
