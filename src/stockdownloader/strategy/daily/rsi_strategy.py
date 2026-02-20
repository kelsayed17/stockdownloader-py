"""Relative Strength Index strategy.

Generates BUY when RSI crosses above the oversold threshold
and SELL when RSI crosses below the overbought threshold.
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.util.crossover import crossed_above, crossed_below
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.pinescript_models import (
    Condition, Indicator, Input, StrategyDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


class RSIStrategy(TradingStrategy):
    """RSI-based strategy with configurable oversold/overbought thresholds."""

    def __init__(self, period: int, oversold: float, overbought: float, hub: IndicatorHub | None = None) -> None:
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
