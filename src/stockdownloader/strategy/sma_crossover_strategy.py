"""Simple Moving Average crossover strategy.

Generates BUY on golden cross (short SMA crosses above long SMA)
and SELL on death cross (short SMA crosses below long SMA).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.util.moving_average_calculator import sma

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

    def get_name(self) -> str:
        return f"SMA Crossover ({self._short_period}/{self._long_period})"

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self._long_period:
            return Signal.HOLD

        current_short_sma = sma(data, current_index, self._short_period)
        current_long_sma = sma(data, current_index, self._long_period)
        prev_short_sma = sma(data, current_index - 1, self._short_period)
        prev_long_sma = sma(data, current_index - 1, self._long_period)

        short_above_long_now = current_short_sma > current_long_sma
        short_above_long_prev = prev_short_sma > prev_long_sma

        if short_above_long_now and not short_above_long_prev:
            return Signal.BUY
        if not short_above_long_now and short_above_long_prev:
            return Signal.SELL
        return Signal.HOLD

    def get_warmup_period(self) -> int:
        return self._long_period
