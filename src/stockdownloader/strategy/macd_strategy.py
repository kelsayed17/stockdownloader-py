"""Moving Average Convergence Divergence strategy.

Generates BUY on bullish crossover (MACD crosses above signal line)
and SELL on bearish crossover (MACD crosses below signal line).
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.util.moving_average_calculator import ema

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData

_SCALE = 10


class MACDStrategy(TradingStrategy):
    """MACD signal-line crossover strategy."""

    def __init__(self, fast_period: int, slow_period: int, signal_period: int) -> None:
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("All periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period

    def get_name(self) -> str:
        return f"MACD ({self._fast_period}/{self._slow_period}/{self._signal_period})"

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        min_required = self._slow_period + self._signal_period
        if current_index < min_required:
            return Signal.HOLD

        current_macd = self._calculate_macd(data, current_index)
        current_signal = self._calculate_signal_line(data, current_index)
        prev_macd = self._calculate_macd(data, current_index - 1)
        prev_signal = self._calculate_signal_line(data, current_index - 1)

        macd_above_signal_now = current_macd > current_signal
        macd_above_signal_prev = prev_macd > prev_signal

        if macd_above_signal_now and not macd_above_signal_prev:
            return Signal.BUY
        if not macd_above_signal_now and macd_above_signal_prev:
            return Signal.SELL
        return Signal.HOLD

    def get_warmup_period(self) -> int:
        return self._slow_period + self._signal_period

    def _calculate_macd(self, data: list[PriceData], end_index: int) -> Decimal:
        fast_ema = ema(data, end_index, self._fast_period)
        slow_ema = ema(data, end_index, self._slow_period)
        return fast_ema - slow_ema

    def _calculate_signal_line(self, data: list[PriceData], end_index: int) -> Decimal:
        multiplier = Decimal(str(2.0 / (self._signal_period + 1)))
        one_minus_multiplier = Decimal("1") - multiplier

        start_index = max(self._slow_period, end_index - self._signal_period + 1)

        total = Decimal("0")
        count = 0
        i = start_index
        while i < start_index + self._signal_period and i <= end_index:
            total += self._calculate_macd(data, i)
            count += 1
            i += 1

        if count == 0:
            return Decimal("0")

        signal_ema = (total / Decimal(str(count))).quantize(
            Decimal(10) ** -_SCALE, rounding=ROUND_HALF_UP
        )

        for j in range(start_index + count, end_index + 1):
            macd_val = self._calculate_macd(data, j)
            signal_ema = (
                macd_val * multiplier + signal_ema * one_minus_multiplier
            ).quantize(Decimal(10) ** -_SCALE, rounding=ROUND_HALF_UP)

        return signal_ema
