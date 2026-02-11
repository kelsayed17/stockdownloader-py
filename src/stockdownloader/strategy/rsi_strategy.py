"""Relative Strength Index strategy.

Generates BUY when RSI crosses above the oversold threshold
and SELL when RSI crosses below the overbought threshold.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData

_RSI_MAX = Decimal("100")
_CALC_SCALE = 10


class RSIStrategy(TradingStrategy):
    """RSI-based strategy with configurable oversold/overbought thresholds."""

    def __init__(self, period: int, oversold: float, overbought: float) -> None:
        if period <= 0:
            raise ValueError("Period must be positive")
        if oversold < 0 or overbought > 100 or oversold >= overbought:
            raise ValueError(
                "Invalid threshold values: oversold must be < overbought, within [0, 100]"
            )
        self._period = period
        self._oversold_threshold = Decimal(str(oversold))
        self._overbought_threshold = Decimal(str(overbought))

    def get_name(self) -> str:
        return f"RSI ({self._period}) [{self._oversold_threshold}/{self._overbought_threshold}]"

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self._period + 1:
            return Signal.HOLD

        current_rsi = self._calculate_rsi(data, current_index)
        prev_rsi = self._calculate_rsi(data, current_index - 1)

        if current_rsi > self._oversold_threshold and prev_rsi <= self._oversold_threshold:
            return Signal.BUY
        if current_rsi < self._overbought_threshold and prev_rsi >= self._overbought_threshold:
            return Signal.SELL
        return Signal.HOLD

    def get_warmup_period(self) -> int:
        return self._period + 1

    def _calculate_rsi(self, data: list[PriceData], end_index: int) -> Decimal:
        avg_gain = Decimal("0")
        avg_loss = Decimal("0")

        for i in range(end_index - self._period + 1, end_index + 1):
            change = data[i].close - data[i - 1].close
            if change > Decimal("0"):
                avg_gain += change
            else:
                avg_loss += abs(change)

        period_bd = Decimal(str(self._period))
        avg_gain = (avg_gain / period_bd).quantize(
            Decimal(10) ** -_CALC_SCALE, rounding=ROUND_HALF_UP
        )
        avg_loss = (avg_loss / period_bd).quantize(
            Decimal(10) ** -_CALC_SCALE, rounding=ROUND_HALF_UP
        )

        if avg_loss == Decimal("0"):
            return _RSI_MAX

        rs = (avg_gain / avg_loss).quantize(
            Decimal(10) ** -_CALC_SCALE, rounding=ROUND_HALF_UP
        )
        return _RSI_MAX - (
            _RSI_MAX / (Decimal("1") + rs)
        ).quantize(Decimal(10) ** -6, rounding=ROUND_HALF_UP)
