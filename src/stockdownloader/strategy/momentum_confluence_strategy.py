"""Momentum/trend-following strategy combining MACD, ADX, and EMA.

BUY when: MACD bullish crossover AND ADX > 25 (strong trend) AND Price > EMA(200)
SELL when: MACD bearish crossover OR ADX falls below 20 (trend weakening)

Also uses OBV as volume confirmation: only enter if OBV is rising (accumulation).
Position sizing should be scaled by ATR externally (smaller in volatile markets).
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.util.moving_average_calculator import ema
from stockdownloader.util.technical_indicators import (
    adx,
    is_obv_rising,
    macd_line,
    macd_signal,
)

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


class MomentumConfluenceStrategy(TradingStrategy):
    """Momentum confluence strategy using MACD + ADX + EMA + OBV."""

    def __init__(
        self,
        fast_ema: int = 12,
        slow_ema: int = 26,
        signal_period: int = 9,
        ema_trend_filter: int = 200,
        adx_strength_threshold: float = 25,
        adx_weak_threshold: float = 20,
    ) -> None:
        self._fast_ema = fast_ema
        self._slow_ema = slow_ema
        self._signal_period = signal_period
        self._ema_trend_filter = ema_trend_filter
        self._adx_strength_threshold = adx_strength_threshold
        self._adx_weak_threshold = adx_weak_threshold

    def get_name(self) -> str:
        return (
            f"Momentum (MACD {self._fast_ema}/{self._slow_ema}/{self._signal_period} "
            f"+ ADX>{self._adx_strength_threshold:.0f} + EMA{self._ema_trend_filter})"
        )

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self.get_warmup_period():
            return Signal.HOLD

        close = data[current_index].close

        # MACD crossover detection
        curr_macd_line = macd_line(data, current_index, self._fast_ema, self._slow_ema)
        curr_macd_signal = macd_signal(
            data, current_index, self._fast_ema, self._slow_ema, self._signal_period
        )
        prev_macd_line = macd_line(data, current_index - 1, self._fast_ema, self._slow_ema)
        prev_macd_signal = macd_signal(
            data, current_index - 1, self._fast_ema, self._slow_ema, self._signal_period
        )

        macd_bullish_cross = (
            curr_macd_line > curr_macd_signal
            and prev_macd_line <= prev_macd_signal
        )
        macd_bearish_cross = (
            curr_macd_line < curr_macd_signal
            and prev_macd_line >= prev_macd_signal
        )

        # ADX for trend strength
        adx_result = adx(data, current_index)
        strong_trend = float(adx_result.adx) > self._adx_strength_threshold
        weak_trend = float(adx_result.adx) < self._adx_weak_threshold
        bullish_di = adx_result.plus_di > adx_result.minus_di

        # EMA trend filter
        ema_val = ema(data, current_index, self._ema_trend_filter)
        above_trend_ema = close > ema_val

        # OBV confirmation
        obv_confirm = is_obv_rising(data, current_index, 5)

        # BUY: MACD bullish crossover + strong uptrend + above EMA + volume confirmation
        if macd_bullish_cross and strong_trend and bullish_di and above_trend_ema and obv_confirm:
            return Signal.BUY

        # SELL: MACD bearish crossover OR trend weakening
        if macd_bearish_cross or (weak_trend and not above_trend_ema):
            return Signal.SELL

        return Signal.HOLD

    def get_warmup_period(self) -> int:
        return max(self._ema_trend_filter, self._slow_ema + self._signal_period) + 1
