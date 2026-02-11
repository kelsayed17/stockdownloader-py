"""Mean reversion strategy combining Bollinger Bands, RSI, and Stochastic Oscillator.

BUY when: Price touches lower Bollinger Band AND RSI < oversold AND Stochastic %K < 20
SELL when: Price touches upper Bollinger Band AND RSI > overbought AND Stochastic %K > 80

Uses ADX as a trend filter: only trades when ADX < 25 (range-bound market).
This prevents mean reversion trades during strong trends where price can continue
moving in one direction for extended periods.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.util.technical_indicators import (
    adx,
    bollinger_bands,
    rsi,
    stochastic,
)

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


class BollingerBandRSIStrategy(TradingStrategy):
    """BB + RSI mean-reversion strategy with ADX trend filter."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        adx_threshold: float = 25,
    ) -> None:
        self._bb_period = bb_period
        self._bb_std_dev = bb_std_dev
        self._rsi_period = rsi_period
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._adx_threshold = adx_threshold

    def get_name(self) -> str:
        return (
            f"BB+RSI Mean Reversion (BB{self._bb_period}, RSI{self._rsi_period} "
            f"[{self._rsi_oversold:.0f}/{self._rsi_overbought:.0f}], "
            f"ADX<{self._adx_threshold:.0f})"
        )

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self.get_warmup_period():
            return Signal.HOLD

        close = data[current_index].close
        bb = bollinger_bands(data, current_index, self._bb_period, self._bb_std_dev)
        rsi_val = rsi(data, current_index, self._rsi_period)
        stoch = stochastic(data, current_index)
        adx_result = adx(data, current_index)

        # Only trade in range-bound markets (ADX < threshold)
        is_range_bound = float(adx_result.adx) < self._adx_threshold

        # Previous values for crossover detection
        prev_close = data[current_index - 1].close
        prev_rsi = rsi(data, current_index - 1, self._rsi_period)

        # BUY: Price at/below lower BB, RSI crosses above oversold, Stochastic < 20
        price_at_lower_bb = (
            close <= bb.lower
            or (prev_close < bb.lower and close >= bb.lower)
        )
        rsi_recovering = (
            float(rsi_val) > self._rsi_oversold
            and float(prev_rsi) <= self._rsi_oversold
        )
        stoch_oversold = float(stoch.percent_k) < 20

        if is_range_bound and (price_at_lower_bb or rsi_recovering) and stoch_oversold:
            return Signal.BUY

        # SELL: Price at/above upper BB, RSI crosses below overbought, Stochastic > 80
        price_at_upper_bb = (
            close >= bb.upper
            or (prev_close > bb.upper and close <= bb.upper)
        )
        rsi_topping = (
            float(rsi_val) < self._rsi_overbought
            and float(prev_rsi) >= self._rsi_overbought
        )
        stoch_overbought = float(stoch.percent_k) > 80

        if is_range_bound and (price_at_upper_bb or rsi_topping) and stoch_overbought:
            return Signal.SELL

        return Signal.HOLD

    def get_warmup_period(self) -> int:
        return max(self._bb_period, max(self._rsi_period + 1, 28))
