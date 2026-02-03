"""Multi-indicator confluence strategy.

Scores buy/sell signals across trend, momentum, volatility, and volume
indicator categories.

Each indicator that agrees with a direction adds to the confluence score.
A trade is only taken when the score meets or exceeds the threshold.

Scored indicators (8 total):
  Trend:      EMA(12) > EMA(26), Price > SMA(200), Ichimoku above cloud
  Momentum:   RSI recovery from oversold, MACD bullish crossover, Stochastic oversold
  Volume:     OBV rising, MFI oversold
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.model.indicator_values import IndicatorValues

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


class MultiIndicatorStrategy(TradingStrategy):
    """Multi-indicator confluence strategy with configurable score thresholds."""

    def __init__(self, buy_threshold: int = 4, sell_threshold: int = 4) -> None:
        if buy_threshold < 1 or sell_threshold < 1:
            raise ValueError("Thresholds must be >= 1")
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold

    def get_name(self) -> str:
        return (
            f"Multi-Indicator Confluence "
            f"(Buy>={self._buy_threshold}, Sell>={self._sell_threshold})"
        )

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self.get_warmup_period():
            return Signal.HOLD

        current = IndicatorValues.compute(data, current_index)
        previous = IndicatorValues.compute(data, current_index - 1)

        buy_score = self._compute_buy_score(current, previous)
        sell_score = self._compute_sell_score(current, previous)

        if buy_score >= self._buy_threshold and buy_score > sell_score:
            return Signal.BUY
        if sell_score >= self._sell_threshold and sell_score > buy_score:
            return Signal.SELL

        return Signal.HOLD

    def get_warmup_period(self) -> int:
        return 201  # Need 200 bars for SMA(200) + 1 for crossover

    @staticmethod
    def _compute_buy_score(current: IndicatorValues, previous: IndicatorValues) -> int:
        score = 0

        # Trend: EMA(12) > EMA(26) -- short-term trend bullish
        if current.ema12 > current.ema26:
            score += 1

        # Trend: Price > SMA(200) -- long-term uptrend
        if current.sma200 > Decimal("0") and current.close > current.sma200:
            score += 1

        # Trend: Ichimoku -- price above cloud
        if current.price_above_cloud:
            score += 1

        # Momentum: RSI recovers from oversold (crosses above 30)
        if float(current.rsi14) > 30 and float(previous.rsi14) <= 30:
            score += 1

        # Momentum: MACD bullish crossover
        if (
            current.macd_line > current.macd_signal
            and previous.macd_line <= previous.macd_signal
        ):
            score += 1

        # Momentum: Stochastic %K crossing above %D from oversold zone
        if (
            float(current.stoch_k) < 30
            and current.stoch_k > current.stoch_d
            and previous.stoch_k <= previous.stoch_d
        ):
            score += 1

        # Volume: OBV rising (accumulation)
        if current.obv_rising:
            score += 1

        # Volume: MFI oversold recovery (< 20 or recovering from < 20)
        if float(current.mfi14) < 30 and float(current.mfi14) > float(previous.mfi14):
            score += 1

        return score

    @staticmethod
    def _compute_sell_score(current: IndicatorValues, previous: IndicatorValues) -> int:
        score = 0

        # Trend: EMA(12) < EMA(26) -- short-term trend bearish
        if current.ema12 < current.ema26:
            score += 1

        # Trend: Price < SMA(200) -- long-term downtrend
        if current.sma200 > Decimal("0") and current.close < current.sma200:
            score += 1

        # Trend: Ichimoku -- price below cloud
        if not current.price_above_cloud and current.ichimoku_span_a > Decimal("0"):
            score += 1

        # Momentum: RSI falls from overbought (crosses below 70)
        if float(current.rsi14) < 70 and float(previous.rsi14) >= 70:
            score += 1

        # Momentum: MACD bearish crossover
        if (
            current.macd_line < current.macd_signal
            and previous.macd_line >= previous.macd_signal
        ):
            score += 1

        # Momentum: Stochastic %K crossing below %D from overbought zone
        if (
            float(current.stoch_k) > 70
            and current.stoch_k < current.stoch_d
            and previous.stoch_k >= previous.stoch_d
        ):
            score += 1

        # Volume: OBV falling (distribution)
        if not current.obv_rising:
            score += 1

        # Volume: MFI overbought reversal (> 80 or falling from > 80)
        if float(current.mfi14) > 70 and float(current.mfi14) < float(previous.mfi14):
            score += 1

        return score
