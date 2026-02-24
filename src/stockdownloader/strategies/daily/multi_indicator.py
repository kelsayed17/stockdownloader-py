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

from stockdownloader.strategies.base import Signal, TradingStrategy
from stockdownloader.core.models.indicator import IndicatorValues
from stockdownloader.indicators.hub import IndicatorHub
from stockdownloader.pinescript.models import (
    Condition, Indicator, Input, StrategyDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.core.models.price import PriceData


class MultiIndicatorStrategy(TradingStrategy):
    """Multi-indicator confluence strategy with configurable score thresholds."""

    def __init__(self, buy_threshold: int = 4, sell_threshold: int = 4, hub: IndicatorHub | None = None) -> None:
        if buy_threshold < 1 or sell_threshold < 1:
            raise ValueError("Thresholds must be >= 1")
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._hub = hub or IndicatorHub()
        # Cache the last computed IndicatorValues to avoid recomputing 28
        # indicators for the previous bar on every call.
        self._cached_index: int = -1
        self._cached_values: IndicatorValues | None = None

    @property
    def name(self) -> str:
        return (
            f"Multi-Indicator Confluence "
            f"(Buy>={self._buy_threshold}, Sell>={self._sell_threshold})"
        )

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self.warmup_period:
            return Signal.HOLD

        # Reuse the cached snapshot for ``current_index - 1`` when possible.
        # On sequential calls (the normal backtest path) this avoids
        # recomputing all 28 indicators for the previous bar.
        prev_index = current_index - 1
        if self._cached_index == prev_index and self._cached_values is not None:
            previous = self._cached_values
        else:
            previous = IndicatorValues.compute(data, prev_index, self._hub)

        current = IndicatorValues.compute(data, current_index, self._hub)
        self._cached_index = current_index
        self._cached_values = current

        buy_score, buy_cats = self._compute_buy_score(current, previous)
        sell_score, sell_cats = self._compute_sell_score(current, previous)

        # Anti-overfitting: require signals from at least 2 of 3 categories
        # (trend, momentum, volume).  This prevents correlated trend
        # indicators from single-handedly triggering a trade.
        if buy_score >= self._buy_threshold and buy_score > sell_score and buy_cats >= 2:
            return Signal.BUY
        if sell_score >= self._sell_threshold and sell_score > buy_score and sell_cats >= 2:
            return Signal.SELL

        return Signal.HOLD

    @property
    def warmup_period(self) -> int:
        return 201  # Need 200 bars for SMA(200) + 1 for crossover

    def to_pinescript(self) -> StrategyDefinition:
        # Build scoring logic as extra_code
        score_code = [
            "// --- Buy/Sell confluence scoring ---",
            "int buyScore = 0",
            "int sellScore = 0",
            "",
            "// Trend: EMA(12) vs EMA(26)",
            "if ema12 > ema26",
            "    buyScore += 1",
            "if ema12 < ema26",
            "    sellScore += 1",
            "",
            "// Trend: Price vs SMA(200)",
            "if sma200 > 0 and close > sma200",
            "    buyScore += 1",
            "if sma200 > 0 and close < sma200",
            "    sellScore += 1",
            "",
            "// Trend: Ichimoku cloud",
            "if aboveCloud",
            "    buyScore += 1",
            "if not aboveCloud and senkouA > 0",
            "    sellScore += 1",
            "",
            "// Momentum: RSI crosses above 30 / below 70",
            "if rsiVal > 30 and rsiVal[1] <= 30",
            "    buyScore += 1",
            "if rsiVal < 70 and rsiVal[1] >= 70",
            "    sellScore += 1",
            "",
            "// Momentum: MACD crossover",
            "if macdLine > macdSignal and macdLine[1] <= macdSignal[1]",
            "    buyScore += 1",
            "if macdLine < macdSignal and macdLine[1] >= macdSignal[1]",
            "    sellScore += 1",
            "",
            "// Momentum: Stochastic from oversold/overbought",
            "if stochK < 30 and stochK > stochD and stochK[1] <= stochD[1]",
            "    buyScore += 1",
            "if stochK > 70 and stochK < stochD and stochK[1] >= stochD[1]",
            "    sellScore += 1",
            "",
            "// Volume: OBV rising/falling",
            "float obvEma = ta.ema(obvVal, 5)",
            "if obvEma > obvEma[1]",
            "    buyScore += 1",
            "if obvEma <= obvEma[1]",
            "    sellScore += 1",
            "",
            "// Volume: MFI oversold recovery / overbought reversal",
            "if mfiVal < 30 and mfiVal > mfiVal[1]",
            "    buyScore += 1",
            "if mfiVal > 70 and mfiVal < mfiVal[1]",
            "    sellScore += 1",
        ]

        return StrategyDefinition(
            name="Multi-Indicator Confluence",
            short_name="MULTI",
            description=(
                "8-indicator confluence scoring strategy.\n"
                "Scores EMA trend, SMA(200), Ichimoku, RSI, MACD,\n"
                "Stochastic, OBV, and MFI.\n"
                "Buy/Sell when score meets threshold and exceeds opposite."
            ),
            inputs=[
                Input.int_("buyThreshold", self._buy_threshold,
                           "Buy Score Threshold"),
                Input.int_("sellThreshold", self._sell_threshold,
                           "Sell Score Threshold"),
            ],
            indicators=[
                Indicator.ema("ema12", "close", "12", plot=False),
                Indicator.ema("ema26", "close", "26", plot=False),
                Indicator.sma("sma200", "close", "200",
                              plot=True, color="color.gray"),
                *Indicator.ichimoku(),
                Indicator.rsi("rsiVal", "close", "14"),
                *Indicator.macd("12", "26", "9"),
                *Indicator.stochastic(),
                Indicator.obv("obvVal"),
                Indicator.mfi("mfiVal", "14"),
            ],
            extra_code=score_code,
            long_entry=Condition(
                "buyScore >= buyThreshold and buyScore > sellScore",
                "Confluence score meets buy threshold and exceeds sell score",
            ),
            short_entry=Condition(
                "sellScore >= sellThreshold and sellScore > buyScore",
                "Confluence score meets sell threshold and exceeds buy score",
            ),
        )

    @staticmethod
    def _compute_buy_score(
        current: IndicatorValues, previous: IndicatorValues,
    ) -> tuple[int, int]:
        """Compute buy confluence score with category caps.

        Returns (total_score, categories_active) where categories_active is
        the number of distinct indicator categories (trend/momentum/volume)
        that contributed at least one point.

        Category caps prevent correlated trend indicators (EMA, SMA200,
        Ichimoku) from dominating the score — trend is capped at 2 points.
        """
        trend = 0
        momentum = 0
        volume = 0

        # Trend: EMA(12) > EMA(26) -- short-term trend bullish
        if current.ema12 > current.ema26:
            trend += 1

        # Trend: Price > SMA(200) -- long-term uptrend
        if current.sma200 > Decimal("0") and current.close > current.sma200:
            trend += 1

        # Trend: Ichimoku -- price above cloud
        if current.price_above_cloud:
            trend += 1

        # Cap trend at 2 — EMA/SMA200/Ichimoku are correlated
        trend = min(trend, 2)

        # Momentum: RSI recovers from oversold (crosses above 30)
        if float(current.rsi14) > 30 and float(previous.rsi14) <= 30:
            momentum += 1

        # Momentum: MACD bullish crossover
        if (
            current.macd_line > current.macd_signal
            and previous.macd_line <= previous.macd_signal
        ):
            momentum += 1

        # Momentum: Stochastic %K crossing above %D from oversold zone
        if (
            float(current.stoch_k) < 30
            and current.stoch_k > current.stoch_d
            and previous.stoch_k <= previous.stoch_d
        ):
            momentum += 1

        # Volume: OBV rising (accumulation)
        if current.obv_rising:
            volume += 1

        # Volume: MFI oversold recovery (< 20 or recovering from < 20)
        if float(current.mfi14) < 30 and float(current.mfi14) > float(previous.mfi14):
            volume += 1

        total = trend + momentum + volume
        cats = (1 if trend > 0 else 0) + (1 if momentum > 0 else 0) + (1 if volume > 0 else 0)
        return total, cats

    @staticmethod
    def _compute_sell_score(
        current: IndicatorValues, previous: IndicatorValues,
    ) -> tuple[int, int]:
        """Compute sell confluence score with category caps.

        Returns (total_score, categories_active).  See :meth:`_compute_buy_score`.
        """
        trend = 0
        momentum = 0
        volume = 0

        # Trend: EMA(12) < EMA(26) -- short-term trend bearish
        if current.ema12 < current.ema26:
            trend += 1

        # Trend: Price < SMA(200) -- long-term downtrend
        if current.sma200 > Decimal("0") and current.close < current.sma200:
            trend += 1

        # Trend: Ichimoku -- price below cloud
        if not current.price_above_cloud and current.ichimoku_span_a > Decimal("0"):
            trend += 1

        # Cap trend at 2 — correlated indicators
        trend = min(trend, 2)

        # Momentum: RSI falls from overbought (crosses below 70)
        if float(current.rsi14) < 70 and float(previous.rsi14) >= 70:
            momentum += 1

        # Momentum: MACD bearish crossover
        if (
            current.macd_line < current.macd_signal
            and previous.macd_line >= previous.macd_signal
        ):
            momentum += 1

        # Momentum: Stochastic %K crossing below %D from overbought zone
        if (
            float(current.stoch_k) > 70
            and current.stoch_k < current.stoch_d
            and previous.stoch_k >= previous.stoch_d
        ):
            momentum += 1

        # Volume: OBV falling (distribution)
        if not current.obv_rising:
            volume += 1

        # Volume: MFI overbought reversal (> 80 or falling from > 80)
        if float(current.mfi14) > 70 and float(current.mfi14) < float(previous.mfi14):
            volume += 1

        total = trend + momentum + volume
        cats = (1 if trend > 0 else 0) + (1 if momentum > 0 else 0) + (1 if volume > 0 else 0)
        return total, cats
