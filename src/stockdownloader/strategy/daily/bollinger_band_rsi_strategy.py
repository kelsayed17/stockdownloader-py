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
from stockdownloader.util.indicators.hub import IndicatorHub
from stockdownloader.util.pinescript.models import (
    Condition, Indicator, Input, StrategyDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.core.models.price import PriceData


class BollingerBandRSIStrategy(TradingStrategy):
    """BB + RSI mean-reversion strategy with ADX trend filter and regime detection.

    Anti-overfitting improvements:
    * **AND logic** — both BB touch AND RSI condition must be met (not OR).
      OR logic produced loose entries that fit noise in-sample.
    * **BB width regime filter** — only trades when BB width is in a "normal"
      range.  Extremely tight bands (pre-breakout squeezes) and extremely
      wide bands (high-vol breakouts) are filtered out because mean-reversion
      fails in both regimes.
    * **Confirmation candle** — requires the entry bar to close inside the
      bands (reversal confirmation) rather than just touching.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        adx_threshold: float = 25,
        hub: IndicatorHub | None = None,
    ) -> None:
        self._bb_period = bb_period
        self._bb_std_dev = bb_std_dev
        self._rsi_period = rsi_period
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._adx_threshold = adx_threshold
        self._hub = hub or IndicatorHub()

    @property
    def name(self) -> str:
        return (
            f"BB+RSI Mean Reversion (BB{self._bb_period}, RSI{self._rsi_period} "
            f"[{self._rsi_oversold:.0f}/{self._rsi_overbought:.0f}], "
            f"ADX<{self._adx_threshold:.0f})"
        )

    def _bb_width_ok(self, bb_upper: float, bb_lower: float, bb_mid: float) -> bool:
        """Return True if BB width is in a normal mean-reversion regime.

        Filters out:
        * Squeeze: width < P10 → bands too tight, breakout imminent.
        * Blow-out: width > P95 → bands too wide, trending/volatile.

        Thresholds are calibrated for 5-minute bars where median BB width
        is ~0.4% (vs ~3% on daily).  Hard-coded to prevent optimizer gaming.
        """
        if bb_mid <= 0:
            return False
        width_pct = (bb_upper - bb_lower) / bb_mid * 100.0
        return 0.15 <= width_pct <= 1.5

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self.warmup_period:
            return Signal.HOLD

        close = data[current_index].close
        bb = self._hub.bollinger_bands(data, current_index, self._bb_period, self._bb_std_dev)
        rsi_val = self._hub.rsi(data, current_index, self._rsi_period)
        stoch = self._hub.stochastic(data, current_index)
        adx_result = self._hub.adx(data, current_index)

        # Only trade in range-bound markets (ADX < threshold)
        is_range_bound = float(adx_result.adx) < self._adx_threshold

        # Regime filter: BB width must be in normal range
        if not self._bb_width_ok(float(bb.upper), float(bb.lower), float(bb.middle)):
            return Signal.HOLD

        # Previous values for crossover detection
        prev_close = data[current_index - 1].close
        prev_rsi = self._hub.rsi(data, current_index - 1, self._rsi_period)

        # BUY: Price touched/crossed lower BB AND RSI recovering from oversold
        # AND Stochastic oversold AND close is back inside bands (confirmation)
        price_touched_lower_bb = (
            float(prev_close) <= float(bb.lower)
            or float(data[current_index].low) <= float(bb.lower)
        )
        close_inside_bands = float(close) > float(bb.lower)
        rsi_recovering = (
            float(rsi_val) > self._rsi_oversold
            and float(prev_rsi) <= self._rsi_oversold
        )
        stoch_oversold = float(stoch.percent_k) < 20

        if (
            is_range_bound
            and price_touched_lower_bb
            and close_inside_bands
            and rsi_recovering
            and stoch_oversold
        ):
            return Signal.BUY

        # SELL: Price touched/crossed upper BB AND RSI falling from overbought
        # AND Stochastic overbought AND close is back inside bands
        price_touched_upper_bb = (
            float(prev_close) >= float(bb.upper)
            or float(data[current_index].high) >= float(bb.upper)
        )
        close_inside_bands_sell = float(close) < float(bb.upper)
        rsi_topping = (
            float(rsi_val) < self._rsi_overbought
            and float(prev_rsi) >= self._rsi_overbought
        )
        stoch_overbought = float(stoch.percent_k) > 80

        if (
            is_range_bound
            and price_touched_upper_bb
            and close_inside_bands_sell
            and rsi_topping
            and stoch_overbought
        ):
            return Signal.SELL

        return Signal.HOLD

    @property
    def warmup_period(self) -> int:
        return max(self._bb_period, max(self._rsi_period + 1, 28))

    def to_pinescript(self) -> StrategyDefinition:
        return StrategyDefinition(
            name="BB + RSI Mean Reversion",
            short_name="BB-RSI",
            description=(
                "Mean reversion using Bollinger Bands, RSI, and Stochastic.\n"
                "Trades only in range-bound markets (ADX below threshold).\n"
                "Buy at lower band + RSI oversold + Stoch < 20.\n"
                "Sell at upper band + RSI overbought + Stoch > 80."
            ),
            inputs=[
                Input.int_("bbPeriod", self._bb_period, "BB Period"),
                Input.float_("bbMult", self._bb_std_dev, "BB Std Dev",
                             min_val=0.5, step=0.1),
                Input.int_("rsiPeriod", self._rsi_period, "RSI Period"),
                Input.float_("rsiOS", self._rsi_oversold,
                             "RSI Oversold", step=1.0),
                Input.float_("rsiOB", self._rsi_overbought,
                             "RSI Overbought", step=1.0),
                Input.float_("adxMax", self._adx_threshold,
                             "ADX Max (Range-Bound)", step=1.0),
            ],
            indicators=[
                Indicator.bbands("bbMid", "bbUpper", "bbLower",
                                 "close", "bbPeriod", "bbMult"),
                Indicator.rsi("rsiVal", "close", "rsiPeriod"),
                *Indicator.stochastic(),
                Indicator.dmi("14", "14"),
            ],
            long_entry=Condition(
                "(close <= bbLower or ta.crossover(rsiVal, rsiOS)) "
                "and stochK < 20 and adxValue < adxMax",
                "Price at lower BB or RSI crosses above oversold, "
                "Stochastic oversold, ranging market",
            ),
            short_entry=Condition(
                "(close >= bbUpper or ta.crossunder(rsiVal, rsiOB)) "
                "and stochK > 80 and adxValue < adxMax",
                "Price at upper BB or RSI crosses below overbought, "
                "Stochastic overbought, ranging market",
            ),
        )
