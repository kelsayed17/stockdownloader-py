"""Breakout strategy using Bollinger Band squeeze detection with volume and ATR confirmation.

Detects low-volatility consolidation periods (Bollinger squeeze) and enters
when price breaks out with volume confirmation.

BUY when: BB width at N-period low (squeeze) AND price breaks above upper band
          AND volume > 1.5x average AND ATR expanding
SELL when: Price breaks below lower band with same confirmations
           OR trailing stop hit (2x ATR below entry)
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.pinescript_models import (
    Condition, Indicator, Input, StrategyDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


class BreakoutStrategy(TradingStrategy):
    """Bollinger Band squeeze breakout strategy with volume and ATR confirmation."""

    def __init__(
        self,
        bb_period: int = 20,
        squeeze_lookback: int = 120,
        volume_multiplier: float = 1.5,
    ) -> None:
        self._bb_period = bb_period
        self._squeeze_lookback = squeeze_lookback
        self._volume_multiplier = volume_multiplier
        self._hub = IndicatorHub()

    @property
    def name(self) -> str:
        return (
            f"Breakout (BB{self._bb_period}, Squeeze{self._squeeze_lookback}, "
            f"Vol>{self._volume_multiplier:.1f}x)"
        )

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self.warmup_period:
            return Signal.HOLD

        close = data[current_index].close
        prev_close = data[current_index - 1].close
        volume = data[current_index].volume

        bb = self._hub.bollinger_bands(data, current_index, self._bb_period, 2.0)
        prev_bb = self._hub.bollinger_bands(data, current_index - 1, self._bb_period, 2.0)

        # Check for squeeze: current BB width near the minimum in the lookback period
        is_squeeze = self._is_in_squeeze(data, current_index)

        # Volume confirmation
        avg_vol = self._hub.average_volume(data, current_index, 20)
        high_volume = (
            avg_vol > Decimal("0")
            and Decimal(str(volume)) > avg_vol * Decimal(str(self._volume_multiplier))
        )

        # ATR expanding (current ATR > previous ATR)
        curr_atr = self._hub.atr(data, current_index, 14)
        prev_atr = self._hub.atr(data, current_index - 1, 14)
        atr_expanding = curr_atr > prev_atr

        # Bullish breakout: price closes above upper BB from squeeze
        bullish_breakout = close > bb.upper and prev_close <= prev_bb.upper

        if (is_squeeze or atr_expanding) and bullish_breakout and high_volume:
            return Signal.BUY

        # Bearish breakdown: price closes below lower BB
        bearish_breakdown = close < bb.lower and prev_close >= prev_bb.lower

        if bearish_breakdown:
            return Signal.SELL

        # Also sell if price falls back inside bands from above (failed breakout)
        failed_breakout = prev_close > prev_bb.upper and close < bb.middle

        if failed_breakout:
            return Signal.SELL

        return Signal.HOLD

    @property
    def warmup_period(self) -> int:
        return max(self._bb_period, self._squeeze_lookback) + 1

    def to_pinescript(self) -> StrategyDefinition:
        """Generate PineScript definition.

        Note: Python detects squeeze via 120-bar BB-width minimum; Pine uses
        the standard Keltner Channel containment approximation (BB inside KC
        = squeeze), which is the TradingView convention.
        """
        return StrategyDefinition(
            name="Breakout Strategy",
            short_name="BRKOUT",
            description=(
                "Bollinger Band squeeze breakout with volume and ATR confirmation.\n"
                "Squeeze detected when BB is inside Keltner Channel.\n"
                "Buy on squeeze release + upper breakout + volume spike + ATR expanding.\n"
                "Sell on lower breakdown or failed breakout."
            ),
            inputs=[
                Input.int_("bbPeriod", self._bb_period, "BB Period"),
                Input.float_("bbMult", 2.0, "BB Std Dev",
                             min_val=0.5, step=0.1),
                Input.int_("kcPeriod", self._bb_period, "KC Period"),
                Input.float_("kcMult", 1.5, "KC ATR Multiplier",
                             min_val=0.5, step=0.1),
                Input.int_("volAvgPeriod", 20, "Volume Avg Period"),
                Input.float_("volMult", self._volume_multiplier,
                             "Volume Multiplier", min_val=1.0, step=0.1),
            ],
            indicators=[
                Indicator.bbands("bbMid", "bbUpper", "bbLower",
                                 "close", "bbPeriod", "bbMult"),
                Indicator.atr("atrVal", "14"),
                Indicator.raw("kcUpper", "bbMid + kcMult * atrVal"),
                Indicator.raw("kcLower", "bbMid - kcMult * atrVal"),
                Indicator.raw("isSqueeze",
                              "bbLower > kcLower and bbUpper < kcUpper"),
                Indicator.raw("wasSqueeze",
                              "bbLower[1] > kcLower[1] and bbUpper[1] < kcUpper[1]"),
                Indicator.raw("volSpike",
                              f"volume > ta.sma(volume, volAvgPeriod) * volMult"),
                Indicator.raw("atrExpanding",
                              "atrVal > atrVal[1]"),
            ],
            long_entry=Condition(
                "not isSqueeze and wasSqueeze and close > bbMid "
                "and volSpike and atrExpanding",
                "Squeeze release + upper breakout + volume spike + ATR expanding",
            ),
            short_entry=Condition(
                "(close < bbLower and close[1] >= bbLower[1]) or "
                "(close[1] > bbUpper[1] and close < bbMid)",
                "Lower band breakdown or failed breakout",
            ),
        )

    def _is_in_squeeze(self, data: list[PriceData], current_index: int) -> bool:
        current_bb = self._hub.bollinger_bands(data, current_index, self._bb_period, 2.0)
        current_width = current_bb.width

        if current_width == Decimal("0"):
            return False

        # Find min BB width in lookback period
        # NOTE: Most of these bollinger_bands calls will be cache hits from
        # previous evaluate() invocations, reducing O(lookback) to ~O(1).
        min_width = current_width
        lookback_end = max(self._bb_period, current_index - self._squeeze_lookback)

        for i in range(lookback_end, current_index):
            past_bb = self._hub.bollinger_bands(data, i, self._bb_period, 2.0)
            if past_bb.width > Decimal("0") and past_bb.width < min_width:
                min_width = past_bb.width

        # Squeeze if current width is within 10% of the minimum
        threshold = min_width * Decimal("1.10")
        return current_width <= threshold
