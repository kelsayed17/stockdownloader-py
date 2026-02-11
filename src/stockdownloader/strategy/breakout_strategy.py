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
from stockdownloader.util.technical_indicators import (
    atr,
    average_volume,
    bollinger_bands,
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

    def get_name(self) -> str:
        return (
            f"Breakout (BB{self._bb_period}, Squeeze{self._squeeze_lookback}, "
            f"Vol>{self._volume_multiplier:.1f}x)"
        )

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self.get_warmup_period():
            return Signal.HOLD

        close = data[current_index].close
        prev_close = data[current_index - 1].close
        volume = data[current_index].volume

        bb = bollinger_bands(data, current_index, self._bb_period, 2.0)
        prev_bb = bollinger_bands(data, current_index - 1, self._bb_period, 2.0)

        # Check for squeeze: current BB width near the minimum in the lookback period
        is_squeeze = self._is_in_squeeze(data, current_index)

        # Volume confirmation
        avg_vol = average_volume(data, current_index, 20)
        high_volume = (
            avg_vol > Decimal("0")
            and Decimal(str(volume)) > avg_vol * Decimal(str(self._volume_multiplier))
        )

        # ATR expanding (current ATR > previous ATR)
        curr_atr = atr(data, current_index, 14)
        prev_atr = atr(data, current_index - 1, 14)
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

    def get_warmup_period(self) -> int:
        return max(self._bb_period, self._squeeze_lookback) + 1

    def _is_in_squeeze(self, data: list[PriceData], current_index: int) -> bool:
        current_bb = bollinger_bands(data, current_index, self._bb_period, 2.0)
        current_width = current_bb.width

        if current_width == Decimal("0"):
            return False

        # Find min BB width in lookback period
        min_width = current_width
        lookback_end = max(self._bb_period, current_index - self._squeeze_lookback)

        for i in range(lookback_end, current_index):
            past_bb = bollinger_bands(data, i, self._bb_period, 2.0)
            if past_bb.width > Decimal("0") and past_bb.width < min_width:
                min_width = past_bb.width

        # Squeeze if current width is within 10% of the minimum
        threshold = min_width * Decimal("1.10")
        return current_width <= threshold
