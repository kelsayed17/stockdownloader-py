"""Protective put strategy: buy OTM puts to hedge a long stock position.

Provides downside protection at the cost of the put premium.

Entry signal: price drops below moving average or momentum weakens.
Exit signal: price recovers above MA or put approaches expiration with no value.

Strike selection: buys puts at a configurable percentage below current price.
Expiration: targets a configurable number of days out (typically 30-60 DTE).
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR
from typing import TYPE_CHECKING

from stockdownloader.model.option_type import OptionType
from stockdownloader.strategy.options_strategy import OptionsSignal, OptionsStrategy
from stockdownloader.util.moving_average_calculator import sma

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


class ProtectivePutStrategy(OptionsStrategy):
    """Protective put strategy that buys OTM puts for downside protection."""

    def __init__(
        self,
        ma_period: int = 20,
        otm_percent: Decimal | None = None,
        days_to_expiry: int = 45,
        momentum_lookback: int = 5,
    ) -> None:
        """Initialize the protective put strategy.

        Args:
            ma_period: Moving average period for trend detection.
            otm_percent: Percentage OTM for strike selection (e.g., 0.05 = 5% OTM).
            days_to_expiry: Target days to expiration.
            momentum_lookback: Bars to look back for momentum calculation.
        """
        if ma_period <= 0:
            raise ValueError("ma_period must be positive")
        if days_to_expiry <= 0:
            raise ValueError("days_to_expiry must be positive")
        if momentum_lookback <= 0:
            raise ValueError("momentum_lookback must be positive")
        self._ma_period = ma_period
        self._otm_percent = otm_percent if otm_percent is not None else Decimal("0.05")
        self._days_to_expiry = days_to_expiry
        self._momentum_lookback = momentum_lookback

    def get_name(self) -> str:
        otm_display = (self._otm_percent * Decimal("100")).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
        return f"Protective Put (MA{self._ma_period}, {otm_display}% OTM, {self._days_to_expiry}DTE)"

    def evaluate(self, data: list[PriceData], current_index: int) -> OptionsSignal:
        if current_index < max(self._ma_period, self._momentum_lookback):
            return OptionsSignal.HOLD

        current_price = data[current_index].close
        ma = sma(data, current_index, self._ma_period)

        # Calculate momentum: percentage change over lookback period
        lookback_price = data[current_index - self._momentum_lookback].close
        momentum = Decimal("0")
        if lookback_price > Decimal("0"):
            momentum = ((current_price - lookback_price) / lookback_price).quantize(
                Decimal(10) ** -6, rounding=ROUND_HALF_UP
            )

        # Open: price crosses below MA or momentum turns negative
        price_below_ma = current_price < ma
        prev_above_ma = data[current_index - 1].close >= sma(
            data, current_index - 1, self._ma_period
        )
        negative_momentum = momentum < Decimal("-0.02")

        if price_below_ma and prev_above_ma:
            return OptionsSignal.OPEN
        if negative_momentum and price_below_ma:
            return OptionsSignal.OPEN

        # Close: price recovers above MA (protection no longer needed)
        if current_price > ma:
            pct_above_ma = ((current_price - ma) / ma).quantize(
                Decimal(10) ** -6, rounding=ROUND_HALF_UP
            )
            if pct_above_ma > Decimal("0.02"):
                return OptionsSignal.CLOSE

        return OptionsSignal.HOLD

    def get_option_type(self) -> OptionType:
        return OptionType.PUT

    def is_short(self) -> bool:
        return False

    def get_target_strike(self, current_price: Decimal) -> Decimal:
        return (current_price * (Decimal("1") - self._otm_percent)).quantize(
            Decimal("1"), rounding=ROUND_FLOOR
        )

    def get_target_days_to_expiry(self) -> int:
        return self._days_to_expiry

    def get_warmup_period(self) -> int:
        return max(self._ma_period, self._momentum_lookback)
