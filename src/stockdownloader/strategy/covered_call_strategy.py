"""Covered call strategy: sell OTM calls against a long stock position.

Generates income from premium collection while capping upside.

Entry signal: price is above the moving average (bullish bias, but not strongly trending)
Exit signal: approaching expiration or price moves significantly against the position

Strike selection: sells calls at a configurable percentage above current price.
Expiration: targets a configurable number of days out (typically 30-45 DTE).
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
from typing import TYPE_CHECKING

from stockdownloader.model.option_type import OptionType
from stockdownloader.strategy.options_strategy import OptionsSignal, OptionsStrategy
from stockdownloader.util.moving_average_calculator import sma

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData


class CoveredCallStrategy(OptionsStrategy):
    """Covered call strategy that sells OTM calls above a moving average."""

    def __init__(
        self,
        ma_period: int = 20,
        otm_percent: Decimal | None = None,
        days_to_expiry: int = 30,
        exit_threshold: Decimal | None = None,
    ) -> None:
        """Initialize the covered call strategy.

        Args:
            ma_period: Moving average period for trend filter.
            otm_percent: Percentage OTM for strike selection (e.g., 0.05 = 5% OTM).
            days_to_expiry: Target days to expiration.
            exit_threshold: Percentage move that triggers early exit (e.g., 0.03 = 3%).
        """
        if ma_period <= 0:
            raise ValueError("ma_period must be positive")
        if days_to_expiry <= 0:
            raise ValueError("days_to_expiry must be positive")
        self._ma_period = ma_period
        self._otm_percent = otm_percent if otm_percent is not None else Decimal("0.05")
        self._days_to_expiry = days_to_expiry
        self._exit_threshold = exit_threshold if exit_threshold is not None else Decimal("0.03")

    def get_name(self) -> str:
        otm_display = (self._otm_percent * Decimal("100")).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
        return f"Covered Call (MA{self._ma_period}, {otm_display}% OTM, {self._days_to_expiry}DTE)"

    def evaluate(self, data: list[PriceData], current_index: int) -> OptionsSignal:
        if current_index < self._ma_period:
            return OptionsSignal.HOLD

        current_price = data[current_index].close
        ma = sma(data, current_index, self._ma_period)

        # Open: price above MA (mild bullish / neutral trend)
        if current_price > ma:
            prev_price = data[current_index - 1].close
            prev_ma = sma(data, current_index - 1, self._ma_period)

            # Only open when price just crossed above MA or is consolidating
            if prev_price <= prev_ma:
                return OptionsSignal.OPEN

        # Close: price drops significantly below MA (trend reversal)
        pct_below_ma = (ma - current_price) / ma
        pct_below_ma = pct_below_ma.quantize(Decimal(10) ** -6, rounding=ROUND_HALF_UP)
        if pct_below_ma > self._exit_threshold:
            return OptionsSignal.CLOSE

        return OptionsSignal.HOLD

    def get_option_type(self) -> OptionType:
        return OptionType.CALL

    def is_short(self) -> bool:
        return True

    def get_target_strike(self, current_price: Decimal) -> Decimal:
        return (current_price * (Decimal("1") + self._otm_percent)).quantize(
            Decimal("1"), rounding=ROUND_CEILING
        )

    def get_target_days_to_expiry(self) -> int:
        return self._days_to_expiry

    def get_warmup_period(self) -> int:
        return self._ma_period
