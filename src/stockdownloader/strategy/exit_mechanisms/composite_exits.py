"""Composite and time-based exit mechanisms."""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import Direction
from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import TrailingExitBase
from stockdownloader.util.math import ZERO
from stockdownloader.util.indicator_hub import IndicatorHub

if TYPE_CHECKING:
    from stockdownloader.model.price_data import IntradayPriceData
    from stockdownloader.model.trade import TournamentTrade


# ============================================================================
# Time-Based Trailing Stop
# ============================================================================


class TimeDecayExit(TrailingExitBase):
    """Trailing stop that tightens by time-of-day.

    Args:
        activation_r: R-multiple threshold for breakeven trigger.
        early_trail_r: Trail width before 11:00 (default 0.6R).
        mid_trail_r: Trail width 11:00-13:00 (default 0.4R).
        late_trail_r: Trail width after 13:00 (default 0.2R).
        be_buffer_pct: Percentage of 1R added as breakeven buffer.
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("1"),
        early_trail_r: Decimal = Decimal("0.6"),
        mid_trail_r: Decimal = Decimal("0.4"),
        late_trail_r: Decimal = Decimal("0.2"),
        be_buffer_pct: Decimal = Decimal("0.05"),
    ) -> None:
        super().__init__()
        self._activation_r = activation_r
        self._early_trail_r = early_trail_r
        self._mid_trail_r = mid_trail_r
        self._late_trail_r = late_trail_r
        self._be_buffer_pct = be_buffer_pct

    @property
    def name(self) -> str:
        return "TIME_DECAY"

    def _trail_multiplier(self, time_str: str) -> Decimal:
        """Choose trail width based on time of day.

        *time_str* is expected in ``HH:MM:SS`` format.
        """
        hour_str = time_str[:2]
        try:
            hour = int(hour_str)
        except ValueError:
            return self._mid_trail_r

        if hour < 11:
            return self._early_trail_r
        if hour < 13:
            return self._mid_trail_r
        return self._late_trail_r

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        stop_dist = trade.stop_distance
        trail_mult = self._trail_multiplier(bar.time_str)
        return self._evaluate_trail(
            bar=bar,
            direction=trade.direction,
            entry=trade.entry_price,
            stop_dist=stop_dist,
            trail_buffer=stop_dist * trail_mult,
            activation_r=self._activation_r,
            be_buffer=stop_dist * self._be_buffer_pct,
        )


# ============================================================================
# Hybrid ATR + VWAP Exit
# ============================================================================


class HybridExit(TrailingExitBase):
    """ATR trail + VWAP cross, whichever triggers first.

    Delegates activation and ATR trailing logic to the base class via
    :meth:`_evaluate_trail`, and injects VWAP-cross detection through
    :meth:`_post_activation_exit_check`.

    Args:
        activation_r: R-multiple threshold before both trails activate.
        atr_multiplier: Fraction of ATR to use as trail distance.
        atr_period: ATR lookback period.
        name: Display name override.
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("0.5"),
        atr_multiplier: Decimal = Decimal("0.3"),
        atr_period: int = 14,
        name: str | None = None,
        hub: IndicatorHub | None = None,
    ) -> None:
        super().__init__()
        self._activation_r = activation_r
        self._atr_multiplier = atr_multiplier
        self._atr_period = atr_period
        self._name = name or "HYBRID"
        self._hub = hub or IndicatorHub()

    @property
    def name(self) -> str:
        return self._name

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        stop_dist = trade.stop_distance
        data_index = self._data_offset + bar_index

        # Compute adaptive ATR trail distance
        current_atr = ZERO
        if self._data and data_index >= self._atr_period:
            current_atr = self._hub.atr(self._data, data_index, self._atr_period)
        if current_atr <= ZERO:
            current_atr = stop_dist

        trail_dist = current_atr * self._atr_multiplier
        be_buffer = stop_dist * Decimal("0.05")

        return self._evaluate_trail(
            bar=bar,
            direction=trade.direction,
            entry=trade.entry_price,
            stop_dist=stop_dist,
            trail_buffer=trail_dist,
            activation_r=self._activation_r,
            be_buffer=be_buffer,
            data_index=data_index,
        )

    def _post_activation_exit_check(
        self,
        bar: IntradayPriceData,
        direction: Direction,
        data_index: int,
    ) -> tuple[Decimal, str] | None:
        if not self._data:
            return None
        vwap_val = self._hub.session_vwap(self._data, data_index)
        if direction == Direction.LONG and bar.close < vwap_val:
            return (bar.close, "vwap_cross")
        if direction == Direction.SHORT and bar.close > vwap_val:
            return (bar.close, "vwap_cross")
        return None
