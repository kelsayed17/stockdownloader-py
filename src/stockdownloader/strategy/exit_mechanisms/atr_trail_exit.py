"""Adaptive ATR-based trailing stop exit mechanism."""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import TrailingExitBase
from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.indicator_hub import IndicatorHub

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade


class AtrTrailExit(TrailingExitBase):
    """0.3x ATR(14) trailing stop from peak after activation.

    Args:
        atr_multiplier: Fraction of ATR to use as trail distance.
        activation_r: R-multiple threshold before trailing begins.
        atr_period: ATR lookback period.
    """

    def __init__(
        self,
        atr_multiplier: Decimal = Decimal("0.3"),
        activation_r: Decimal = Decimal("1"),
        atr_period: int = 14,
    ) -> None:
        super().__init__()
        self._atr_multiplier = atr_multiplier
        self._activation_r = activation_r
        self._atr_period = atr_period
        self._hub = IndicatorHub()

    @property
    def name(self) -> str:
        return "ATR_TRAIL"

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        stop_dist = trade.stop_distance
        data_index = self._data_offset + bar_index

        # Compute current ATR for adaptive trail distance
        current_atr = ZERO
        if self._data and data_index >= self._atr_period:
            current_atr = self._hub.atr(self._data, data_index, self._atr_period)
        if current_atr <= ZERO:
            current_atr = stop_dist  # fallback

        trail_dist = current_atr * self._atr_multiplier

        return self._evaluate_trail(
            bar=bar,
            direction=trade.direction,
            entry=trade.entry_price,
            stop_dist=stop_dist,
            trail_buffer=trail_dist,
            activation_r=self._activation_r,
            be_buffer=None,  # ATR trail doesn't set breakeven stop
        )
