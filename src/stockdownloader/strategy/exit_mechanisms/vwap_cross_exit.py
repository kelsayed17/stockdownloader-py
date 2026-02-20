"""Exit on session VWAP cross after activation threshold."""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import Direction
from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import TrailingExitBase
from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.indicator_hub import IndicatorHub

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade


class VwapCrossExit(TrailingExitBase):
    """Exit on VWAP cross after activation threshold.

    Uses the base class activation logic via :meth:`_evaluate_trail` and
    injects VWAP-cross detection through :meth:`_post_activation_exit_check`.

    Args:
        activation_r: R-multiple threshold before VWAP trailing begins.
        name: Display name override (defaults to ``'VWAP_CROSS'``).
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("0.5"),
        name: str | None = None,
        hub: IndicatorHub | None = None,
    ) -> None:
        super().__init__()
        self._activation_r = activation_r
        self._name = name or "VWAP_CROSS"
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
        return self._evaluate_trail(
            bar=bar,
            direction=trade.direction,
            entry=trade.entry_price,
            stop_dist=trade.stop_distance,
            trail_buffer=ZERO,
            activation_r=self._activation_r,
            be_buffer=None,
            data_index=self._data_offset + bar_index,
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
