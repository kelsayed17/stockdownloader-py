"""Fixed R-based trailing stop exit mechanism."""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import TrailingExitBase

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade


class TrailingStopExit(TrailingExitBase):
    """0.5R trailing stop from peak after 1R breakeven trigger.

    Args:
        activation_r: R-multiple at which the breakeven stop activates.
        trail_r: R-multiple width of the trailing stop from peak.
        be_buffer_pct: Percentage of 1R to add as a breakeven buffer.
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("1"),
        trail_r: Decimal = Decimal("0.5"),
        be_buffer_pct: Decimal = Decimal("0.05"),
    ) -> None:
        super().__init__()
        self._activation_r = activation_r
        self._trail_r = trail_r
        self._be_buffer_pct = be_buffer_pct

    @property
    def name(self) -> str:
        return "CURRENT_TRAIL"

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        stop_dist = trade.stop_distance
        return self._evaluate_trail(
            bar=bar,
            direction=trade.direction,
            entry=trade.entry_price,
            stop_dist=stop_dist,
            trail_buffer=stop_dist * self._trail_r,
            activation_r=self._activation_r,
            be_buffer=stop_dist * self._be_buffer_pct,
        )
