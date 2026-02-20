"""Trailing stop that tightens by time-of-day."""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import TrailingExitBase

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade


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
