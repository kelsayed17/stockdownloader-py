"""Trade imported from TradingView for exit tournament evaluation."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.model.trade import Direction
from stockdownloader.util.big_decimal_math import HUNDRED, ZERO

class TournamentTrade:
    """Represents a trade imported from TradingView for the exit tournament.

    Stores the original entry/exit parameters plus the initial stop level so
    that R-multiples can be computed.  Unlike the regular :class:`Trade` model
    this class is read-only and focused on analysis rather than position
    management.
    """

    def __init__(
        self,
        trade_id: int,
        direction: Direction,
        signal_type: str,
        entry_datetime: str,
        exit_datetime: str,
        entry_price: Decimal,
        original_exit_price: Decimal,
        original_pnl: Decimal,
        stop_distance: Decimal,
        shares: int = 1,
    ) -> None:
        if direction is None:
            raise ValueError("direction must not be None")
        if entry_datetime is None:
            raise ValueError("entry_datetime must not be None")
        if entry_price is None or entry_price <= ZERO:
            raise ValueError("entry_price must be positive")
        if stop_distance is None or stop_distance <= ZERO:
            raise ValueError("stop_distance must be positive")
        if shares <= 0:
            raise ValueError("shares must be positive")

        self._trade_id = trade_id
        self._direction = direction
        self._signal_type = signal_type
        self._entry_datetime = entry_datetime
        self._exit_datetime = exit_datetime
        self._entry_price = entry_price
        self._original_exit_price = original_exit_price
        self._original_pnl = original_pnl
        self._stop_distance = stop_distance
        self._shares = shares

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trade_id(self) -> int:
        return self._trade_id

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def signal_type(self) -> str:
        return self._signal_type

    @property
    def entry_datetime(self) -> str:
        return self._entry_datetime

    @property
    def exit_datetime(self) -> str:
        return self._exit_datetime

    @property
    def entry_price(self) -> Decimal:
        return self._entry_price

    @property
    def original_exit_price(self) -> Decimal:
        return self._original_exit_price

    @property
    def original_pnl(self) -> Decimal:
        return self._original_pnl

    @property
    def stop_distance(self) -> Decimal:
        """The 1R distance: ``|entry_price - initial_stop|``."""
        return self._stop_distance

    @property
    def shares(self) -> int:
        return self._shares

    @property
    def entry_trading_date(self) -> str:
        """Calendar date portion of entry datetime (``YYYY-MM-DD``)."""
        return self._entry_datetime[:10]

    @property
    def stop_price(self) -> Decimal:
        """The initial hard-stop price implied by the stop distance."""
        if self._direction == Direction.LONG:
            return self._entry_price - self._stop_distance
        return self._entry_price + self._stop_distance

    @property
    def original_r_multiple(self) -> Decimal:
        """How many R the original trade captured."""
        if self._stop_distance == ZERO:
            return ZERO
        return (self._original_pnl / self._stop_distance).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

    def __str__(self) -> str:
        return (
            f"TournamentTrade #{self._trade_id} "
            f"{self._direction.value} {self._signal_type}: "
            f"entry {self._entry_datetime} @ ${self._entry_price} "
            f"stop_dist=${self._stop_distance} "
            f"original_pnl=${self._original_pnl}"
        )

    def __repr__(self) -> str:
        return (
            f"TournamentTrade(trade_id={self._trade_id}, "
            f"direction={self._direction!r}, "
            f"signal_type={self._signal_type!r}, "
            f"entry_price={self._entry_price})"
        )
