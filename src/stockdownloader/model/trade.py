"""Trade-related types: positions, directions, and intraday signals."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

from stockdownloader.util.math import HUNDRED, ZERO


class Direction(Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    """Trade status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class Trade:
    """Tracks an individual trade position from entry to exit."""

    def __init__(
        self,
        direction: Direction,
        entry_date: str,
        entry_price: Decimal,
        shares: int,
    ) -> None:
        if direction is None:
            raise ValueError("direction must not be null")
        if entry_date is None:
            raise ValueError("entry_date must not be null")
        if entry_price is None:
            raise ValueError("entry_price must not be null")
        if shares <= 0:
            raise ValueError("shares must be positive")

        self._direction: Direction = direction
        self._entry_date: str = entry_date
        self._entry_price: Decimal = entry_price
        self._shares: int = shares

        self._status: TradeStatus = TradeStatus.OPEN
        self._exit_date: str | None = None
        self._exit_price: Decimal | None = None
        self._profit_loss: Decimal = Decimal("0")
        self._return_pct: Decimal = Decimal("0")

    def close(self, exit_date: str, exit_price: Decimal) -> None:
        """Close the trade at the given price."""
        if exit_date is None:
            raise ValueError("exit_date must not be null")
        if exit_price is None:
            raise ValueError("exit_price must not be null")
        if self._status == TradeStatus.CLOSED:
            raise RuntimeError("Trade is already closed")

        self._exit_date = exit_date
        self._exit_price = exit_price
        self._status = TradeStatus.CLOSED

        if self._direction == Direction.LONG:
            price_diff = exit_price - self._entry_price
        else:
            price_diff = self._entry_price - exit_price

        self._profit_loss = price_diff * Decimal(self._shares)
        self._return_pct = (
            price_diff / self._entry_price
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * Decimal("100")

    def is_win(self) -> bool:
        """Return True if the trade was profitable."""
        return self._profit_loss > Decimal("0")

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def status(self) -> TradeStatus:
        return self._status

    @property
    def entry_date(self) -> str:
        return self._entry_date

    @property
    def exit_date(self) -> str | None:
        return self._exit_date

    @property
    def entry_price(self) -> Decimal:
        return self._entry_price

    @property
    def exit_price(self) -> Decimal | None:
        return self._exit_price

    @property
    def shares(self) -> int:
        return self._shares

    @property
    def profit_loss(self) -> Decimal:
        return self._profit_loss

    @property
    def return_pct(self) -> Decimal:
        return self._return_pct

    def __str__(self) -> str:
        exit_date_str = self._exit_date if self._exit_date is not None else "N/A"
        exit_price_str = (
            f"${self._exit_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
            if self._exit_price is not None
            else "N/A"
        )
        return (
            f"{self._direction.value} {self._status.value}: "
            f"Entry {self._entry_date} @ "
            f"${self._entry_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} -> "
            f"Exit {exit_date_str} @ {exit_price_str} | "
            f"P/L: ${self._profit_loss.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} "
            f"({float(self._return_pct):.2f}%)"
        )


# ---------------------------------------------------------------------------
# Intraday signal types
# ---------------------------------------------------------------------------


class IntradayAction(Enum):
    """Action produced by an intraday strategy evaluation."""

    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class IntradaySignal:
    """Signal produced by an intraday strategy on each bar.

    Carries all the information an execution layer needs: direction,
    stop/target levels, confluence score, and a human-readable reason.
    """

    action: IntradayAction
    mode: str = ""
    stop_loss: Decimal = ZERO
    take_profit: Decimal = ZERO
    confluence_score: int = 0
    max_score: int = 0
    risk_per_share: Decimal = ZERO
    reason: str = ""


# Convenience singleton for the common "do nothing" case.
HOLD = IntradaySignal(action=IntradayAction.HOLD)


# ---------------------------------------------------------------------------
# Tournament trade (imported from TradingView for exit evaluation)
# ---------------------------------------------------------------------------


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
