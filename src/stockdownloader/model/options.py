"""Unified options models: OptionType, OptionContract, OptionsChain, and OptionsTrade."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum


# ---------------------------------------------------------------------------
# OptionType
# ---------------------------------------------------------------------------

class OptionType(Enum):
    """Option contract type: CALL gives the right to buy, PUT gives the right to sell."""

    CALL = "CALL"
    PUT = "PUT"


# ---------------------------------------------------------------------------
# OptionContract
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OptionContract:
    """Immutable representation of a single options contract with full greeks and volume data."""

    contract_symbol: str
    type: OptionType
    strike: Decimal
    expiration_date: str
    last_price: Decimal
    bid: Decimal
    ask: Decimal
    volume: int
    open_interest: int
    implied_volatility: Decimal
    delta: Decimal
    gamma: Decimal
    theta: Decimal
    vega: Decimal
    in_the_money: bool

    def __post_init__(self) -> None:
        # Required-field null checks — Python's type system doesn't enforce
        # these at runtime, but callers may pass None from untyped code.
        for field_name in (
            "contract_symbol", "type", "strike", "expiration_date",
            "last_price", "bid", "ask", "implied_volatility",
            "delta", "gamma", "theta", "vega",
        ):
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must not be null")
        if self.volume < 0:
            raise ValueError("volume must not be negative")
        if self.open_interest < 0:
            raise ValueError("open_interest must not be negative")

    def mid_price(self) -> Decimal:
        """Return the mid-price between bid and ask."""
        return (self.bid + self.ask) / Decimal("2")

    def spread(self) -> Decimal:
        """Return the bid-ask spread."""
        return self.ask - self.bid

    def notional_value(self) -> Decimal:
        """Return the notional value per contract (premium * 100 shares)."""
        return self.last_price * Decimal("100")

    def __str__(self) -> str:
        return (
            f"{self.contract_symbol} {self.type.value} "
            f"${self.strike} exp:{self.expiration_date} "
            f"last:${self.last_price} bid:${self.bid} ask:${self.ask} "
            f"vol:{self.volume} OI:{self.open_interest} "
            f"IV:{float(self.implied_volatility) * 100:.2f}%"
        )


# ---------------------------------------------------------------------------
# OptionsChain
# ---------------------------------------------------------------------------

class OptionsChain:
    """Full options chain for an underlying symbol, organized by expiration date.

    Tracks calls and puts separately with aggregate volume metrics.
    """

    def __init__(self, underlying_symbol: str) -> None:
        if underlying_symbol is None:
            raise ValueError("underlying_symbol must not be null")

        self._underlying_symbol: str = underlying_symbol
        self._underlying_price: Decimal = Decimal("0")
        self._expiration_dates: list[str] = []
        self._calls_by_expiration: OrderedDict[str, list[OptionContract]] = OrderedDict()
        self._puts_by_expiration: OrderedDict[str, list[OptionContract]] = OrderedDict()

    @property
    def underlying_symbol(self) -> str:
        return self._underlying_symbol

    @property
    def underlying_price(self) -> Decimal:
        return self._underlying_price

    @underlying_price.setter
    def underlying_price(self, price: Decimal) -> None:
        if price is None:
            raise ValueError("price must not be null")
        self._underlying_price = price

    @property
    def expiration_dates(self) -> list[str]:
        return list(self._expiration_dates)

    @property
    def calls_by_expiration(self) -> dict[str, list[OptionContract]]:
        return dict(self._calls_by_expiration)

    @property
    def puts_by_expiration(self) -> dict[str, list[OptionContract]]:
        return dict(self._puts_by_expiration)

    def add_expiration_date(self, date: str) -> None:
        """Add an expiration date if not already present."""
        if date not in self._expiration_dates:
            self._expiration_dates.append(date)

    def add_call(self, expiration: str, contract: OptionContract) -> None:
        """Add a call contract for the given expiration."""
        self._calls_by_expiration.setdefault(expiration, []).append(contract)

    def add_put(self, expiration: str, contract: OptionContract) -> None:
        """Add a put contract for the given expiration."""
        self._puts_by_expiration.setdefault(expiration, []).append(contract)

    def get_calls(self, expiration: str) -> list[OptionContract]:
        """Return a copy of calls for the given expiration."""
        return list(self._calls_by_expiration.get(expiration, []))

    def get_puts(self, expiration: str) -> list[OptionContract]:
        """Return a copy of puts for the given expiration."""
        return list(self._puts_by_expiration.get(expiration, []))

    @property
    def all_calls(self) -> list[OptionContract]:
        """Return all call contracts across all expirations."""
        return [c for contracts in self._calls_by_expiration.values() for c in contracts]

    @property
    def all_puts(self) -> list[OptionContract]:
        """Return all put contracts across all expirations."""
        return [p for contracts in self._puts_by_expiration.values() for p in contracts]

    @property
    def total_call_volume(self) -> int:
        """Return total volume across all call contracts."""
        return sum(c.volume for c in self.all_calls)

    @property
    def total_put_volume(self) -> int:
        """Return total volume across all put contracts."""
        return sum(p.volume for p in self.all_puts)

    @property
    def total_volume(self) -> int:
        """Return total volume across all contracts."""
        return self.total_call_volume + self.total_put_volume

    @property
    def total_call_open_interest(self) -> int:
        """Return total open interest across all call contracts."""
        return sum(c.open_interest for c in self.all_calls)

    @property
    def total_put_open_interest(self) -> int:
        """Return total open interest across all put contracts."""
        return sum(p.open_interest for p in self.all_puts)

    @property
    def put_call_ratio(self) -> Decimal:
        """Put/Call ratio based on volume. Values > 1 indicate bearish sentiment."""
        call_vol = self.total_call_volume
        if call_vol == 0:
            return Decimal("0")
        return (Decimal(self.total_put_volume) / Decimal(call_vol)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

    def find_nearest_strike(
        self,
        expiration: str,
        option_type: OptionType,
        target_price: Decimal,
    ) -> OptionContract | None:
        """Find the nearest strike to a target price for a given expiration and type.

        Returns the OptionContract or None if no contracts exist.
        """
        contracts = (
            self.get_calls(expiration)
            if option_type == OptionType.CALL
            else self.get_puts(expiration)
        )
        if not contracts:
            return None
        return min(contracts, key=lambda c: abs(c.strike - target_price))

    def get_contracts_at_strike(
        self, expiration: str, strike: Decimal
    ) -> list[OptionContract]:
        """Get contracts for a specific strike and expiration."""
        result: list[OptionContract] = []
        for c in self.get_calls(expiration):
            if c.strike == strike:
                result.append(c)
        for p in self.get_puts(expiration):
            if p.strike == strike:
                result.append(p)
        return result


# ---------------------------------------------------------------------------
# OptionsTrade
# ---------------------------------------------------------------------------

class OptionsDirection(Enum):
    """Options trade direction."""

    BUY = "BUY"
    SELL = "SELL"


class OptionsTradeStatus(Enum):
    """Options trade status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


CONTRACT_MULTIPLIER: int = 100


class OptionsTrade:
    """Tracks an individual options trade from entry to exit.

    Each contract represents 100 shares of the underlying.
    """

    def __init__(
        self,
        option_type: OptionType,
        direction: OptionsDirection,
        strike: Decimal,
        expiration_date: str,
        entry_date: str,
        entry_premium: Decimal,
        contracts: int,
        entry_volume: int,
    ) -> None:
        if option_type is None:
            raise ValueError("option_type must not be null")
        if direction is None:
            raise ValueError("direction must not be null")
        if strike is None:
            raise ValueError("strike must not be null")
        if expiration_date is None:
            raise ValueError("expiration_date must not be null")
        if entry_date is None:
            raise ValueError("entry_date must not be null")
        if entry_premium is None:
            raise ValueError("entry_premium must not be null")
        if contracts <= 0:
            raise ValueError("contracts must be positive")

        self._option_type: OptionType = option_type
        self._direction: OptionsDirection = direction
        self._strike: Decimal = strike
        self._expiration_date: str = expiration_date
        self._entry_date: str = entry_date
        self._entry_premium: Decimal = entry_premium
        self._contracts: int = contracts
        self._entry_volume: int = entry_volume

        self._status: OptionsTradeStatus = OptionsTradeStatus.OPEN
        self._exit_date: str | None = None
        self._exit_premium: Decimal | None = None
        self._profit_loss: Decimal = Decimal("0")
        self._return_pct: Decimal = Decimal("0")

    def _calculate_profit_loss(self) -> None:
        """Calculate P/L and return percentage after close or expiry."""
        assert self._exit_premium is not None

        if self._direction == OptionsDirection.BUY:
            premium_diff = self._exit_premium - self._entry_premium
        else:
            premium_diff = self._entry_premium - self._exit_premium

        self._profit_loss = (
            premium_diff * Decimal(self._contracts) * Decimal(CONTRACT_MULTIPLIER)
        )

        total_cost = (
            self._entry_premium
            * Decimal(self._contracts)
            * Decimal(CONTRACT_MULTIPLIER)
        )
        if total_cost != Decimal("0"):
            self._return_pct = (
                self._profit_loss / total_cost
            ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * Decimal("100")

    def close(self, exit_date: str, exit_premium: Decimal) -> None:
        """Close the trade at the given premium."""
        if exit_date is None:
            raise ValueError("exit_date must not be null")
        if exit_premium is None:
            raise ValueError("exit_premium must not be null")
        if self._status != OptionsTradeStatus.OPEN:
            raise RuntimeError(f"Trade is not open, current status: {self._status.value}")

        self._exit_date = exit_date
        self._exit_premium = exit_premium
        self._status = OptionsTradeStatus.CLOSED
        self._calculate_profit_loss()

    def expire(self, expiry_date: str, settlement_premium: Decimal) -> None:
        """Mark the trade as expired (option expired worthless or exercised)."""
        if expiry_date is None:
            raise ValueError("expiry_date must not be null")
        if settlement_premium is None:
            raise ValueError("settlement_premium must not be null")
        if self._status != OptionsTradeStatus.OPEN:
            raise RuntimeError(f"Trade is not open, current status: {self._status.value}")

        self._exit_date = expiry_date
        self._exit_premium = settlement_premium
        self._status = OptionsTradeStatus.EXPIRED
        self._calculate_profit_loss()

    def is_win(self) -> bool:
        """Return True if the trade was profitable."""
        return self._profit_loss > Decimal("0")

    def total_entry_cost(self) -> Decimal:
        """Total premium paid/received at entry (contracts * 100 * premium)."""
        return self._entry_premium * Decimal(self._contracts * CONTRACT_MULTIPLIER)

    @property
    def option_type(self) -> OptionType:
        return self._option_type

    @property
    def direction(self) -> OptionsDirection:
        return self._direction

    @property
    def strike(self) -> Decimal:
        return self._strike

    @property
    def expiration_date(self) -> str:
        return self._expiration_date

    @property
    def entry_date(self) -> str:
        return self._entry_date

    @property
    def entry_premium(self) -> Decimal:
        return self._entry_premium

    @property
    def contracts(self) -> int:
        return self._contracts

    @property
    def entry_volume(self) -> int:
        return self._entry_volume

    @property
    def status(self) -> OptionsTradeStatus:
        return self._status

    @property
    def exit_date(self) -> str | None:
        return self._exit_date

    @property
    def exit_premium(self) -> Decimal | None:
        return self._exit_premium

    @property
    def profit_loss(self) -> Decimal:
        return self._profit_loss

    @property
    def return_pct(self) -> Decimal:
        return self._return_pct

    def __str__(self) -> str:
        exit_date_str = self._exit_date if self._exit_date is not None else "N/A"
        exit_premium_str = (
            f"${self._exit_premium.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
            if self._exit_premium is not None
            else "N/A"
        )
        return (
            f"{self._direction.value} {self._option_type.value} "
            f"{self._status.value} "
            f"${self._strike.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} "
            f"exp:{self._expiration_date} | "
            f"Entry:{self._entry_date} "
            f"@${self._entry_premium.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} -> "
            f"Exit:{exit_date_str} @{exit_premium_str} | "
            f"P/L:${self._profit_loss.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} "
            f"({float(self._return_pct):.2f}%) "
            f"vol:{self._entry_volume}"
        )
