"""Immutable representation of a single options contract with full greeks and volume data."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.model.option_type import OptionType


@dataclass(frozen=True)
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
        if self.contract_symbol is None:
            raise ValueError("contract_symbol must not be null")
        if self.type is None:
            raise ValueError("type must not be null")
        if self.strike is None:
            raise ValueError("strike must not be null")
        if self.expiration_date is None:
            raise ValueError("expiration_date must not be null")
        if self.last_price is None:
            raise ValueError("last_price must not be null")
        if self.bid is None:
            raise ValueError("bid must not be null")
        if self.ask is None:
            raise ValueError("ask must not be null")
        if self.implied_volatility is None:
            raise ValueError("implied_volatility must not be null")
        if self.delta is None:
            raise ValueError("delta must not be null")
        if self.gamma is None:
            raise ValueError("gamma must not be null")
        if self.theta is None:
            raise ValueError("theta must not be null")
        if self.vega is None:
            raise ValueError("vega must not be null")
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
