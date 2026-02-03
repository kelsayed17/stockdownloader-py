"""Full options chain for an underlying symbol, organized by expiration date."""

from __future__ import annotations

from collections import OrderedDict
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from stockdownloader.model.option_contract import OptionContract
from stockdownloader.model.option_type import OptionType


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

    def get_all_calls(self) -> list[OptionContract]:
        """Return all call contracts across all expirations."""
        return [c for contracts in self._calls_by_expiration.values() for c in contracts]

    def get_all_puts(self) -> list[OptionContract]:
        """Return all put contracts across all expirations."""
        return [p for contracts in self._puts_by_expiration.values() for p in contracts]

    def get_total_call_volume(self) -> int:
        """Return total volume across all call contracts."""
        return sum(c.volume for c in self.get_all_calls())

    def get_total_put_volume(self) -> int:
        """Return total volume across all put contracts."""
        return sum(p.volume for p in self.get_all_puts())

    def get_total_volume(self) -> int:
        """Return total volume across all contracts."""
        return self.get_total_call_volume() + self.get_total_put_volume()

    def get_total_call_open_interest(self) -> int:
        """Return total open interest across all call contracts."""
        return sum(c.open_interest for c in self.get_all_calls())

    def get_total_put_open_interest(self) -> int:
        """Return total open interest across all put contracts."""
        return sum(p.open_interest for p in self.get_all_puts())

    def get_put_call_ratio(self) -> Decimal:
        """Put/Call ratio based on volume. Values > 1 indicate bearish sentiment."""
        call_vol = self.get_total_call_volume()
        if call_vol == 0:
            return Decimal("0")
        return (Decimal(self.get_total_put_volume()) / Decimal(call_vol)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

    def find_nearest_strike(
        self,
        expiration: str,
        option_type: OptionType,
        target_price: Decimal,
    ) -> Optional[OptionContract]:
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
