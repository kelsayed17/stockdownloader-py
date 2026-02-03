"""Historical price data and derived pattern information for a stock ticker."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal


class HistoricalData:
    """Historical price data and derived pattern information for a stock ticker."""

    def __init__(self, ticker: str) -> None:
        self._ticker: str = ticker

        self.highest_price_this_qtr: Decimal = Decimal("0")
        self.lowest_price_this_qtr: Decimal = Decimal("0")
        self.highest_price_last_qtr: Decimal = Decimal("0")
        self.lowest_price_last_qtr: Decimal = Decimal("0")

        self.historical_prices: list[str] = []
        self.patterns: defaultdict[str, set[str]] = defaultdict(set)

        self.incomplete: bool = False
        self.error: bool = False

    @property
    def ticker(self) -> str:
        return self._ticker
