"""Unified view of all financial data for a single symbol."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from stockdownloader.model.financial_data import FinancialData
from stockdownloader.model.historical_data import HistoricalData
from stockdownloader.model.options_chain import OptionsChain
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.quote_data import QuoteData


class UnifiedMarketData:
    """Unified view of all financial data for a single symbol, consolidating
    price data, quote data, historical data, financial data, and options chain
    into a single cohesive model.
    """

    def __init__(self, symbol: str) -> None:
        if symbol is None:
            raise ValueError("symbol must not be null")

        self._symbol: str = symbol
        self.latest_price: Optional[PriceData] = None
        self.price_history: list[PriceData] = []
        self.quote: Optional[QuoteData] = None
        self.historical: Optional[HistoricalData] = None
        self.financials: Optional[FinancialData] = None
        self.options_chain: Optional[OptionsChain] = None

    @property
    def symbol(self) -> str:
        return self._symbol

    # --- Volume aggregation (unified across all data sources) ---

    def get_equity_volume(self) -> Decimal:
        """Latest equity trading volume from quote data."""
        if self.quote is not None:
            return self.quote.volume
        return Decimal("0")

    def get_options_volume(self) -> int:
        """Total options volume (calls + puts) across all expirations."""
        if self.options_chain is not None:
            return self.options_chain.get_total_volume()
        return 0

    def get_call_volume(self) -> int:
        """Total call volume across all expirations."""
        if self.options_chain is not None:
            return self.options_chain.get_total_call_volume()
        return 0

    def get_put_volume(self) -> int:
        """Total put volume across all expirations."""
        if self.options_chain is not None:
            return self.options_chain.get_total_put_volume()
        return 0

    def get_total_combined_volume(self) -> Decimal:
        """Combined volume: equity trading volume + options volume."""
        return self.get_equity_volume() + Decimal(self.get_options_volume())

    def get_average_daily_volume(self, days: int) -> Decimal:
        """Average daily equity volume from price history."""
        if not self.price_history:
            return Decimal("0")
        limit = min(days, len(self.price_history))
        total_vol = sum(
            self.price_history[i].volume
            for i in range(len(self.price_history) - limit, len(self.price_history))
        )
        return (Decimal(total_vol) / Decimal(limit)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )

    def get_total_open_interest(self) -> int:
        """Total open interest across all options."""
        if self.options_chain is None:
            return 0
        return (
            self.options_chain.get_total_call_open_interest()
            + self.options_chain.get_total_put_open_interest()
        )

    def get_put_call_ratio(self) -> Decimal:
        """Put/call ratio based on volume."""
        if self.options_chain is not None:
            return self.options_chain.get_put_call_ratio()
        return Decimal("0")

    # --- Price metrics ---

    def get_current_price(self) -> Decimal:
        """Return the current price from quote data or latest price."""
        if (
            self.quote is not None
            and self.quote.last_trade_price_only > Decimal("0")
        ):
            return self.quote.last_trade_price_only
        if self.latest_price is not None:
            return self.latest_price.close
        return Decimal("0")

    def get_market_cap(self) -> Decimal:
        """Return the market capitalization."""
        if self.quote is not None:
            return Decimal(self.quote.market_capitalization)
        return Decimal("0")

    # --- Completeness checks ---

    def has_quote_data(self) -> bool:
        return self.quote is not None and not self.quote.error

    def has_price_history(self) -> bool:
        return bool(self.price_history)

    def has_financial_data(self) -> bool:
        return self.financials is not None and not self.financials.error

    def has_options_chain(self) -> bool:
        return (
            self.options_chain is not None
            and bool(self.options_chain.expiration_dates)
        )

    def has_historical_data(self) -> bool:
        return self.historical is not None and not self.historical.error

    def is_complete(self) -> bool:
        """Return True if all data sources are populated and error-free."""
        return (
            self.has_quote_data()
            and self.has_price_history()
            and self.has_financial_data()
            and self.has_options_chain()
            and self.has_historical_data()
        )

    def __str__(self) -> str:
        return (
            f"UnifiedMarketData[{self._symbol}] "
            f"price=${self.get_current_price()} "
            f"eqVol={self.get_equity_volume()} "
            f"optVol={self.get_options_volume()} "
            f"OI={self.get_total_open_interest()} "
            f"P/C={float(self.get_put_call_ratio()):.4f}"
        )
