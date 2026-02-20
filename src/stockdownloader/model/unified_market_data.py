"""Unified view of all financial data for a single symbol.

Also contains the ``HistoricalData`` and ``FinancialData`` helper classes
that were previously in their own modules.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING

from stockdownloader.model.options import OptionsChain
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.financial_models import QuoteData


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


class FinancialData:
    """Fundamental financial data model holding revenue, shares outstanding,
    and derived revenue-per-share metrics.

    Arrays are indexed 0-5 corresponding to Qtr1-5 + TTM.
    """

    def __init__(self) -> None:
        self.revenue: list[int] = [0] * 6
        self.basic_shares: list[int] = [0] * 6
        self.diluted_shares: list[int] = [0] * 6
        self.revenue_per_share: list[Decimal] = [Decimal("0")] * 6
        self.revenue_per_share_ttm_last_qtr: Decimal = Decimal("0")
        self.fiscal_quarters: list[str] = [""] * 6

        self.incomplete: bool = False
        self.error: bool = False

    @staticmethod
    def _divide_revenue(revenue: int, shares: int) -> Decimal:
        """Divide revenue by shares with CEILING rounding, 2 decimal places."""
        if shares == 0:
            return Decimal("0")
        return Decimal(revenue) / Decimal(shares)

    def compute_revenue_per_share(self) -> None:
        """Compute revenue per share for each quarter and TTM last quarter."""
        for i in range(6):
            if self.diluted_shares[i] == 0:
                self.diluted_shares[i] = self.basic_shares[i]

        for i in range(6):
            self.revenue_per_share[i] = self._divide_revenue(
                self.revenue[i], self.diluted_shares[i]
            ).quantize(Decimal("0.01"), rounding=ROUND_CEILING)

        self.revenue_per_share_ttm_last_qtr = (
            self.revenue_per_share[0]
            + self.revenue_per_share[1]
            + self.revenue_per_share[2]
            + self.revenue_per_share[3]
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @property
    def revenue_per_share_ttm(self) -> Decimal:
        """Convenience accessor for TTM revenue per share (index 5)."""
        return self.revenue_per_share[5]


class UnifiedMarketData:
    """Unified view of all financial data for a single symbol, consolidating
    price data, quote data, historical data, financial data, and options chain
    into a single cohesive model.
    """

    def __init__(self, symbol: str) -> None:
        if symbol is None:
            raise ValueError("symbol must not be null")

        self._symbol: str = symbol
        self.latest_price: PriceData | None = None
        self.price_history: list[PriceData] = []
        self.quote: QuoteData | None = None
        self.historical: HistoricalData | None = None
        self.financials: FinancialData | None = None
        self.options_chain: OptionsChain | None = None

    @property
    def symbol(self) -> str:
        return self._symbol

    # --- Volume aggregation (unified across all data sources) ---

    @property
    def equity_volume(self) -> Decimal:
        """Latest equity trading volume from quote data."""
        if self.quote is not None:
            return self.quote.volume
        return Decimal("0")

    @property
    def options_volume(self) -> int:
        """Total options volume (calls + puts) across all expirations."""
        if self.options_chain is not None:
            return self.options_chain.total_volume
        return 0

    @property
    def call_volume(self) -> int:
        """Total call volume across all expirations."""
        if self.options_chain is not None:
            return self.options_chain.total_call_volume
        return 0

    @property
    def put_volume(self) -> int:
        """Total put volume across all expirations."""
        if self.options_chain is not None:
            return self.options_chain.total_put_volume
        return 0

    @property
    def total_combined_volume(self) -> Decimal:
        """Combined volume: equity trading volume + options volume."""
        return self.equity_volume + Decimal(self.options_volume)

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

    @property
    def total_open_interest(self) -> int:
        """Total open interest across all options."""
        if self.options_chain is None:
            return 0
        return (
            self.options_chain.total_call_open_interest
            + self.options_chain.total_put_open_interest
        )

    @property
    def put_call_ratio(self) -> Decimal:
        """Put/call ratio based on volume."""
        if self.options_chain is not None:
            return self.options_chain.put_call_ratio
        return Decimal("0")

    # --- Price metrics ---

    @property
    def current_price(self) -> Decimal:
        """Return the current price from quote data or latest price."""
        if (
            self.quote is not None
            and self.quote.last_trade_price_only > Decimal("0")
        ):
            return self.quote.last_trade_price_only
        if self.latest_price is not None:
            return self.latest_price.close
        return Decimal("0")

    @property
    def market_cap(self) -> Decimal:
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
            f"price=${self.current_price} "
            f"eqVol={self.equity_volume} "
            f"optVol={self.options_volume} "
            f"OI={self.total_open_interest} "
            f"P/C={float(self.put_call_ratio):.4f}"
        )
