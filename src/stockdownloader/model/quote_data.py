"""Real-time stock quote data model with price, volume, and valuation fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class QuoteData:
    """Real-time stock quote data model with price, volume, and valuation fields."""

    price_sales: Decimal = field(default_factory=lambda: Decimal("0"))
    trailing_annual_dividend_yield: Decimal = field(default_factory=lambda: Decimal("0"))
    diluted_eps: Decimal = field(default_factory=lambda: Decimal("0"))
    eps_estimate_next_year: Decimal = field(default_factory=lambda: Decimal("0"))
    last_trade_price_only: Decimal = field(default_factory=lambda: Decimal("0"))
    year_high: Decimal = field(default_factory=lambda: Decimal("0"))
    year_low: Decimal = field(default_factory=lambda: Decimal("0"))
    fifty_day_moving_average: Decimal = field(default_factory=lambda: Decimal("0"))
    two_hundred_day_moving_average: Decimal = field(default_factory=lambda: Decimal("0"))
    previous_close: Decimal = field(default_factory=lambda: Decimal("0"))
    open: Decimal = field(default_factory=lambda: Decimal("0"))
    days_high: Decimal = field(default_factory=lambda: Decimal("0"))
    days_low: Decimal = field(default_factory=lambda: Decimal("0"))
    volume: Decimal = field(default_factory=lambda: Decimal("0"))
    year_range: str = ""
    market_capitalization_str: str = ""
    market_capitalization: int = 0
    incomplete: bool = False
    error: bool = False
