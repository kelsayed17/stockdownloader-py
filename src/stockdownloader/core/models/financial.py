"""Financial data models for the value-screening pipeline.

Combines real-time quote data, extended fundamental metrics, and scored
screening results into a single cohesive module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

_ZERO = Decimal("0")


@dataclass(slots=True)
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
    trailing_pe: Decimal = field(default_factory=lambda: Decimal("0"))
    forward_pe: Decimal = field(default_factory=lambda: Decimal("0"))
    price_to_book: Decimal = field(default_factory=lambda: Decimal("0"))
    book_value: Decimal = field(default_factory=lambda: Decimal("0"))
    market_capitalization_str: str = ""
    market_capitalization: int = 0
    incomplete: bool = False
    error: bool = False


@dataclass(slots=True)
class DetailedFinancialData:
    """Extended fundamental data for deep value analysis.

    All :class:`Decimal` fields default to zero and all :class:`int` fields
    default to ``0``.  Set ``incomplete = True`` when the upstream API
    response is partial or missing modules.
    """

    symbol: str = ""

    # -- Profitability --------------------------------------------------------
    profit_margin: Decimal = field(default_factory=lambda: _ZERO)
    operating_margin: Decimal = field(default_factory=lambda: _ZERO)
    return_on_equity: Decimal = field(default_factory=lambda: _ZERO)
    return_on_assets: Decimal = field(default_factory=lambda: _ZERO)

    # -- Balance sheet --------------------------------------------------------
    total_debt: int = 0
    total_equity: int = 0
    debt_to_equity: Decimal = field(default_factory=lambda: _ZERO)
    current_ratio: Decimal = field(default_factory=lambda: _ZERO)

    # -- Growth ---------------------------------------------------------------
    revenue_growth: Decimal = field(default_factory=lambda: _ZERO)
    earnings_growth: Decimal = field(default_factory=lambda: _ZERO)

    # -- Cash flow ------------------------------------------------------------
    free_cash_flow: int = 0
    operating_cash_flow: int = 0

    # -- Earnings trend -------------------------------------------------------
    earnings_estimate_next_qtr: Decimal = field(default_factory=lambda: _ZERO)
    earnings_estimate_next_year: Decimal = field(default_factory=lambda: _ZERO)

    # -- Enterprise valuation (from defaultKeyStatistics) ---------------------
    enterprise_value: int = 0
    enterprise_to_revenue: Decimal = field(default_factory=lambda: _ZERO)
    enterprise_to_ebitda: Decimal = field(default_factory=lambda: _ZERO)
    peg_ratio: Decimal = field(default_factory=lambda: _ZERO)
    beta: Decimal = field(default_factory=lambda: _ZERO)
    shares_outstanding: int = 0
    payout_ratio: Decimal = field(default_factory=lambda: _ZERO)
    forward_eps: Decimal = field(default_factory=lambda: _ZERO)

    # -- Additional financialData fields --------------------------------------
    ebitda: int = 0
    total_cash: int = 0
    gross_margins: Decimal = field(default_factory=lambda: _ZERO)
    gross_profits: int = 0
    revenue_per_share: Decimal = field(default_factory=lambda: _ZERO)
    target_mean_price: Decimal = field(default_factory=lambda: _ZERO)

    # -- Source quality -------------------------------------------------------
    incomplete: bool = False


@dataclass(slots=True)
class ValueScreenerResult:
    """Scored deep-value result for a single ticker.

    Dimension scores range from 0 to 100.  The ``composite_score`` is a
    weighted average of the individual dimension scores.
    """

    symbol: str
    price: Decimal
    market_cap: int

    # -- Dimension scores (0-100) ---------------------------------------------
    valuation_score: float = 0.0
    growth_score: float = 0.0
    quality_score: float = 0.0
    income_score: float = 0.0

    # -- Composite ------------------------------------------------------------
    composite_score: float = 0.0

    # -- Key metrics for display ----------------------------------------------
    trailing_pe: Decimal = field(default_factory=lambda: _ZERO)
    forward_pe: Decimal = field(default_factory=lambda: _ZERO)
    price_to_book: Decimal = field(default_factory=lambda: _ZERO)
    price_to_sales: Decimal = field(default_factory=lambda: _ZERO)
    dividend_yield: Decimal = field(default_factory=lambda: _ZERO)
    graham_number: Decimal = field(default_factory=lambda: _ZERO)
    margin_of_safety: Decimal = field(default_factory=lambda: _ZERO)
    debt_to_equity: Decimal = field(default_factory=lambda: _ZERO)
    return_on_equity: Decimal = field(default_factory=lambda: _ZERO)
    free_cash_flow: int = 0

    # -- New industry-standard metrics ----------------------------------------
    earnings_yield: Decimal = field(default_factory=lambda: _ZERO)
    fcf_yield: Decimal = field(default_factory=lambda: _ZERO)
    peg_ratio: Decimal = field(default_factory=lambda: _ZERO)
    ev_ebitda: Decimal = field(default_factory=lambda: _ZERO)
    piotroski_score: int = 0
    operating_margin: Decimal = field(default_factory=lambda: _ZERO)
    return_on_assets: Decimal = field(default_factory=lambda: _ZERO)
    current_ratio: Decimal = field(default_factory=lambda: _ZERO)
    beta: Decimal = field(default_factory=lambda: _ZERO)
    payout_ratio: Decimal = field(default_factory=lambda: _ZERO)
    enterprise_value: int = 0
    revenue_growth: Decimal = field(default_factory=lambda: _ZERO)
    earnings_growth: Decimal = field(default_factory=lambda: _ZERO)
