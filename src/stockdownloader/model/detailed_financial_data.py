"""Extended fundamental financial data model for deep value analysis.

Holds profitability, balance sheet, growth, and cash flow metrics fetched
from Yahoo Finance v10 quoteSummary API (``financialData``,
``earningsTrend``, and ``balanceSheetHistory`` modules).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

_ZERO = Decimal("0")


@dataclass(slots=True)
class DetailedFinancialData:
    """Extended fundamental data for deep value analysis.

    All :class:`Decimal` fields default to zero and all :class:`int` fields
    default to ``0``.  Set ``incomplete = True`` when the upstream API
    response is partial or missing modules.
    """

    symbol: str = ""

    # ── Profitability ────────────────────────────────────────────────
    profit_margin: Decimal = field(default_factory=lambda: _ZERO)
    operating_margin: Decimal = field(default_factory=lambda: _ZERO)
    return_on_equity: Decimal = field(default_factory=lambda: _ZERO)
    return_on_assets: Decimal = field(default_factory=lambda: _ZERO)

    # ── Balance sheet ────────────────────────────────────────────────
    total_debt: int = 0
    total_equity: int = 0
    debt_to_equity: Decimal = field(default_factory=lambda: _ZERO)
    current_ratio: Decimal = field(default_factory=lambda: _ZERO)

    # ── Growth ───────────────────────────────────────────────────────
    revenue_growth: Decimal = field(default_factory=lambda: _ZERO)
    earnings_growth: Decimal = field(default_factory=lambda: _ZERO)

    # ── Cash flow ────────────────────────────────────────────────────
    free_cash_flow: int = 0
    operating_cash_flow: int = 0

    # ── Earnings trend ───────────────────────────────────────────────
    earnings_estimate_next_qtr: Decimal = field(default_factory=lambda: _ZERO)
    earnings_estimate_next_year: Decimal = field(default_factory=lambda: _ZERO)

    # ── Enterprise valuation (from defaultKeyStatistics) ───────────
    enterprise_value: int = 0
    enterprise_to_revenue: Decimal = field(default_factory=lambda: _ZERO)
    enterprise_to_ebitda: Decimal = field(default_factory=lambda: _ZERO)
    peg_ratio: Decimal = field(default_factory=lambda: _ZERO)
    beta: Decimal = field(default_factory=lambda: _ZERO)
    shares_outstanding: int = 0
    payout_ratio: Decimal = field(default_factory=lambda: _ZERO)
    forward_eps: Decimal = field(default_factory=lambda: _ZERO)

    # ── Additional financialData fields ────────────────────────────
    ebitda: int = 0
    total_cash: int = 0
    gross_margins: Decimal = field(default_factory=lambda: _ZERO)
    gross_profits: int = 0
    revenue_per_share: Decimal = field(default_factory=lambda: _ZERO)
    target_mean_price: Decimal = field(default_factory=lambda: _ZERO)

    # ── Source quality ───────────────────────────────────────────────
    incomplete: bool = False
