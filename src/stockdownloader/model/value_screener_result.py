"""Scored deep-value screening result for a single ticker.

Produced by :class:`~stockdownloader.analysis.value_screener.ValueScreener`
after evaluating quote and financial data against multi-factor value criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

_ZERO = Decimal("0")


@dataclass(slots=True)
class ValueScreenerResult:
    """Scored deep-value result for a single ticker.

    Dimension scores range from 0 to 100.  The ``composite_score`` is a
    weighted average of the individual dimension scores.
    """

    symbol: str
    price: Decimal
    market_cap: int

    # ── Dimension scores (0-100) ─────────────────────────────────────
    valuation_score: float = 0.0
    growth_score: float = 0.0
    quality_score: float = 0.0
    income_score: float = 0.0

    # ── Composite ────────────────────────────────────────────────────
    composite_score: float = 0.0

    # ── Key metrics for display ──────────────────────────────────────
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

    # ── New industry-standard metrics ────────────────────────────────
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
