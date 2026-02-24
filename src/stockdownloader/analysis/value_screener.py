"""Multi-factor deep value scoring engine using industry-standard formulas.

Accepts pre-fetched data models and produces scored results.  Does NOT
call any APIs directly — fully decoupled from the data layer.

Scoring methodology includes:
- **Valuation**: P/E, Earnings Yield, PEG ratio, EV/EBITDA, P/B, P/S, Graham MoS
- **Quality**: Piotroski F-Score (0-9), ROE, ROA, operating margin, profit margin, D/E, current ratio
- **Growth**: EPS growth, forward P/E improvement, revenue growth, earnings growth, PEG bonus
- **Income**: Dividend yield, FCF yield, payout ratio, earnings yield

Usage::

    from stockdownloader.analysis.value_screener import ValueScreener
    from stockdownloader.model import QuoteData

    screener = ValueScreener()
    quotes: dict[str, QuoteData] = ...            # from YahooFinanceClient
    survivors = screener.coarse_filter(quotes)

    for sym in survivors:
        result = screener.score(sym, quotes[sym])  # Phase 1 (coarse)
        # or with detailed financials:
        result = screener.score(sym, quotes[sym], detailed)
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.analysis.value_scoring import (
    continuous_score,
    compute_piotroski,
    graham_number,
    score_growth,
    score_income,
    score_quality,
    score_valuation,
)

if TYPE_CHECKING:
    from stockdownloader.model.financial_models import DetailedFinancialData
    from stockdownloader.model.financial_models import QuoteData
    from stockdownloader.model.financial_models import ValueScreenerResult

_ZERO = Decimal("0")
_ONE = Decimal("1")

# Default scoring weights
_DEFAULT_WEIGHTS: dict[str, float] = {
    "valuation": 0.35,
    "growth": 0.20,
    "quality": 0.25,
    "income": 0.20,
}


class ValueScreener:
    """Multi-factor deep value scoring engine.

    Accepts pre-fetched data models and produces
    :class:`ValueScreenerResult` instances.  Does NOT call any APIs
    directly — fully decoupled from the data layer.

    Parameters
    ----------
    weights:
        Optional dict overriding the default dimension weights.
        Keys: ``"valuation"``, ``"growth"``, ``"quality"``, ``"income"``.
        Values must sum to 1.0.
    """

    def __init__(self, *, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or dict(_DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Phase 1: Coarse filter
    # ------------------------------------------------------------------

    def coarse_filter(
        self,
        quotes: dict[str, QuoteData],
        *,
        min_market_cap: int = 100_000_000,
    ) -> list[str]:
        """Filter tickers by basic criteria using quote data only.

        Returns list of symbols that pass:

        - Market cap >= *min_market_cap* (default $100M)
        - Positive EPS (``diluted_eps > 0``)
        - Trailing P/E > 0 and < 50
        - Price > $1 (excludes penny stocks)
        """
        survivors: list[str] = []

        for symbol, q in quotes.items():
            # Skip incomplete / errored quotes
            if q.incomplete or q.error:
                continue

            # Minimum market cap
            if q.market_capitalization < min_market_cap:
                continue

            # Positive EPS
            if q.diluted_eps <= _ZERO:
                continue

            # Reasonable P/E range
            pe = q.trailing_pe
            if pe <= _ZERO or pe > Decimal("50"):
                continue

            # No penny stocks
            if q.last_trade_price_only <= _ONE:
                continue

            survivors.append(symbol)

        return survivors

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def score(
        self,
        symbol: str,
        quote: QuoteData,
        detailed: DetailedFinancialData | None = None,
    ) -> ValueScreenerResult:
        """Compute all dimension scores and weighted composite.

        If *detailed* is ``None``, ``quality_score`` defaults to 0 and
        growth / valuation / income scores use quote-only data (Phase 1
        coarse scoring mode).
        """
        from stockdownloader.model.financial_models import ValueScreenerResult

        val_score = self._score_valuation(quote, detailed)
        growth_score = self._score_growth(quote, detailed)
        quality_score = self._score_quality(detailed, quote) if detailed else 0.0
        income_score = self._score_income(quote, detailed)

        # Weighted composite — if quality is unavailable, redistribute
        if detailed is not None:
            composite = (
                self._weights["valuation"] * val_score
                + self._weights["growth"] * growth_score
                + self._weights["quality"] * quality_score
                + self._weights["income"] * income_score
            )
        else:
            # Without detailed data, use only valuation, growth, income
            total_w = (
                self._weights["valuation"]
                + self._weights["growth"]
                + self._weights["income"]
            )
            if total_w > 0:
                composite = (
                    self._weights["valuation"] * val_score
                    + self._weights["growth"] * growth_score
                    + self._weights["income"] * income_score
                ) / total_w
            else:
                composite = 0.0

        # Build Graham Number & margin of safety
        gn = self.graham_number(quote.diluted_eps, quote.book_value)
        mos = _ZERO
        if gn > _ZERO and quote.last_trade_price_only > _ZERO:
            mos = (gn - quote.last_trade_price_only) / quote.last_trade_price_only

        # Derived metrics
        price = float(quote.last_trade_price_only)
        eps = float(quote.diluted_eps)
        pe = float(quote.trailing_pe)
        earnings_yield = (
            Decimal(str(round(eps / price, 4)))
            if price > 0 and eps > 0
            else _ZERO
        )

        # FCF yield — clamped to [-1.0, 1.0] to cap data anomalies
        fcf_yield = _ZERO
        if detailed and quote.market_capitalization > 0:
            fcf_y = detailed.free_cash_flow / quote.market_capitalization
            fcf_yield = Decimal(str(round(max(-1.0, min(1.0, fcf_y)), 4)))

        piotroski = self._compute_piotroski(quote, detailed) if detailed else 0

        # Normalize D/E from Yahoo's percentage format
        de = detailed.debt_to_equity if detailed else _ZERO
        if float(de) > 10:
            de = Decimal(str(round(float(de) / 100.0, 4)))

        # PEG ratio — with fallback computation when Yahoo omits
        peg = detailed.peg_ratio if detailed else _ZERO
        if detailed and float(peg) == 0:
            if (
                pe > 0
                and eps > 0
                and float(quote.eps_estimate_next_year) > eps
            ):
                # Fallback 1: from forward EPS growth
                growth_pct = (
                    (float(quote.eps_estimate_next_year) - eps) / eps * 100.0
                )
                if growth_pct > 0:
                    peg = Decimal(str(round(pe / growth_pct, 2)))
            elif pe > 0 and float(detailed.earnings_growth) > 0:
                # Fallback 2: from Yahoo earnings_growth
                eg_pct = float(detailed.earnings_growth) * 100.0
                if eg_pct > 0:
                    peg = Decimal(str(round(pe / eg_pct, 2)))

        # P/S ratio — with fallback from revenue_per_share
        ps = quote.price_sales
        if (
            float(ps) == 0
            and detailed
            and float(detailed.revenue_per_share) > 0
            and price > 0
        ):
            ps = Decimal(str(round(price / float(detailed.revenue_per_share), 2)))

        # Payout ratio — with fallback from dividend yield + EPS
        payout = detailed.payout_ratio if detailed else _ZERO
        # Clamp to [0, 1.5]
        payout = min(payout, Decimal("1.5"))
        if (
            detailed
            and float(payout) == 0
            and float(quote.trailing_annual_dividend_yield) > 0
        ):
            div_yield_f = min(
                float(quote.trailing_annual_dividend_yield), 0.30
            )
            if price > 0 and eps > 0:
                computed_pr = (div_yield_f * price) / eps
                payout = Decimal(str(round(min(computed_pr, 1.5), 4)))

        # Dividend yield — clamped at 30%
        clamped_div_yield = min(
            quote.trailing_annual_dividend_yield, Decimal("0.30")
        )

        return ValueScreenerResult(
            symbol=symbol,
            price=quote.last_trade_price_only,
            market_cap=quote.market_capitalization,
            valuation_score=round(val_score, 2),
            growth_score=round(growth_score, 2),
            quality_score=round(quality_score, 2),
            income_score=round(income_score, 2),
            composite_score=round(composite, 2),
            # Existing metrics
            trailing_pe=quote.trailing_pe,
            forward_pe=quote.forward_pe,
            price_to_book=quote.price_to_book,
            price_to_sales=ps,
            dividend_yield=clamped_div_yield,
            graham_number=gn,
            margin_of_safety=mos,
            debt_to_equity=de,
            return_on_equity=detailed.return_on_equity if detailed else _ZERO,
            free_cash_flow=detailed.free_cash_flow if detailed else 0,
            # New industry-standard metrics
            earnings_yield=earnings_yield,
            fcf_yield=fcf_yield,
            peg_ratio=peg,
            ev_ebitda=detailed.enterprise_to_ebitda if detailed else _ZERO,
            piotroski_score=piotroski,
            operating_margin=detailed.operating_margin if detailed else _ZERO,
            return_on_assets=detailed.return_on_assets if detailed else _ZERO,
            current_ratio=detailed.current_ratio if detailed else _ZERO,
            beta=detailed.beta if detailed else _ZERO,
            payout_ratio=payout,
            enterprise_value=detailed.enterprise_value if detailed else 0,
            revenue_growth=detailed.revenue_growth if detailed else _ZERO,
            earnings_growth=detailed.earnings_growth if detailed else _ZERO,
        )

    # ------------------------------------------------------------------
    # Dimension scoring (0-100 each) — delegated to value_scoring module
    # ------------------------------------------------------------------

    def _score_valuation(
        self,
        quote: QuoteData,
        detailed: DetailedFinancialData | None = None,
    ) -> float:
        return score_valuation(self._weights, quote, detailed)

    def _score_growth(
        self,
        quote: QuoteData,
        detailed: DetailedFinancialData | None = None,
    ) -> float:
        return score_growth(self._weights, quote, detailed)

    def _score_quality(
        self,
        detailed: DetailedFinancialData,
        quote: QuoteData,
    ) -> float:
        return score_quality(self._weights, detailed, quote)

    def _score_income(
        self,
        quote: QuoteData,
        detailed: DetailedFinancialData | None = None,
    ) -> float:
        return score_income(self._weights, quote, detailed)

    # ------------------------------------------------------------------
    # Piotroski F-Score — delegated to value_scoring module
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_piotroski(
        quote: QuoteData,
        detailed: DetailedFinancialData,
    ) -> int:
        return compute_piotroski(quote, detailed)

    # ------------------------------------------------------------------
    # Helpers — delegated to value_scoring module
    # ------------------------------------------------------------------

    @staticmethod
    def _continuous_score(
        value: float,
        best: float,
        worst: float,
        max_points: float,
    ) -> float:
        return continuous_score(value, best, worst, max_points)

    @staticmethod
    def graham_number(eps: Decimal, book_value: Decimal) -> Decimal:
        return graham_number(eps, book_value)
