"""Scoring dimension functions extracted from :class:`ValueScreener`.

Contains all dimension scorers (valuation, growth, quality, income),
the Piotroski F-Score computation, and helper utilities (continuous
linear scorer, Graham Number calculator).  These are pure functions
that accept weights and data models as parameters.
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.model.financial_models import DetailedFinancialData
    from stockdownloader.model.financial_models import QuoteData

_ZERO = Decimal("0")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def continuous_score(
    value: float,
    best: float,
    worst: float,
    max_points: float,
) -> float:
    """Map *value* linearly to ``[0, max_points]``.

    If ``best < worst`` (lower-is-better metrics like P/E), a value
    at or below ``best`` earns full points, at or above ``worst``
    earns 0.  If ``best > worst`` (higher-is-better metrics like
    ROE), the logic reverses accordingly.

    The result is always clamped to ``[0, max_points]``.
    """
    if best == worst:
        return max_points if value == best else 0.0
    ratio = (value - worst) / (best - worst)
    return max(0.0, min(max_points, ratio * max_points))


def graham_number(eps: Decimal, book_value: Decimal) -> Decimal:
    """Compute Graham Number = sqrt(22.5 * EPS * BookValue).

    Returns ``Decimal("0")`` if either input is non-positive.
    """
    if eps <= _ZERO or book_value <= _ZERO:
        return _ZERO

    product = Decimal("22.5") * eps * book_value
    return Decimal(str(round(math.sqrt(float(product)), 2)))


# ------------------------------------------------------------------
# Piotroski F-Score
# ------------------------------------------------------------------


def compute_piotroski(
    quote: QuoteData,
    detailed: DetailedFinancialData,
) -> int:
    """Compute Piotroski F-Score (0-9) using available data.

    Industry-standard 9-point composite quality score.  Some criteria
    use proxy tests since we lack multi-year historical data:

    1. Net income > 0 (EPS positive)
    2. Operating cash flow > 0
    3. ROA > 0 (proxy for "ROA increasing")
    4. OCF quality: OCF > 0 AND FCF > 0 (accrual quality proxy)
    5. D/E < 1 (proxy for "leverage decreasing")
    6. Current ratio > 1 (proxy for "current ratio increasing")
    7. *Skipped* — no share dilution history available
    8. Gross margin > 0 (proxy for "gross margin increasing")
    9. Revenue growth > 0 (proxy for "asset turnover increasing")
    """
    f_score = 0
    eps = float(quote.diluted_eps)

    # 1. Net income positive
    if eps > 0:
        f_score += 1

    # 2. Operating cash flow positive
    if detailed.operating_cash_flow > 0:
        f_score += 1

    # 3. ROA positive (proxy for increasing)
    if float(detailed.return_on_assets) > 0:
        f_score += 1

    # 4. Accrual quality: OCF > 0 and FCF > 0
    if detailed.operating_cash_flow > 0 and detailed.free_cash_flow > 0:
        f_score += 1

    # 5. Leverage: D/E < 1 (normalize Yahoo percentage format)
    de = float(detailed.debt_to_equity)
    if de > 10:
        de = de / 100.0
    if de >= 0:
        if de < 1.0:
            f_score += 1
    elif detailed.total_debt == 0:
        f_score += 1  # No debt is good

    # 6. Current ratio > 1
    if float(detailed.current_ratio) > 1.0:
        f_score += 1

    # 7. Skipped (no share dilution history)

    # 8. Gross margin positive
    if float(detailed.gross_margins) > 0:
        f_score += 1

    # 9. Revenue growth positive (asset turnover proxy)
    if float(detailed.revenue_growth) > 0:
        f_score += 1

    return f_score


# ------------------------------------------------------------------
# Dimension scorers (0-100 each)
# ------------------------------------------------------------------


def score_valuation(
    w: dict[str, float],  # noqa: ARG001 — reserved for future weighted sub-scoring
    quote: QuoteData,
    detailed: DetailedFinancialData | None = None,
) -> float:
    """Score valuation using P/E, Earnings Yield, PEG, EV/EBITDA, P/B, P/S, Graham MoS.

    Uses continuous linear scoring (not discrete tiers) for smooth
    differentiation between companies.
    """
    score = 0.0
    pe = float(quote.trailing_pe)
    price = float(quote.last_trade_price_only)
    eps = float(quote.diluted_eps)

    # P/E ratio (0-15): lower is better
    if pe > 0:
        score += continuous_score(pe, best=5.0, worst=40.0, max_points=15.0)

    # Earnings Yield (0-15): EPS/Price — Greenblatt Magic Formula metric
    if price > 0 and eps > 0:
        ey = eps / price
        score += continuous_score(ey, best=0.20, worst=0.02, max_points=15.0)

    # PEG ratio (0-15): Peter Lynch's metric — lower is better
    if detailed is not None and float(detailed.peg_ratio) > 0:
        peg = float(detailed.peg_ratio)
        score += continuous_score(peg, best=0.5, worst=3.0, max_points=15.0)
    elif pe > 0 and eps > 0 and float(quote.eps_estimate_next_year) > eps:
        # Fallback 1: Compute PEG from forward EPS estimates
        growth_rate = (
            float(quote.eps_estimate_next_year - quote.diluted_eps)
            / eps
            * 100.0
        )
        if growth_rate > 0:
            computed_peg = pe / growth_rate
            score += continuous_score(
                computed_peg, best=0.5, worst=3.0, max_points=15.0
            )
    elif (
        pe > 0
        and detailed is not None
        and float(detailed.earnings_growth) > 0
    ):
        # Fallback 2: Compute PEG from Yahoo earnings_growth
        eg_pct = float(detailed.earnings_growth) * 100.0  # 0.10 -> 10%
        if eg_pct > 0:
            computed_peg = pe / eg_pct
            score += continuous_score(
                computed_peg, best=0.5, worst=3.0, max_points=15.0
            )

    # EV/EBITDA (0-15): Warren Buffett's preferred metric — lower is better
    if detailed is not None:
        ev_ebitda = float(detailed.enterprise_to_ebitda)
        if ev_ebitda > 0:
            score += continuous_score(
                ev_ebitda, best=4.0, worst=20.0, max_points=15.0
            )

    # P/B ratio (0-12): lower is better
    pb = float(quote.price_to_book)
    if pb > 0:
        score += continuous_score(pb, best=0.5, worst=5.0, max_points=12.0)

    # P/S ratio (0-10): lower is better
    ps = float(quote.price_sales)
    # Fallback: compute from revenue_per_share if Yahoo omits P/S
    if ps == 0 and detailed is not None and float(detailed.revenue_per_share) > 0:
        ps = price / float(detailed.revenue_per_share)
    if ps > 0:
        score += continuous_score(ps, best=0.5, worst=6.0, max_points=10.0)

    # Graham Margin of Safety (0-18)
    gn = graham_number(quote.diluted_eps, quote.book_value)
    if gn > _ZERO and quote.last_trade_price_only > _ZERO:
        mos = float(
            (gn - quote.last_trade_price_only) / quote.last_trade_price_only
        )
        score += continuous_score(
            mos, best=0.60, worst=-0.20, max_points=18.0
        )

    return min(score, 100.0)


def score_growth(
    w: dict[str, float],  # noqa: ARG001 — reserved for future weighted sub-scoring
    quote: QuoteData,
    detailed: DetailedFinancialData | None = None,
) -> float:
    """Score growth using EPS growth, forward P/E improvement, revenue/earnings growth.

    Moderate positive growth preferred (value investors want stable
    growers, not hyper-growth).
    """
    score = 0.0

    # Forward EPS growth (0-30): higher is better
    if quote.diluted_eps > _ZERO and quote.eps_estimate_next_year > _ZERO:
        eps_growth = float(
            (quote.eps_estimate_next_year - quote.diluted_eps)
            / quote.diluted_eps
        )
        score += continuous_score(
            eps_growth, best=0.15, worst=-0.05, max_points=30.0
        )

    # Forward P/E vs trailing P/E (0-15): improvement is better
    if quote.trailing_pe > _ZERO and quote.forward_pe > _ZERO:
        if quote.forward_pe < quote.trailing_pe:
            improvement = float(
                (quote.trailing_pe - quote.forward_pe) / quote.trailing_pe
            )
            score += continuous_score(
                improvement, best=0.20, worst=0.0, max_points=15.0
            )

    if detailed is not None:
        # Revenue growth (0-25): higher is better
        rev_g = float(detailed.revenue_growth)
        if rev_g > 0:
            score += continuous_score(
                rev_g, best=0.20, worst=0.0, max_points=25.0
            )

        # Earnings growth (0-25): higher is better
        earn_g = float(detailed.earnings_growth)
        if earn_g > 0:
            score += continuous_score(
                earn_g, best=0.20, worst=0.0, max_points=25.0
            )

        # PEG bonus (0-5): PEG < 1 is a growth-at-value signal
        peg = float(detailed.peg_ratio)
        if 0 < peg < 1.0:
            score += 5.0

    return min(score, 100.0)


def score_quality(
    w: dict[str, float],  # noqa: ARG001 — reserved for future weighted sub-scoring
    detailed: DetailedFinancialData,
    quote: QuoteData,
) -> float:
    """Score quality using Piotroski F-Score + individual profitability/solvency metrics.

    Requires detailed financial data.
    """
    score = 0.0

    # Piotroski F-Score (0-30): 3.33 points per F-Score point
    f_score = compute_piotroski(quote, detailed)
    score += f_score * (30.0 / 9.0)

    # ROE (0-15): higher is always better (FIX: no longer inverted)
    roe = float(detailed.return_on_equity)
    if roe > 0:
        score += continuous_score(
            roe, best=0.30, worst=0.0, max_points=15.0
        )

    # ROA (0-10): higher is better
    roa = float(detailed.return_on_assets)
    if roa > 0:
        score += continuous_score(
            roa, best=0.15, worst=0.0, max_points=10.0
        )

    # Operating margin (0-10): higher is better
    op_margin = float(detailed.operating_margin)
    if op_margin > 0:
        score += continuous_score(
            op_margin, best=0.30, worst=0.0, max_points=10.0
        )

    # Profit margin (0-10): higher is better
    pm = float(detailed.profit_margin)
    if pm > 0:
        score += continuous_score(
            pm, best=0.25, worst=0.0, max_points=10.0
        )

    # D/E ratio (0-15): lower is better
    de = float(detailed.debt_to_equity)
    # Yahoo sometimes reports D/E as percentage (e.g., 173 = 1.73x)
    if de > 10:
        de = de / 100.0
    if de >= 0:
        score += continuous_score(
            de, best=0.0, worst=3.0, max_points=15.0
        )

    # Current ratio (0-10): tent with peak at 2.0
    cr = float(detailed.current_ratio)
    if cr > 0:
        if cr <= 2.0:
            score += continuous_score(
                cr, best=2.0, worst=0.5, max_points=10.0
            )
        else:
            score += continuous_score(
                cr, best=2.0, worst=5.0, max_points=10.0
            )

    return min(score, 100.0)


def score_income(
    w: dict[str, float],  # noqa: ARG001 — reserved for future weighted sub-scoring
    quote: QuoteData,
    detailed: DetailedFinancialData | None = None,
) -> float:
    """Score income/yield dimension using fundamental metrics only.

    Unlike the old ``_score_yield``, this excludes technical/momentum
    indicators (52-week proximity, 200MA discount) which are not
    value fundamentals.

    Sub-metrics: dividend yield, FCF yield, payout ratio, earnings yield.
    """
    score = 0.0
    price = float(quote.last_trade_price_only)
    eps = float(quote.diluted_eps)

    # Dividend yield (0-30): tent with sweet spot at 3-5%
    # Clamp at 30% — yields above this are data errors or distress
    div_yield = min(float(quote.trailing_annual_dividend_yield), 0.30)
    if div_yield > 0:
        if div_yield <= 0.05:
            # Rising toward sweet spot
            score += continuous_score(
                div_yield, best=0.04, worst=0.0, max_points=30.0
            )
        else:
            # Excess yield may signal distress — decay
            score += continuous_score(
                div_yield, best=0.05, worst=0.12, max_points=30.0
            )

    # FCF Yield (0-25): FCF / Market Cap — higher is better
    if detailed is not None and quote.market_capitalization > 0:
        fcf_yield = detailed.free_cash_flow / quote.market_capitalization
        # Clamp to [-1.0, 1.0] — values beyond this are data anomalies
        fcf_yield = max(-1.0, min(1.0, fcf_yield))
        if fcf_yield > 0:
            score += continuous_score(
                fcf_yield, best=0.10, worst=0.0, max_points=25.0
            )

    # Payout ratio (0-20): tent with peak at 30-60%
    if detailed is not None:
        pr = min(float(detailed.payout_ratio), 1.5)
        # Fallback: compute from dividend yield and EPS when Yahoo omits
        if pr == 0 and div_yield > 0 and price > 0 and eps > 0:
            annual_div = div_yield * price
            pr = min(annual_div / eps, 1.5)
        if pr > 0:
            if pr <= 0.60:
                # Rising toward sustainable range
                score += continuous_score(
                    pr, best=0.45, worst=0.0, max_points=20.0
                )
            else:
                # Declining as payout becomes unsustainable
                score += continuous_score(
                    pr, best=0.60, worst=1.0, max_points=20.0
                )

    # Earnings Yield (0-25): EPS / Price — higher is better
    if price > 0 and eps > 0:
        ey = eps / price
        score += continuous_score(
            ey, best=0.15, worst=0.02, max_points=25.0
        )

    return min(score, 100.0)
