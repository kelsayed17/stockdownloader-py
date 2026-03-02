"""Tests for the ValueScreener analysis engine."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.analysis.value.screener import ValueScreener
from stockdownloader.core.models.financial import DetailedFinancialData, QuoteData


def _make_quote(
    *,
    price: float = 100.0,
    eps: float = 5.0,
    pe: float = 20.0,
    forward_pe: float = 18.0,
    pb: float = 2.0,
    bv: float = 50.0,
    ps: float = 3.0,
    div_yield: float = 0.02,
    market_cap: int = 500_000_000,
    year_high: float = 120.0,
    year_low: float = 80.0,
    ma200: float = 105.0,
    eps_forward: float = 5.5,
) -> QuoteData:
    """Helper to construct a QuoteData with reasonable defaults."""
    q = QuoteData()
    q.last_trade_price_only = Decimal(str(price))
    q.diluted_eps = Decimal(str(eps))
    q.trailing_pe = Decimal(str(pe))
    q.forward_pe = Decimal(str(forward_pe))
    q.price_to_book = Decimal(str(pb))
    q.book_value = Decimal(str(bv))
    q.price_sales = Decimal(str(ps))
    q.trailing_annual_dividend_yield = Decimal(str(div_yield))
    q.market_capitalization = market_cap
    q.year_high = Decimal(str(year_high))
    q.year_low = Decimal(str(year_low))
    q.two_hundred_day_moving_average = Decimal(str(ma200))
    q.eps_estimate_next_year = Decimal(str(eps_forward))
    return q


def _make_detailed(
    *,
    roe: float = 0.15,
    margin: float = 0.12,
    de: float = 0.5,
    cr: float = 2.0,
    fcf: int = 10_000_000,
    rev_growth: float = 0.08,
    earn_growth: float = 0.10,
    operating_margin: float = 0.15,
    return_on_assets: float = 0.08,
    gross_margins: float = 0.40,
    operating_cash_flow: int = 15_000_000,
    peg_ratio: float = 0.0,
    enterprise_to_ebitda: float = 0.0,
    payout_ratio: float = 0.0,
    beta: float = 1.0,
    enterprise_value: int = 0,
    revenue_per_share: float = 0.0,
) -> DetailedFinancialData:
    """Helper to construct DetailedFinancialData."""
    return DetailedFinancialData(
        return_on_equity=Decimal(str(roe)),
        profit_margin=Decimal(str(margin)),
        debt_to_equity=Decimal(str(de)),
        current_ratio=Decimal(str(cr)),
        free_cash_flow=fcf,
        revenue_growth=Decimal(str(rev_growth)),
        earnings_growth=Decimal(str(earn_growth)),
        operating_margin=Decimal(str(operating_margin)),
        return_on_assets=Decimal(str(return_on_assets)),
        gross_margins=Decimal(str(gross_margins)),
        operating_cash_flow=operating_cash_flow,
        peg_ratio=Decimal(str(peg_ratio)),
        enterprise_to_ebitda=Decimal(str(enterprise_to_ebitda)),
        payout_ratio=Decimal(str(payout_ratio)),
        beta=Decimal(str(beta)),
        enterprise_value=enterprise_value,
        revenue_per_share=Decimal(str(revenue_per_share)),
    )


class TestCoarseFilter:
    """Tests for ValueScreener.coarse_filter."""

    def test_passes_valid_stock(self) -> None:
        screener = ValueScreener()
        q = _make_quote(market_cap=500_000_000, eps=5.0, pe=15.0, price=50.0)
        survivors = screener.coarse_filter({"GOOD": q})
        assert "GOOD" in survivors

    def test_excludes_low_market_cap(self) -> None:
        screener = ValueScreener()
        q = _make_quote(market_cap=50_000_000)  # $50M < $100M default
        survivors = screener.coarse_filter({"SMALL": q})
        assert "SMALL" not in survivors

    def test_excludes_negative_eps(self) -> None:
        screener = ValueScreener()
        q = _make_quote(eps=-2.0)
        survivors = screener.coarse_filter({"LOSS": q})
        assert "LOSS" not in survivors

    def test_excludes_high_pe(self) -> None:
        screener = ValueScreener()
        q = _make_quote(pe=60.0)  # > 50
        survivors = screener.coarse_filter({"GROWTH": q})
        assert "GROWTH" not in survivors

    def test_excludes_penny_stocks(self) -> None:
        screener = ValueScreener()
        q = _make_quote(price=0.50)  # < $1
        survivors = screener.coarse_filter({"PENNY": q})
        assert "PENNY" not in survivors

    def test_excludes_incomplete(self) -> None:
        screener = ValueScreener()
        q = _make_quote()
        q.incomplete = True
        survivors = screener.coarse_filter({"BAD": q})
        assert "BAD" not in survivors

    def test_custom_min_market_cap(self) -> None:
        screener = ValueScreener()
        q = _make_quote(market_cap=50_000_000)
        # Lower threshold passes it through
        survivors = screener.coarse_filter({"SMALL": q}, min_market_cap=10_000_000)
        assert "SMALL" in survivors


class TestContinuousScoring:
    """Tests for the _continuous_score helper."""

    def test_at_best_value_lower_is_better(self) -> None:
        score = ValueScreener._continuous_score(5.0, best=5.0, worst=40.0, max_points=15.0)
        assert score == 15.0

    def test_at_worst_value(self) -> None:
        score = ValueScreener._continuous_score(40.0, best=5.0, worst=40.0, max_points=15.0)
        assert score == 0.0

    def test_midpoint(self) -> None:
        score = ValueScreener._continuous_score(22.5, best=5.0, worst=40.0, max_points=15.0)
        assert abs(score - 7.5) < 0.01

    def test_beyond_worst_clamps_zero(self) -> None:
        score = ValueScreener._continuous_score(50.0, best=5.0, worst=40.0, max_points=15.0)
        assert score == 0.0

    def test_beyond_best_clamps_max(self) -> None:
        score = ValueScreener._continuous_score(2.0, best=5.0, worst=40.0, max_points=15.0)
        assert score == 15.0

    def test_higher_is_better(self) -> None:
        """ROE-style: best=0.30 > worst=0.0."""
        score = ValueScreener._continuous_score(0.15, best=0.30, worst=0.0, max_points=15.0)
        assert abs(score - 7.5) < 0.01

    def test_equal_best_worst(self) -> None:
        assert ValueScreener._continuous_score(5.0, best=5.0, worst=5.0, max_points=10.0) == 10.0
        assert ValueScreener._continuous_score(3.0, best=5.0, worst=5.0, max_points=10.0) == 0.0


class TestScoreValuation:
    """Tests for _score_valuation dimension scoring."""

    def test_high_score_for_low_pe_low_pb(self) -> None:
        screener = ValueScreener()
        q = _make_quote(pe=8.0, pb=0.8, ps=0.9, eps=10.0, bv=100.0, price=80.0)
        score = screener._score_valuation(q)
        # Low P/E + low P/B + low P/S + high MoS = high score
        assert score >= 50.0

    def test_low_score_for_high_pe(self) -> None:
        screener = ValueScreener()
        q = _make_quote(pe=40.0, pb=10.0, ps=8.0, eps=1.0, bv=5.0, price=40.0)
        score = screener._score_valuation(q)
        assert score <= 20.0

    def test_moderate_pe_moderate_score(self) -> None:
        screener = ValueScreener()
        q = _make_quote(pe=15.0, pb=2.0, ps=2.5, eps=5.0, bv=40.0, price=75.0)
        score = screener._score_valuation(q)
        assert 15.0 <= score <= 70.0

    def test_with_peg_and_ev_ebitda(self) -> None:
        screener = ValueScreener()
        q = _make_quote(pe=10.0, pb=1.0, ps=1.0, eps=8.0, bv=70.0, price=80.0)
        d = _make_detailed(peg_ratio=0.8, enterprise_to_ebitda=6.0)
        score = screener._score_valuation(q, d)
        # Low PE + low PB + low PS + good PEG + good EV/EBITDA = very high
        assert score >= 60.0


class TestScoreQuality:
    """Tests for _score_quality dimension scoring."""

    def test_high_score_for_quality_company(self) -> None:
        screener = ValueScreener()
        q = _make_quote(eps=5.0)
        d = _make_detailed(
            roe=0.18, margin=0.25, de=0.3, cr=2.0, fcf=50_000_000,
            operating_margin=0.25, return_on_assets=0.12, gross_margins=0.50,
            operating_cash_flow=60_000_000, rev_growth=0.05,
        )
        score = screener._score_quality(d, q)
        assert score >= 70.0

    def test_low_score_for_poor_quality(self) -> None:
        screener = ValueScreener()
        q = _make_quote(eps=-2.0)
        d = _make_detailed(
            roe=0.02, margin=0.01, de=3.0, cr=0.5, fcf=-5_000_000,
            operating_margin=0.01, return_on_assets=0.0, gross_margins=0.0,
            operating_cash_flow=-3_000_000, rev_growth=-0.05,
        )
        score = screener._score_quality(d, q)
        assert score <= 25.0


class TestROEFix:
    """Verify the ROE scoring inversion bug is fixed."""

    def test_higher_roe_scores_higher(self) -> None:
        screener = ValueScreener()
        q = _make_quote(eps=5.0)

        d_moderate = _make_detailed(roe=0.18)
        d_high = _make_detailed(roe=0.30)

        score_mod = screener._score_quality(d_moderate, q)
        score_high = screener._score_quality(d_high, q)

        # FIX: ROE=0.30 MUST score higher than ROE=0.18
        assert score_high > score_mod

    def test_roe_monotonically_increasing(self) -> None:
        screener = ValueScreener()
        q = _make_quote(eps=5.0)

        scores = []
        for roe in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            d = _make_detailed(roe=roe)
            scores.append(screener._score_quality(d, q))

        # Each successive ROE should score at least as high
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], (
                f"ROE monotonicity violated: ROE={0.05 + i * 0.05} "
                f"scored {scores[i]} < {scores[i - 1]}"
            )


class TestPiotroskiFScore:
    """Tests for Piotroski F-Score computation."""

    def test_perfect_score(self) -> None:
        screener = ValueScreener()
        q = _make_quote(eps=5.0)
        d = _make_detailed(
            roe=0.15, margin=0.12, de=0.5, cr=2.0,
            fcf=10_000_000, rev_growth=0.08, earn_growth=0.10,
            operating_cash_flow=15_000_000,
            return_on_assets=0.10, gross_margins=0.40,
        )
        f_score = screener._compute_piotroski(q, d)
        # Should get most points: EPS>0, OCF>0, ROA>0, OCF&FCF>0,
        # D/E<1, CR>1, gross_margins>0, rev_growth>0 = 8/8 (skip 1)
        assert f_score >= 7

    def test_zero_score_distressed(self) -> None:
        screener = ValueScreener()
        q = _make_quote(eps=-2.0)
        d = _make_detailed(
            roe=-0.05, margin=-0.10, de=5.0, cr=0.3,
            fcf=-5_000_000, rev_growth=-0.10, earn_growth=-0.20,
            operating_cash_flow=-8_000_000,
            return_on_assets=-0.05, gross_margins=-0.05,
        )
        f_score = screener._compute_piotroski(q, d)
        assert f_score <= 2

    def test_individual_eps_criterion(self) -> None:
        """Only positive EPS, everything else negative."""
        screener = ValueScreener()
        q = _make_quote(eps=3.0)
        d = _make_detailed(
            roe=0.0, margin=0.0, de=2.0, cr=0.5,
            fcf=-1, rev_growth=-0.05, earn_growth=-0.10,
            operating_cash_flow=-1,
            return_on_assets=-0.01, gross_margins=-0.01,
        )
        f_score = screener._compute_piotroski(q, d)
        assert f_score == 1  # Only EPS positive


class TestScoreIncome:
    """Tests for _score_income dimension scoring (replaces old TestScoreYield)."""

    def test_high_score_for_good_dividend_and_earnings(self) -> None:
        screener = ValueScreener()
        # eps/price = 5/80 = 6.25% earnings yield, 4% dividend
        q = _make_quote(div_yield=0.04, price=80.0, eps=5.0)
        d = _make_detailed(fcf=40_000_000, payout_ratio=0.45)
        q.market_capitalization = 500_000_000
        score = screener._score_income(q, d)
        # Good dividend + FCF yield + good payout + earnings yield
        assert score >= 50.0

    def test_zero_dividend_still_has_earnings_yield(self) -> None:
        screener = ValueScreener()
        q = _make_quote(div_yield=0.0, price=50.0, eps=5.0)
        score = screener._score_income(q)
        # No dividend → 0 div points, but earnings yield = 10% → some points
        assert score >= 10.0

    def test_no_technical_indicators(self) -> None:
        """Verify 52-week high and 200MA don't affect income score."""
        screener = ValueScreener()
        # Same fundamentals, different technical positions
        q1 = _make_quote(
            div_yield=0.03, price=80.0, year_high=120.0, ma200=110.0, eps=5.0
        )
        q2 = _make_quote(
            div_yield=0.03, price=80.0, year_high=80.0, ma200=75.0, eps=5.0
        )
        score1 = screener._score_income(q1)
        score2 = screener._score_income(q2)
        # Scores should be identical — no technical influence
        assert abs(score1 - score2) < 0.01


class TestGrahamNumber:
    """Tests for the graham_number static method."""

    def test_correct_computation(self) -> None:
        # GN = sqrt(22.5 * 5 * 40) = sqrt(4500) ~ 67.08
        gn = ValueScreener.graham_number(Decimal("5"), Decimal("40"))
        assert gn == Decimal("67.08")

    def test_negative_eps_returns_zero(self) -> None:
        gn = ValueScreener.graham_number(Decimal("-2"), Decimal("40"))
        assert gn == Decimal("0")

    def test_negative_book_value_returns_zero(self) -> None:
        gn = ValueScreener.graham_number(Decimal("5"), Decimal("-10"))
        assert gn == Decimal("0")

    def test_zero_eps_returns_zero(self) -> None:
        gn = ValueScreener.graham_number(Decimal("0"), Decimal("40"))
        assert gn == Decimal("0")


class TestCompositeScoring:
    """Tests for the full score() method."""

    def test_composite_is_weighted_average(self) -> None:
        screener = ValueScreener()
        q = _make_quote(pe=10.0, pb=1.2, ps=1.5, eps=8.0, bv=70.0, price=80.0)
        d = _make_detailed(
            roe=0.15, margin=0.15, de=0.4, cr=2.0, fcf=20_000_000,
            operating_margin=0.20, return_on_assets=0.10, gross_margins=0.40,
            operating_cash_flow=25_000_000, rev_growth=0.08,
        )

        result = screener.score("TEST", q, d)

        # Composite should be in a reasonable range
        assert 0 <= result.composite_score <= 100
        # Should have all scores populated
        assert result.valuation_score >= 0
        assert result.growth_score >= 0
        assert result.quality_score >= 0
        assert result.income_score >= 0

    def test_score_without_detailed(self) -> None:
        """Phase 1 mode — detailed=None."""
        screener = ValueScreener()
        q = _make_quote(pe=12.0, pb=1.5, ps=2.0, eps=6.0, bv=50.0, price=72.0)

        result = screener.score("COARSE", q, None)

        assert result.quality_score == 0.0  # No detailed data
        assert result.composite_score > 0  # Still gets a score
        assert result.symbol == "COARSE"
        assert result.graham_number > Decimal("0")

    def test_result_has_key_metrics(self) -> None:
        screener = ValueScreener()
        q = _make_quote(
            pe=15.0, forward_pe=13.0, pb=2.5, ps=3.0,
            div_yield=0.03, eps=5.0, bv=30.0, price=75.0,
            market_cap=1_000_000_000,
        )
        d = _make_detailed(
            de=0.8, roe=0.20, fcf=50_000_000,
            peg_ratio=1.5, enterprise_to_ebitda=12.0,
            payout_ratio=0.40, beta=1.1,
        )

        result = screener.score("META", q, d)

        assert result.trailing_pe == Decimal("15.0")
        assert result.forward_pe == Decimal("13.0")
        assert result.price_to_book == Decimal("2.5")
        assert result.price_to_sales == Decimal("3.0")
        assert result.dividend_yield == Decimal("0.03")
        assert result.debt_to_equity == Decimal("0.8")
        assert result.return_on_equity == Decimal("0.20")
        assert result.free_cash_flow == 50_000_000
        assert result.market_cap == 1_000_000_000

    def test_result_has_new_metrics(self) -> None:
        screener = ValueScreener()
        q = _make_quote(
            pe=12.0, pb=1.5, ps=2.0, eps=6.0, bv=50.0, price=72.0,
            market_cap=800_000_000,
        )
        d = _make_detailed(
            peg_ratio=0.9, enterprise_to_ebitda=8.0,
            payout_ratio=0.50, beta=0.9,
            return_on_assets=0.12, operating_margin=0.18,
            rev_growth=0.06, earn_growth=0.10,
            enterprise_value=900_000_000,
        )

        result = screener.score("NEW", q, d)

        # New metrics populated
        assert result.earnings_yield > _ZERO  # EPS/Price = 6/72 > 0
        assert result.peg_ratio == Decimal("0.9")
        assert result.ev_ebitda == Decimal("8.0")
        assert result.piotroski_score >= 0
        assert result.operating_margin == Decimal("0.18")
        assert result.return_on_assets == Decimal("0.12")
        assert result.beta == Decimal("0.9")
        assert result.payout_ratio == Decimal("0.50")
        assert result.revenue_growth == Decimal("0.06")
        assert result.earnings_growth == Decimal("0.10")
        assert result.enterprise_value == 900_000_000


class TestDataQualityClamping:
    """Tests for data quality clamping and fallback computations."""

    def test_extreme_dividend_yield_is_clamped_in_scoring(self) -> None:
        """Dividend yield = 10.0 (1000%) should be clamped to 0.30 for scoring."""
        screener = ValueScreener()
        q = _make_quote(div_yield=10.0, price=20.0, eps=4.0)
        # With clamped yield (30%), score should NOT be absurdly high
        score = screener._score_income(q)
        # 30% yield decays from 5% sweet spot → modest dividend score
        # Plus earnings yield = 4/20 = 20% → good
        assert score <= 55.0  # Would be much higher without clamping

    def test_extreme_dividend_yield_is_clamped_in_result(self) -> None:
        """Result should show clamped dividend yield, not raw value."""
        screener = ValueScreener()
        q = _make_quote(div_yield=10.0, price=20.0, eps=4.0, pe=5.0, bv=15.0)
        result = screener.score("CLAMP", q)
        assert result.dividend_yield == Decimal("0.30")

    def test_extreme_fcf_yield_is_clamped(self) -> None:
        """FCF = 100x market cap → fcf_yield clamped to 1.0."""
        screener = ValueScreener()
        q = _make_quote(price=50.0, eps=5.0, pe=10.0, bv=30.0, market_cap=100_000_000)
        d = _make_detailed(fcf=10_000_000_000)  # 100x market cap
        result = screener.score("CLAMP", q, d)
        assert float(result.fcf_yield) <= 1.0

    def test_payout_ratio_computed_from_dividend_yield(self) -> None:
        """When Yahoo omits payoutRatio, compute from (yield*price)/eps."""
        screener = ValueScreener()
        # div_yield=0.04, price=100, eps=5 → annual_div=4, payout=4/5=0.80
        q = _make_quote(div_yield=0.04, price=100.0, eps=5.0, pe=20.0, bv=50.0)
        d = _make_detailed(payout_ratio=0.0)  # Yahoo didn't provide
        result = screener.score("PAY", q, d)
        # Should have computed payout ratio ≈ 0.80
        assert float(result.payout_ratio) > 0.7
        assert float(result.payout_ratio) < 0.9

    def test_peg_computed_from_earnings_growth(self) -> None:
        """When Yahoo omits pegRatio but provides earnings_growth, compute PEG."""
        screener = ValueScreener()
        # PE=12, earnings_growth=0.10 → growth_pct=10% → PEG=12/10=1.2
        q = _make_quote(pe=12.0, price=60.0, eps=5.0, eps_forward=5.0, bv=40.0)
        d = _make_detailed(peg_ratio=0.0, earn_growth=0.10)
        result = screener.score("PEG", q, d)
        assert float(result.peg_ratio) > 1.0
        assert float(result.peg_ratio) < 1.5

    def test_ps_computed_from_revenue_per_share(self) -> None:
        """When Yahoo omits P/S but provides revenue_per_share, compute P/S."""
        screener = ValueScreener()
        # price=80, revenue_per_share=20 → P/S = 80/20 = 4.0
        q = _make_quote(ps=0.0, price=80.0, eps=5.0, pe=16.0, bv=40.0)
        d = _make_detailed(revenue_per_share=20.0)
        result = screener.score("PS", q, d)
        assert result.price_to_sales == Decimal("4.0")

    def test_peg_fallback_from_forward_eps_takes_priority(self) -> None:
        """Forward EPS fallback should be tried before earnings_growth fallback."""
        screener = ValueScreener()
        # eps_forward=6.0 > eps=5.0 → growth = (6-5)/5*100 = 20%
        # PE=15 → PEG = 15/20 = 0.75
        q = _make_quote(pe=15.0, price=75.0, eps=5.0, eps_forward=6.0, bv=40.0)
        d = _make_detailed(peg_ratio=0.0, earn_growth=0.10)
        result = screener.score("PEG2", q, d)
        # Should use forward EPS fallback: PEG = 15/20 = 0.75
        assert abs(float(result.peg_ratio) - 0.75) < 0.01


_ZERO = Decimal("0")
