"""Tests for ValueScreenerResult model."""
from decimal import Decimal

from stockdownloader.model.financial_models import ValueScreenerResult


class TestValueScreenerResult:
    """Tests for the ValueScreenerResult dataclass."""

    def test_construction_required_fields(self) -> None:
        result = ValueScreenerResult(
            symbol="AAPL",
            price=Decimal("150.00"),
            market_cap=2_500_000_000_000,
        )
        assert result.symbol == "AAPL"
        assert result.price == Decimal("150.00")
        assert result.market_cap == 2_500_000_000_000
        # Defaults
        assert result.valuation_score == 0.0
        assert result.growth_score == 0.0
        assert result.quality_score == 0.0
        assert result.income_score == 0.0
        assert result.composite_score == 0.0
        assert result.free_cash_flow == 0

    def test_score_ranges(self) -> None:
        result = ValueScreenerResult(
            symbol="JNJ",
            price=Decimal("160.00"),
            market_cap=400_000_000_000,
            valuation_score=75.0,
            growth_score=50.0,
            quality_score=85.0,
            income_score=60.0,
            composite_score=68.5,
        )
        assert 0 <= result.valuation_score <= 100
        assert 0 <= result.growth_score <= 100
        assert 0 <= result.quality_score <= 100
        assert 0 <= result.income_score <= 100
        assert 0 <= result.composite_score <= 100

    def test_all_metric_fields(self) -> None:
        result = ValueScreenerResult(
            symbol="PG",
            price=Decimal("155.00"),
            market_cap=360_000_000_000,
            trailing_pe=Decimal("24.5"),
            forward_pe=Decimal("22.1"),
            price_to_book=Decimal("7.2"),
            price_to_sales=Decimal("4.3"),
            dividend_yield=Decimal("0.025"),
            graham_number=Decimal("45.00"),
            margin_of_safety=Decimal("-0.71"),
            debt_to_equity=Decimal("0.8"),
            return_on_equity=Decimal("0.30"),
            free_cash_flow=15_000_000_000,
        )
        assert result.trailing_pe == Decimal("24.5")
        assert result.graham_number == Decimal("45.00")
        assert result.free_cash_flow == 15_000_000_000

    def test_new_industry_standard_metrics(self) -> None:
        result = ValueScreenerResult(
            symbol="MSFT",
            price=Decimal("420.00"),
            market_cap=3_100_000_000_000,
            earnings_yield=Decimal("0.0274"),
            fcf_yield=Decimal("0.0195"),
            peg_ratio=Decimal("2.1"),
            ev_ebitda=Decimal("22.5"),
            piotroski_score=7,
            operating_margin=Decimal("0.43"),
            return_on_assets=Decimal("0.19"),
            current_ratio=Decimal("1.77"),
            beta=Decimal("0.89"),
            payout_ratio=Decimal("0.27"),
            enterprise_value=3_200_000_000_000,
            revenue_growth=Decimal("0.16"),
            earnings_growth=Decimal("0.10"),
        )
        assert result.earnings_yield == Decimal("0.0274")
        assert result.fcf_yield == Decimal("0.0195")
        assert result.peg_ratio == Decimal("2.1")
        assert result.ev_ebitda == Decimal("22.5")
        assert result.piotroski_score == 7
        assert result.operating_margin == Decimal("0.43")
        assert result.return_on_assets == Decimal("0.19")
        assert result.current_ratio == Decimal("1.77")
        assert result.beta == Decimal("0.89")
        assert result.payout_ratio == Decimal("0.27")
        assert result.enterprise_value == 3_200_000_000_000
        assert result.revenue_growth == Decimal("0.16")
        assert result.earnings_growth == Decimal("0.10")

    def test_new_fields_default_to_zero(self) -> None:
        result = ValueScreenerResult(
            symbol="TEST",
            price=Decimal("100.00"),
            market_cap=1_000_000_000,
        )
        assert result.earnings_yield == Decimal("0")
        assert result.fcf_yield == Decimal("0")
        assert result.peg_ratio == Decimal("0")
        assert result.ev_ebitda == Decimal("0")
        assert result.piotroski_score == 0
        assert result.operating_margin == Decimal("0")
        assert result.return_on_assets == Decimal("0")
        assert result.current_ratio == Decimal("0")
        assert result.beta == Decimal("0")
        assert result.payout_ratio == Decimal("0")
        assert result.enterprise_value == 0
        assert result.revenue_growth == Decimal("0")
        assert result.earnings_growth == Decimal("0")
