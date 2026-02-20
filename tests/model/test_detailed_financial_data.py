"""Tests for DetailedFinancialData model."""
from decimal import Decimal

from stockdownloader.model.financial_models import DetailedFinancialData


class TestDetailedFinancialData:
    """Tests for the DetailedFinancialData dataclass."""

    def test_default_construction(self) -> None:
        data = DetailedFinancialData()
        assert data.symbol == ""
        assert data.profit_margin == Decimal("0")
        assert data.operating_margin == Decimal("0")
        assert data.return_on_equity == Decimal("0")
        assert data.return_on_assets == Decimal("0")
        assert data.total_debt == 0
        assert data.total_equity == 0
        assert data.debt_to_equity == Decimal("0")
        assert data.current_ratio == Decimal("0")
        assert data.revenue_growth == Decimal("0")
        assert data.earnings_growth == Decimal("0")
        assert data.free_cash_flow == 0
        assert data.operating_cash_flow == 0
        assert data.earnings_estimate_next_qtr == Decimal("0")
        assert data.earnings_estimate_next_year == Decimal("0")
        assert data.incomplete is False

    def test_field_assignment(self) -> None:
        data = DetailedFinancialData(
            symbol="AAPL",
            profit_margin=Decimal("0.25"),
            return_on_equity=Decimal("0.18"),
            debt_to_equity=Decimal("1.5"),
            free_cash_flow=90_000_000_000,
        )
        assert data.symbol == "AAPL"
        assert data.profit_margin == Decimal("0.25")
        assert data.return_on_equity == Decimal("0.18")
        assert data.debt_to_equity == Decimal("1.5")
        assert data.free_cash_flow == 90_000_000_000

    def test_incomplete_flag(self) -> None:
        data = DetailedFinancialData(incomplete=True)
        assert data.incomplete is True
