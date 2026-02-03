"""Fundamental financial data model holding revenue, shares outstanding,
and derived revenue-per-share metrics."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING


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
