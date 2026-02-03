"""Calculates stock valuation metrics including Graham Number, intrinsic value,
margin of safety, P/E ratios, and projected returns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.util.big_decimal_math import (
    add,
    average,
    divide,
    multiply,
    scale2,
    subtract,
)

# Type imports only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.model.financial_data import FinancialData
    from stockdownloader.model.historical_data import HistoricalData
    from stockdownloader.model.quote_data import QuoteData

_ZERO = Decimal("0")
_ONE = Decimal("1")

DEFAULT_FIXED_EPS_GROWTH = Decimal("0.06")
DEFAULT_DESIRED_RETURN = Decimal("0.05")
DEFAULT_CORPORATE_BONDS_YIELD = Decimal("4.09")
DEFAULT_RATE_OF_RETURN = Decimal("4.4")


@dataclass
class ValuationInputs:
    """Required EPS-related inputs that must be provided before calculation."""

    eps_year_one: Decimal
    eps_year_five: Decimal
    eps_estimate_next_year: Decimal
    eps_growth: Decimal
    five_year_period: float


@dataclass
class FormulaCalculator:
    """Calculates stock valuation metrics including Graham Number, intrinsic value,
    margin of safety, P/E ratios, and projected returns.
    """

    # Fixed rates
    fixed_eps_growth: Decimal = field(default=_ZERO, init=False)
    desired_return_per_year: Decimal = field(default=_ZERO, init=False)
    corporate_bonds_yield: Decimal = field(default=_ZERO, init=False)
    rate_of_return: Decimal = field(default=_ZERO, init=False)

    # PS ratio derived values
    difference_from_price_at_min_ps_ratio: Decimal = field(default=_ZERO, init=False)
    difference_from_price_at_max_ps_ratio: Decimal = field(default=_ZERO, init=False)

    # Growth
    growth_multiple: Decimal = field(default=_ZERO, init=False)
    five_year_growth_multiple: Decimal = field(default=_ZERO, init=False)

    # Range
    year_low_difference: Decimal = field(default=_ZERO, init=False)
    years_range_difference: Decimal = field(default=_ZERO, init=False)

    # Growth rates
    compound_annual_growth_rate: Decimal = field(default=_ZERO, init=False)
    fool_eps_growth: Decimal = field(default=_ZERO, init=False)

    # Valuation
    intrinsic_value: Decimal = field(default=_ZERO, init=False)
    graham_margin_of_safety: Decimal = field(default=_ZERO, init=False)
    buffett_margin_of_safety: Decimal = field(default=_ZERO, init=False)

    # PE ratios
    pe_ratio_ttm: Decimal = field(default=_ZERO, init=False)
    forward_pe_ratio: Decimal = field(default=_ZERO, init=False)
    assumed_forward_pe: Decimal = field(default=_ZERO, init=False)

    # Projections
    eps_over_holding_period_year_one: Decimal = field(default=_ZERO, init=False)
    eps_over_holding_period_year_two: Decimal = field(default=_ZERO, init=False)
    eps_over_holding_period_year_three: Decimal = field(default=_ZERO, init=False)
    eps_over_holding_period_total: Decimal = field(default=_ZERO, init=False)
    expected_share_price_in_three_years: Decimal = field(default=_ZERO, init=False)
    dividend_payout_ratio: Decimal = field(default=_ZERO, init=False)
    total_dividends_per_share_over_three_years: Decimal = field(default=_ZERO, init=False)
    expected_share_value_at_end_of_three_years: Decimal = field(default=_ZERO, init=False)
    present_share_value_for_good_value: Decimal = field(default=_ZERO, init=False)
    latest_price_sales: Decimal = field(default=_ZERO, init=False)

    # PS ratio values
    max_ps_ratio_this_qtr: Decimal = field(default=_ZERO, init=False)
    min_ps_ratio_this_qtr: Decimal = field(default=_ZERO, init=False)
    max_ps_ratio_last_qtr: Decimal = field(default=_ZERO, init=False)
    min_ps_ratio_last_qtr: Decimal = field(default=_ZERO, init=False)
    price_at_max_ps_ratio_this_qtr: Decimal = field(default=_ZERO, init=False)
    price_at_max_ps_ratio_last_qtr: Decimal = field(default=_ZERO, init=False)
    price_at_min_ps_ratio_this_qtr: Decimal = field(default=_ZERO, init=False)
    price_at_min_ps_ratio_last_qtr: Decimal = field(default=_ZERO, init=False)

    def calculate(
        self,
        yf: QuoteData,
        ms: FinancialData,
        yh: HistoricalData,
        inputs: ValuationInputs,
    ) -> None:
        """Run the full valuation calculation."""
        revenue_per_share_ttm = ms.revenue_per_share_ttm
        revenue_per_share_ttm_last_qtr = ms.revenue_per_share_ttm_last_qtr

        self._compute_ps_ratios(yh, revenue_per_share_ttm, revenue_per_share_ttm_last_qtr)
        self._compute_fixed_rates()
        self._compute_valuation_metrics(yf, inputs)
        self._compute_projections(yf, ms, inputs)

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    def _compute_ps_ratios(
        self,
        yh: HistoricalData,
        rev_ttm: Decimal,
        rev_ttm_last_qtr: Decimal,
    ) -> None:
        self.max_ps_ratio_this_qtr = divide(yh.highest_price_this_qtr, rev_ttm, 2)
        self.min_ps_ratio_this_qtr = divide(yh.lowest_price_this_qtr, rev_ttm, 2)
        self.max_ps_ratio_last_qtr = divide(yh.highest_price_last_qtr, rev_ttm_last_qtr, 2)
        self.min_ps_ratio_last_qtr = divide(yh.lowest_price_last_qtr, rev_ttm_last_qtr, 2)

        max_ratio = max(self.max_ps_ratio_this_qtr, self.max_ps_ratio_last_qtr)
        min_ratio = min(self.min_ps_ratio_this_qtr, self.min_ps_ratio_last_qtr)

        self.price_at_max_ps_ratio_this_qtr = scale2(max_ratio * rev_ttm)
        self.price_at_max_ps_ratio_last_qtr = scale2(max_ratio * rev_ttm_last_qtr)
        self.price_at_min_ps_ratio_this_qtr = scale2(min_ratio * rev_ttm)
        self.price_at_min_ps_ratio_last_qtr = scale2(min_ratio * rev_ttm_last_qtr)

    def _compute_fixed_rates(self) -> None:
        self.fixed_eps_growth = DEFAULT_FIXED_EPS_GROWTH
        self.desired_return_per_year = DEFAULT_DESIRED_RETURN
        self.corporate_bonds_yield = DEFAULT_CORPORATE_BONDS_YIELD
        self.rate_of_return = DEFAULT_RATE_OF_RETURN

    def _compute_valuation_metrics(self, yf: QuoteData, inputs: ValuationInputs) -> None:
        price = yf.last_trade_price_only
        eps = yf.diluted_eps

        self.difference_from_price_at_min_ps_ratio = scale2(
            subtract(_ONE, divide(self.price_at_min_ps_ratio_this_qtr, price))
        )
        self.difference_from_price_at_max_ps_ratio = scale2(
            subtract(_ONE, divide(price, self.price_at_max_ps_ratio_this_qtr))
        )
        self.growth_multiple = scale2(divide(inputs.eps_year_five, inputs.eps_year_one))
        self.five_year_growth_multiple = scale2(
            Decimal(str(math.pow(abs(float(abs(self.growth_multiple))), inputs.five_year_period)))
        )
        self.year_low_difference = scale2(subtract(_ONE, divide(yf.year_low, price)))
        self.years_range_difference = scale2(subtract(yf.year_high, yf.year_low))
        self.compound_annual_growth_rate = scale2(
            multiply(subtract(self.five_year_growth_multiple, _ONE), Decimal("100"))
        )
        self.fool_eps_growth = scale2(divide(subtract(inputs.eps_estimate_next_year, eps), eps))

        adjusted_growth = add(
            Decimal("8.5"),
            multiply(Decimal("2"), multiply(inputs.eps_growth, Decimal("100"))),
        )
        self.intrinsic_value = scale2(
            divide(multiply(multiply(eps, adjusted_growth), self.rate_of_return), self.corporate_bonds_yield)
        )
        self.graham_margin_of_safety = scale2(divide(self.intrinsic_value, price))
        self.buffett_margin_of_safety = scale2(multiply(self.intrinsic_value, Decimal("0.75")))

        self.pe_ratio_ttm = scale2(divide(price, eps))
        self.forward_pe_ratio = scale2(divide(price, inputs.eps_estimate_next_year))
        self.assumed_forward_pe = scale2(average(self.pe_ratio_ttm, self.forward_pe_ratio))

    def _compute_projections(
        self,
        yf: QuoteData,
        ms: FinancialData,
        inputs: ValuationInputs,
    ) -> None:
        eps = yf.diluted_eps
        growth_factor = add(inputs.eps_growth, _ONE)

        self.eps_over_holding_period_year_one = scale2(multiply(eps, growth_factor))
        self.eps_over_holding_period_year_two = scale2(
            multiply(self.eps_over_holding_period_year_one, growth_factor)
        )
        self.eps_over_holding_period_year_three = scale2(
            multiply(self.eps_over_holding_period_year_two, growth_factor)
        )
        self.eps_over_holding_period_total = scale2(
            add(
                self.eps_over_holding_period_year_one,
                add(self.eps_over_holding_period_year_two, self.eps_over_holding_period_year_three),
            )
        )

        self.expected_share_price_in_three_years = scale2(
            multiply(self.eps_over_holding_period_year_three, self.assumed_forward_pe)
        )
        self.dividend_payout_ratio = scale2(
            divide(yf.trailing_annual_dividend_yield, self.eps_over_holding_period_year_three)
        )
        self.total_dividends_per_share_over_three_years = scale2(
            multiply(self.dividend_payout_ratio, self.eps_over_holding_period_total)
        )
        self.expected_share_value_at_end_of_three_years = scale2(
            add(self.total_dividends_per_share_over_three_years, self.expected_share_price_in_three_years)
        )
        self.present_share_value_for_good_value = scale2(
            divide(
                self.expected_share_value_at_end_of_three_years,
                add(_ONE, self.desired_return_per_year) ** 3,
            )
        )
        self.latest_price_sales = scale2(divide(yf.last_trade_price_only, ms.revenue_per_share_ttm))
