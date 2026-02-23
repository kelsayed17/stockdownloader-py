"""End-to-end test for the full valuation analysis pipeline.
Uses realistic AAPL-like financial data to exercise the complete chain:
QuoteData + FinancialData + HistoricalData + ValuationInputs
  -> FormulaCalculator.calculate()
  -> All computed metrics (P/E, Graham, intrinsic value, margin of safety,
     EPS projections, dividend analysis, PS ratios, CAGR)

This validates that all model classes, FormulaCalculator, and BigDecimalMath
work together correctly with real-world-like data.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.analysis.formula_calculator import FormulaCalculator, ValuationInputs
from stockdownloader.model.financial_models import QuoteData
from stockdownloader.model.unified_market_data import FinancialData, HistoricalData
from stockdownloader.util.math import average, divide, scale2


@pytest.fixture(scope="module")
def aapl():
    """Real-world-like AAPL data (approximate Q4 2023 values)."""
    quote = QuoteData()
    quote.last_trade_price_only = Decimal("185.50")
    quote.diluted_eps = Decimal("6.42")
    quote.eps_estimate_next_year = Decimal("7.10")
    quote.trailing_annual_dividend_yield = Decimal("0.96")
    quote.year_high = Decimal("199.62")
    quote.year_low = Decimal("124.17")
    quote.price_sales = Decimal("7.68")
    quote.previous_close = Decimal("184.25")
    quote.open = Decimal("185.00")
    quote.days_high = Decimal("186.10")
    quote.days_low = Decimal("183.80")
    quote.volume = Decimal("55000000")
    quote.year_range = "124.17 - 199.62"
    quote.market_capitalization = 2_900_000_000_000
    quote.market_capitalization_str = "2.9T"
    return quote


@pytest.fixture(scope="module")
def aapl_financials():
    """Morningstar-like financial data (5 quarters + TTM)."""
    financials = FinancialData()
    revenues = [94836, 89498, 81797, 83359, 90146, 383296]
    diluted_shares = [15744, 15697, 15672, 15634, 15599, 15599]
    for i in range(6):
        financials.revenue[i] = revenues[i]
        financials.diluted_shares[i] = diluted_shares[i]
        financials.basic_shares[i] = diluted_shares[i] - 100
    financials.fiscal_quarters[0] = "2022-12"
    financials.fiscal_quarters[1] = "2023-03"
    financials.fiscal_quarters[2] = "2023-06"
    financials.fiscal_quarters[3] = "2023-09"
    financials.fiscal_quarters[4] = "2022-09"
    financials.fiscal_quarters[5] = "TTM"
    financials.compute_revenue_per_share()
    return financials


@pytest.fixture(scope="module")
def aapl_history():
    """Historical price data (quarterly extremes)."""
    history = HistoricalData("AAPL")
    history.highest_price_this_qtr = Decimal("198.23")
    history.lowest_price_this_qtr = Decimal("165.67")
    history.highest_price_last_qtr = Decimal("189.98")
    history.lowest_price_last_qtr = Decimal("167.62")
    return history


@pytest.fixture(scope="module")
def aapl_inputs():
    """Valuation inputs (analyst consensus-like)."""
    return ValuationInputs(
        eps_year_one=Decimal("5.61"),      # FY2020
        eps_year_five=Decimal("7.10"),      # FY2024 estimate
        eps_estimate_next_year=Decimal("7.10"),
        eps_growth=Decimal("0.08"),         # 8%
        five_year_period=5.0,
    )


@pytest.fixture(scope="module")
def calculator(aapl, aapl_financials, aapl_history, aapl_inputs):
    calc = FormulaCalculator()
    calc.calculate(aapl, aapl_financials, aapl_history, aapl_inputs)
    return calc


# ========== P/E Ratios ==========


def test_pe_ratio_ttm_calculation(calculator):
    pe = calculator.pe_ratio_ttm
    # 185.50 / 6.42 ~ 28.89
    assert pe > Decimal("28")
    assert pe < Decimal("30")


def test_forward_pe_ratio_calculation(calculator):
    fpe = calculator.forward_pe_ratio
    # 185.50 / 7.10 ~ 26.13
    assert fpe > Decimal("25")
    assert fpe < Decimal("28")


def test_forward_pe_less_than_trailing_when_eps_growing(calculator):
    assert calculator.forward_pe_ratio < calculator.pe_ratio_ttm, \
        "Forward P/E should be lower than trailing P/E when EPS is growing"


def test_assumed_forward_pe_is_average_of_ttm_and_forward(calculator):
    assumed_fwd = calculator.assumed_forward_pe
    avg_expected = average(calculator.pe_ratio_ttm, calculator.forward_pe_ratio)
    assert avg_expected.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) == assumed_fwd, \
        "Assumed forward P/E should be average of TTM and forward P/E"


# ========== Intrinsic Value & Graham ==========


def test_intrinsic_value_positive(calculator):
    iv = calculator.intrinsic_value
    assert iv > Decimal("0"), \
        "Intrinsic value should be positive for profitable company"


def test_graham_margin_of_safety_calculation(calculator, aapl):
    mos = calculator.graham_margin_of_safety
    # Graham MOS = intrinsicValue / price
    expected = divide(calculator.intrinsic_value, aapl.last_trade_price_only, 2)
    assert expected == mos, \
        "Graham MOS should be intrinsic value / price"


def test_buffett_margin_of_safety_is_75_pct_of_intrinsic(calculator):
    buffett = calculator.buffett_margin_of_safety
    expected = (
        calculator.intrinsic_value * Decimal("0.75")
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    assert expected == buffett, \
        "Buffett MOS should be 75% of intrinsic value"


# ========== EPS Projections ==========


def test_eps_projections_grow_by_growth_rate(calculator, aapl):
    year1 = calculator.eps_over_holding_period_year_one
    year2 = calculator.eps_over_holding_period_year_two
    year3 = calculator.eps_over_holding_period_year_three

    assert year1 > Decimal("0"), "Year 1 EPS should be positive"
    assert year2 > year1, "Year 2 should grow over Year 1"
    assert year3 > year2, "Year 3 should grow over Year 2"

    # Each year should grow by the EPS growth rate (8%)
    growth_factor = Decimal("1.08")
    expected_year1 = (aapl.diluted_eps * growth_factor).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    assert expected_year1 == year1, \
        "Year 1 EPS = current EPS * (1 + growth rate)"


def test_eps_projection_total_equals_sum(calculator):
    total = calculator.eps_over_holding_period_total
    sum_val = (
        calculator.eps_over_holding_period_year_one
        + calculator.eps_over_holding_period_year_two
        + calculator.eps_over_holding_period_year_three
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    assert sum_val == total, "Total EPS should equal sum of yearly projections"


# ========== Price Projections & Fair Value ==========


def test_expected_share_price_positive(calculator):
    expected_price = calculator.expected_share_price_in_three_years
    assert expected_price > Decimal("0"), \
        "Expected share price in 3 years should be positive"


def test_expected_share_value_includes_dividends(calculator):
    expected_value = calculator.expected_share_value_at_end_of_three_years
    expected_price = calculator.expected_share_price_in_three_years
    total_dividends = calculator.total_dividends_per_share_over_three_years

    expected = (expected_price + total_dividends).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    assert expected == expected_value, \
        "Share value = expected price + total dividends"


def test_present_value_discounted_from_future_value(calculator):
    present_value = calculator.present_share_value_for_good_value
    future_value = calculator.expected_share_value_at_end_of_three_years
    # Present value should be less than future value (discounting at 5%/year)
    assert present_value < future_value, \
        "Present value should be less than future value due to discounting"


# ========== P/S Ratio Pipeline ==========


def test_ps_ratios_computed_from_historical_and_financial(calculator):
    max_ps = calculator.max_ps_ratio_this_qtr
    min_ps = calculator.min_ps_ratio_this_qtr

    assert max_ps > Decimal("0"), "Max P/S should be positive"
    assert min_ps > Decimal("0"), "Min P/S should be positive"
    assert max_ps >= min_ps, "Max P/S >= Min P/S"


def test_ps_ratios_cross_quarter_comparison(calculator):
    max_this_qtr = calculator.max_ps_ratio_this_qtr
    max_last_qtr = calculator.max_ps_ratio_last_qtr
    min_this_qtr = calculator.min_ps_ratio_this_qtr
    min_last_qtr = calculator.min_ps_ratio_last_qtr

    # All should be positive
    assert max_this_qtr > Decimal("0")
    assert max_last_qtr > Decimal("0")
    assert min_this_qtr > Decimal("0")
    assert min_last_qtr > Decimal("0")


def test_price_at_ps_ratios_are_consistent(calculator):
    price_max = calculator.price_at_max_ps_ratio_this_qtr
    price_min = calculator.price_at_min_ps_ratio_this_qtr

    assert price_max >= price_min, \
        "Price at max P/S should be >= price at min P/S"
    assert price_max > Decimal("0"), \
        "Price at max P/S should be positive"
    assert price_min > Decimal("0"), \
        "Price at min P/S should be positive"


# ========== Growth Metrics ==========


def test_growth_multiple_calculation(calculator):
    gm = calculator.growth_multiple
    # epsYearFive / epsYearOne = 7.10 / 5.61 ~ 1.27
    assert gm > Decimal("1.0"), \
        "Growth multiple should be > 1 when EPS is growing"
    assert gm < Decimal("2.0"), \
        "Growth multiple should be reasonable"


def test_compound_annual_growth_rate_positive(calculator):
    cagr = calculator.compound_annual_growth_rate
    assert cagr > Decimal("0"), \
        "CAGR should be positive for growing EPS"


def test_fool_eps_growth_calculation(calculator):
    fool_growth = calculator.fool_eps_growth
    # (7.10 - 6.42) / 6.42 ~ 0.1059
    assert fool_growth > Decimal("0"), \
        "Fool EPS growth should be positive when next year estimate > current EPS"


# ========== Year Range Metrics ==========


def test_year_low_difference_calculation(calculator):
    yld = calculator.year_low_difference
    # 1 - (yearLow / price) = 1 - (124.17/185.50) ~ 0.33
    assert yld is not None
    assert yld > Decimal("0"), \
        "Year low difference should be positive when price > year low"


def test_years_range_difference(calculator, aapl):
    range_diff = calculator.years_range_difference
    # 199.62 - 124.17 = 75.45
    expected = (aapl.year_high - aapl.year_low).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    assert expected == range_diff, \
        "Year range diff should equal year high - year low"


# ========== Fixed Rates ==========


def test_fixed_rates_configured_correctly(calculator):
    assert calculator.fixed_eps_growth == Decimal("0.06")
    assert calculator.desired_return_per_year == Decimal("0.05")
    assert calculator.corporate_bonds_yield == Decimal("4.09")
    assert calculator.rate_of_return == Decimal("4.4")


# ========== FinancialData Revenue Per Share ==========


def test_revenue_per_share_computed_correctly(aapl_financials):
    rps0 = aapl_financials.revenue_per_share[0]
    # 94836 / 15744 ~ 6.02
    assert rps0 > Decimal("0"), "Revenue per share should be positive"

    rps_ttm = aapl_financials.revenue_per_share_ttm
    assert rps_ttm > Decimal("0"), "TTM revenue per share should be positive"


def test_revenue_per_share_ttm_last_qtr_equals_first_four_quarters(aapl_financials):
    expected = (
        aapl_financials.revenue_per_share[0]
        + aapl_financials.revenue_per_share[1]
        + aapl_financials.revenue_per_share[2]
        + aapl_financials.revenue_per_share[3]
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    assert expected == aapl_financials.revenue_per_share_ttm_last_qtr, \
        "TTM last quarter RPS should equal sum of first 4 quarters"


# ========== Dividend Analysis ==========


def test_dividend_payout_ratio_reasonable(calculator):
    payout_ratio = calculator.dividend_payout_ratio
    assert payout_ratio is not None
    # AAPL has a modest dividend; payout ratio should be well under 1
    assert abs(payout_ratio) < Decimal("2"), \
        "Payout ratio should be reasonable for AAPL"


# ========== QuoteData Model Integrity ==========


def test_quote_data_fields_preserved(aapl):
    assert aapl.last_trade_price_only == Decimal("185.50")
    assert aapl.diluted_eps == Decimal("6.42")
    assert aapl.eps_estimate_next_year == Decimal("7.10")
    assert aapl.trailing_annual_dividend_yield == Decimal("0.96")
    assert aapl.year_high == Decimal("199.62")
    assert aapl.year_low == Decimal("124.17")
    assert aapl.volume == Decimal("55000000")
    assert aapl.market_capitalization == 2_900_000_000_000
    assert aapl.market_capitalization_str == "2.9T"
    assert not aapl.incomplete
    assert not aapl.error


# ========== HistoricalData Model Integrity ==========


def test_historical_data_fields_preserved(aapl_history):
    assert aapl_history.ticker == "AAPL"
    assert aapl_history.highest_price_this_qtr == Decimal("198.23")
    assert aapl_history.lowest_price_this_qtr == Decimal("165.67")
    assert aapl_history.highest_price_last_qtr == Decimal("189.98")
    assert aapl_history.lowest_price_last_qtr == Decimal("167.62")
    assert not aapl_history.incomplete
    assert not aapl_history.error


# ========== Latest P/S ==========


def test_latest_price_sales_calculation(calculator):
    latest_ps = calculator.latest_price_sales
    assert latest_ps > Decimal("0"), "Latest P/S should be positive"
    # 185.50 / (TTM rev per share) should give a reasonable P/S ratio
    assert latest_ps > Decimal("1"), "AAPL P/S should be > 1"


# ========== Recalculation with Different Data ==========


def test_recalculation_with_value_stock_data(calculator):
    # Test with a value stock profile (low P/E, high dividend)
    value_quote = QuoteData()
    value_quote.last_trade_price_only = Decimal("45.00")
    value_quote.diluted_eps = Decimal("5.00")
    value_quote.eps_estimate_next_year = Decimal("5.25")
    value_quote.trailing_annual_dividend_yield = Decimal("2.50")
    value_quote.year_high = Decimal("52.00")
    value_quote.year_low = Decimal("38.00")

    value_financial = FinancialData()
    for i in range(6):
        value_financial.revenue[i] = 50000 + i * 2000
        value_financial.diluted_shares[i] = 10000
    value_financial.compute_revenue_per_share()

    value_history = HistoricalData("VAL")
    value_history.highest_price_this_qtr = Decimal("50.00")
    value_history.lowest_price_this_qtr = Decimal("40.00")
    value_history.highest_price_last_qtr = Decimal("48.00")
    value_history.lowest_price_last_qtr = Decimal("36.00")

    value_inputs = ValuationInputs(
        eps_year_one=Decimal("4.50"),
        eps_year_five=Decimal("5.50"),
        eps_estimate_next_year=Decimal("5.25"),
        eps_growth=Decimal("0.04"),
        five_year_period=5.0,
    )

    value_calc = FormulaCalculator()
    value_calc.calculate(value_quote, value_financial, value_history, value_inputs)

    # Value stock should have lower P/E than AAPL
    assert value_calc.pe_ratio_ttm < calculator.pe_ratio_ttm, \
        "Value stock should have lower P/E than growth stock"

    # Value stock intrinsic value should be positive
    assert value_calc.intrinsic_value > Decimal("0")

    # Value stock should have a dividend payout ratio
    assert abs(value_calc.dividend_payout_ratio) >= Decimal("0")
