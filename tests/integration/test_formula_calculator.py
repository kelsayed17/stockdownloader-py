"""Integration test for the full valuation calculation pipeline:
QuoteData + FinancialData + HistoricalData + ValuationInputs -> FormulaCalculator -> computed metrics.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.analysis.formula_calculator import FormulaCalculator, ValuationInputs
from stockdownloader.model.financial_models import QuoteData
from stockdownloader.model.unified_market_data import FinancialData, HistoricalData


def _create_standard_quote() -> QuoteData:
    quote = QuoteData()
    quote.last_trade_price_only = Decimal("150.00")
    quote.diluted_eps = Decimal("6.00")
    quote.eps_estimate_next_year = Decimal("7.00")
    quote.trailing_annual_dividend_yield = Decimal("0.88")
    quote.year_high = Decimal("180.00")
    quote.year_low = Decimal("120.00")
    quote.price_sales = Decimal("7.5")
    return quote


def _create_standard_financial() -> FinancialData:
    financial = FinancialData()
    for i in range(6):
        financial.revenue[i] = 80000 + i * 4000
        financial.diluted_shares[i] = 15000
    financial.compute_revenue_per_share()
    return financial


def _create_standard_historical() -> HistoricalData:
    historical = HistoricalData("TEST")
    historical.highest_price_this_qtr = Decimal("175.00")
    historical.lowest_price_this_qtr = Decimal("130.00")
    historical.highest_price_last_qtr = Decimal("170.00")
    historical.lowest_price_last_qtr = Decimal("125.00")
    return historical


def test_full_valuation_pipeline():
    # Set up QuoteData (simulating Yahoo Finance data)
    quote = QuoteData()
    quote.last_trade_price_only = Decimal("150.00")
    quote.diluted_eps = Decimal("6.50")
    quote.eps_estimate_next_year = Decimal("7.20")
    quote.trailing_annual_dividend_yield = Decimal("0.92")
    quote.year_high = Decimal("180.00")
    quote.year_low = Decimal("120.00")
    quote.price_sales = Decimal("7.5")

    # Set up FinancialData (simulating Morningstar data)
    financial = FinancialData()
    for i in range(6):
        financial.revenue[i] = 90000 + i * 5000
        financial.diluted_shares[i] = 16000
    financial.compute_revenue_per_share()

    # Set up HistoricalData (simulating historical price data)
    historical = HistoricalData("AAPL")
    historical.highest_price_this_qtr = Decimal("175.00")
    historical.lowest_price_this_qtr = Decimal("130.00")
    historical.highest_price_last_qtr = Decimal("170.00")
    historical.lowest_price_last_qtr = Decimal("125.00")

    # Set up ValuationInputs
    inputs = ValuationInputs(
        eps_year_one=Decimal("4.50"),
        eps_year_five=Decimal("7.50"),
        eps_estimate_next_year=Decimal("7.20"),
        eps_growth=Decimal("0.10"),
        five_year_period=5.0,
    )

    # Run the calculation
    calculator = FormulaCalculator()
    calculator.calculate(quote, financial, historical, inputs)

    # Verify P/E ratios
    pe_ratio_ttm = calculator.pe_ratio_ttm
    assert pe_ratio_ttm is not None
    assert pe_ratio_ttm > Decimal("0"), "P/E ratio TTM should be positive"
    # 150/6.5 ~ 23.08
    assert pe_ratio_ttm > Decimal("22")
    assert pe_ratio_ttm < Decimal("24")

    forward_pe = calculator.forward_pe_ratio
    assert forward_pe is not None
    assert forward_pe > Decimal("0"), "Forward P/E should be positive"
    # 150/7.2 ~ 20.83
    assert forward_pe > Decimal("20")
    assert forward_pe < Decimal("22")

    # Forward P/E should be less than TTM P/E when EPS is growing
    assert forward_pe < pe_ratio_ttm, \
        "Forward P/E should be less than TTM P/E when EPS is growing"


def test_intrinsic_value_calculation():
    quote = _create_standard_quote()
    financial = _create_standard_financial()
    historical = _create_standard_historical()
    inputs = ValuationInputs(
        eps_year_one=Decimal("5.00"),
        eps_year_five=Decimal("8.00"),
        eps_estimate_next_year=Decimal("7.00"),
        eps_growth=Decimal("0.08"),
        five_year_period=5.0,
    )

    calculator = FormulaCalculator()
    calculator.calculate(quote, financial, historical, inputs)

    intrinsic_value = calculator.intrinsic_value
    assert intrinsic_value is not None
    # Intrinsic value should be a reasonable stock price
    assert intrinsic_value > Decimal("0"), \
        "Intrinsic value should be positive for positive EPS"


def test_margin_of_safety_calculation():
    quote = _create_standard_quote()
    financial = _create_standard_financial()
    historical = _create_standard_historical()
    inputs = ValuationInputs(
        eps_year_one=Decimal("5.00"),
        eps_year_five=Decimal("8.00"),
        eps_estimate_next_year=Decimal("7.00"),
        eps_growth=Decimal("0.08"),
        five_year_period=5.0,
    )

    calculator = FormulaCalculator()
    calculator.calculate(quote, financial, historical, inputs)

    graham_mos = calculator.graham_margin_of_safety
    buffett_mos = calculator.buffett_margin_of_safety

    assert graham_mos is not None
    assert buffett_mos is not None

    # Buffett MOS should be 75% of intrinsic value
    intrinsic = calculator.intrinsic_value
    expected_buffett = (intrinsic * Decimal("0.75")).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    assert expected_buffett == buffett_mos, \
        "Buffett MOS should be 75% of intrinsic value"


def test_eps_projections_grow_over_time():
    quote = _create_standard_quote()
    financial = _create_standard_financial()
    historical = _create_standard_historical()
    inputs = ValuationInputs(
        eps_year_one=Decimal("5.00"),
        eps_year_five=Decimal("8.00"),
        eps_estimate_next_year=Decimal("7.00"),
        eps_growth=Decimal("0.10"),
        five_year_period=5.0,
    )

    calculator = FormulaCalculator()
    calculator.calculate(quote, financial, historical, inputs)

    year1 = calculator.eps_over_holding_period_year_one
    year2 = calculator.eps_over_holding_period_year_two
    year3 = calculator.eps_over_holding_period_year_three

    assert year1 > Decimal("0")
    assert year2 > year1, \
        "Year 2 EPS should be greater than Year 1 with positive growth"
    assert year3 > year2, \
        "Year 3 EPS should be greater than Year 2 with positive growth"

    # Total should equal sum
    total = calculator.eps_over_holding_period_total
    expected_total = (year1 + year2 + year3).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    assert expected_total == total, \
        "EPS total should equal sum of yearly projections"


def test_ps_ratio_calculation():
    quote = _create_standard_quote()
    financial = _create_standard_financial()
    historical = _create_standard_historical()
    inputs = ValuationInputs(
        eps_year_one=Decimal("5.00"),
        eps_year_five=Decimal("8.00"),
        eps_estimate_next_year=Decimal("7.00"),
        eps_growth=Decimal("0.08"),
        five_year_period=5.0,
    )

    calculator = FormulaCalculator()
    calculator.calculate(quote, financial, historical, inputs)

    # Max PS ratio should be >= min PS ratio
    max_ps = calculator.max_ps_ratio_this_qtr
    min_ps = calculator.min_ps_ratio_this_qtr
    assert max_ps >= min_ps, "Max P/S ratio should be >= min P/S ratio"

    # Price at max ratio should be >= price at min ratio
    price_max = calculator.price_at_max_ps_ratio_this_qtr
    price_min = calculator.price_at_min_ps_ratio_this_qtr
    assert price_max >= price_min, \
        "Price at max ratio should be >= price at min ratio"


def test_fixed_rates_are_set_correctly():
    quote = _create_standard_quote()
    financial = _create_standard_financial()
    historical = _create_standard_historical()
    inputs = ValuationInputs(
        eps_year_one=Decimal("5.00"),
        eps_year_five=Decimal("8.00"),
        eps_estimate_next_year=Decimal("7.00"),
        eps_growth=Decimal("0.08"),
        five_year_period=5.0,
    )

    calculator = FormulaCalculator()
    calculator.calculate(quote, financial, historical, inputs)

    assert calculator.fixed_eps_growth == Decimal("0.06")
    assert calculator.desired_return_per_year == Decimal("0.05")
    assert calculator.corporate_bonds_yield == Decimal("4.09")
    assert calculator.rate_of_return == Decimal("4.4")


def test_year_range_difference():
    quote = _create_standard_quote()
    financial = _create_standard_financial()
    historical = _create_standard_historical()
    inputs = ValuationInputs(
        eps_year_one=Decimal("5.00"),
        eps_year_five=Decimal("8.00"),
        eps_estimate_next_year=Decimal("7.00"),
        eps_growth=Decimal("0.08"),
        five_year_period=5.0,
    )

    calculator = FormulaCalculator()
    calculator.calculate(quote, financial, historical, inputs)

    range_diff = calculator.years_range_difference
    # 180 - 120 = 60
    assert Decimal("60.00") == range_diff, \
        "Year range difference should be year high minus year low"
