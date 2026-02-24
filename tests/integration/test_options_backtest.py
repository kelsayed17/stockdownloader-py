"""Integration tests that exercise the full options backtesting pipeline:
data loading -> strategy evaluation -> engine execution -> result analysis.
"""
from __future__ import annotations

import math
from decimal import Decimal

import pytest

from stockdownloader.backtest.options_backtest_engine import OptionsBacktestEngine
from stockdownloader.backtest.backtest_result import OptionsBacktestResult
from stockdownloader.data.parsers import CsvPriceDataLoader
from stockdownloader.core.models.options import OptionContract, OptionType, OptionsChain
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.models.financial import QuoteData
from stockdownloader.core.models.market_data import (
    FinancialData,
    HistoricalData,
    UnifiedMarketData,
)
from stockdownloader.strategy.options.options_strategies import CoveredCallStrategy, ProtectivePutStrategy
from stockdownloader.strategy.options.options_strategies import OptionsStrategy
from stockdownloader.analysis.options.pricing import (
    price as bs_price,
    estimate_volatility,
    delta as bs_delta,
)


def _generate_synthetic_data(days: int) -> list[PriceData]:
    data = []
    price = 450.0
    for i in range(days):
        price += math.sin(i * 0.1) * 2 + 0.1
        p = Decimal(str(price))
        h = p + Decimal("2")
        low = p - Decimal("2")
        data.append(PriceData(
            f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            p, h, low, p, p, 500000 + int(100000 * (i % 7) / 7),
        ))
    return data


def _build_sample_chain(current_price: Decimal) -> OptionsChain:
    chain = OptionsChain("SPY")
    chain.underlying_price = current_price
    chain.add_expiration_date("2024-03-15")

    strikes = [
        current_price - Decimal("20"),
        current_price - Decimal("10"),
        current_price,
        current_price + Decimal("10"),
        current_price + Decimal("20"),
    ]

    for strike in strikes:
        chain.add_call("2024-03-15", OptionContract(
            f"SPY-C-{strike}", OptionType.CALL, strike, "2024-03-15",
            Decimal("5.00"), Decimal("4.90"), Decimal("5.10"),
            1000, 5000,
            Decimal("0.20"),
            Decimal("0.50"), Decimal("0.02"),
            Decimal("-0.10"), Decimal("0.15"),
            strike < current_price,
        ))

        chain.add_put("2024-03-15", OptionContract(
            f"SPY-P-{strike}", OptionType.PUT, strike, "2024-03-15",
            Decimal("4.00"), Decimal("3.90"), Decimal("4.10"),
            800, 4000,
            Decimal("0.22"),
            Decimal("-0.40"), Decimal("0.02"),
            Decimal("-0.08"), Decimal("0.12"),
            strike > current_price,
        ))

    return chain


@pytest.fixture(scope="module")
def test_data(test_data_path):
    data = CsvPriceDataLoader.load_from_file(str(test_data_path))
    if not data:
        # Fall back to synthetic data if no CSV available
        data = _generate_synthetic_data(200)
    return data


def test_covered_call_backtest_produces_valid_results(test_data):
    strategy = CoveredCallStrategy(20, Decimal("0.05"), 30, Decimal("0.03"))
    engine = OptionsBacktestEngine(Decimal("100000"), Decimal("0"))

    result = engine.run(strategy, test_data)

    assert result is not None
    assert result.final_capital is not None
    assert len(result.equity_curve) == len(test_data)
    assert result.start_date == test_data[0].date
    assert result.end_date == test_data[-1].date


def test_protective_put_backtest_produces_valid_results(test_data):
    strategy = ProtectivePutStrategy(20, Decimal("0.05"), 30, 5)
    engine = OptionsBacktestEngine(Decimal("100000"), Decimal("0"))

    result = engine.run(strategy, test_data)

    assert result is not None
    assert result.final_capital is not None


def test_multiple_strategies_compare(test_data):
    strategies = [
        CoveredCallStrategy(20, Decimal("0.03"), 30, Decimal("0.03")),
        CoveredCallStrategy(20, Decimal("0.05"), 30, Decimal("0.03")),
        ProtectivePutStrategy(20, Decimal("0.05"), 30, 5),
        ProtectivePutStrategy(20, Decimal("0.03"), 45, 10),
    ]

    engine = OptionsBacktestEngine(Decimal("100000"), Decimal("0.65"))
    results = []

    for strategy in strategies:
        result = engine.run(strategy, test_data)
        results.append(result)
        assert result.strategy_name is not None

    assert len(results) == 4


def test_volume_is_captured_through_pipeline(test_data):
    strategy = CoveredCallStrategy(5, Decimal("0.05"), 10, Decimal("0.03"))
    engine = OptionsBacktestEngine(Decimal("100000"), Decimal("0"))

    result = engine.run(strategy, test_data)

    # Verify volume is tracked
    for trade in result.trades:
        assert trade.entry_volume >= 0, "Every trade should capture volume"


def test_unified_market_data_integration(test_data):
    unified = UnifiedMarketData("SPY")

    # Set up all data sources
    unified.price_history = test_data
    unified.latest_price = test_data[-1]

    quote = QuoteData()
    quote.last_trade_price_only = test_data[-1].close
    quote.volume = Decimal(str(test_data[-1].volume))
    unified.quote = quote

    chain = _build_sample_chain(test_data[-1].close)
    unified.options_chain = chain

    unified.historical = HistoricalData("SPY")
    unified.financials = FinancialData()

    # Verify unified data
    assert unified.is_complete()
    assert unified.current_price > Decimal("0")
    assert unified.has_price_history()
    assert unified.has_options_chain()

    # Equity + options volume
    assert unified.total_combined_volume > Decimal("0")
    assert unified.options_volume > 0
    assert unified.get_average_daily_volume(20) > Decimal("0")


def test_black_scholes_integration_with_backtest(test_data):
    spot = test_data[-1].close
    strike = spot * Decimal("1.05")
    time_to_expiry = Decimal("0.0822")  # ~30 days
    rate = Decimal("0.05")

    close_prices = [d.close for d in test_data]
    vol = estimate_volatility(close_prices, 20)

    call_price = bs_price(OptionType.CALL, spot, strike, time_to_expiry, rate, vol)
    put_price = bs_price(OptionType.PUT, spot, strike, time_to_expiry, rate, vol)

    assert float(call_price) > 0, "Call should have positive price"
    assert float(put_price) > 0, "Put should have positive price"
    assert float(vol) > 0, "Estimated vol should be positive"

    d = bs_delta(OptionType.CALL, spot, strike, time_to_expiry, rate, vol)
    assert 0 < float(d) < 1


def test_options_chain_search_integration(test_data):
    price = test_data[-1].close
    chain = _build_sample_chain(price)

    # Find nearest strike to current price
    nearest = chain.find_nearest_strike("2024-03-15", OptionType.CALL, price)
    assert nearest is not None

    # Get all contracts at a specific strike
    target_strike = nearest.strike
    contracts = chain.get_contracts_at_strike("2024-03-15", target_strike)
    assert len(contracts) > 0

    # Volume metrics
    assert chain.total_volume > 0
    assert chain.total_call_open_interest > 0
