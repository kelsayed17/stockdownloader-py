"""Integration test that runs multiple strategies against the same dataset
and validates cross-strategy consistency and relative metric properties.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.data.parsers import CsvPriceDataLoader
from stockdownloader.core.models.price import PriceData
from stockdownloader.strategy.daily.simple_strategies import MACDStrategy
from stockdownloader.strategy.daily.simple_strategies import RSIStrategy
from stockdownloader.strategy.daily.simple_strategies import SMACrossoverStrategy
from stockdownloader.strategy.trading_strategy import TradingStrategy

INITIAL_CAPITAL = Decimal("100000.00")


@pytest.fixture(scope="module")
def price_data(test_data_path):
    data = CsvPriceDataLoader.load_from_file(str(test_data_path))
    assert len(data) > 0
    return data


@pytest.fixture(scope="module")
def strategies():
    return [
        SMACrossoverStrategy(20, 50),
        SMACrossoverStrategy(50, 200),
        RSIStrategy(14, 30.0, 70.0),
        RSIStrategy(14, 25.0, 75.0),
        MACDStrategy(12, 26, 9),
    ]


@pytest.fixture(scope="module")
def results(price_data, strategies):
    engine = BacktestEngine(INITIAL_CAPITAL, Decimal("0"))
    return [engine.run(strategy, price_data) for strategy in strategies]


def test_all_strategies_produce_results(strategies, results):
    assert len(results) == len(strategies)
    for result in results:
        assert result is not None
        assert result.strategy_name is not None
        assert result.final_capital is not None


def test_all_strategies_have_same_initial_capital(results):
    for result in results:
        assert result.initial_capital == INITIAL_CAPITAL


def test_all_strategies_have_same_date_range(price_data, results):
    expected_start = price_data[0].date
    expected_end = price_data[-1].date

    for result in results:
        assert result.start_date == expected_start
        assert result.end_date == expected_end


def test_all_strategies_have_same_buy_and_hold_return(price_data, results):
    expected = results[0].buy_and_hold_return(price_data)
    for result in results:
        assert expected == result.buy_and_hold_return(price_data), \
            "Buy and hold return should be identical across strategies"


def test_equity_curve_size_matches_data_size(price_data, results):
    for result in results:
        assert len(result.equity_curve) == len(price_data), \
            f"Equity curve size should match data size for {result.strategy_name}"


def test_metrics_are_within_reasonable_bounds(results):
    for result in results:
        name = result.strategy_name

        total_return = result.total_return
        assert total_return >= Decimal("-100"), f"{name} return should be >= -100%"
        assert total_return <= Decimal("1000"), \
            f"{name} return should be <= 1000% for ~1 year of data"

        max_drawdown = result.max_drawdown
        assert max_drawdown >= Decimal("0"), f"{name} max drawdown should be >= 0"
        assert max_drawdown <= Decimal("100"), f"{name} max drawdown should be <= 100%"

        win_rate = result.win_rate
        assert win_rate >= Decimal("0"), f"{name} win rate should be >= 0"
        assert win_rate <= Decimal("100"), f"{name} win rate should be <= 100"


def test_shorter_sma_period_generates_more_trades(results):
    # SMA(20/50) should generally generate >= trades than SMA(50/200) on same data
    sma_20_50 = results[0]
    sma_50_200 = results[1]

    assert sma_20_50.total_trades >= sma_50_200.total_trades, \
        "SMA(20/50) should generate >= trades than SMA(50/200)"


def test_profit_factor_consistency_across_strategies(results):
    for result in results:
        profit_factor = result.profit_factor
        assert profit_factor is not None, \
            f"Profit factor should not be null for {result.strategy_name}"
        assert profit_factor >= Decimal("0"), \
            f"Profit factor should be non-negative for {result.strategy_name}"


def test_determinism_same_input_same_output(price_data):
    # Running the same strategy twice should produce identical results
    engine = BacktestEngine(INITIAL_CAPITAL, Decimal("0"))
    strategy = SMACrossoverStrategy(20, 50)

    run1 = engine.run(strategy, price_data)
    run2 = engine.run(strategy, price_data)

    assert run1.final_capital == run2.final_capital
    assert run1.total_trades == run2.total_trades
    assert run1.total_return == run2.total_return
    assert run1.max_drawdown == run2.max_drawdown


def test_rsi_wider_thresholds_fewer_trades(results):
    # RSI(14, 25/75) is wider than RSI(14, 30/70), so should generate <= signals
    rsi_narrow = results[2]  # 30/70
    rsi_wide = results[3]    # 25/75

    assert rsi_narrow.total_trades >= rsi_wide.total_trades, \
        "Wider RSI thresholds should produce fewer or equal trades"
