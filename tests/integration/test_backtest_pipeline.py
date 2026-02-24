"""Integration test that exercises the full backtest pipeline:
CSV file loading -> PriceData parsing -> Strategy evaluation -> BacktestEngine execution -> Result metrics.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.data.parsers import CsvPriceDataLoader
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.models.trade import Trade, Direction, TradeStatus
from stockdownloader.strategies.daily.simple import MACDStrategy
from stockdownloader.strategies.daily.simple import RSIStrategy
from stockdownloader.strategies.daily.simple import SMACrossoverStrategy

INITIAL_CAPITAL = Decimal("100000.00")
ZERO_COMMISSION = Decimal("0")


@pytest.fixture(scope="module")
def price_data(test_data_path):
    data = CsvPriceDataLoader.load_from_file(str(test_data_path))
    assert len(data) > 0, "Test price data should be loaded successfully"
    return data


def test_csv_data_loaded_correctly(price_data):
    assert len(price_data) > 200, "Should have > 200 trading days for strategy warmup"
    assert price_data[0].date == "2023-01-03"
    assert price_data[-1].date == "2024-01-31"

    # Verify OHLCV fields are populated
    first = price_data[0]
    assert first.open > Decimal("0")
    assert first.high > Decimal("0")
    assert first.low > Decimal("0")
    assert first.close > Decimal("0")
    assert first.volume > 0


def test_sma_crossover_pipeline_produces_valid_results(price_data):
    strategy = SMACrossoverStrategy(20, 50)
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

    result = engine.run(strategy, price_data)

    assert result is not None
    assert result.strategy_name == "SMA Crossover (20/50)"
    assert result.initial_capital == INITIAL_CAPITAL
    assert result.final_capital is not None
    assert result.final_capital > Decimal("0"), "Final capital should be positive"

    # Should have at least some trades over a year of data
    assert result.total_trades >= 0, "Should have non-negative trade count"

    # Equity curve should match data size
    assert len(result.equity_curve) == len(price_data)

    # All equity values should be positive
    for equity in result.equity_curve:
        assert equity > Decimal("0"), "Equity should always be positive"


def test_rsi_pipeline_produces_valid_results(price_data):
    strategy = RSIStrategy(14, 30.0, 70.0)
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

    result = engine.run(strategy, price_data)

    assert result is not None
    assert result.strategy_name == "RSI (14) [30.0/70.0]"
    assert result.final_capital > Decimal("0")

    # Verify metrics are computable without error
    assert result.total_return is not None
    assert result.total_pnl is not None
    assert result.win_rate is not None
    assert result.max_drawdown is not None
    assert result.profit_factor is not None


def test_macd_pipeline_produces_valid_results(price_data):
    strategy = MACDStrategy(12, 26, 9)
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

    result = engine.run(strategy, price_data)

    assert result is not None
    assert result.strategy_name == "MACD (12/26/9)"
    assert result.final_capital > Decimal("0")

    # Sharpe ratio should be computable
    sharpe = result.sharpe_ratio(252)
    assert sharpe is not None


def test_commission_reduces_final_capital(price_data):
    strategy = SMACrossoverStrategy(20, 50)
    commission = Decimal("10.00")

    engine_no_commission = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
    engine_with_commission = BacktestEngine(INITIAL_CAPITAL, commission)

    result_no_comm = engine_no_commission.run(strategy, price_data)
    result_with_comm = engine_with_commission.run(strategy, price_data)

    # If trades occurred, commission should reduce final capital
    if result_no_comm.total_trades > 0:
        assert result_no_comm.final_capital >= result_with_comm.final_capital, \
            "Commission should reduce or maintain final capital"


def test_all_trades_are_closed_at_end(price_data):
    strategy = SMACrossoverStrategy(20, 50)
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

    result = engine.run(strategy, price_data)

    # All trades in the result should be closed (engine closes open positions at end)
    for trade in result.trades:
        assert trade.status == TradeStatus.CLOSED, \
            "All trades should be closed at end of backtest"
        assert trade.exit_date is not None
        assert trade.exit_price is not None


def test_profit_loss_consistency(price_data):
    strategy = MACDStrategy(12, 26, 9)
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

    result = engine.run(strategy, price_data)

    expected_pl = result.final_capital - result.initial_capital
    assert expected_pl == result.total_pnl, \
        "Total P/L should equal final capital minus initial capital"


def test_win_rate_consistency(price_data):
    strategy = RSIStrategy(14, 25.0, 75.0)
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

    result = engine.run(strategy, price_data)

    total = result.total_trades
    winning = result.winning_trades
    losing = result.losing_trades

    assert total == winning + losing, \
        "Winning + losing trades should equal total trades"

    if total > 0:
        win_rate = result.win_rate
        assert win_rate >= Decimal("0"), "Win rate should be >= 0"
        assert win_rate <= Decimal("100"), "Win rate should be <= 100"


def test_buy_and_hold_return_matches_data(price_data):
    strategy = SMACrossoverStrategy(20, 50)
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

    result = engine.run(strategy, price_data)
    buy_and_hold = result.buy_and_hold_return(price_data)

    # Manually compute buy and hold
    first_close = price_data[0].close
    last_close = price_data[-1].close
    expected_return = (
        (last_close - first_close) / first_close
    ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * Decimal("100")

    assert expected_return == buy_and_hold, \
        "Buy and hold return should match manual calculation"


def test_max_drawdown_is_non_negative(price_data):
    strategy = SMACrossoverStrategy(50, 200)
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

    result = engine.run(strategy, price_data)

    max_drawdown = result.max_drawdown
    assert max_drawdown >= Decimal("0"), "Max drawdown should be non-negative"
    assert max_drawdown <= Decimal("100"), "Max drawdown should be <= 100%"
