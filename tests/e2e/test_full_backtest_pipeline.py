"""End-to-end test that replicates the full SPYBacktestApp pipeline:
Load real SPY CSV data -> configure all 5 strategies -> run BacktestEngine
-> compute BacktestResult metrics -> generate reports and comparison.

Uses real historical SPY data from test-price-data.csv (272 trading days).
"""
from __future__ import annotations

import io
import sys
from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.report_formatter import print_daily_report, print_daily_comparison
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.data.parsers import CsvPriceDataLoader
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.models.trade import Trade, Direction, TradeStatus
from stockdownloader.strategies.daily.simple import MACDStrategy
from stockdownloader.strategies.daily.simple import RSIStrategy
from stockdownloader.strategies.daily.simple import SMACrossoverStrategy
from stockdownloader.strategies.base import TradingStrategy
from stockdownloader.core.math import percent_change
from stockdownloader.indicators import sma, ema

INITIAL_CAPITAL = Decimal("100000.00")
ZERO_COMMISSION = Decimal("0")
TRADING_DAYS_PER_YEAR = 252


@pytest.fixture(scope="module")
def spy_data(test_data_path):
    data = CsvPriceDataLoader.load_from_file(str(test_data_path))
    assert len(data) > 0, "Real SPY data must be loaded"
    assert len(data) > 250, "Should have ~272 trading days of real data"
    return data


@pytest.fixture(scope="module")
def strategies():
    return [
        SMACrossoverStrategy(50, 200),
        SMACrossoverStrategy(20, 50),
        RSIStrategy(14, 30.0, 70.0),
        RSIStrategy(14, 25.0, 75.0),
        MACDStrategy(12, 26, 9),
    ]


@pytest.fixture(scope="module")
def results(spy_data, strategies):
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
    res = [engine.run(strategy, spy_data) for strategy in strategies]
    assert len(res) == 5, "All 5 strategies should produce results"
    return res


# ========== Data Loading Verification ==========


def test_real_spy_data_has_correct_date_range(spy_data):
    assert spy_data[0].date == "2023-01-03"
    assert spy_data[-1].date == "2024-01-31"


def test_real_spy_data_ohlcv_integrity(spy_data):
    for bar in spy_data:
        assert bar.date is not None
        assert bar.low <= bar.high, f"Low should be <= High on {bar.date}"
        assert bar.open <= bar.high, f"Open should be <= High on {bar.date}"
        assert bar.open >= bar.low, f"Open should be >= Low on {bar.date}"
        assert bar.close <= bar.high, f"Close should be <= High on {bar.date}"
        assert bar.close >= bar.low, f"Close should be >= Low on {bar.date}"
        assert bar.volume > 0, f"Volume should be positive on {bar.date}"


def test_real_spy_data_chronological_order(spy_data):
    for i in range(1, len(spy_data)):
        assert spy_data[i].date > spy_data[i - 1].date, \
            f"Data should be chronologically ordered at index {i}"


# ========== Strategy Name Verification ==========


def test_all_strategy_names_match_expected(results):
    assert results[0].strategy_name == "SMA Crossover (50/200)"
    assert results[1].strategy_name == "SMA Crossover (20/50)"
    assert results[2].strategy_name == "RSI (14) [30.0/70.0]"
    assert results[3].strategy_name == "RSI (14) [25.0/75.0]"
    assert results[4].strategy_name == "MACD (12/26/9)"


# ========== SMA Crossover (50/200) End-to-End ==========


def test_sma_50_200_produces_valid_end_to_end_results(spy_data, results):
    result = results[0]

    assert result.initial_capital == INITIAL_CAPITAL
    assert result.final_capital > Decimal("0")
    assert len(result.equity_curve) == len(spy_data)
    assert result.start_date == "2023-01-03"
    assert result.end_date == "2024-01-31"

    # With only 272 days, SMA(50/200) may have very few trades since warmup needs 200 bars
    assert result.total_trades >= 0

    # Verify moving averages are correctly computed for the last bar
    last_idx = len(spy_data) - 1
    sma50 = sma(spy_data, last_idx, 50)
    sma200 = sma(spy_data, last_idx, 200)
    assert sma50 > Decimal("0")
    assert sma200 > Decimal("0")


# ========== SMA Crossover (20/50) End-to-End ==========


def test_sma_20_50_generates_trades_with_real_data(spy_data, results):
    result = results[1]

    # With 272 days, SMA(20/50) should have enough warmup to generate signals
    assert result.total_trades > 0, \
        "SMA(20/50) should generate at least one trade with 272 days of real SPY data"

    # Verify every trade has valid entry/exit data
    for trade in result.closed_trades:
        assert trade.direction == Direction.LONG, "Engine only produces LONG trades"
        assert trade.status == TradeStatus.CLOSED
        assert trade.entry_date is not None
        assert trade.exit_date is not None
        assert trade.entry_price > Decimal("0")
        assert trade.exit_price > Decimal("0")
        assert trade.shares > 0
        # Verify P/L calculation: (exitPrice - entryPrice) * shares for LONG
        expected_pl = (trade.exit_price - trade.entry_price) * Decimal(str(trade.shares))
        assert expected_pl == trade.profit_loss, \
            "Trade P/L should match manual calculation"


# ========== RSI Strategy End-to-End ==========


def test_rsi_strategies_produce_valid_signals(results):
    rsi_30_70 = results[2]
    rsi_25_75 = results[3]

    # Both RSI strategies should produce valid results
    assert rsi_30_70.final_capital > Decimal("0")
    assert rsi_25_75.final_capital > Decimal("0")

    # Wider thresholds (25/75) should generate fewer or equal trades than (30/70)
    assert rsi_30_70.total_trades >= rsi_25_75.total_trades, \
        "RSI(30/70) should have >= trades than RSI(25/75)"


# ========== MACD Strategy End-to-End ==========


def test_macd_strategy_full_pipeline(results):
    result = results[4]

    assert result.final_capital > Decimal("0")

    # MACD warmup = 26 + 9 = 35 bars, so plenty of room for signals in 272 days
    assert result.total_trades > 0, \
        "MACD should generate at least one trade with real SPY data"

    # Verify Sharpe ratio is computable
    sharpe = result.sharpe_ratio(TRADING_DAYS_PER_YEAR)
    assert sharpe is not None


# ========== Cross-Strategy Consistency ==========


def test_all_strategies_share_buy_and_hold_benchmark(spy_data, results):
    first_close = spy_data[0].close
    last_close = spy_data[-1].close
    expected_buy_hold = (
        (last_close - first_close) / first_close
    ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * Decimal("100")

    for result in results:
        buy_hold = result.buy_and_hold_return(spy_data)
        assert expected_buy_hold == buy_hold, \
            f"Buy-and-hold return should be identical for all strategies on {result.strategy_name}"


def test_capital_conservation_across_strategies(results):
    for result in results:
        total_pl = result.total_pnl
        expected_final = INITIAL_CAPITAL + total_pl
        assert expected_final == result.final_capital, \
            f"Final capital should equal initial + total P/L for {result.strategy_name}"


def test_equity_curve_starts_at_initial_capital(results):
    for result in results:
        first_equity = result.equity_curve[0]
        assert INITIAL_CAPITAL == first_equity, \
            f"Equity curve should start at initial capital for {result.strategy_name}"


def test_equity_curve_all_positive_values(results):
    for result in results:
        for i, equity in enumerate(result.equity_curve):
            assert equity > Decimal("0"), \
                f"Equity should be positive at index {i} for {result.strategy_name}"


# ========== Metrics Computation Verification ==========


def test_total_return_consistency_across_strategies(results):
    for result in results:
        # totalReturn = (finalCapital - initialCapital) / initialCapital * 100
        expected = (
            (result.final_capital - result.initial_capital)
            / result.initial_capital
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * Decimal("100")
        assert expected == result.total_return, \
            f"Total return formula consistency for {result.strategy_name}"


def test_win_loss_trade_counts_add_up(results):
    for result in results:
        assert result.total_trades == result.winning_trades + result.losing_trades, \
            f"Win + Loss should equal total trades for {result.strategy_name}"


def test_max_drawdown_bounds(results):
    for result in results:
        max_dd = result.max_drawdown
        assert max_dd >= Decimal("0"), \
            f"Max drawdown >= 0 for {result.strategy_name}"
        assert max_dd <= Decimal("100"), \
            f"Max drawdown <= 100% for {result.strategy_name}"


def test_profit_factor_valid_values(results):
    for result in results:
        pf = result.profit_factor
        assert pf >= Decimal("0"), \
            f"Profit factor should be non-negative for {result.strategy_name}"


# ========== Report Generation E2E ==========


def test_individual_reports_generate_without_errors(spy_data, results):
    capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = capture

    try:
        for result in results:
            print_daily_report(result, spy_data)
    finally:
        sys.stdout = old_stdout

    output = capture.getvalue()
    # Verify each strategy's report was generated
    assert "BACKTEST REPORT: SMA Crossover (50/200)" in output
    assert "BACKTEST REPORT: SMA Crossover (20/50)" in output
    assert "BACKTEST REPORT: RSI (14) [30.0/70.0]" in output
    assert "BACKTEST REPORT: RSI (14) [25.0/75.0]" in output
    assert "BACKTEST REPORT: MACD (12/26/9)" in output

    # Verify report sections exist
    assert "PERFORMANCE METRICS" in output
    assert "TRADE STATISTICS" in output
    assert "Total Return:" in output
    assert "Buy & Hold Return:" in output
    assert "Sharpe Ratio:" in output
    assert "Max Drawdown:" in output
    assert "Win Rate:" in output


def test_comparison_report_generates_without_errors(spy_data, results):
    capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = capture

    try:
        print_daily_comparison(results, spy_data)
    finally:
        sys.stdout = old_stdout

    output = capture.getvalue()
    assert "STRATEGY COMPARISON SUMMARY" in output
    assert "Buy & Hold (Benchmark)" in output
    assert "Best performing strategy:" in output
    assert "DISCLAIMER" in output

    # All strategy names should appear in comparison
    for result in results:
        assert result.strategy_name in output, \
            f"Comparison should include {result.strategy_name}"


def test_best_strategy_identified_correctly(spy_data, results):
    capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = capture

    try:
        print_daily_comparison(results, spy_data)
    finally:
        sys.stdout = old_stdout

    # Find the best strategy manually
    best = max(results, key=lambda r: r.total_return)

    output = capture.getvalue()
    assert best is not None
    assert f"Best performing strategy: {best.strategy_name}" in output


# ========== Commission Impact E2E ==========


def test_commission_impact_on_all_strategies(spy_data, strategies, results):
    commission = Decimal("9.99")
    engine_with_comm = BacktestEngine(INITIAL_CAPITAL, commission)

    for i, strategy in enumerate(strategies):
        no_comm = results[i]
        with_comm = engine_with_comm.run(strategy, spy_data)

        if no_comm.total_trades > 0:
            assert no_comm.final_capital >= with_comm.final_capital, \
                f"Commission should reduce final capital for {strategy.name}"


# ========== Determinism E2E ==========


def test_full_pipeline_is_deterministic(spy_data, strategies, results):
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
    rerun = [engine.run(strategy, spy_data) for strategy in strategies]

    for i in range(len(results)):
        r1 = results[i]
        r2 = rerun[i]

        assert r1.final_capital == r2.final_capital, \
            f"Determinism: final capital for {r1.strategy_name}"
        assert r1.total_trades == r2.total_trades, \
            f"Determinism: trade count for {r1.strategy_name}"
        assert r1.total_return == r2.total_return, \
            f"Determinism: total return for {r1.strategy_name}"
        assert len(r1.equity_curve) == len(r2.equity_curve), \
            f"Determinism: equity curve size for {r1.strategy_name}"


# ========== Moving Average Integration ==========


def test_moving_averages_correctly_computed_on_real_data(spy_data):
    # Verify SMA/EMA computations on real SPY data match expected behavior
    idx = 100  # Well past any warmup period
    sma20 = sma(spy_data, idx, 20)
    sma50 = sma(spy_data, idx, 50)
    ema12 = ema(spy_data, idx, 12)
    ema26 = ema(spy_data, idx, 26)

    # All should be in reasonable SPY price range ($350-$500)
    lower_bound = Decimal("350")
    upper_bound = Decimal("500")

    assert lower_bound < sma20 < upper_bound, "SMA(20) should be in SPY price range"
    assert lower_bound < sma50 < upper_bound, "SMA(50) should be in SPY price range"
    assert lower_bound < ema12 < upper_bound, "EMA(12) should be in SPY price range"
    assert lower_bound < ema26 < upper_bound, "EMA(26) should be in SPY price range"


# ========== BigDecimalMath Integration ==========


def test_big_decimal_math_used_in_metric_computations(spy_data, results):
    # Verify BigDecimalMath.percentChange gives same result as BacktestResult.getBuyAndHoldReturn
    first_close = spy_data[0].close
    last_close = spy_data[-1].close

    math_pct_change = percent_change(first_close, last_close)
    result_pct_change = results[0].buy_and_hold_return(spy_data)

    assert math_pct_change == result_pct_change, \
        "BigDecimalMath.percentChange should match BacktestResult.getBuyAndHoldReturn"


# ========== Varying Capital Sizes ==========


def test_different_capital_sizes_produce_proportional_results(spy_data):
    strategy = SMACrossoverStrategy(20, 50)
    small_cap = Decimal("10000.00")
    large_cap = Decimal("1000000.00")

    engine_small = BacktestEngine(small_cap, ZERO_COMMISSION)
    engine_large = BacktestEngine(large_cap, ZERO_COMMISSION)

    result_small = engine_small.run(strategy, spy_data)
    result_large = engine_large.run(strategy, spy_data)

    # Total return percentage should be very close (within 1% due to share rounding)
    return_diff = abs(result_small.total_return - result_large.total_return)
    assert return_diff < Decimal("1.0"), \
        f"Return % should be nearly proportional across capital sizes, diff={return_diff}"
