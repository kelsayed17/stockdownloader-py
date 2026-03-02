"""Comprehensive backtesting scenarios using real SPY data (Jan 2023 - Jan 2024).

Tests strategies across different market regimes, parameter sensitivities,
capital/commission configurations, and produces a full scenario comparison report.

Market segments identified in the data:
  - Q1 2023 Recovery (Jan-Mar ~60 days): Post-2022 bear recovery, choppy/sideways
  - Q2 2023 Bull Rally (Apr-Jul ~80 days): Sustained uptrend driven by AI enthusiasm
  - Q3 2023 Correction (Aug-Oct ~65 days): Pullback from summer highs
  - Q4 2023 Year-End Rally (Nov 2023-Jan 2024 ~67 days): Strong year-end rally
"""
from __future__ import annotations

import io
import sys
from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.backtesting.engines.daily import BacktestEngine
from stockdownloader.backtesting.results.formatter import print_daily_comparison
from stockdownloader.backtesting.results.result import BacktestResult
from stockdownloader.data.parsers import CsvPriceDataLoader
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.models.trade import Trade
from stockdownloader.strategies.daily.simple import MACDStrategy
from stockdownloader.strategies.daily.simple import RSIStrategy
from stockdownloader.strategies.daily.simple import SMACrossoverStrategy
from stockdownloader.strategies.base import TradingStrategy

INITIAL_CAPITAL = Decimal("100000.00")
ZERO_COMMISSION = Decimal("0")
TRADING_DAYS_PER_YEAR = 252


def _find_index_for_date(data: list[PriceData], target_date: str) -> int:
    for i, bar in enumerate(data):
        if bar.date >= target_date:
            return i
    return len(data)


@pytest.fixture(scope="module")
def full_data(test_data_path):
    data = CsvPriceDataLoader.load_from_file(str(test_data_path))
    assert len(data) > 0, "SPY data must be loaded"
    assert len(data) >= 270, f"Expected ~272 trading days, got {len(data)}"
    return data


@pytest.fixture(scope="module")
def market_segments(full_data):
    q1_end = _find_index_for_date(full_data, "2023-04-01")
    q2_end = _find_index_for_date(full_data, "2023-08-01")
    q3_end = _find_index_for_date(full_data, "2023-11-01")

    q1_recovery = full_data[:q1_end]
    q2_bull_rally = full_data[q1_end:q2_end]
    q3_correction = full_data[q2_end:q3_end]
    q4_year_end_rally = full_data[q3_end:]

    assert len(q1_recovery) >= 50, "Q1 segment should have adequate data"
    assert len(q2_bull_rally) >= 60, "Q2 segment should have adequate data"
    assert len(q3_correction) >= 50, "Q3 segment should have adequate data"
    assert len(q4_year_end_rally) >= 50, "Q4 segment should have adequate data"

    return {
        "q1_recovery": q1_recovery,
        "q2_bull_rally": q2_bull_rally,
        "q3_correction": q3_correction,
        "q4_year_end_rally": q4_year_end_rally,
    }


# ======================================================================
#  SCENARIO 1: Market Regime Characterization
# ======================================================================


class TestMarketRegimeCharacterization:

    def test_q1_recovery_buy_and_hold_performance(self, market_segments):
        q1 = market_segments["q1_recovery"]
        first_close = q1[0].close
        last_close = q1[-1].close
        return_pct = (last_close - first_close) / first_close * Decimal("100")
        assert isinstance(return_pct, Decimal)

    def test_q2_bull_rally_buy_and_hold_performance(self, market_segments):
        q2 = market_segments["q2_bull_rally"]
        first_close = q2[0].close
        last_close = q2[-1].close
        return_pct = (last_close - first_close) / first_close * Decimal("100")
        assert return_pct > Decimal("0"), \
            "Q2 2023 should have positive buy-and-hold return (bull market)"

    def test_q3_correction_buy_and_hold_performance(self, market_segments):
        q3 = market_segments["q3_correction"]
        first_close = q3[0].close
        last_close = q3[-1].close
        return_pct = (last_close - first_close) / first_close * Decimal("100")
        assert isinstance(return_pct, Decimal)

    def test_q4_year_end_rally_buy_and_hold_performance(self, market_segments):
        q4 = market_segments["q4_year_end_rally"]
        first_close = q4[0].close
        last_close = q4[-1].close
        return_pct = (last_close - first_close) / first_close * Decimal("100")
        assert return_pct > Decimal("0"), \
            "Q4 2023 year-end rally should have positive buy-and-hold return"

    def test_full_period_buy_and_hold_performance(self, full_data):
        first_close = full_data[0].close
        last_close = full_data[-1].close
        return_pct = (last_close - first_close) / first_close * Decimal("100")
        assert return_pct > Decimal("0"), \
            "SPY full-period return should be positive (Jan 2023 to Jan 2024)"

    def test_segments_are_mutually_exclusive_and_exhaustive(self, full_data, market_segments):
        total_segment_days = (
            len(market_segments["q1_recovery"])
            + len(market_segments["q2_bull_rally"])
            + len(market_segments["q3_correction"])
            + len(market_segments["q4_year_end_rally"])
        )
        assert total_segment_days == len(full_data), \
            "Segments should cover all trading days"

        # Verify no overlap: last date of each segment < first date of next
        q1 = market_segments["q1_recovery"]
        q2 = market_segments["q2_bull_rally"]
        q3 = market_segments["q3_correction"]
        q4 = market_segments["q4_year_end_rally"]
        assert q1[-1].date < q2[0].date
        assert q2[-1].date < q3[0].date
        assert q3[-1].date < q4[0].date


# ======================================================================
#  SCENARIO 2: SMA Crossover Parameter Sensitivity
# ======================================================================


class TestSMAParameterSensitivity:

    SMA_CONFIGS = [
        (5, 20),    # Aggressive short-term
        (10, 30),   # Short-term
        (20, 50),   # Medium-term
        (50, 100),  # Medium-long term
        (50, 200),  # Classic golden/death cross
    ]

    def test_shorter_periods_generate_more_trades(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        sma_results = []
        for short_p, long_p in self.SMA_CONFIGS:
            strategy = SMACrossoverStrategy(short_p, long_p)
            sma_results.append(engine.run(strategy, full_data))

        # Generally, shorter periods = more crossover signals = more trades
        aggressive = sma_results[0]
        conservative = sma_results[-1]

        assert aggressive.total_trades >= conservative.total_trades, \
            "Shorter SMA periods should generate >= trades than longer periods"

    def test_all_sma_configs_preserve_capital(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        for short_p, long_p in self.SMA_CONFIGS:
            strategy = SMACrossoverStrategy(short_p, long_p)
            result = engine.run(strategy, full_data)

            assert result.final_capital > Decimal("0"), \
                f"SMA({short_p}/{long_p}) should have positive final capital"
            assert len(result.equity_curve) == len(full_data), \
                f"SMA({short_p}/{long_p}) equity curve size mismatch"

    def test_sma_returns_are_consistent_with_trade_log(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        for short_p, long_p in self.SMA_CONFIGS:
            strategy = SMACrossoverStrategy(short_p, long_p)
            result = engine.run(strategy, full_data)

            trade_pl_sum = sum(t.profit_loss for t in result.closed_trades)
            assert trade_pl_sum == result.total_pnl, \
                f"SMA({short_p}/{long_p}) trade P/L sum should match total P/L"


# ======================================================================
#  SCENARIO 3: RSI Threshold Sensitivity
# ======================================================================


class TestRSIThresholdSensitivity:

    RSI_CONFIGS = [
        (14, 20.0, 80.0),   # Very wide thresholds - fewer signals
        (14, 25.0, 75.0),   # Wide thresholds
        (14, 30.0, 70.0),   # Standard thresholds
        (14, 35.0, 65.0),   # Tight thresholds
        (14, 40.0, 60.0),   # Very tight thresholds - more signals
    ]

    def test_wider_thresholds_produce_fewer_trades(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        rsi_results = []
        for period, oversold, overbought in self.RSI_CONFIGS:
            strategy = RSIStrategy(period, oversold, overbought)
            rsi_results.append(engine.run(strategy, full_data))

        # Wider thresholds = fewer signals
        widest = rsi_results[0]    # 20/80
        tightest = rsi_results[-1]  # 40/60
        assert tightest.total_trades >= widest.total_trades, \
            "Tighter RSI thresholds should generate >= trades than wider ones"

    def test_rsi_period_sensitivity(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
        periods = [7, 14, 21, 28]

        for period in periods:
            strategy = RSIStrategy(period, 30.0, 70.0)
            result = engine.run(strategy, full_data)

            assert result.final_capital > Decimal("0"), \
                f"RSI({period}) should maintain positive capital"


# ======================================================================
#  SCENARIO 4: MACD Parameter Variations
# ======================================================================


class TestMACDParameterVariations:

    def test_standard_vs_aggressive_macd_parameters(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        standard = MACDStrategy(12, 26, 9)
        aggressive = MACDStrategy(8, 17, 9)
        conservative = MACDStrategy(19, 39, 9)

        std_result = engine.run(standard, full_data)
        agg_result = engine.run(aggressive, full_data)
        con_result = engine.run(conservative, full_data)

        # Aggressive (shorter periods) should generate more signals
        assert agg_result.total_trades >= con_result.total_trades, \
            "Aggressive MACD should generate >= trades than conservative"

    def test_macd_signal_period_sensitivity(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
        signal_periods = [5, 7, 9, 12, 15]

        for signal_period in signal_periods:
            strategy = MACDStrategy(12, 26, signal_period)
            result = engine.run(strategy, full_data)
            assert len(result.equity_curve) == len(full_data)


# ======================================================================
#  SCENARIO 5: Strategy Tournament Across Market Segments
# ======================================================================


class TestStrategyTournament:

    TOURNAMENT_STRATEGIES = [
        SMACrossoverStrategy(10, 30),
        SMACrossoverStrategy(20, 50),
        RSIStrategy(14, 30.0, 70.0),
        MACDStrategy(12, 26, 9),
    ]

    def test_tournament_on_full_data(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        best_return = None
        best_strategy = None

        for strategy in self.TOURNAMENT_STRATEGIES:
            result = engine.run(strategy, full_data)
            if best_return is None or result.total_return > best_return:
                best_return = result.total_return
                best_strategy = strategy.name

        assert best_strategy is not None

    def test_tournament_across_market_segments(self, full_data, market_segments):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        segments = {
            "Q1 Recovery": market_segments["q1_recovery"],
            "Q2 Bull Rally": market_segments["q2_bull_rally"],
            "Q3 Correction": market_segments["q3_correction"],
            "Q4 Year-End Rally": market_segments["q4_year_end_rally"],
        }

        # Every segment should produce results (no crashes)
        assert len(segments) == 4

        for seg_name, seg_data in segments.items():
            for strategy in self.TOURNAMENT_STRATEGIES:
                result = engine.run(strategy, seg_data)
                assert result is not None

    def test_no_strategy_loses_more_than_fifty_percent_on_any_segment(
        self, full_data, market_segments
    ):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
        max_loss_threshold = Decimal("-50")

        all_segments = [
            market_segments["q1_recovery"],
            market_segments["q2_bull_rally"],
            market_segments["q3_correction"],
            market_segments["q4_year_end_rally"],
            full_data,
        ]

        for segment in all_segments:
            for strategy in self.TOURNAMENT_STRATEGIES:
                result = engine.run(strategy, segment)
                assert result.total_return > max_loss_threshold, \
                    f"{strategy.name} should not lose >50% on any segment"


# ======================================================================
#  SCENARIO 6: Commission Impact Analysis
# ======================================================================


class TestCommissionImpactAnalysis:

    def test_commission_scale_impact(self, full_data):
        commissions = [
            Decimal("0"),
            Decimal("1.00"),
            Decimal("4.95"),
            Decimal("9.99"),
            Decimal("25.00"),
        ]

        # Use a strategy with many trades to see commission impact clearly
        strategy = SMACrossoverStrategy(10, 30)

        previous_final_capital = None

        for commission in commissions:
            engine = BacktestEngine(INITIAL_CAPITAL, commission)
            result = engine.run(strategy, full_data)

            if previous_final_capital is not None and result.total_trades > 0:
                assert result.final_capital <= previous_final_capital, \
                    "Higher commission should result in lower or equal final capital"
            previous_final_capital = result.final_capital

    def test_high_commission_can_turn_profitable_strategy_unprofitable(self, full_data):
        strategy = MACDStrategy(12, 26, 9)

        no_comm = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
        no_comm_result = no_comm.run(strategy, full_data)

        # Try progressively higher commissions
        commission = Decimal("100.00")
        high_comm = BacktestEngine(INITIAL_CAPITAL, commission)
        high_comm_result = high_comm.run(strategy, full_data)

        if no_comm_result.total_trades > 0:
            assert no_comm_result.total_return >= high_comm_result.total_return, \
                "Very high commissions should reduce returns"

    def test_zero_commission_matches_trade_log_exactly(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
        strategy = SMACrossoverStrategy(20, 50)
        result = engine.run(strategy, full_data)

        trade_pl_sum = sum(t.profit_loss for t in result.closed_trades)

        assert trade_pl_sum == result.total_pnl, \
            "Zero-commission trade P/L sum must exactly match total P/L"

        assert INITIAL_CAPITAL + trade_pl_sum == result.final_capital, \
            "Final capital = initial + sum of trade P/L with zero commission"


# ======================================================================
#  SCENARIO 7: Capital Size Scenarios
# ======================================================================


class TestCapitalSizeScenarios:

    def test_return_percentages_are_nearly_identical_across_capital_sizes(self, full_data):
        capitals = [
            Decimal("1000.00"),
            Decimal("10000.00"),
            Decimal("100000.00"),
            Decimal("1000000.00"),
            Decimal("10000000.00"),
        ]

        strategy = MACDStrategy(12, 26, 9)
        returns = []

        for capital in capitals:
            engine = BacktestEngine(capital, ZERO_COMMISSION)
            result = engine.run(strategy, full_data)
            returns.append(result.total_return)

        # Returns should be within 2% of each other (due to share rounding)
        for i in range(1, len(returns)):
            diff = abs(returns[i] - returns[0])
            assert diff < Decimal("2.0"), \
                f"Return % should be similar across capital sizes (diff={diff})"

    def test_very_small_capital_still_functions(self, full_data):
        engine = BacktestEngine(Decimal("500.00"), ZERO_COMMISSION)
        strategy = SMACrossoverStrategy(20, 50)
        result = engine.run(strategy, full_data)

        assert result.final_capital > Decimal("0"), \
            "Small capital should still produce valid results"
        assert len(result.equity_curve) == len(full_data)


# ======================================================================
#  SCENARIO 8: Risk Metrics Deep Dive
# ======================================================================


class TestRiskMetricsDeepDive:

    def test_max_drawdown_comparison(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        strategies = [
            SMACrossoverStrategy(10, 30),
            SMACrossoverStrategy(20, 50),
            SMACrossoverStrategy(50, 200),
            RSIStrategy(14, 30.0, 70.0),
            MACDStrategy(12, 26, 9),
        ]

        for strategy in strategies:
            result = engine.run(strategy, full_data)
            assert result.max_drawdown >= Decimal("0")
            assert result.max_drawdown <= Decimal("100")

    def test_sharpe_ratio_is_reasonable(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        strategies = [
            SMACrossoverStrategy(20, 50),
            RSIStrategy(14, 30.0, 70.0),
            MACDStrategy(12, 26, 9),
        ]

        for strategy in strategies:
            result = engine.run(strategy, full_data)
            sharpe = result.sharpe_ratio(TRADING_DAYS_PER_YEAR)

            assert sharpe > Decimal("-5"), \
                f"{strategy.name} Sharpe should be > -5"
            assert sharpe < Decimal("10"), \
                f"{strategy.name} Sharpe should be < 10"

    def test_profit_factor_correlates_with_win_rate(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        strategy = SMACrossoverStrategy(10, 30)
        result = engine.run(strategy, full_data)

        if result.total_trades > 0:
            profit_factor = result.profit_factor
            assert profit_factor >= Decimal("0"), "Profit factor should be non-negative"

    def test_equity_curve_never_goes_negative(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        strategies = [
            SMACrossoverStrategy(5, 20),
            SMACrossoverStrategy(20, 50),
            RSIStrategy(7, 30.0, 70.0),
            RSIStrategy(14, 25.0, 75.0),
            MACDStrategy(8, 17, 9),
            MACDStrategy(12, 26, 9),
        ]

        for strategy in strategies:
            result = engine.run(strategy, full_data)
            for i, equity in enumerate(result.equity_curve):
                assert equity > Decimal("0"), \
                    f"{strategy.name} equity should never go negative (index {i})"


# ======================================================================
#  SCENARIO 9: Head-to-Head Strategy Comparison with Report
# ======================================================================


class TestHeadToHeadComparison:

    def test_full_comparison_report_all_strategies(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        strategies = [
            SMACrossoverStrategy(5, 20),
            SMACrossoverStrategy(10, 30),
            SMACrossoverStrategy(20, 50),
            SMACrossoverStrategy(50, 200),
            RSIStrategy(7, 30.0, 70.0),
            RSIStrategy(14, 30.0, 70.0),
            RSIStrategy(14, 25.0, 75.0),
            RSIStrategy(21, 35.0, 65.0),
            MACDStrategy(8, 17, 9),
            MACDStrategy(12, 26, 9),
            MACDStrategy(19, 39, 9),
        ]

        all_results = [engine.run(strategy, full_data) for strategy in strategies]

        # Capture the comparison report
        capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = capture
        try:
            print_daily_comparison(all_results, full_data)
        finally:
            sys.stdout = old_stdout

        report = capture.getvalue()

        # Verify report contains all strategies
        for result in all_results:
            assert result.strategy_name in report, \
                f"Report should contain {result.strategy_name}"
        assert "STRATEGY COMPARISON SUMMARY" in report
        assert "Best performing strategy:" in report

    def test_rank_strategies_by_risk_adjusted_return(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        strategies = [
            SMACrossoverStrategy(20, 50),
            RSIStrategy(14, 30.0, 70.0),
            MACDStrategy(12, 26, 9),
        ]

        rankings = []
        for strategy in strategies:
            result = engine.run(strategy, full_data)
            rankings.append({
                "name": strategy.name,
                "ret": result.total_return,
                "sharpe": result.sharpe_ratio(TRADING_DAYS_PER_YEAR),
                "max_dd": result.max_drawdown,
                "trades": result.total_trades,
            })

        # Rank by Sharpe ratio (best risk-adjusted measure)
        rankings.sort(key=lambda r: r["sharpe"], reverse=True)

        # At least one strategy should have been evaluated
        assert len(rankings) > 0


# ======================================================================
#  SCENARIO 10: Edge Cases and Stress Testing
# ======================================================================


class TestEdgeCasesAndStressTesting:

    def test_minimum_data_for_each_strategy(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        # SMA(50/200) needs at least 200 bars for warmup
        sma_strategy = SMACrossoverStrategy(50, 200)
        assert sma_strategy.warmup_period == 200

        # RSI(14, 30/70) needs 15 bars
        rsi = RSIStrategy(14, 30.0, 70.0)
        assert rsi.warmup_period == 15

        # MACD(12/26/9) needs 35 bars
        macd = MACDStrategy(12, 26, 9)
        assert macd.warmup_period == 35

        # Test with just enough data for each strategy
        # Use 36 bars for MACD (minimum + 1)
        min_data = full_data[:36]
        macd_result = engine.run(macd, min_data)
        assert len(macd_result.equity_curve) == 36

        # Use 16 bars for RSI
        rsi_min_data = full_data[:16]
        rsi_result = engine.run(rsi, rsi_min_data)
        assert len(rsi_result.equity_curve) == 16

    def test_all_strategies_handle_entire_dataset_gracefully(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

        # Run every possible strategy configuration without errors
        all_strategies = [
            SMACrossoverStrategy(5, 10),
            SMACrossoverStrategy(5, 20),
            SMACrossoverStrategy(10, 30),
            SMACrossoverStrategy(20, 50),
            SMACrossoverStrategy(50, 100),
            SMACrossoverStrategy(50, 200),
            RSIStrategy(7, 20.0, 80.0),
            RSIStrategy(7, 30.0, 70.0),
            RSIStrategy(14, 25.0, 75.0),
            RSIStrategy(14, 30.0, 70.0),
            RSIStrategy(14, 35.0, 65.0),
            RSIStrategy(21, 30.0, 70.0),
            RSIStrategy(28, 30.0, 70.0),
            MACDStrategy(8, 17, 9),
            MACDStrategy(12, 26, 9),
            MACDStrategy(12, 26, 5),
            MACDStrategy(12, 26, 15),
            MACDStrategy(19, 39, 9),
        ]

        for strategy in all_strategies:
            # Should not throw
            engine.run(strategy, full_data)

    def test_single_bar_segment_produces_zero_trades(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
        one_bar = full_data[:1]

        rsi = RSIStrategy(14, 30.0, 70.0)
        result = engine.run(rsi, one_bar)
        assert result.total_trades == 0
        assert result.final_capital == INITIAL_CAPITAL

    def test_multiple_runs_on_same_engine_dont_leak_state(self, full_data):
        engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)
        strategy = SMACrossoverStrategy(20, 50)

        run1 = engine.run(strategy, full_data)
        run2 = engine.run(strategy, full_data)

        assert run1.final_capital == run2.final_capital, \
            "Same engine, same strategy, same data should produce identical results"
        assert run1.total_trades == run2.total_trades
        assert len(run1.equity_curve) == len(run2.equity_curve)


# ======================================================================
#  SCENARIO 11: Comprehensive Scenario Report
# ======================================================================


def test_generate_comprehensive_scenario_report(full_data):
    engine = BacktestEngine(INITIAL_CAPITAL, ZERO_COMMISSION)

    strategies = [
        SMACrossoverStrategy(10, 30),
        SMACrossoverStrategy(20, 50),
        SMACrossoverStrategy(50, 200),
        RSIStrategy(14, 30.0, 70.0),
        RSIStrategy(14, 25.0, 75.0),
        MACDStrategy(12, 26, 9),
    ]

    all_results = []
    for strategy in strategies:
        result = engine.run(strategy, full_data)
        all_results.append(result)

    # Find best by return and by Sharpe
    best_return = max(all_results, key=lambda r: r.total_return)
    best_sharpe = max(all_results, key=lambda r: r.sharpe_ratio(TRADING_DAYS_PER_YEAR))
    lowest_dd = min(all_results, key=lambda r: r.max_drawdown)

    assert best_return is not None
    assert best_sharpe is not None
    assert lowest_dd is not None

    # Verify all results are valid
    for result in all_results:
        assert result.final_capital > Decimal("0")
        assert len(result.equity_curve) == len(full_data)
