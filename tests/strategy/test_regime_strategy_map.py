"""Tests for RegimeStrategyMapper.

Validates recording trades, querying best strategy per regime,
performance matrix, and edge cases.
"""

import pytest

from stockdownloader.strategy.regime.regime_detector import MarketRegime
from stockdownloader.strategy.regime.regime_strategy_map import (
    RegimePerformance,
    RegimeStrategyMapper,
)


# =========================================================================
# RegimePerformance
# =========================================================================


def test_regime_performance_avg_pnl():
    perf = RegimePerformance(
        regime=MarketRegime.MEAN_REVERTING,
        strategy_name="rsi",
        total_pnl=300.0,
        trade_count=3,
        wins=2,
    )
    assert perf.avg_pnl == 100.0


def test_regime_performance_win_rate():
    perf = RegimePerformance(
        regime=MarketRegime.STRONG_TREND_UP,
        strategy_name="sma",
        total_pnl=500.0,
        trade_count=10,
        wins=7,
    )
    assert perf.win_rate == 0.7


def test_regime_performance_no_trades():
    perf = RegimePerformance(
        regime=MarketRegime.WEAK_TREND,
        strategy_name="test",
    )
    assert perf.avg_pnl == 0.0
    assert perf.win_rate == 0.0


# =========================================================================
# RegimeStrategyMapper — recording
# =========================================================================


def test_record_single_trade():
    mapper = RegimeStrategyMapper()
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=150.0)

    perf = mapper.get_performance("rsi", MarketRegime.MEAN_REVERTING)
    assert perf is not None
    assert perf.trade_count == 1
    assert perf.total_pnl == 150.0
    assert perf.wins == 1  # inferred from pnl > 0


def test_record_multiple_trades():
    mapper = RegimeStrategyMapper()
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=100.0)
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=-50.0)
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=75.0)

    perf = mapper.get_performance("rsi", MarketRegime.MEAN_REVERTING)
    assert perf is not None
    assert perf.trade_count == 3
    assert perf.total_pnl == 125.0
    assert perf.wins == 2  # two positive trades
    assert perf.avg_pnl == pytest.approx(41.67, abs=0.01)


def test_record_explicit_is_win():
    mapper = RegimeStrategyMapper()
    # Record a zero-pnl trade as a win (e.g. break-even counted as success)
    mapper.record("sma", MarketRegime.WEAK_TREND, pnl=0.0, is_win=True)

    perf = mapper.get_performance("sma", MarketRegime.WEAK_TREND)
    assert perf is not None
    assert perf.wins == 1


def test_record_different_regimes():
    mapper = RegimeStrategyMapper()
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=100.0)
    mapper.record("rsi", MarketRegime.STRONG_TREND_UP, pnl=-50.0)

    assert mapper.get_performance("rsi", MarketRegime.MEAN_REVERTING).trade_count == 1
    assert mapper.get_performance("rsi", MarketRegime.STRONG_TREND_UP).trade_count == 1


def test_record_different_strategies():
    mapper = RegimeStrategyMapper()
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=100.0)
    mapper.record("sma", MarketRegime.MEAN_REVERTING, pnl=200.0)

    names = mapper.strategy_names()
    assert "rsi" in names
    assert "sma" in names


# =========================================================================
# Best strategy per regime
# =========================================================================


def test_best_strategy_single():
    mapper = RegimeStrategyMapper()
    for _ in range(5):
        mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=100.0)

    best = mapper.best_strategy_for_regime(MarketRegime.MEAN_REVERTING)
    assert best == "rsi"


def test_best_strategy_among_multiple():
    mapper = RegimeStrategyMapper()
    # RSI: avg 50
    for _ in range(5):
        mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=50.0)
    # SMA: avg 120
    for _ in range(5):
        mapper.record("sma", MarketRegime.MEAN_REVERTING, pnl=120.0)
    # MACD: avg 80
    for _ in range(5):
        mapper.record("macd", MarketRegime.MEAN_REVERTING, pnl=80.0)

    best = mapper.best_strategy_for_regime(MarketRegime.MEAN_REVERTING)
    assert best == "sma"


def test_best_strategy_min_trades_filter():
    mapper = RegimeStrategyMapper()
    # RSI: 2 trades (below min_trades=3)
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=1000.0)
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=1000.0)
    # SMA: 3 trades (meets min_trades=3)
    for _ in range(3):
        mapper.record("sma", MarketRegime.MEAN_REVERTING, pnl=50.0)

    best = mapper.best_strategy_for_regime(MarketRegime.MEAN_REVERTING, min_trades=3)
    assert best == "sma"  # RSI filtered out despite higher avg_pnl


def test_best_strategy_no_candidates():
    mapper = RegimeStrategyMapper()
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=100.0)

    # No trades in STRONG_TREND_UP
    best = mapper.best_strategy_for_regime(MarketRegime.STRONG_TREND_UP)
    assert best is None


def test_best_strategy_empty_mapper():
    mapper = RegimeStrategyMapper()
    best = mapper.best_strategy_for_regime(MarketRegime.WEAK_TREND)
    assert best is None


# =========================================================================
# Performance matrix
# =========================================================================


def test_performance_matrix():
    mapper = RegimeStrategyMapper()
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=100.0)
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=50.0)
    mapper.record("sma", MarketRegime.STRONG_TREND_UP, pnl=200.0)

    matrix = mapper.performance_matrix
    assert "rsi" in matrix
    assert "mean_reverting" in matrix["rsi"]
    assert matrix["rsi"]["mean_reverting"] == 75.0  # avg of 100 and 50


# =========================================================================
# Regime coverage
# =========================================================================


def test_regime_coverage():
    mapper = RegimeStrategyMapper()
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=100.0)
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=50.0)
    mapper.record("rsi", MarketRegime.STRONG_TREND_UP, pnl=200.0)

    coverage = mapper.regime_coverage("rsi")
    assert coverage[MarketRegime.MEAN_REVERTING] == 2
    assert coverage[MarketRegime.STRONG_TREND_UP] == 1


def test_regime_coverage_unknown_strategy():
    mapper = RegimeStrategyMapper()
    coverage = mapper.regime_coverage("nonexistent")
    assert coverage == {}


# =========================================================================
# All performances
# =========================================================================


def test_all_performances():
    mapper = RegimeStrategyMapper()
    mapper.record("rsi", MarketRegime.MEAN_REVERTING, pnl=100.0)
    mapper.record("sma", MarketRegime.STRONG_TREND_UP, pnl=200.0)

    all_perfs = mapper.all_performances()
    assert len(all_perfs) == 2
    assert all(isinstance(p, RegimePerformance) for p in all_perfs)
