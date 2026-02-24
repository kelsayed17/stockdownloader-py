"""Tests for regime-aware tournament analysis."""
from __future__ import annotations

import pytest

from stockdownloader.backtest.tournament_engine import (
    ComboKey,
    RegimeAnalysis,
    RegimeTradeStats,
    _compute_regime_bonus,
    run_regime_analysis,
)
from stockdownloader.strategies.regime.detector import MarketRegime


class TestRegimeTradeStats:
    def test_construction(self):
        stats = RegimeTradeStats(
            regime=MarketRegime.STRONG_TREND_UP,
            trade_count=10,
            total_pnl=500.0,
            win_count=7,
            avg_pnl=50.0,
            win_rate=0.7,
        )
        assert stats.trade_count == 10
        assert stats.avg_pnl == 50.0
        assert stats.win_rate == 0.7
        assert stats.regime == MarketRegime.STRONG_TREND_UP

    def test_frozen(self):
        stats = RegimeTradeStats(
            regime=MarketRegime.WEAK_TREND,
            trade_count=5,
            total_pnl=100.0,
            win_count=3,
            avg_pnl=20.0,
            win_rate=0.6,
        )
        with pytest.raises(AttributeError):
            stats.trade_count = 99  # type: ignore[misc]


class TestComputeRegimeBonus:
    def test_all_regimes_covered(self):
        """Strategy trades in all 5 regimes with positive avg_pnl."""
        per_regime = {
            MarketRegime.STRONG_TREND_UP: RegimeTradeStats(
                MarketRegime.STRONG_TREND_UP, 10, 500.0, 7, 50.0, 0.7,
            ),
            MarketRegime.STRONG_TREND_DOWN: RegimeTradeStats(
                MarketRegime.STRONG_TREND_DOWN, 8, 400.0, 5, 50.0, 0.625,
            ),
            MarketRegime.WEAK_TREND: RegimeTradeStats(
                MarketRegime.WEAK_TREND, 15, 600.0, 9, 40.0, 0.6,
            ),
            MarketRegime.MEAN_REVERTING: RegimeTradeStats(
                MarketRegime.MEAN_REVERTING, 12, 360.0, 7, 30.0, 0.583,
            ),
            MarketRegime.HIGH_VOLATILITY: RegimeTradeStats(
                MarketRegime.HIGH_VOLATILITY, 5, 250.0, 3, 50.0, 0.6,
            ),
        }
        bonus = _compute_regime_bonus(per_regime)
        # Coverage: 5 * 1.0 = 5.0, plus consistency bonus, no collapse
        assert bonus > 5.0

    def test_single_regime_penalized(self):
        """Strategy only trades in 1 regime — gets penalized."""
        per_regime = {
            MarketRegime.STRONG_TREND_UP: RegimeTradeStats(
                MarketRegime.STRONG_TREND_UP, 50, 5000.0, 35, 100.0, 0.7,
            ),
        }
        bonus = _compute_regime_bonus(per_regime)
        # Coverage: 1.0, single-regime penalty: -2.0
        assert bonus < 0

    def test_collapse_penalty(self):
        """Strategy collapses in one regime (avg_pnl < -$50)."""
        per_regime = {
            MarketRegime.STRONG_TREND_UP: RegimeTradeStats(
                MarketRegime.STRONG_TREND_UP, 10, 500.0, 7, 50.0, 0.7,
            ),
            MarketRegime.MEAN_REVERTING: RegimeTradeStats(
                MarketRegime.MEAN_REVERTING, 10, -2000.0, 2, -200.0, 0.2,
            ),
            MarketRegime.WEAK_TREND: RegimeTradeStats(
                MarketRegime.WEAK_TREND, 10, 300.0, 6, 30.0, 0.6,
            ),
        }
        bonus = _compute_regime_bonus(per_regime)
        # Collapse penalty for -200 avg_pnl: abs(-200 + 50) * 0.1 = 15.0
        assert bonus < 0

    def test_no_collapse_penalty_when_above_threshold(self):
        """No collapse penalty when worst regime avg_pnl > -$50."""
        per_regime = {
            MarketRegime.STRONG_TREND_UP: RegimeTradeStats(
                MarketRegime.STRONG_TREND_UP, 10, 500.0, 7, 50.0, 0.7,
            ),
            MarketRegime.WEAK_TREND: RegimeTradeStats(
                MarketRegime.WEAK_TREND, 10, -100.0, 4, -10.0, 0.4,
            ),
            MarketRegime.MEAN_REVERTING: RegimeTradeStats(
                MarketRegime.MEAN_REVERTING, 10, 200.0, 6, 20.0, 0.6,
            ),
        }
        bonus = _compute_regime_bonus(per_regime)
        # No collapse penalty (worst is -10, above -50 threshold)
        assert bonus > 0

    def test_consistency_bonus_higher_for_low_variance(self):
        """More consistent avg_pnl across regimes gives bigger bonus."""
        # Consistent: all around 50
        consistent = {
            MarketRegime.STRONG_TREND_UP: RegimeTradeStats(
                MarketRegime.STRONG_TREND_UP, 10, 500.0, 7, 50.0, 0.7,
            ),
            MarketRegime.WEAK_TREND: RegimeTradeStats(
                MarketRegime.WEAK_TREND, 10, 520.0, 7, 52.0, 0.7,
            ),
            MarketRegime.MEAN_REVERTING: RegimeTradeStats(
                MarketRegime.MEAN_REVERTING, 10, 480.0, 7, 48.0, 0.7,
            ),
        }
        consistent_bonus = _compute_regime_bonus(consistent)

        # Inconsistent: wide spread
        inconsistent = {
            MarketRegime.STRONG_TREND_UP: RegimeTradeStats(
                MarketRegime.STRONG_TREND_UP, 10, 2000.0, 9, 200.0, 0.9,
            ),
            MarketRegime.WEAK_TREND: RegimeTradeStats(
                MarketRegime.WEAK_TREND, 10, 100.0, 5, 10.0, 0.5,
            ),
            MarketRegime.MEAN_REVERTING: RegimeTradeStats(
                MarketRegime.MEAN_REVERTING, 10, 50.0, 5, 5.0, 0.5,
            ),
        }
        inconsistent_bonus = _compute_regime_bonus(inconsistent)

        assert consistent_bonus > inconsistent_bonus

    def test_empty_per_regime(self):
        """No crash on empty dict."""
        bonus = _compute_regime_bonus({})
        assert bonus == 0.0

    def test_insufficient_trades_ignored(self):
        """Regimes with < 3 trades are not counted as active."""
        per_regime = {
            MarketRegime.STRONG_TREND_UP: RegimeTradeStats(
                MarketRegime.STRONG_TREND_UP, 2, 100.0, 2, 50.0, 1.0,
            ),
            MarketRegime.WEAK_TREND: RegimeTradeStats(
                MarketRegime.WEAK_TREND, 1, 10.0, 1, 10.0, 1.0,
            ),
        }
        bonus = _compute_regime_bonus(per_regime)
        assert bonus == 0.0  # No active regimes


class TestRunRegimeAnalysis:
    def test_basic_analysis(self):
        """run_regime_analysis returns valid RegimeAnalysis."""
        key = ComboKey("rsi", "5m")
        trades = [
            ("2025-01-15 09:35:00", 100.0, True),
            ("2025-01-15 10:00:00", -50.0, False),
            ("2025-01-15 10:30:00", 75.0, True),
            ("2025-01-16 09:35:00", 120.0, True),
            ("2025-01-16 10:00:00", -30.0, False),
        ]
        regime_map = {
            "2025-01-15 09:35:00": MarketRegime.STRONG_TREND_UP,
            "2025-01-15 10:00:00": MarketRegime.WEAK_TREND,
            "2025-01-15 10:30:00": MarketRegime.STRONG_TREND_UP,
            "2025-01-16 09:35:00": MarketRegime.MEAN_REVERTING,
            "2025-01-16 10:00:00": MarketRegime.MEAN_REVERTING,
        }

        result_key, analysis, elapsed, error = run_regime_analysis(
            key, trades, regime_map,
        )
        assert error is None
        assert analysis is not None
        assert result_key == key
        assert len(analysis.per_regime) >= 2
        assert isinstance(analysis.regime_bonus, float)

    def test_empty_trades(self):
        """Gracefully handles empty trade list."""
        key = ComboKey("rsi", "5m")
        result_key, analysis, elapsed, error = run_regime_analysis(
            key, [], {},
        )
        assert error is None
        assert analysis is None

    def test_missing_dates_fallback(self):
        """Falls back to WEAK_TREND when trade date not in regime map."""
        key = ComboKey("rsi", "5m")
        trades = [
            ("2025-01-15 09:35:00", 100.0, True),
            ("2025-01-15 10:00:00", -50.0, False),
            ("2025-01-15 10:30:00", 75.0, True),
        ]
        # Empty regime map — all trades should fall back to WEAK_TREND
        regime_map: dict[str, MarketRegime] = {}

        _, analysis, _, error = run_regime_analysis(key, trades, regime_map)
        assert error is None
        assert analysis is not None
        assert MarketRegime.WEAK_TREND in analysis.per_regime
