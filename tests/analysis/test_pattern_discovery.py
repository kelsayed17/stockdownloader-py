"""Tests for pattern_discovery — mining, filtering, catalog."""
from __future__ import annotations

import math
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from stockdownloader.analysis.pattern_discovery import (
    DiscoveredPattern,
    FilterConfig,
    PatternCatalog,
    PatternMiner,
    PatternOutcome,
    PatternStats,
    apply_fdr_correction,
    deduplicate_patterns,
    filter_patterns,
)
from stockdownloader.analysis.pattern_encoder import BarEncoder, BarFeatures, PatternContext


def _make_bar(
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: int = 1000,
    date: str = "2025-01-15 10:30:00-05:00",
) -> MagicMock:
    """Create a mock IntradayPriceData bar."""
    bar = MagicMock()
    bar.close = Decimal(str(close))
    bar.open = Decimal(str(open_ or close - 0.2))
    bar.high = Decimal(str(high or close + 0.5))
    bar.low = Decimal(str(low or close - 0.5))
    bar.volume = volume
    bar.date = date
    bar.datetime_parsed = MagicMock()
    bar.datetime_parsed.weekday.return_value = 2
    return bar


def _make_features(
    body_type: str = "bull",
    body_strength: str = "moderate",
    wick_signal: str = "no_wick",
    relative_size: str = "normal",
    volume_profile: str = "normal",
) -> BarFeatures:
    return BarFeatures(body_type, body_strength, wick_signal, relative_size, volume_profile)


def _make_discovered(
    direction: str = "long",
    t_stat: float = 2.5,
    p_value: float = 0.01,
    occurrences: int = 30,
    avg_return: float = 0.05,
    win_rate: float = 0.58,
    key: tuple | None = None,
    best_horizon: int = 5,
) -> DiscoveredPattern:
    if key is None:
        key = (_make_features(),)
    return DiscoveredPattern(
        key=key,
        direction=direction,
        best_horizon=best_horizon,
        avg_return=avg_return,
        win_rate=win_rate,
        occurrences=occurrences,
        t_stat=t_stat,
        p_value=p_value,
        avg_mae=-0.1,
        avg_mfe=0.15,
        risk_reward=1.5,
        walk_forward_stable=True,
        human_label=f"{len(key)}-bar: test",
    )


# ======================================================================
# PatternOutcome
# ======================================================================


class TestPatternOutcome:
    def test_construction(self):
        ctx = PatternContext("morning", 2, "weak_trend", 1)
        outcome = PatternOutcome(
            bar_index=100,
            returns={1: 0.05, 5: 0.10},
            mae=-0.08,
            mfe=0.12,
            context=ctx,
        )
        assert outcome.bar_index == 100
        assert outcome.returns[1] == pytest.approx(0.05)
        assert outcome.mae == pytest.approx(-0.08)

    def test_empty_returns(self):
        ctx = PatternContext("open", 0, "unknown", 0)
        outcome = PatternOutcome(
            bar_index=0,
            returns={},
            mae=0.0,
            mfe=0.0,
            context=ctx,
        )
        assert len(outcome.returns) == 0


# ======================================================================
# PatternStats
# ======================================================================


class TestPatternStats:
    def test_default_fields(self):
        stats = PatternStats(
            key=(_make_features(),),
            occurrences=10,
            outcomes=[],
        )
        assert stats.avg_return == {}
        assert stats.t_stat == {}
        assert stats.avg_mae == 0.0


# ======================================================================
# FilterConfig
# ======================================================================


class TestFilterConfig:
    def test_defaults(self):
        fc = FilterConfig()
        assert fc.min_occurrences == 20
        assert fc.max_p_value == 0.05
        assert fc.min_win_rate == 0.52
        assert fc.max_patterns == 50
        assert fc.min_effect_size == 0.2
        assert fc.min_risk_reward == 0.8
        assert fc.min_walk_forward_folds == 2
        assert fc.walk_forward_tolerance == 0.15
        assert fc.apply_fdr is False
        assert fc.regime_filter is None
        assert fc.min_regime_purity == 0.0

    def test_for_timeframe_5m_unchanged(self):
        fc = FilterConfig(min_occurrences=20)
        scaled = fc.for_timeframe("5m")
        assert scaled.min_occurrences == 20  # 1.0 scale

    def test_for_timeframe_1h_scaled(self):
        fc = FilterConfig(min_occurrences=20)
        scaled = fc.for_timeframe("1h")
        # 20 * 0.4 = 8 → clamped to 15
        assert scaled.min_occurrences == 15

    def test_for_timeframe_daily_floor(self):
        fc = FilterConfig(min_occurrences=20)
        scaled = fc.for_timeframe("1d")
        # 20 * 0.15 = 3 → clamped to 15
        assert scaled.min_occurrences == 15

    def test_for_timeframe_preserves_other_fields(self):
        fc = FilterConfig(
            min_occurrences=50,
            max_p_value=0.01,
            min_win_rate=0.55,
            min_abs_return=0.03,
            min_risk_reward=1.5,
            walk_forward_tolerance=0.20,
            max_patterns=25,
            min_effect_size=0.4,
            min_walk_forward_folds=3,
            apply_fdr=True,
            regime_filter="mean_reverting",
            min_regime_purity=0.5,
        )
        scaled = fc.for_timeframe("15m")
        assert scaled.max_p_value == 0.01
        assert scaled.min_win_rate == 0.55
        assert scaled.min_abs_return == 0.03
        assert scaled.min_risk_reward == 1.5
        assert scaled.walk_forward_tolerance == 0.20
        assert scaled.max_patterns == 25
        assert scaled.min_effect_size == 0.4
        assert scaled.min_walk_forward_folds == 3
        assert scaled.apply_fdr is True
        assert scaled.regime_filter == "mean_reverting"
        assert scaled.min_regime_purity == 0.5
        # min_occurrences: 50 * 0.6 = 30
        assert scaled.min_occurrences == 30

    def test_for_timeframe_unknown_uses_1x(self):
        fc = FilterConfig(min_occurrences=20)
        scaled = fc.for_timeframe("unknown_tf")
        assert scaled.min_occurrences == 20


# ======================================================================
# Pattern filtering
# ======================================================================


class TestFilterPatterns:
    def _make_stats(
        self,
        occurrences: int = 60,
        avg_return: float = 0.05,
        win_rate: float = 0.58,
        t_stat: float = 2.5,
        p_value: float = 0.005,
        mae: float = -0.1,
        mfe: float = 0.15,
        fh_wr: float = 0.55,
        sh_wr: float = 0.56,
        std_return: float = 0.10,
        fold_wrs: list[float] | None = None,
        regime: str = "unknown",
    ) -> PatternStats:
        ctx = PatternContext("morning", 2, regime, 0)
        outcomes = [
            PatternOutcome(i, {5: avg_return}, mae, mfe, ctx)
            for i in range(occurrences)
        ]
        stats = PatternStats(
            key=(_make_features(),),
            occurrences=occurrences,
            outcomes=outcomes,
            avg_return={5: avg_return},
            median_return={5: avg_return},
            win_rate={5: win_rate},
            std_return={5: std_return},
            t_stat={5: t_stat},
            p_value={5: p_value},
            first_half_wr={5: fh_wr},
            second_half_wr={5: sh_wr},
            fold_win_rates={5: fold_wrs or [0.55, 0.56, 0.57]},
            avg_mae=mae,
            avg_mfe=mfe,
        )
        return stats

    # Use relaxed FilterConfig for unit tests (disable effect-size gate)
    _LENIENT = FilterConfig(
        min_effect_size=0.0,
    )

    def test_passes_good_pattern(self):
        key = (_make_features(),)
        stats = self._make_stats()
        result = filter_patterns({key: stats}, self._LENIENT)
        assert len(result) == 1
        assert result[0].direction == "long"

    def test_passes_good_pattern_with_defaults(self):
        """Pattern with strong stats passes with default config."""
        key = (_make_features(),)
        stats = self._make_stats(
            occurrences=30,
            avg_return=0.05,
            std_return=0.10,
            t_stat=2.5,
            p_value=0.01,
        )
        result = filter_patterns({key: stats}, FilterConfig())
        assert len(result) == 1
        assert result[0].effect_size == pytest.approx(0.5, abs=0.01)

    def test_rejects_low_occurrences(self):
        key = (_make_features(),)
        stats = self._make_stats(occurrences=5)
        result = filter_patterns({key: stats}, FilterConfig(min_occurrences=20))
        assert len(result) == 0

    def test_rejects_high_p_value(self):
        key = (_make_features(),)
        stats = self._make_stats(p_value=0.10)
        result = filter_patterns({key: stats}, FilterConfig(max_p_value=0.05))
        assert len(result) == 0

    def test_rejects_low_return(self):
        key = (_make_features(),)
        stats = self._make_stats(avg_return=0.001, std_return=0.001)
        result = filter_patterns({key: stats}, FilterConfig(
            min_occurrences=20, max_p_value=0.05, min_effect_size=0.0,
            min_abs_return=0.02,
        ))
        assert len(result) == 0

    def test_rejects_low_win_rate(self):
        key = (_make_features(),)
        stats = self._make_stats(win_rate=0.48)
        result = filter_patterns({key: stats}, FilterConfig(
            min_occurrences=20, max_p_value=0.05, min_effect_size=0.0,
            min_win_rate=0.52,
        ))
        assert len(result) == 0

    def test_rejects_low_risk_reward(self):
        key = (_make_features(),)
        stats = self._make_stats(mae=-0.2, mfe=0.1)  # RR = 0.5
        result = filter_patterns({key: stats}, FilterConfig(
            min_occurrences=20, max_p_value=0.05, min_effect_size=0.0,
            min_risk_reward=1.0,
        ))
        assert len(result) == 0

    def test_rejects_walk_forward_failure(self):
        key = (_make_features(),)
        stats = self._make_stats(
            fh_wr=0.55, sh_wr=0.45,
            fold_wrs=[0.55, 0.48, 0.45],  # only 1 of 3 folds profitable (< min 2)
        )
        result = filter_patterns({key: stats}, self._LENIENT)
        assert len(result) == 0

    def test_short_direction(self):
        key = (_make_features("bear"),)
        stats = self._make_stats(
            avg_return=-0.05, win_rate=0.42,
            fh_wr=0.42, sh_wr=0.40,
            fold_wrs=[0.42, 0.40, 0.38],
        )
        result = filter_patterns({key: stats}, self._LENIENT)
        assert len(result) == 1
        assert result[0].direction == "short"

    def test_max_patterns_cap(self):
        patterns = {}
        for i in range(10):
            key = (_make_features(body_strength=f"strength_{i}"),)
            stats = self._make_stats(t_stat=2.0 + i * 0.1)
            patterns[key] = stats
        result = filter_patterns(patterns, FilterConfig(
            min_occurrences=20, max_p_value=0.05, min_effect_size=0.0,
            max_patterns=3,
        ))
        assert len(result) == 3

    def test_empty_patterns(self):
        result = filter_patterns({}, FilterConfig())
        assert result == []

    def test_rejects_low_effect_size(self):
        """Cohen's d below threshold should reject pattern."""
        key = (_make_features(),)
        # avg_return=0.01, std=0.10 → d=0.1 < 0.3
        stats = self._make_stats(avg_return=0.01, std_return=0.10)
        result = filter_patterns({key: stats}, FilterConfig(
            min_occurrences=20, max_p_value=0.05,
            min_effect_size=0.3,
        ))
        assert len(result) == 0

    def test_passes_high_effect_size(self):
        """Cohen's d above threshold should pass."""
        key = (_make_features(),)
        # avg_return=0.05, std=0.10 → d=0.5 >= 0.3
        stats = self._make_stats(avg_return=0.05, std_return=0.10)
        result = filter_patterns({key: stats}, FilterConfig(
            min_occurrences=20, max_p_value=0.05,
            min_effect_size=0.3,
        ))
        assert len(result) == 1
        assert result[0].effect_size == pytest.approx(0.5, abs=0.01)

    def test_regime_filter_keeps_matching(self):
        """Regime filter only counts matching outcomes."""
        key = (_make_features(),)
        # Need varied returns for regime-filtered recomputation of t-test
        ctx = PatternContext("morning", 2, "mean_reverting", 0)
        outcomes = []
        import random
        rng = random.Random(42)
        for i in range(60):
            ret = 0.05 + rng.gauss(0, 0.05)
            outcomes.append(PatternOutcome(i, {5: ret}, -0.1, 0.15, ctx))
        stats = PatternStats(
            key=key,
            occurrences=60,
            outcomes=outcomes,
            avg_return={5: 0.05},
            win_rate={5: 0.58},
            std_return={5: 0.05},
            t_stat={5: 2.5},
            p_value={5: 0.005},
            first_half_wr={5: 0.55},
            second_half_wr={5: 0.56},
            fold_win_rates={5: [0.55, 0.56, 0.57]},
            avg_mae=-0.1,
            avg_mfe=0.15,
        )
        result = filter_patterns({key: stats}, FilterConfig(
            min_occurrences=20, max_p_value=0.05, min_effect_size=0.0,
            regime_filter="mean_reverting",
        ))
        assert len(result) == 1
        assert result[0].dominant_regime == "mean_reverting"

    def test_regime_filter_rejects_wrong_regime(self):
        """Regime filter rejects patterns in wrong regime."""
        key = (_make_features(),)
        stats = self._make_stats(occurrences=60, regime="strong_trend_up")
        result = filter_patterns({key: stats}, FilterConfig(
            min_occurrences=20, max_p_value=0.05, min_effect_size=0.0,
            regime_filter="mean_reverting",
        ))
        assert len(result) == 0

    def test_three_fold_all_profitable_passes(self):
        """Pattern with all 3 folds profitable should pass."""
        key = (_make_features(),)
        stats = self._make_stats(fold_wrs=[0.55, 0.56, 0.58])
        result = filter_patterns({key: stats}, self._LENIENT)
        assert len(result) == 1

    def test_three_fold_one_unprofitable_passes_with_majority(self):
        """Pattern with 2/3 profitable folds passes (min_walk_forward_folds=2)."""
        key = (_make_features(),)
        stats = self._make_stats(fold_wrs=[0.55, 0.49, 0.56])
        result = filter_patterns({key: stats}, self._LENIENT)
        assert len(result) == 1

    def test_three_fold_two_unprofitable_rejects(self):
        """Pattern with only 1/3 profitable folds should be rejected."""
        key = (_make_features(),)
        stats = self._make_stats(fold_wrs=[0.55, 0.45, 0.48])
        result = filter_patterns({key: stats}, self._LENIENT)
        assert len(result) == 0

    def test_three_fold_all_required_when_configured(self):
        """min_walk_forward_folds=3 rejects pattern with 1 bad fold."""
        key = (_make_features(),)
        stats = self._make_stats(fold_wrs=[0.55, 0.49, 0.56])
        result = filter_patterns({key: stats}, FilterConfig(
            min_effect_size=0.0, min_walk_forward_folds=3,
        ))
        assert len(result) == 0

    def test_regime_breakdown_stored(self):
        """Regime breakdown should be stored on discovered pattern."""
        key = (_make_features(),)
        stats = self._make_stats(occurrences=60)
        result = filter_patterns({key: stats}, self._LENIENT)
        assert len(result) == 1
        assert result[0].regime_breakdown is not None
        assert "unknown" in result[0].regime_breakdown


# ======================================================================
# FDR correction
# ======================================================================


class TestFDRCorrection:
    def test_all_significant(self):
        patterns = [
            _make_discovered(p_value=0.001),
            _make_discovered(p_value=0.002),
            _make_discovered(p_value=0.003),
        ]
        result = apply_fdr_correction(patterns, alpha=0.05)
        assert len(result) == 3

    def test_removes_insignificant(self):
        patterns = [
            _make_discovered(p_value=0.001),
            _make_discovered(p_value=0.08),  # BH threshold = 2/2 * 0.05 = 0.05 → 0.08 > 0.05
        ]
        result = apply_fdr_correction(patterns, alpha=0.05)
        # First one passes (0.001 <= 1/2 * 0.05 = 0.025)
        # Second one fails (0.08 > 2/2 * 0.05 = 0.05)
        assert len(result) == 1

    def test_empty_list(self):
        assert apply_fdr_correction([]) == []

    def test_preserves_original_order(self):
        p1 = _make_discovered(p_value=0.02, t_stat=3.0)
        p2 = _make_discovered(p_value=0.01, t_stat=2.0)
        result = apply_fdr_correction([p1, p2], alpha=0.05)
        # Both should survive; original order preserved
        assert len(result) == 2
        assert result[0].t_stat == 3.0  # p1 first
        assert result[1].t_stat == 2.0  # p2 second

    def test_total_hypotheses_makes_correction_lenient(self):
        """Using total_hypotheses > len(patterns) makes BH more lenient."""
        patterns = [
            _make_discovered(p_value=0.01),
            _make_discovered(p_value=0.04),
        ]
        # With m=2 (default): threshold = rank/2 * 0.05
        # rank 1: 0.01 <= 0.025 → pass, rank 2: 0.04 <= 0.05 → pass
        assert len(apply_fdr_correction(patterns, alpha=0.05)) == 2
        # With m=100: threshold = rank/100 * 0.05
        # rank 1: 0.01 <= 0.0005 → fail
        result_strict = apply_fdr_correction(
            patterns, alpha=0.05, total_hypotheses=100,
        )
        assert len(result_strict) == 0

    def test_total_hypotheses_none_uses_len(self):
        """When total_hypotheses is None, uses len(patterns)."""
        patterns = [_make_discovered(p_value=0.001)]
        # m=1: threshold = 1/1 * 0.05 = 0.05 → 0.001 passes
        assert len(apply_fdr_correction(patterns, alpha=0.05)) == 1
        assert len(
            apply_fdr_correction(patterns, alpha=0.05, total_hypotheses=None)
        ) == 1


# ======================================================================
# Deduplication
# ======================================================================


class TestDeduplication:
    def test_removes_longer_superset(self):
        a = _make_features("bull")
        b = _make_features("doji")
        c = _make_features("bear")

        short = _make_discovered(key=(b, c), t_stat=3.0)
        long_ = _make_discovered(key=(a, b, c), t_stat=2.5)  # suffix matches short

        result = deduplicate_patterns([short, long_])
        assert len(result) == 1
        assert len(result[0].key) == 2  # shorter kept

    def test_keeps_longer_if_better(self):
        a = _make_features("bull")
        b = _make_features("doji")
        c = _make_features("bear")

        short = _make_discovered(key=(b, c), t_stat=2.0)
        long_ = _make_discovered(key=(a, b, c), t_stat=3.5)  # strictly better

        result = deduplicate_patterns([short, long_])
        assert len(result) == 2  # both kept

    def test_empty(self):
        assert deduplicate_patterns([]) == []

    def test_no_overlap(self):
        a = _make_features("bull")
        b = _make_features("bear")

        p1 = _make_discovered(key=(a,))
        p2 = _make_discovered(key=(b,))

        result = deduplicate_patterns([p1, p2])
        assert len(result) == 2


# ======================================================================
# PatternCatalog
# ======================================================================


class TestPatternCatalog:
    def test_lookup_match(self):
        bf_a = _make_features("bull")
        bf_b = _make_features("doji")
        pattern = _make_discovered(key=(bf_a, bf_b), t_stat=3.0)
        catalog = PatternCatalog(patterns=(pattern,))

        recent = (bf_a, bf_b)
        result = catalog.lookup(recent)
        assert result is not None
        assert result.t_stat == 3.0

    def test_lookup_no_match(self):
        bf_a = _make_features("bull")
        bf_b = _make_features("doji")
        pattern = _make_discovered(key=(bf_a, bf_b))
        catalog = PatternCatalog(patterns=(pattern,))

        # Different features
        recent = (_make_features("bear"), _make_features("bear"))
        assert catalog.lookup(recent) is None

    def test_lookup_longest_match_wins(self):
        bf_a = _make_features("bull")
        bf_b = _make_features("doji")
        bf_c = _make_features("bear")

        short = _make_discovered(key=(bf_b, bf_c), t_stat=2.0)
        long_ = _make_discovered(key=(bf_a, bf_b, bf_c), t_stat=4.0)
        catalog = PatternCatalog(patterns=(short, long_))

        # Both match — longer has higher t-stat so wins
        recent = (bf_a, bf_b, bf_c)
        result = catalog.lookup(recent)
        assert result is not None
        assert result.t_stat == 4.0

    def test_lookup_insufficient_features(self):
        bf_a = _make_features("bull")
        bf_b = _make_features("doji")
        pattern = _make_discovered(key=(bf_a, bf_b))
        catalog = PatternCatalog(patterns=(pattern,))

        # Only 1 feature, pattern needs 2
        assert catalog.lookup((bf_a,)) is None

    def test_empty_catalog(self):
        catalog = PatternCatalog(patterns=())
        assert catalog.lookup((_make_features(),)) is None

    def test_serialization_roundtrip(self):
        bf = _make_features("bull")
        pattern = _make_discovered(key=(bf,))
        catalog = PatternCatalog(
            patterns=(pattern,),
            discovery_date="2025-01-15",
            data_range="2024-01-01 to 2025-01-15",
            total_bars_scanned=10000,
            total_patterns_tested=500,
        )

        d = catalog.to_dict()
        restored = PatternCatalog.from_dict(d)

        assert len(restored.patterns) == 1
        assert restored.patterns[0].direction == "long"
        assert restored.patterns[0].key[0].body_type == "bull"
        assert restored.discovery_date == "2025-01-15"
        assert restored.total_bars_scanned == 10000

    def test_save_load_roundtrip(self, tmp_path):
        bf = _make_features("bear")
        pattern = _make_discovered(key=(bf,), direction="short")
        catalog = PatternCatalog(patterns=(pattern,))

        path = tmp_path / "catalog.json"
        catalog.save(path)
        loaded = PatternCatalog.load(path)

        assert len(loaded.patterns) == 1
        assert loaded.patterns[0].direction == "short"
        assert loaded.patterns[0].key[0].body_type == "bear"


# ======================================================================
# PatternMiner (unit tests with mock data)
# ======================================================================


class TestPatternMiner:
    def test_mine_empty_data(self):
        hub = MagicMock()
        hub.atr.return_value = Decimal("1.0")
        hub.rel_vol.return_value = Decimal("1.0")
        hub.ema.return_value = Decimal("100.0")
        encoder = BarEncoder(hub)
        miner = PatternMiner(encoder)
        result = miner.mine([], start=0, end=0)
        assert result == {}

    def test_mine_insufficient_data(self):
        hub = MagicMock()
        hub.atr.return_value = Decimal("1.0")
        hub.rel_vol.return_value = Decimal("1.0")
        hub.ema.return_value = Decimal("100.0")
        encoder = BarEncoder(hub)
        miner = PatternMiner(encoder, pattern_lengths=(2,), horizons=(1,))
        # Only 1 bar — can't make a 2-bar pattern
        bars = [_make_bar(100.0)]
        result = miner.mine(bars, start=0, end=1)
        assert result == {}

    def test_mine_produces_patterns(self):
        """Two identical bars should produce a 2-bar pattern."""
        hub = MagicMock()
        hub.atr.return_value = Decimal("1.0")
        hub.rel_vol.return_value = Decimal("1.0")
        hub.ema.return_value = Decimal("100.0")
        encoder = BarEncoder(hub)
        miner = PatternMiner(encoder, pattern_lengths=(2,), horizons=(1,))

        # 5 bars on same day, close increasing slightly
        bars = []
        for i in range(5):
            bars.append(_make_bar(
                close=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                open_=99.8 + i * 0.1,
                date=f"2025-01-15 {10 + i}:30:00-05:00",
            ))

        result = miner.mine(bars, start=0, end=len(bars))
        assert len(result) > 0  # should find at least 1 pattern

    def test_session_boundary_stops_measurement(self):
        """Returns should stop at session boundary."""
        hub = MagicMock()
        hub.atr.return_value = Decimal("1.0")
        hub.rel_vol.return_value = Decimal("1.0")
        hub.ema.return_value = Decimal("100.0")
        encoder = BarEncoder(hub)
        miner = PatternMiner(encoder, pattern_lengths=(2,), horizons=(1, 3))

        # 2 bars day 1, 2 bars day 2
        bars = [
            _make_bar(100.0, date="2025-01-15 15:50:00-05:00"),
            _make_bar(100.1, date="2025-01-15 15:55:00-05:00"),
            _make_bar(100.5, date="2025-01-16 09:30:00-05:00"),
            _make_bar(100.6, date="2025-01-16 09:35:00-05:00"),
        ]

        result = miner.mine(bars, start=0, end=len(bars))

        # Pattern at bars [0,1] — horizon 1 would be bar 2 (next day)
        # Session boundary should prevent measurement
        for key, stats in result.items():
            for outcome in stats.outcomes:
                # No outcome from bar 1 should have returns at any horizon
                # because bar 2 is on a different day
                if outcome.bar_index == 1:
                    assert len(outcome.returns) == 0 or all(
                        h_bar <= 0 for h_bar in outcome.returns
                    )


# ======================================================================
# Label generation
# ======================================================================


class TestGenerateLabel:
    def test_basic_label(self):
        from stockdownloader.analysis.pattern_discovery import _generate_label

        bf = _make_features("bull", "strong")
        label = _generate_label((bf,), "long", 5)
        assert "bull-strong" in label
        assert "LONG" in label
        assert "5b" in label

    def test_doji_label(self):
        from stockdownloader.analysis.pattern_discovery import _generate_label

        bf = _make_features("doji", "weak")
        label = _generate_label((bf,), "short", 3)
        assert "doji" in label
        assert "SHORT" in label

    def test_multi_bar_label(self):
        from stockdownloader.analysis.pattern_discovery import _generate_label

        bf1 = _make_features("bull", "strong")
        bf2 = _make_features("doji")
        label = _generate_label((bf1, bf2), "long", 5)
        assert "/" in label  # separator between bars
        assert "2-bar" in label


# ======================================================================
# Confirmation analysis
# ======================================================================


class TestConfirmationAnalysis:
    """Tests for _find_best_confirmation()."""

    def test_finds_best_single_field(self):
        from stockdownloader.analysis.pattern_discovery import _find_best_confirmation

        # Create outcomes where rsi_zone="neutral" has much better WR
        outcomes = []
        for i in range(30):
            # Neutral RSI: 80% win rate for long direction
            ctx = PatternContext(
                "morning", 2, "unknown", 1,
                rsi_zone="neutral",
                vwap_position="above",
            )
            ret = 0.05 if i < 24 else -0.05  # 24/30 = 80%
            outcomes.append(PatternOutcome(i, {5: ret}, -0.1, 0.15, ctx))

        for i in range(20):
            # Overbought RSI: 40% win rate
            ctx = PatternContext(
                "morning", 2, "unknown", 1,
                rsi_zone="overbought",
                vwap_position="above",
            )
            ret = 0.05 if i < 8 else -0.05  # 8/20 = 40%
            outcomes.append(PatternOutcome(i + 30, {5: ret}, -0.1, 0.15, ctx))

        base_wr = 32 / 50  # 0.64 overall
        result = _find_best_confirmation(outcomes, "long", 5, base_wr, min_subset_size=8)
        assert result is not None
        assert "rsi_zone" in result
        assert result["rsi_zone"] == "neutral"

    def test_no_improvement_returns_none(self):
        from stockdownloader.analysis.pattern_discovery import _find_best_confirmation

        # All outcomes have roughly same WR regardless of indicators
        outcomes = []
        for i in range(40):
            zone = "neutral" if i % 2 == 0 else "overbought"
            ctx = PatternContext(
                "morning", 2, "unknown", 1,
                rsi_zone=zone,
                vwap_position="above",
            )
            ret = 0.05 if i < 24 else -0.05  # 60% WR for both groups
            outcomes.append(PatternOutcome(i, {5: ret}, -0.1, 0.15, ctx))

        result = _find_best_confirmation(outcomes, "long", 5, 0.60, min_subset_size=8)
        assert result is None

    def test_insufficient_subset_size(self):
        from stockdownloader.analysis.pattern_discovery import _find_best_confirmation

        # Good indicator value but too few samples
        outcomes = []
        for i in range(5):
            ctx = PatternContext(
                "morning", 2, "unknown", 0,  # trend_dir=0
                rsi_zone="oversold",
            )
            outcomes.append(PatternOutcome(i, {5: 0.10}, -0.1, 0.15, ctx))

        for i in range(30):
            # Mix trend_dir so it doesn't create a spurious signal
            td = 1 if i % 2 == 0 else -1
            ctx = PatternContext(
                "morning", 2, "unknown", td,
                rsi_zone="neutral",
            )
            ret = 0.05 if i < 18 else -0.05
            outcomes.append(PatternOutcome(i + 5, {5: ret}, -0.1, 0.15, ctx))

        result = _find_best_confirmation(
            outcomes, "long", 5, 0.60, min_subset_size=8,
        )
        # "oversold" group has only 5 samples (< 8), so ignored
        # "neutral" group has 60% WR = base WR, no improvement
        # trend_dir is mixed so no single value should improve WR significantly
        assert result is None

    def test_empty_outcomes(self):
        from stockdownloader.analysis.pattern_discovery import _find_best_confirmation

        result = _find_best_confirmation([], "long", 5, 0.60)
        assert result is None

    def test_no_context_fields_populated(self):
        from stockdownloader.analysis.pattern_discovery import _find_best_confirmation

        # All indicator fields are None, trend_dir mixed so no signal
        outcomes = []
        for i in range(20):
            td = 1 if i % 2 == 0 else -1
            ctx = PatternContext("morning", 2, "unknown", td)
            ret = 0.05 if i < 12 else -0.05  # 60% WR overall
            outcomes.append(PatternOutcome(i, {5: ret}, -0.1, 0.15, ctx))

        result = _find_best_confirmation(outcomes, "long", 5, 0.60)
        assert result is None


# ======================================================================
# DiscoveredPattern enhancement fields
# ======================================================================


class TestDiscoveredPatternEnhancement:
    def test_default_confirmation_none(self):
        p = _make_discovered()
        assert p.confirmation is None

    def test_default_timeframe_5m(self):
        p = _make_discovered()
        assert p.timeframe == "5m"

    def test_with_confirmation(self):
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=30,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.1,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"rsi_zone": "neutral", "trend_dir": "1"},
            timeframe="15m",
        )
        assert p.confirmation == {"rsi_zone": "neutral", "trend_dir": "1"}
        assert p.timeframe == "15m"

    def test_serialization_with_confirmation(self):
        """Confirmation + timeframe survive to_dict/from_dict roundtrip."""
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=30,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.1,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            confirmation={"rsi_zone": "neutral"},
            timeframe="1h",
        )
        catalog = PatternCatalog(patterns=(p,))
        d = catalog.to_dict()
        restored = PatternCatalog.from_dict(d)
        rp = restored.patterns[0]
        assert rp.confirmation == {"rsi_zone": "neutral"}
        assert rp.timeframe == "1h"

    def test_serialization_without_confirmation(self):
        """Patterns without confirmation still roundtrip correctly."""
        bf = _make_features()
        p = _make_discovered(key=(bf,))
        catalog = PatternCatalog(patterns=(p,))
        d = catalog.to_dict()
        restored = PatternCatalog.from_dict(d)
        rp = restored.patterns[0]
        assert rp.confirmation is None
        assert rp.timeframe == "5m"  # default

    def test_backward_compat_old_catalog_json(self):
        """Loading a catalog JSON without confirmation/timeframe keys works."""
        bf = _make_features()
        d = {
            "patterns": [{
                "key": [{
                    "body_type": bf.body_type,
                    "body_strength": bf.body_strength,
                    "wick_signal": bf.wick_signal,
                    "relative_size": bf.relative_size,
                    "volume_profile": bf.volume_profile,
                }],
                "direction": "long",
                "best_horizon": 5,
                "avg_return": 0.05,
                "win_rate": 0.60,
                "occurrences": 30,
                "t_stat": 3.0,
                "p_value": 0.003,
                "avg_mae": -0.1,
                "avg_mfe": 0.15,
                "risk_reward": 1.5,
                "walk_forward_stable": True,
                "human_label": "1-bar: test",
                # NO "confirmation", "timeframe", "effect_size", "dominant_regime", etc.
            }],
        }
        catalog = PatternCatalog.from_dict(d)
        rp = catalog.patterns[0]
        assert rp.confirmation is None
        assert rp.timeframe == "5m"
        assert rp.effect_size == 0.0
        assert rp.dominant_regime is None
        assert rp.regime_breakdown is None

    def test_serialization_with_new_fields(self):
        """New fields (effect_size, regime) survive roundtrip."""
        bf = _make_features()
        p = DiscoveredPattern(
            key=(bf,),
            direction="long",
            best_horizon=5,
            avg_return=0.05,
            win_rate=0.60,
            occurrences=30,
            t_stat=3.0,
            p_value=0.003,
            avg_mae=-0.1,
            avg_mfe=0.15,
            risk_reward=1.5,
            walk_forward_stable=True,
            human_label="1-bar: test",
            effect_size=0.45,
            dominant_regime="mean_reverting",
            regime_breakdown={"mean_reverting": 20, "weak_trend": 10},
        )
        catalog = PatternCatalog(patterns=(p,))
        d = catalog.to_dict()
        restored = PatternCatalog.from_dict(d)
        rp = restored.patterns[0]
        assert rp.effect_size == pytest.approx(0.45)
        assert rp.dominant_regime == "mean_reverting"
        assert rp.regime_breakdown == {"mean_reverting": 20, "weak_trend": 10}


# ======================================================================
# Filter patterns with timeframe
# ======================================================================


class TestFilterPatternsTimeframe:
    _LENIENT = FilterConfig(
        min_occurrences=20,
        max_p_value=0.05,
        min_effect_size=0.0,
    )

    def _make_stats_with_context(
        self,
        occurrences: int = 60,
        avg_return: float = 0.05,
        win_rate: float = 0.58,
        t_stat: float = 2.5,
        p_value: float = 0.005,
        mae: float = -0.1,
        mfe: float = 0.15,
        fh_wr: float = 0.55,
        sh_wr: float = 0.56,
        std_return: float = 0.10,
    ) -> PatternStats:
        ctx = PatternContext("morning", 2, "unknown", 0, rsi_zone="neutral")
        outcomes = [
            PatternOutcome(i, {5: avg_return}, mae, mfe, ctx)
            for i in range(occurrences)
        ]
        stats = PatternStats(
            key=(_make_features(),),
            occurrences=occurrences,
            outcomes=outcomes,
            avg_return={5: avg_return},
            win_rate={5: win_rate},
            std_return={5: std_return},
            t_stat={5: t_stat},
            p_value={5: p_value},
            first_half_wr={5: fh_wr},
            second_half_wr={5: sh_wr},
            fold_win_rates={5: [0.55, 0.56, 0.57]},
            avg_mae=mae,
            avg_mfe=mfe,
        )
        return stats

    def test_timeframe_stored_on_pattern(self):
        key = (_make_features(),)
        stats = self._make_stats_with_context()
        result = filter_patterns({key: stats}, self._LENIENT, timeframe="15m")
        assert len(result) == 1
        assert result[0].timeframe == "15m"

    def test_default_timeframe_5m(self):
        key = (_make_features(),)
        stats = self._make_stats_with_context()
        result = filter_patterns({key: stats}, self._LENIENT)
        assert len(result) == 1
        assert result[0].timeframe == "5m"
