"""Tests for pattern_discovery — filtering, FDR correction, deduplication."""
from __future__ import annotations

import pytest

from stockdownloader.analysis.pattern_discovery import (
    DiscoveredPattern,
    FilterConfig,
    PatternOutcome,
    PatternStats,
    apply_fdr_correction,
    deduplicate_patterns,
    filter_patterns,
)
from stockdownloader.analysis.pattern_encoder import BarFeatures, PatternContext


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
        # avg_return=0.01, std=0.10 -> d=0.1 < 0.3
        stats = self._make_stats(avg_return=0.01, std_return=0.10)
        result = filter_patterns({key: stats}, FilterConfig(
            min_occurrences=20, max_p_value=0.05,
            min_effect_size=0.3,
        ))
        assert len(result) == 0

    def test_passes_high_effect_size(self):
        """Cohen's d above threshold should pass."""
        key = (_make_features(),)
        # avg_return=0.05, std=0.10 -> d=0.5 >= 0.3
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
            _make_discovered(p_value=0.08),  # BH threshold = 2/2 * 0.05 = 0.05 -> 0.08 > 0.05
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
        # rank 1: 0.01 <= 0.025 -> pass, rank 2: 0.04 <= 0.05 -> pass
        assert len(apply_fdr_correction(patterns, alpha=0.05)) == 2
        # With m=100: threshold = rank/100 * 0.05
        # rank 1: 0.01 <= 0.0005 -> fail
        result_strict = apply_fdr_correction(
            patterns, alpha=0.05, total_hypotheses=100,
        )
        assert len(result_strict) == 0

    def test_total_hypotheses_none_uses_len(self):
        """When total_hypotheses is None, uses len(patterns)."""
        patterns = [_make_discovered(p_value=0.001)]
        # m=1: threshold = 1/1 * 0.05 = 0.05 -> 0.001 passes
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
