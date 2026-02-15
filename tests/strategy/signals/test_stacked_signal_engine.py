"""Tests for StackedSignalEngine with all 5 aggregation modes."""
from __future__ import annotations

import math

import pytest

from stockdownloader.strategy.signals.signal_generator import (
    SignalDirection,
    SignalResult,
)
from stockdownloader.strategy.signals.multi_timeframe_aligner import AlignedSignal
from stockdownloader.strategy.signals.stacked_signal_engine import (
    AggregationMode,
    StackConfig,
    StackedSignalEngine,
    StackResult,
)
from stockdownloader.util.timeframe_aggregator import Timeframe


# ======================================================================
# Helpers
# ======================================================================


def _make_aligned(
    score: float,
    fired: bool = True,
    weight: float = 1.0,
    name: str = "gen",
    category: str = "momentum",
    timeframe: Timeframe = Timeframe.M5,
) -> AlignedSignal:
    direction = (
        SignalDirection.BULLISH if score > 0.1
        else SignalDirection.BEARISH if score < -0.1
        else SignalDirection.NEUTRAL
    )
    return AlignedSignal(
        result=SignalResult(
            score=score,
            direction=direction,
            fired=fired,
        ),
        timeframe=timeframe,
        generator_name=name,
        category=category,
        weight=weight,
    )


# ======================================================================
# AggregationMode enum tests
# ======================================================================


class TestAggregationMode:
    def test_enum_values(self):
        assert AggregationMode.WEIGHTED_AVERAGE.value == "weighted_average"
        assert AggregationMode.WEIGHTED_SUM.value == "weighted_sum"
        assert AggregationMode.UNANIMOUS.value == "unanimous"
        assert AggregationMode.MAJORITY_VOTE.value == "majority_vote"
        assert AggregationMode.MAX_CONFLUENCE.value == "max_confluence"

    def test_enum_count(self):
        assert len(AggregationMode) == 5


# ======================================================================
# StackConfig tests
# ======================================================================


class TestStackConfig:
    def test_defaults(self):
        cfg = StackConfig()
        assert cfg.buy_threshold == 0.3
        assert cfg.sell_threshold == 0.3
        assert cfg.mode == AggregationMode.WEIGHTED_AVERAGE
        assert cfg.require_fire is True
        assert cfg.min_fire_count == 1
        assert cfg.category_weights == {}

    def test_custom_values(self):
        cfg = StackConfig(
            buy_threshold=0.5,
            sell_threshold=0.4,
            mode=AggregationMode.UNANIMOUS,
            require_fire=False,
            min_fire_count=3,
            category_weights={"momentum": 1.5, "trend": 2.0},
        )
        assert cfg.buy_threshold == 0.5
        assert cfg.sell_threshold == 0.4
        assert cfg.mode == AggregationMode.UNANIMOUS
        assert cfg.require_fire is False
        assert cfg.min_fire_count == 3
        assert cfg.category_weights == {"momentum": 1.5, "trend": 2.0}


# ======================================================================
# StackResult tests
# ======================================================================


class TestStackResult:
    def test_construction(self):
        result = StackResult(
            composite_score=0.5,
            buy_signal=True,
            sell_signal=False,
            fire_count=2,
            total_signals=3,
            per_signal_scores={"rsi": (0.7, True, 1.0)},
        )
        assert result.composite_score == 0.5
        assert result.buy_signal is True
        assert result.sell_signal is False
        assert result.fire_count == 2
        assert result.total_signals == 3
        assert result.per_signal_scores == {"rsi": (0.7, True, 1.0)}

    def test_default_per_signal(self):
        result = StackResult(
            composite_score=0.0,
            buy_signal=False,
            sell_signal=False,
            fire_count=0,
            total_signals=0,
        )
        assert result.per_signal_scores == {}


# ======================================================================
# StackedSignalEngine — empty input
# ======================================================================


class TestStackedSignalEngineEmpty:
    def test_empty_signals(self):
        engine = StackedSignalEngine(StackConfig())
        result = engine.evaluate([])
        assert result.composite_score == 0.0
        assert result.buy_signal is False
        assert result.sell_signal is False
        assert result.fire_count == 0
        assert result.total_signals == 0

    def test_config_property(self):
        cfg = StackConfig(buy_threshold=0.5)
        engine = StackedSignalEngine(cfg)
        assert engine.config is cfg


# ======================================================================
# WEIGHTED_AVERAGE mode
# ======================================================================


class TestWeightedAverage:
    def _engine(self, **kwargs) -> StackedSignalEngine:
        return StackedSignalEngine(
            StackConfig(mode=AggregationMode.WEIGHTED_AVERAGE, **kwargs),
        )

    def test_single_signal_bullish(self):
        engine = self._engine(buy_threshold=0.3, require_fire=False)
        signals = [_make_aligned(0.7, fired=False, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.7)
        assert result.buy_signal is True
        assert result.sell_signal is False

    def test_single_signal_bearish(self):
        engine = self._engine(sell_threshold=0.3, require_fire=False)
        signals = [_make_aligned(-0.8, fired=False, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(-0.8)
        assert result.buy_signal is False
        assert result.sell_signal is True

    def test_two_equal_weights(self):
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.6, weight=1.0, name="a"),
            _make_aligned(0.4, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        # (0.6*1 + 0.4*1) / (1+1) = 0.5
        assert result.composite_score == pytest.approx(0.5)

    def test_two_unequal_weights(self):
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.8, weight=2.0, name="a"),
            _make_aligned(0.2, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        # (0.8*2 + 0.2*1) / (2+1) = 1.8/3 = 0.6
        assert result.composite_score == pytest.approx(0.6)

    def test_mixed_signals_cancel(self):
        engine = self._engine(buy_threshold=0.3, sell_threshold=0.3, require_fire=False)
        signals = [
            _make_aligned(0.5, weight=1.0, name="bull"),
            _make_aligned(-0.5, weight=1.0, name="bear"),
        ]
        result = engine.evaluate(signals)
        # (0.5 + -0.5) / 2 = 0.0
        assert result.composite_score == pytest.approx(0.0)
        assert result.buy_signal is False
        assert result.sell_signal is False

    def test_below_buy_threshold(self):
        engine = self._engine(buy_threshold=0.5, require_fire=False)
        signals = [_make_aligned(0.3, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.3)
        assert result.buy_signal is False

    def test_exactly_at_buy_threshold(self):
        engine = self._engine(buy_threshold=0.5, require_fire=False)
        signals = [_make_aligned(0.5, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        assert result.buy_signal is True

    def test_category_weights(self):
        engine = StackedSignalEngine(StackConfig(
            mode=AggregationMode.WEIGHTED_AVERAGE,
            require_fire=False,
            category_weights={"momentum": 2.0, "trend": 0.5},
        ))
        signals = [
            _make_aligned(0.4, weight=1.0, name="rsi", category="momentum"),
            _make_aligned(0.4, weight=1.0, name="sma", category="trend"),
        ]
        result = engine.evaluate(signals)
        # Effective weights: rsi=1*2=2, sma=1*0.5=0.5
        # (0.4*2 + 0.4*0.5) / (2+0.5) = 1.0/2.5 = 0.4
        assert result.composite_score == pytest.approx(0.4)

    def test_zero_total_weight(self):
        engine = StackedSignalEngine(StackConfig(
            mode=AggregationMode.WEIGHTED_AVERAGE,
            require_fire=False,
            category_weights={"momentum": 0.0},
        ))
        signals = [
            _make_aligned(0.5, weight=0.0, name="a", category="momentum"),
        ]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.0)


# ======================================================================
# WEIGHTED_SUM mode
# ======================================================================


class TestWeightedSum:
    def _engine(self, **kwargs) -> StackedSignalEngine:
        return StackedSignalEngine(
            StackConfig(mode=AggregationMode.WEIGHTED_SUM, **kwargs),
        )

    def test_single_signal(self):
        engine = self._engine(require_fire=False)
        signals = [_make_aligned(0.5, weight=2.0, name="a")]
        result = engine.evaluate(signals)
        # 0.5 * 2 = 1.0
        assert result.composite_score == pytest.approx(1.0)

    def test_multiple_signals_sum(self):
        engine = self._engine(buy_threshold=1.0, require_fire=False)
        signals = [
            _make_aligned(0.3, weight=1.0, name="a"),
            _make_aligned(0.4, weight=1.0, name="b"),
            _make_aligned(0.5, weight=1.0, name="c"),
        ]
        result = engine.evaluate(signals)
        # 0.3 + 0.4 + 0.5 = 1.2
        assert result.composite_score == pytest.approx(1.2)
        assert result.buy_signal is True

    def test_rewards_more_signals(self):
        """WEIGHTED_SUM grows with more concordant signals (unlike WEIGHTED_AVERAGE)."""
        engine = self._engine(require_fire=False)
        two_signals = [
            _make_aligned(0.5, weight=1.0, name="a"),
            _make_aligned(0.5, weight=1.0, name="b"),
        ]
        three_signals = two_signals + [
            _make_aligned(0.5, weight=1.0, name="c"),
        ]
        r2 = engine.evaluate(two_signals)
        r3 = engine.evaluate(three_signals)
        assert r3.composite_score > r2.composite_score

    def test_bearish_sum(self):
        engine = self._engine(sell_threshold=0.5, require_fire=False)
        signals = [
            _make_aligned(-0.3, weight=1.0, name="a"),
            _make_aligned(-0.4, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        # -0.3 + -0.4 = -0.7
        assert result.composite_score == pytest.approx(-0.7)
        assert result.sell_signal is True


# ======================================================================
# UNANIMOUS mode
# ======================================================================


class TestUnanimous:
    def _engine(self, **kwargs) -> StackedSignalEngine:
        return StackedSignalEngine(
            StackConfig(mode=AggregationMode.UNANIMOUS, **kwargs),
        )

    def test_all_bullish(self):
        engine = self._engine(buy_threshold=0.3, require_fire=False)
        signals = [
            _make_aligned(0.5, weight=1.0, name="a"),
            _make_aligned(0.7, weight=1.0, name="b"),
            _make_aligned(0.3, weight=1.0, name="c"),
        ]
        result = engine.evaluate(signals)
        # All bullish: weighted average of scores
        assert result.composite_score == pytest.approx(0.5)
        assert result.buy_signal is True

    def test_all_bearish(self):
        engine = self._engine(sell_threshold=0.3, require_fire=False)
        signals = [
            _make_aligned(-0.5, weight=1.0, name="a"),
            _make_aligned(-0.7, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        # All bearish
        assert result.composite_score == pytest.approx(-0.6)
        assert result.sell_signal is True

    def test_disagreement_returns_zero(self):
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.5, weight=1.0, name="bull"),
            _make_aligned(-0.5, weight=1.0, name="bear"),
        ]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.0)
        assert result.buy_signal is False
        assert result.sell_signal is False

    def test_neutral_signals_ignored(self):
        """Neutral scores (|score| < 0.01) are ignored in unanimity check."""
        engine = self._engine(buy_threshold=0.3, require_fire=False)
        signals = [
            _make_aligned(0.5, weight=1.0, name="bull"),
            _make_aligned(0.005, weight=1.0, name="neutral"),
        ]
        result = engine.evaluate(signals)
        # Only the bullish signal counts, neutral is ignored
        assert result.composite_score == pytest.approx(0.5)
        assert result.buy_signal is True

    def test_all_neutral(self):
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.005, weight=1.0, name="a"),
            _make_aligned(-0.005, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.0)

    def test_one_dissenter_blocks_signal(self):
        """Even one non-neutral dissenter blocks the signal."""
        engine = self._engine(buy_threshold=0.2, require_fire=False)
        signals = [
            _make_aligned(0.8, weight=1.0, name="a"),
            _make_aligned(0.6, weight=1.0, name="b"),
            _make_aligned(-0.3, weight=1.0, name="dissenter"),
        ]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.0)
        assert result.buy_signal is False


# ======================================================================
# MAJORITY_VOTE mode
# ======================================================================


class TestMajorityVote:
    def _engine(self, **kwargs) -> StackedSignalEngine:
        return StackedSignalEngine(
            StackConfig(mode=AggregationMode.MAJORITY_VOTE, **kwargs),
        )

    def test_clear_bullish_majority(self):
        engine = self._engine(buy_threshold=0.3, require_fire=False)
        signals = [
            _make_aligned(0.5, weight=1.0, name="a"),
            _make_aligned(0.7, weight=1.0, name="b"),
            _make_aligned(-0.3, weight=1.0, name="c"),
        ]
        result = engine.evaluate(signals)
        # 2 bullish vs 1 bearish = bullish majority
        # Returns weighted average of bullish: (0.5 + 0.7) / 2 = 0.6
        assert result.composite_score == pytest.approx(0.6)
        assert result.buy_signal is True

    def test_clear_bearish_majority(self):
        engine = self._engine(sell_threshold=0.3, require_fire=False)
        signals = [
            _make_aligned(-0.4, weight=1.0, name="a"),
            _make_aligned(-0.6, weight=1.0, name="b"),
            _make_aligned(0.3, weight=1.0, name="c"),
        ]
        result = engine.evaluate(signals)
        # 2 bearish vs 1 bullish
        assert result.composite_score == pytest.approx(-0.5)
        assert result.sell_signal is True

    def test_tie_returns_zero(self):
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.5, weight=1.0, name="a"),
            _make_aligned(-0.5, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.0)
        assert result.buy_signal is False
        assert result.sell_signal is False

    def test_all_neutral_abstain(self):
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.005, weight=1.0, name="a"),
            _make_aligned(-0.005, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.0)

    def test_majority_with_neutral_signals(self):
        """Neutral signals abstain from voting."""
        engine = self._engine(buy_threshold=0.3, require_fire=False)
        signals = [
            _make_aligned(0.5, weight=1.0, name="a"),
            _make_aligned(0.005, weight=1.0, name="neutral"),
            _make_aligned(-0.3, weight=1.0, name="bear"),
        ]
        result = engine.evaluate(signals)
        # 1 bull vs 1 bear = tie, neutral abstains
        assert result.composite_score == pytest.approx(0.0)

    def test_weighted_majority(self):
        """Majority returns weighted average of winning side."""
        engine = self._engine(buy_threshold=0.3, require_fire=False)
        signals = [
            _make_aligned(0.8, weight=2.0, name="strong_bull"),
            _make_aligned(0.3, weight=1.0, name="weak_bull"),
            _make_aligned(-0.4, weight=1.0, name="bear"),
        ]
        result = engine.evaluate(signals)
        # 2 bullish vs 1 bearish = bullish majority
        # Weighted avg of bulls: (0.8*2 + 0.3*1) / (2+1) = 1.9/3
        assert result.composite_score == pytest.approx(1.9 / 3.0)


# ======================================================================
# MAX_CONFLUENCE mode
# ======================================================================


class TestMaxConfluence:
    def _engine(self, **kwargs) -> StackedSignalEngine:
        return StackedSignalEngine(
            StackConfig(mode=AggregationMode.MAX_CONFLUENCE, **kwargs),
        )

    def test_single_bullish(self):
        engine = self._engine(require_fire=False)
        signals = [_make_aligned(0.8, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        # Geometric mean of [0.8] = 0.8
        assert result.composite_score == pytest.approx(0.8)

    def test_two_concordant_bullish(self):
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.8, weight=1.0, name="a"),
            _make_aligned(0.6, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        # Geometric mean of [0.8, 0.6] = sqrt(0.8*0.6) = sqrt(0.48)
        expected = math.sqrt(0.48)
        assert result.composite_score == pytest.approx(expected, abs=0.01)

    def test_two_concordant_bearish(self):
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(-0.8, weight=1.0, name="a"),
            _make_aligned(-0.6, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        expected = -math.sqrt(0.48)
        assert result.composite_score == pytest.approx(expected, abs=0.01)

    def test_disagreement_uses_dominant_side(self):
        """With more bullish signals, bearish are excluded."""
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.8, weight=1.0, name="a"),
            _make_aligned(0.6, weight=1.0, name="b"),
            _make_aligned(-0.3, weight=1.0, name="c"),
        ]
        result = engine.evaluate(signals)
        # 2 bull vs 1 bear: dominant is bullish
        # Geometric mean of bullish weighted scores: sqrt(0.8*0.6)
        expected = math.sqrt(0.48)
        assert result.composite_score == pytest.approx(expected, abs=0.01)

    def test_all_neutral(self):
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.005, weight=1.0, name="a"),
            _make_aligned(-0.005, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.0)

    def test_weighted_scores(self):
        """Weights multiply scores before geometric mean."""
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.5, weight=2.0, name="a"),
            _make_aligned(0.5, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        # Bull scores: [0.5*2=1.0, 0.5*1=0.5]
        # Geometric mean: sqrt(1.0*0.5) = sqrt(0.5)
        expected = math.sqrt(0.5)
        assert result.composite_score == pytest.approx(expected, abs=0.01)

    def test_capped_at_one(self):
        """Result is capped at 1.0."""
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(0.9, weight=3.0, name="a"),
            _make_aligned(0.9, weight=3.0, name="b"),
        ]
        result = engine.evaluate(signals)
        assert result.composite_score <= 1.0

    def test_capped_at_negative_one(self):
        """Result is capped at -1.0."""
        engine = self._engine(require_fire=False)
        signals = [
            _make_aligned(-0.9, weight=3.0, name="a"),
            _make_aligned(-0.9, weight=3.0, name="b"),
        ]
        result = engine.evaluate(signals)
        assert result.composite_score >= -1.0


# ======================================================================
# Fire logic
# ======================================================================


class TestFireLogic:
    def test_require_fire_true_with_fires(self):
        engine = StackedSignalEngine(StackConfig(
            buy_threshold=0.3,
            require_fire=True,
            min_fire_count=1,
        ))
        signals = [_make_aligned(0.5, fired=True, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        assert result.buy_signal is True
        assert result.fire_count == 1

    def test_require_fire_true_without_fires(self):
        engine = StackedSignalEngine(StackConfig(
            buy_threshold=0.3,
            require_fire=True,
            min_fire_count=1,
        ))
        signals = [_make_aligned(0.5, fired=False, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        assert result.composite_score == pytest.approx(0.5)
        assert result.buy_signal is False  # score passes but fire required

    def test_require_fire_false(self):
        engine = StackedSignalEngine(StackConfig(
            buy_threshold=0.3,
            require_fire=False,
        ))
        signals = [_make_aligned(0.5, fired=False, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        assert result.buy_signal is True

    def test_min_fire_count(self):
        engine = StackedSignalEngine(StackConfig(
            buy_threshold=0.3,
            require_fire=True,
            min_fire_count=2,
        ))
        signals = [
            _make_aligned(0.5, fired=True, weight=1.0, name="a"),
            _make_aligned(0.5, fired=False, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        assert result.fire_count == 1
        assert result.buy_signal is False  # only 1 fire, need 2

    def test_min_fire_count_met(self):
        engine = StackedSignalEngine(StackConfig(
            buy_threshold=0.3,
            require_fire=True,
            min_fire_count=2,
        ))
        signals = [
            _make_aligned(0.5, fired=True, weight=1.0, name="a"),
            _make_aligned(0.5, fired=True, weight=1.0, name="b"),
        ]
        result = engine.evaluate(signals)
        assert result.fire_count == 2
        assert result.buy_signal is True

    def test_fire_count_tracking(self):
        engine = StackedSignalEngine(StackConfig(require_fire=False))
        signals = [
            _make_aligned(0.3, fired=True, weight=1.0, name="a"),
            _make_aligned(0.3, fired=False, weight=1.0, name="b"),
            _make_aligned(0.3, fired=True, weight=1.0, name="c"),
            _make_aligned(0.3, fired=False, weight=1.0, name="d"),
        ]
        result = engine.evaluate(signals)
        assert result.fire_count == 2
        assert result.total_signals == 4


# ======================================================================
# Per-signal breakdown
# ======================================================================


class TestPerSignalBreakdown:
    def test_per_signal_scores_present(self):
        engine = StackedSignalEngine(StackConfig(require_fire=False))
        signals = [
            _make_aligned(0.5, fired=True, weight=2.0, name="rsi"),
            _make_aligned(-0.3, fired=False, weight=1.0, name="macd"),
        ]
        result = engine.evaluate(signals)
        assert "rsi" in result.per_signal_scores
        assert "macd" in result.per_signal_scores

        rsi_score, rsi_fired, rsi_weight = result.per_signal_scores["rsi"]
        assert rsi_score == 0.5
        assert rsi_fired is True
        assert rsi_weight == 2.0

        macd_score, macd_fired, macd_weight = result.per_signal_scores["macd"]
        assert macd_score == -0.3
        assert macd_fired is False
        assert macd_weight == 1.0


# ======================================================================
# Category weights integration
# ======================================================================


class TestCategoryWeights:
    def test_category_weight_multiplies_effective_weight(self):
        engine = StackedSignalEngine(StackConfig(
            mode=AggregationMode.WEIGHTED_AVERAGE,
            require_fire=False,
            category_weights={"momentum": 3.0},
        ))
        signals = [
            _make_aligned(0.5, weight=1.0, name="rsi", category="momentum"),
            _make_aligned(0.5, weight=1.0, name="sma", category="trend"),
        ]
        result = engine.evaluate(signals)
        # rsi effective weight: 1.0 * 3.0 = 3.0
        # sma effective weight: 1.0 * 1.0 = 1.0 (default)
        # (0.5*3 + 0.5*1) / (3+1) = 2.0/4 = 0.5
        assert result.composite_score == pytest.approx(0.5)

    def test_missing_category_defaults_to_one(self):
        engine = StackedSignalEngine(StackConfig(
            mode=AggregationMode.WEIGHTED_AVERAGE,
            require_fire=False,
            category_weights={"volatility": 2.0},  # no momentum or trend
        ))
        signals = [
            _make_aligned(0.5, weight=1.0, name="rsi", category="momentum"),
        ]
        result = engine.evaluate(signals)
        # No category_weight for momentum => default 1.0
        assert result.composite_score == pytest.approx(0.5)

    def test_per_signal_shows_effective_weight(self):
        engine = StackedSignalEngine(StackConfig(
            mode=AggregationMode.WEIGHTED_AVERAGE,
            require_fire=False,
            category_weights={"momentum": 2.0},
        ))
        signals = [
            _make_aligned(0.5, weight=1.5, name="rsi", category="momentum"),
        ]
        result = engine.evaluate(signals)
        # Effective weight = 1.5 * 2.0 = 3.0
        _, _, weight = result.per_signal_scores["rsi"]
        assert weight == pytest.approx(3.0)


# ======================================================================
# Sell signal logic
# ======================================================================


class TestSellSignal:
    def test_sell_threshold(self):
        engine = StackedSignalEngine(StackConfig(
            sell_threshold=0.4,
            require_fire=False,
        ))
        signals = [_make_aligned(-0.3, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        # |-0.3| = 0.3 < 0.4 threshold
        assert result.sell_signal is False

    def test_sell_meets_threshold(self):
        engine = StackedSignalEngine(StackConfig(
            sell_threshold=0.4,
            require_fire=False,
        ))
        signals = [_make_aligned(-0.5, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        assert result.sell_signal is True

    def test_sell_requires_fire(self):
        engine = StackedSignalEngine(StackConfig(
            sell_threshold=0.3,
            require_fire=True,
            min_fire_count=1,
        ))
        signals = [_make_aligned(-0.5, fired=False, weight=1.0, name="a")]
        result = engine.evaluate(signals)
        assert result.sell_signal is False

    def test_buy_and_sell_exclusive(self):
        """A positive score cannot trigger sell, a negative cannot trigger buy."""
        engine = StackedSignalEngine(StackConfig(
            buy_threshold=0.3,
            sell_threshold=0.3,
            require_fire=False,
        ))
        buy_signals = [_make_aligned(0.5, weight=1.0, name="a")]
        buy_result = engine.evaluate(buy_signals)
        assert buy_result.buy_signal is True
        assert buy_result.sell_signal is False

        sell_signals = [_make_aligned(-0.5, weight=1.0, name="b")]
        sell_result = engine.evaluate(sell_signals)
        assert sell_result.buy_signal is False
        assert sell_result.sell_signal is True
