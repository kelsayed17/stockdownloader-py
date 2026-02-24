"""Composable signal stacking engine.

Combines multiple :class:`AlignedSignal` results into a single composite
score using one of five aggregation modes.  Threshold logic and fire-count
requirements determine whether the composite triggers a BUY, SELL, or HOLD.

Usage::

    from stockdownloader.signals.engine import (
        AggregationMode, StackConfig, StackedSignalEngine,
    )

    config = StackConfig(
        buy_threshold=0.3,
        sell_threshold=0.3,
        mode=AggregationMode.WEIGHTED_AVERAGE,
        require_fire=True,
    )
    engine = StackedSignalEngine(config)
    result = engine.evaluate(aligned_signals)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from stockdownloader.signals.generator import SignalDirection

if TYPE_CHECKING:
    from stockdownloader.signals.timeframe_aligner import AlignedSignal


class AggregationMode(Enum):
    """How to combine multiple signal scores into a composite."""

    WEIGHTED_AVERAGE = "weighted_average"
    """``Σ(score×weight) / Σ(weight)`` — default, balanced."""

    WEIGHTED_SUM = "weighted_sum"
    """``Σ(score×weight)`` — rewards more signals."""

    UNANIMOUS = "unanimous"
    """ALL must agree in direction (high precision, low recall)."""

    MAJORITY_VOTE = "majority_vote"
    """>50% agree in direction (binary voting)."""

    MAX_CONFLUENCE = "max_confluence"
    """Product of concordant scores (nonlinear amplification)."""


@dataclass(frozen=True, slots=True)
class StackConfig:
    """Configuration for signal stacking.

    Attributes
    ----------
    buy_threshold:
        Minimum composite score to trigger a BUY signal.
    sell_threshold:
        Minimum absolute composite score to trigger a SELL signal.
    mode:
        Aggregation mode for combining aligned signals.
    require_fire:
        If True, at least ``min_fire_count`` generators must have
        ``fired=True`` for the composite to trigger.
    min_fire_count:
        Minimum number of fired generators (only when ``require_fire``
        is True).
    category_weights:
        Per-category multiplier applied on top of per-signal weights.
        Missing categories default to 1.0.
    """

    buy_threshold: float = 0.3
    sell_threshold: float = 0.3
    mode: AggregationMode = AggregationMode.WEIGHTED_AVERAGE
    require_fire: bool = True
    min_fire_count: int = 1
    category_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StackResult:
    """Output of the stacking engine.

    Attributes
    ----------
    composite_score:
        Combined score from all signals.
    buy_signal:
        Whether the composite meets BUY criteria.
    sell_signal:
        Whether the composite meets SELL criteria.
    fire_count:
        Number of generators that fired this bar.
    total_signals:
        Total number of aligned signals evaluated.
    per_signal_scores:
        Breakdown: ``{generator_name: (score, fired, weight)}``.
    """

    composite_score: float
    buy_signal: bool
    sell_signal: bool
    fire_count: int
    total_signals: int
    per_signal_scores: dict[str, tuple[float, bool, float]] = field(
        default_factory=dict,
    )


class StackedSignalEngine:
    """Evaluates a set of aligned signals and produces a composite result.

    The engine is a pure function of ``(config, aligned_signals)`` —
    no internal state is carried between calls.

    Parameters
    ----------
    config:
        Stacking configuration (thresholds, mode, fire requirements).
    """

    __slots__ = ("_config",)

    def __init__(self, config: StackConfig) -> None:
        self._config = config

    @property
    def config(self) -> StackConfig:
        return self._config

    def evaluate(self, aligned_signals: list[AlignedSignal]) -> StackResult:
        """Combine aligned signals into a composite result.

        Parameters
        ----------
        aligned_signals:
            Output from :meth:`MultiTimeframeAligner.get_aligned_signals`.

        Returns
        -------
        StackResult
            Composite score, BUY/SELL flags, and per-signal breakdown.
        """
        if not aligned_signals:
            return StackResult(
                composite_score=0.0,
                buy_signal=False,
                sell_signal=False,
                fire_count=0,
                total_signals=0,
            )

        cfg = self._config

        # Apply category weights to effective weights
        effective: list[tuple[float, float, bool, str]] = []
        # (score, effective_weight, fired, generator_name)
        for sig in aligned_signals:
            cat_mult = cfg.category_weights.get(sig.category, 1.0)
            eff_weight = sig.weight * cat_mult
            effective.append((
                sig.result.score,
                eff_weight,
                sig.result.fired,
                sig.generator_name,
            ))

        # Compute fire count
        fire_count = sum(1 for _, _, fired, _ in effective if fired)

        # Compute composite score based on mode
        composite = self._aggregate(effective, cfg.mode)

        # Build per-signal breakdown
        per_signal: dict[str, tuple[float, bool, float]] = {
            name: (score, fired, weight)
            for score, weight, fired, name in effective
        }

        # Threshold + fire logic
        fire_ok = (
            not cfg.require_fire or fire_count >= cfg.min_fire_count
        )
        buy_signal = composite >= cfg.buy_threshold and fire_ok
        sell_signal = composite <= -cfg.sell_threshold and fire_ok

        return StackResult(
            composite_score=composite,
            buy_signal=buy_signal,
            sell_signal=sell_signal,
            fire_count=fire_count,
            total_signals=len(aligned_signals),
            per_signal_scores=per_signal,
        )

    # ------------------------------------------------------------------
    # Aggregation modes
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        signals: list[tuple[float, float, bool, str]],
        mode: AggregationMode,
    ) -> float:
        """Dispatch to the correct aggregation method."""
        if mode == AggregationMode.WEIGHTED_AVERAGE:
            return _weighted_average(signals)
        if mode == AggregationMode.WEIGHTED_SUM:
            return _weighted_sum(signals)
        if mode == AggregationMode.UNANIMOUS:
            return _unanimous(signals)
        if mode == AggregationMode.MAJORITY_VOTE:
            return _majority_vote(signals)
        if mode == AggregationMode.MAX_CONFLUENCE:
            return _max_confluence(signals)
        return 0.0  # pragma: no cover


# ======================================================================
# Free-standing aggregation functions (stateless, testable in isolation)
# ======================================================================

def _weighted_average(
    signals: list[tuple[float, float, bool, str]],
) -> float:
    """Σ(score×weight) / Σ(weight)."""
    total_weight = sum(w for _, w, _, _ in signals)
    if total_weight == 0:
        return 0.0
    return sum(s * w for s, w, _, _ in signals) / total_weight


def _weighted_sum(
    signals: list[tuple[float, float, bool, str]],
) -> float:
    """Σ(score×weight) — unbounded, rewards more signals."""
    return sum(s * w for s, w, _, _ in signals)


def _unanimous(
    signals: list[tuple[float, float, bool, str]],
) -> float:
    """All must agree in direction; returns average score if unanimous, else 0.

    Direction is determined by sign of score.  Neutral (|score| < 0.01) scores
    are ignored.  If no non-neutral scores exist, returns 0.
    """
    non_neutral = [(s, w) for s, w, _, _ in signals if abs(s) >= 0.01]
    if not non_neutral:
        return 0.0

    positive = all(s > 0 for s, _ in non_neutral)
    negative = all(s < 0 for s, _ in non_neutral)

    if not positive and not negative:
        return 0.0  # disagreement

    total_w = sum(w for _, w in non_neutral)
    if total_w == 0:
        return 0.0
    return sum(s * w for s, w in non_neutral) / total_w


def _majority_vote(
    signals: list[tuple[float, float, bool, str]],
) -> float:
    """>50% must agree in direction.  Returns weighted average of majority side.

    Each signal gets one vote based on sign.  Neutral scores abstain.
    If majority agrees, return their weighted average; else return 0.
    """
    bullish_votes = 0
    bearish_votes = 0
    bull_sum = 0.0
    bull_w = 0.0
    bear_sum = 0.0
    bear_w = 0.0

    for s, w, _, _ in signals:
        if s > 0.01:
            bullish_votes += 1
            bull_sum += s * w
            bull_w += w
        elif s < -0.01:
            bearish_votes += 1
            bear_sum += s * w
            bear_w += w

    total_votes = bullish_votes + bearish_votes
    if total_votes == 0:
        return 0.0

    if bullish_votes > total_votes / 2:
        return bull_sum / bull_w if bull_w > 0 else 0.0
    if bearish_votes > total_votes / 2:
        return bear_sum / bear_w if bear_w > 0 else 0.0

    return 0.0  # tied or no majority


def _max_confluence(
    signals: list[tuple[float, float, bool, str]],
) -> float:
    """Product of concordant scores (nonlinear amplification).

    Separate bullish and bearish scores.  For the dominant side,
    return ``sign × product(|scores|)^(1/n)`` (geometric mean to
    keep the result in [-1, 1]).  Returns 0 on disagreement.
    """
    bull_scores: list[float] = []
    bear_scores: list[float] = []

    for s, w, _, _ in signals:
        if s > 0.01:
            bull_scores.append(abs(s) * w)
        elif s < -0.01:
            bear_scores.append(abs(s) * w)

    if not bull_scores and not bear_scores:
        return 0.0

    # Use the dominant side
    if len(bull_scores) >= len(bear_scores):
        if not bull_scores:
            return 0.0
        # Geometric mean of weighted scores
        product = math.prod(bull_scores)
        geo_mean = product ** (1.0 / len(bull_scores))
        return min(1.0, geo_mean)
    else:
        if not bear_scores:
            return 0.0
        product = math.prod(bear_scores)
        geo_mean = product ** (1.0 / len(bear_scores))
        return max(-1.0, -geo_mean)
