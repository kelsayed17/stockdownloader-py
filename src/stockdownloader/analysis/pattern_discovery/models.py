"""Data structures for pattern discovery.

Defines the core dataclasses used throughout the pattern mining pipeline:
:class:`PatternOutcome`, :class:`PatternStats`, :class:`DiscoveredPattern`,
and :class:`FilterConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from stockdownloader.analysis.pattern_encoder import BarFeatures, PatternContext

# Type alias -- a pattern key is a tuple of BarFeatures (hashable, immutable)
PatternKey = tuple[BarFeatures, ...]


# =========================================================================
# Data structures
# =========================================================================


@dataclass(slots=True)
class PatternOutcome:
    """What happened after one occurrence of a pattern."""

    bar_index: int
    """Index where the pattern ended in the data."""

    returns: dict[int, float]
    """Horizon (bars) -> percentage return."""

    mae: float
    """Max adverse excursion (worst drawdown as pct, negative)."""

    mfe: float
    """Max favorable excursion (best move as pct, positive)."""

    context: PatternContext
    """Market context at time of pattern."""


@dataclass(slots=True)
class PatternStats:
    """Aggregated statistics for one pattern across all occurrences."""

    key: PatternKey
    occurrences: int
    outcomes: list[PatternOutcome]

    # Per-horizon stats
    avg_return: dict[int, float] = field(default_factory=dict)
    median_return: dict[int, float] = field(default_factory=dict)
    win_rate: dict[int, float] = field(default_factory=dict)
    std_return: dict[int, float] = field(default_factory=dict)
    t_stat: dict[int, float] = field(default_factory=dict)
    p_value: dict[int, float] = field(default_factory=dict)

    # Walk-forward stability
    first_half_wr: dict[int, float] = field(default_factory=dict)
    second_half_wr: dict[int, float] = field(default_factory=dict)

    # 3-fold expanding walk-forward
    fold_win_rates: dict[int, list[float]] = field(default_factory=dict)
    """Per-horizon list of test-fold win rates from expanding walk-forward."""

    # MAE / MFE averages
    avg_mae: float = 0.0
    avg_mfe: float = 0.0


@dataclass(frozen=True, slots=True)
class DiscoveredPattern:
    """A pattern that passed all statistical filters -- ready for trading."""

    key: PatternKey
    direction: str  # "long" | "short"
    best_horizon: int  # optimal holding period in bars
    avg_return: float
    win_rate: float
    occurrences: int
    t_stat: float
    p_value: float
    avg_mae: float  # for stop-loss sizing
    avg_mfe: float  # for take-profit sizing
    risk_reward: float  # avg_mfe / |avg_mae|
    walk_forward_stable: bool
    human_label: str  # e.g. "3-bar: bull-strong / doji / bull-strong"

    # -- Enhancement fields -------------------------------------------
    confirmation: dict[str, str] | None = None
    """Indicator conditions that improve this pattern's edge,
    e.g. ``{"rsi_zone": "neutral", "trend_dir": "1"}``.
    ``None`` means no confirmation was found or analysis was skipped."""

    timeframe: str = "5m"
    """Timeframe this pattern was mined on."""

    effect_size: float = 0.0
    """Cohen's d -- practical significance of the edge."""

    dominant_regime: str | None = None
    """The market regime in which this pattern appeared most often."""

    regime_breakdown: dict[str, int] | None = None
    """Occurrence count per regime, e.g. ``{"mean_reverting": 30, ...}``."""


_TF_OCCURRENCE_SCALE: dict[str, float] = {
    "5m": 1.0, "15m": 0.6, "30m": 0.5, "1h": 0.4, "4h": 0.25, "1d": 0.15,
}


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Thresholds for pattern significance filtering."""

    min_occurrences: int = 20
    max_p_value: float = 0.05
    min_win_rate: float = 0.52
    min_abs_return: float = 0.02  # minimum avg return (%)
    min_risk_reward: float = 0.8
    walk_forward_tolerance: float = 0.15  # max WR difference between folds
    max_patterns: int = 50
    min_effect_size: float = 0.2  # Cohen's d minimum
    min_walk_forward_folds: int = 2  # min folds (of 3) that must be profitable
    apply_fdr: bool = False  # skip BH FDR when multi-filter pipeline is active

    # -- Regime filtering ---------------------------------------------
    regime_filter: str | None = None
    """Only keep occurrences matching this regime (e.g. ``"mean_reverting"``)."""

    min_regime_purity: float = 0.0
    """Minimum fraction of occurrences in the dominant regime."""

    def for_timeframe(self, tf_label: str) -> FilterConfig:
        """Return a copy with ``min_occurrences`` scaled for *tf_label*.

        Higher timeframes have fewer bars, so the minimum occurrence
        threshold is reduced proportionally.
        """
        scale = _TF_OCCURRENCE_SCALE.get(tf_label, 1.0)
        scaled_min = max(15, int(self.min_occurrences * scale))
        return FilterConfig(
            min_occurrences=scaled_min,
            max_p_value=self.max_p_value,
            min_win_rate=self.min_win_rate,
            min_abs_return=self.min_abs_return,
            min_risk_reward=self.min_risk_reward,
            walk_forward_tolerance=self.walk_forward_tolerance,
            max_patterns=self.max_patterns,
            min_effect_size=self.min_effect_size,
            min_walk_forward_folds=self.min_walk_forward_folds,
            apply_fdr=self.apply_fdr,
            regime_filter=self.regime_filter,
            min_regime_purity=self.min_regime_purity,
        )
