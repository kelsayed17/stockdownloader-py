"""N-gram price action pattern mining with statistical validation.

Scans historical price data for recurring multi-bar patterns (2-5 bars),
measures follow-through returns at multiple horizons, applies statistical
filters (t-test, Benjamini-Hochberg FDR, walk-forward stability), and
produces a :class:`PatternCatalog` of validated patterns.

Usage::

    from stockdownloader.analysis.pattern_discovery import (
        PatternMiner, filter_patterns, apply_fdr_correction, PatternCatalog,
    )

    miner = PatternMiner(encoder)
    raw = miner.mine(data, start=warmup, end=len(data))
    discovered = filter_patterns(raw, FilterConfig())
    discovered = apply_fdr_correction(discovered)
    catalog = PatternCatalog(patterns=tuple(discovered), ...)
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scipy.stats import t as t_dist

from stockdownloader.analysis.pattern_encoder import BarFeatures, PatternContext

if TYPE_CHECKING:
    from stockdownloader.analysis.pattern_encoder import BarEncoder
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.regime.regime_detector import MarketRegime

# Type alias — a pattern key is a tuple of BarFeatures (hashable, immutable)
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
    """A pattern that passed all statistical filters — ready for trading."""

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

    # ── Enhancement fields ───────────────────────────────────────────
    confirmation: dict[str, str] | None = None
    """Indicator conditions that improve this pattern's edge,
    e.g. ``{"rsi_zone": "neutral", "trend_dir": "1"}``.
    ``None`` means no confirmation was found or analysis was skipped."""

    timeframe: str = "5m"
    """Timeframe this pattern was mined on."""

    effect_size: float = 0.0
    """Cohen's d — practical significance of the edge."""

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

    # ── Regime filtering ─────────────────────────────────────────────
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


@dataclass(frozen=True, slots=True)
class PatternCatalog:
    """Immutable catalog of discovered patterns, ready for live evaluation.

    Serializable to/from JSON for persistence between discovery runs.
    """

    patterns: tuple[DiscoveredPattern, ...]
    discovery_date: str = ""
    data_range: str = ""
    total_bars_scanned: int = 0
    total_patterns_tested: int = 0

    def lookup(
        self,
        recent_features: tuple[BarFeatures, ...],
    ) -> DiscoveredPattern | None:
        """Find the best matching pattern for the given recent bar features.

        Checks all pattern lengths (longest first for specificity).
        Returns the pattern with highest |t-stat| if multiple match.
        """
        best: DiscoveredPattern | None = None

        for pattern in self.patterns:
            n = len(pattern.key)
            if len(recent_features) < n:
                continue
            # Check if the last N features match this pattern's key
            if recent_features[-n:] == pattern.key:
                if best is None or abs(pattern.t_stat) > abs(best.t_stat):
                    best = pattern

        return best

    def to_dict(self) -> dict[str, Any]:
        """Serialize catalog to a JSON-compatible dict."""
        patterns_list = []
        for p in self.patterns:
            key_list = [
                {
                    "body_type": bf.body_type,
                    "body_strength": bf.body_strength,
                    "wick_signal": bf.wick_signal,
                    "relative_size": bf.relative_size,
                    "volume_profile": bf.volume_profile,
                }
                for bf in p.key
            ]
            patterns_list.append({
                "key": key_list,
                "direction": p.direction,
                "best_horizon": p.best_horizon,
                "avg_return": p.avg_return,
                "win_rate": p.win_rate,
                "occurrences": p.occurrences,
                "t_stat": p.t_stat,
                "p_value": p.p_value,
                "avg_mae": p.avg_mae,
                "avg_mfe": p.avg_mfe,
                "risk_reward": p.risk_reward,
                "walk_forward_stable": p.walk_forward_stable,
                "human_label": p.human_label,
                "confirmation": p.confirmation,
                "timeframe": p.timeframe,
                "effect_size": p.effect_size,
                "dominant_regime": p.dominant_regime,
                "regime_breakdown": p.regime_breakdown,
            })

        return {
            "patterns": patterns_list,
            "discovery_date": self.discovery_date,
            "data_range": self.data_range,
            "total_bars_scanned": self.total_bars_scanned,
            "total_patterns_tested": self.total_patterns_tested,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatternCatalog:
        """Deserialize catalog from a JSON-compatible dict."""
        patterns = []
        for pd in d.get("patterns", []):
            key = tuple(
                BarFeatures(**bf) for bf in pd["key"]
            )
            patterns.append(DiscoveredPattern(
                key=key,
                direction=pd["direction"],
                best_horizon=pd["best_horizon"],
                avg_return=pd["avg_return"],
                win_rate=pd["win_rate"],
                occurrences=pd["occurrences"],
                t_stat=pd["t_stat"],
                p_value=pd["p_value"],
                avg_mae=pd["avg_mae"],
                avg_mfe=pd["avg_mfe"],
                risk_reward=pd["risk_reward"],
                walk_forward_stable=pd["walk_forward_stable"],
                human_label=pd["human_label"],
                confirmation=pd.get("confirmation"),
                timeframe=pd.get("timeframe", "5m"),
                effect_size=pd.get("effect_size", 0.0),
                dominant_regime=pd.get("dominant_regime"),
                regime_breakdown=pd.get("regime_breakdown"),
            ))

        return cls(
            patterns=tuple(patterns),
            discovery_date=d.get("discovery_date", ""),
            data_range=d.get("data_range", ""),
            total_bars_scanned=d.get("total_bars_scanned", 0),
            total_patterns_tested=d.get("total_patterns_tested", 0),
        )

    def save(self, path: Path) -> None:
        """Save catalog to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PatternCatalog:
        """Load catalog from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =========================================================================
# Pattern mining
# =========================================================================


class PatternMiner:
    """Scans price data for recurring N-bar patterns and measures follow-through.

    Parameters
    ----------
    encoder:
        Bar encoder for converting bars to categorical features.
    pattern_lengths:
        N-gram sizes to scan (default: 2, 3, 4, 5).
    horizons:
        Bars ahead to measure follow-through (default: 1, 3, 5, 10, 20).
    """

    DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
    DEFAULT_PATTERN_LENGTHS = (2, 3)

    def __init__(
        self,
        encoder: BarEncoder,
        *,
        pattern_lengths: tuple[int, ...] = DEFAULT_PATTERN_LENGTHS,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    ) -> None:
        self._encoder = encoder
        self._pattern_lengths = pattern_lengths
        self._horizons = horizons

    def mine(
        self,
        data: list[IntradayPriceData],
        start: int,
        end: int,
        *,
        regime_map: dict[str, MarketRegime] | None = None,
    ) -> dict[PatternKey, PatternStats]:
        """Mine patterns from ``data[start:end]``.

        Algorithm:
        1. Pre-encode all bars into BarFeatures (single forward pass).
        2. For each pattern length N: slide N-bar window, build key,
           measure follow-through at each horizon.
        3. Session boundary enforcement: stop measuring when
           ``trading_date`` changes.
        4. Group outcomes by key, compute statistics.

        Parameters
        ----------
        data:
            Full price data list.
        start:
            First bar index to mine from (after warmup).
        end:
            End index (exclusive).
        regime_map:
            Optional pre-computed ``{date_str: MarketRegime}`` mapping.

        Returns
        -------
        Dict mapping PatternKey to PatternStats.
        """
        end = min(end, len(data))
        if start >= end:
            return {}

        # Step 1: Pre-encode all bars
        features: list[BarFeatures | None] = [None] * len(data)
        bar_of_day_map: list[int] = [0] * len(data)

        current_date = ""
        day_bar = 0

        for i in range(start, end):
            bar = data[i]
            td = bar.date[:10] if hasattr(bar, "date") else ""
            if td != current_date:
                current_date = td
                day_bar = 0
            day_bar += 1
            bar_of_day_map[i] = day_bar
            features[i] = self._encoder.encode(data, i)

        # Step 2: Slide windows and collect outcomes
        raw: dict[PatternKey, list[PatternOutcome]] = defaultdict(list)
        max_horizon = max(self._horizons) if self._horizons else 20

        for n in self._pattern_lengths:
            for i in range(start, end - n + 1):
                # Build pattern key
                key_parts: list[BarFeatures] = []
                valid = True
                for j in range(n):
                    f = features[i + j]
                    if f is None:
                        valid = False
                        break
                    key_parts.append(f)

                if not valid:
                    continue

                pattern_key = tuple(key_parts)
                pattern_end = i + n - 1
                entry_date = data[pattern_end].date[:10]

                # Measure follow-through
                outcome = self._measure_outcome(
                    data, pattern_end, entry_date, max_horizon,
                    bar_of_day_map, regime_map,
                )
                if outcome is not None:
                    raw[pattern_key].append(outcome)

        # Step 3: Compute statistics
        result: dict[PatternKey, PatternStats] = {}
        for key, outcomes in raw.items():
            stats = self._compute_stats(key, outcomes)
            result[key] = stats

        return result

    def _measure_outcome(
        self,
        data: list[IntradayPriceData],
        pattern_end_index: int,
        entry_date: str,
        max_horizon: int,
        bar_of_day_map: list[int],
        regime_map: dict[str, MarketRegime] | None,
    ) -> PatternOutcome | None:
        """Measure what happened after the pattern ended."""
        entry_close = float(data[pattern_end_index].close)
        if entry_close == 0:
            return None

        returns: dict[int, float] = {}
        mae = 0.0
        mfe = 0.0

        # Scan forward up to max_horizon, stopping at session boundary
        actual_max = 0
        for j in range(1, max_horizon + 1):
            idx = pattern_end_index + j
            if idx >= len(data):
                break
            future_date = data[idx].date[:10]
            if future_date != entry_date:
                break  # session boundary — stop

            actual_max = j
            bar = data[idx]
            # Track MAE/MFE using high/low
            low_pct = (float(bar.low) - entry_close) / entry_close * 100.0
            high_pct = (float(bar.high) - entry_close) / entry_close * 100.0
            if low_pct < mae:
                mae = low_pct
            if high_pct > mfe:
                mfe = high_pct

        # Record returns at each horizon that we could measure
        for h in self._horizons:
            target_idx = pattern_end_index + h
            if target_idx >= len(data):
                break
            if data[target_idx].date[:10] != entry_date:
                break
            future_close = float(data[target_idx].close)
            ret = (future_close - entry_close) / entry_close * 100.0
            returns[h] = ret

        if not returns:
            return None  # couldn't measure any horizon

        # Build context
        bod = bar_of_day_map[pattern_end_index]
        regime = None
        if regime_map is not None:
            regime = regime_map.get(data[pattern_end_index].date)

        context = self._encoder.encode_context(
            data, pattern_end_index, bod, regime=regime,
        )

        return PatternOutcome(
            bar_index=pattern_end_index,
            returns=returns,
            mae=mae,
            mfe=mfe,
            context=context,
        )

    def _compute_stats(
        self,
        key: PatternKey,
        outcomes: list[PatternOutcome],
    ) -> PatternStats:
        """Compute aggregate statistics for a pattern."""
        stats = PatternStats(
            key=key,
            occurrences=len(outcomes),
            outcomes=outcomes,
        )

        for horizon in self._horizons:
            horizon_returns = [
                o.returns[horizon]
                for o in outcomes
                if horizon in o.returns
            ]
            n = len(horizon_returns)
            if n < 2:
                continue

            mean = statistics.mean(horizon_returns)
            stdev = statistics.stdev(horizon_returns)
            median = statistics.median(horizon_returns)

            stats.avg_return[horizon] = mean
            stats.median_return[horizon] = median
            stats.win_rate[horizon] = sum(1 for r in horizon_returns if r > 0) / n
            stats.std_return[horizon] = stdev

            # T-test: is mean significantly different from 0?
            if stdev > 0 and n >= 5:
                t = mean / (stdev / math.sqrt(n))
                stats.t_stat[horizon] = t
                stats.p_value[horizon] = float(
                    2 * t_dist.sf(abs(t), df=n - 1)
                )

            # Walk-forward stability: 3-fold expanding window
            #   Fold 1: train [0:33%], test [33%:66%]
            #   Fold 2: train [0:50%], test [50%:75%]
            #   Fold 3: train [0:67%], test [67%:100%]
            fold_splits = [
                (n // 3, 2 * n // 3),
                (n // 2, 3 * n // 4),
                (2 * n // 3, n),
            ]
            horizon_returns_ordered = [
                o.returns[horizon]
                for o in outcomes
                if horizon in o.returns
            ]
            fold_wrs: list[float] = []
            for split_start, split_end in fold_splits:
                test_rets = horizon_returns_ordered[split_start:split_end]
                if test_rets:
                    fold_wr = sum(1 for r in test_rets if r > 0) / len(test_rets)
                    fold_wrs.append(fold_wr)

            if fold_wrs:
                stats.fold_win_rates[horizon] = fold_wrs

            # Backward-compat: first/second half from first/last fold
            mid = len(outcomes) // 2
            first_half = outcomes[:mid]
            second_half = outcomes[mid:]

            fh_returns = [
                o.returns[horizon]
                for o in first_half
                if horizon in o.returns
            ]
            sh_returns = [
                o.returns[horizon]
                for o in second_half
                if horizon in o.returns
            ]
            if fh_returns:
                stats.first_half_wr[horizon] = (
                    sum(1 for r in fh_returns if r > 0) / len(fh_returns)
                )
            if sh_returns:
                stats.second_half_wr[horizon] = (
                    sum(1 for r in sh_returns if r > 0) / len(sh_returns)
                )

        # MAE / MFE averages
        mae_vals = [o.mae for o in outcomes]
        mfe_vals = [o.mfe for o in outcomes]
        if mae_vals:
            stats.avg_mae = statistics.mean(mae_vals)
        if mfe_vals:
            stats.avg_mfe = statistics.mean(mfe_vals)

        return stats


# =========================================================================
# Statistical filtering
# =========================================================================


def _generate_label(key: PatternKey, direction: str, horizon: int) -> str:
    """Generate a human-readable label for a pattern."""
    parts: list[str] = []
    for bf in key:
        label = bf.body_type
        if bf.body_type != "doji":
            label += f"-{bf.body_strength}"
        if bf.wick_signal != "no_wick":
            label += f"+{bf.wick_signal.replace('_', '')}"
        if bf.relative_size in ("large", "huge"):
            label += f"+{bf.relative_size}"
        if bf.volume_profile in ("high", "extreme"):
            label += f"+{bf.volume_profile}vol"
        parts.append(label)

    pattern_str = " / ".join(parts)
    return f"{len(key)}-bar: {pattern_str} → {direction.upper()} {horizon}b"


# Context fields eligible for confirmation analysis
_CONFIRMATION_FIELDS = (
    "rsi_zone", "vwap_position", "obv_trend",
    "adx_level", "macd_signal", "trend_dir",
    "htf_trend", "cvd_direction",
)


def _find_best_confirmation(
    outcomes: list[PatternOutcome],
    direction: str,
    horizon: int,
    base_win_rate: float,
    *,
    min_improvement: float = 0.05,
    min_subset_size: int = 8,
) -> dict[str, str] | None:
    """Find indicator conditions that improve a pattern's win rate.

    Scans each context field for values that produce a meaningfully
    higher win rate than the pattern's overall rate.

    Parameters
    ----------
    outcomes:
        All pattern occurrences with their :class:`PatternContext`.
    direction:
        ``"long"`` or ``"short"``.
    horizon:
        The pattern's best holding horizon.
    base_win_rate:
        The pattern's overall win rate (fraction 0-1).
    min_improvement:
        A candidate must improve WR by at least this much.
    min_subset_size:
        Minimum outcomes with a given field value to consider.

    Returns
    -------
    Dict of ``{field: value}`` for the best condition(s), or ``None``.
    """
    # Build per-field groups: field -> value -> list of returns
    field_groups: dict[str, dict[str, list[float]]] = {}

    for field_name in _CONFIRMATION_FIELDS:
        groups: dict[str, list[float]] = defaultdict(list)
        for o in outcomes:
            if horizon not in o.returns:
                continue
            val = getattr(o.context, field_name, None)
            if val is None:
                continue
            ret = o.returns[horizon]
            groups[str(val)].append(ret)
        if groups:
            field_groups[field_name] = dict(groups)

    if not field_groups:
        return None

    # Score single-field candidates
    best_single: tuple[float, str, str] | None = None  # (improvement, field, value)

    for field_name, groups in field_groups.items():
        for val, returns in groups.items():
            if len(returns) < min_subset_size:
                continue
            if direction == "long":
                subset_wr = sum(1 for r in returns if r > 0) / len(returns)
            else:
                subset_wr = sum(1 for r in returns if r < 0) / len(returns)
            improvement = subset_wr - base_win_rate
            if improvement >= min_improvement:
                if best_single is None or improvement > best_single[0]:
                    best_single = (improvement, field_name, val)

    # Try best pair: combine the best single-field with another field
    best_pair: tuple[float, dict[str, str]] | None = None

    if best_single is not None:
        f1, v1 = best_single[1], best_single[2]
        for field_name, groups in field_groups.items():
            if field_name == f1:
                continue
            for val, _ in groups.items():
                # Filter outcomes matching both conditions
                pair_returns: list[float] = []
                for o in outcomes:
                    if horizon not in o.returns:
                        continue
                    val1 = getattr(o.context, f1, None)
                    val2 = getattr(o.context, field_name, None)
                    if str(val1) == v1 and str(val2) == val:
                        pair_returns.append(o.returns[horizon])

                if len(pair_returns) < min_subset_size:
                    continue
                if direction == "long":
                    pair_wr = sum(1 for r in pair_returns if r > 0) / len(pair_returns)
                else:
                    pair_wr = sum(1 for r in pair_returns if r < 0) / len(pair_returns)
                pair_improvement = pair_wr - base_win_rate
                if pair_improvement > best_single[0] + 0.02:  # pair must beat single by 2%
                    if best_pair is None or pair_improvement > best_pair[0]:
                        best_pair = (pair_improvement, {f1: v1, field_name: val})

    # Return the best option
    if best_pair is not None:
        return best_pair[1]
    if best_single is not None:
        return {best_single[1]: best_single[2]}
    return None


def _compute_regime_breakdown(
    outcomes: list[PatternOutcome],
) -> tuple[dict[str, int], str | None]:
    """Count occurrences per regime and find the dominant one.

    Returns
    -------
    (regime_counts, dominant_regime)
    """
    regime_counts: dict[str, int] = defaultdict(int)
    for o in outcomes:
        regime_counts[o.context.regime] += 1

    dominant: str | None = None
    if regime_counts:
        dominant = max(regime_counts, key=lambda k: regime_counts[k])

    return dict(regime_counts), dominant


def compute_pattern_stats(
    outcomes: list[PatternOutcome],
    horizon: int,
    *,
    regime_filter: str | None = None,
) -> tuple[float, float, float, float, int]:
    """Compute return stats for a pattern at a given horizon.

    Optionally filters to only outcomes matching *regime_filter*.

    Returns
    -------
    (avg_return, std_return, win_rate, t_stat, count)
    """
    filtered = outcomes
    if regime_filter:
        filtered = [
            o for o in outcomes
            if o.context.regime == regime_filter
        ]

    horizon_returns = [
        o.returns[horizon]
        for o in filtered
        if horizon in o.returns
    ]
    n = len(horizon_returns)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0, n

    mean = statistics.mean(horizon_returns)
    stdev = statistics.stdev(horizon_returns)
    wr = sum(1 for r in horizon_returns if r > 0) / n

    t = 0.0
    if stdev > 0 and n >= 5:
        t = mean / (stdev / math.sqrt(n))

    return mean, stdev, wr, t, n


def filter_patterns(
    all_patterns: dict[PatternKey, PatternStats],
    config: FilterConfig,
    *,
    timeframe: str = "5m",
) -> list[DiscoveredPattern]:
    """Apply multi-stage filtering to find genuinely significant patterns.

    Stages:
    1. Minimum sample size (optionally regime-filtered)
    2. Statistical significance (p-value)
    3. Effect size (Cohen's d)
    4. Minimum return magnitude
    5. Win rate threshold
    6. Risk/reward from MAE/MFE
    7. Walk-forward stability (3-fold expanding, fall back to 50/50)
    8. Indicator confirmation analysis (optional — enriches patterns)

    Parameters
    ----------
    all_patterns:
        Raw mined patterns from :meth:`PatternMiner.mine`.
    config:
        Filter thresholds.
    timeframe:
        Timeframe label stored on discovered patterns.

    Returns
    -------
    List of :class:`DiscoveredPattern` sorted by edge strength.
    """
    candidates: list[DiscoveredPattern] = []

    for key, stats in all_patterns.items():
        # ── Regime breakdown (computed for all patterns) ─────────────
        regime_breakdown, dominant_regime = _compute_regime_breakdown(
            stats.outcomes,
        )

        # ── Regime-aware stats ──────────────────────────────────────
        # If regime_filter is set, re-compute stats on the filtered subset
        effective_outcomes = stats.outcomes
        if config.regime_filter:
            effective_outcomes = [
                o for o in stats.outcomes
                if o.context.regime == config.regime_filter
            ]

        # Stage 1: Sample size (on effective outcomes)
        effective_count = len(effective_outcomes)
        if effective_count < config.min_occurrences:
            continue

        # Regime purity check
        if config.min_regime_purity > 0 and dominant_regime:
            dom_count = regime_breakdown.get(dominant_regime, 0)
            purity = dom_count / stats.occurrences if stats.occurrences > 0 else 0
            if purity < config.min_regime_purity:
                continue

        # Stage 2: Find best horizon (highest |t-stat| with p < threshold)
        # Re-compute stats when regime filtering is active
        if config.regime_filter:
            best_horizon: int | None = None
            best_t = 0.0
            best_avg_ret = 0.0
            best_std = 0.0
            best_wr = 0.0
            best_p = 1.0
            for h in sorted(stats.p_value.keys()):
                mean, stdev, wr, t, n = compute_pattern_stats(
                    stats.outcomes, h, regime_filter=config.regime_filter,
                )
                if n < 5 or stdev <= 0:
                    continue
                p_val = float(2 * t_dist.sf(abs(t), df=n - 1))
                if p_val <= config.max_p_value and abs(t) > abs(best_t):
                    best_horizon = h
                    best_t = t
                    best_avg_ret = mean
                    best_std = stdev
                    best_wr = wr
                    best_p = p_val
        else:
            best_horizon = None
            best_t = 0.0
            for h in sorted(stats.p_value.keys()):
                if stats.p_value[h] <= config.max_p_value:
                    if abs(stats.t_stat.get(h, 0.0)) > abs(best_t):
                        best_horizon = h
                        best_t = stats.t_stat.get(h, 0.0)

        if best_horizon is None:
            continue

        # Get stats for best horizon
        if config.regime_filter:
            avg_ret = best_avg_ret
            std_ret = best_std
            wr = best_wr
            p_val_final = best_p
        else:
            avg_ret = stats.avg_return.get(best_horizon, 0.0)
            std_ret = stats.std_return.get(best_horizon, 0.0)
            wr = stats.win_rate.get(best_horizon, 0.0)
            p_val_final = stats.p_value[best_horizon]

        # Stage 3: Effect size (Cohen's d)
        if std_ret > 0:
            cohens_d = abs(avg_ret) / std_ret
        else:
            cohens_d = 0.0
        if cohens_d < config.min_effect_size:
            continue

        # Stage 4: Minimum return magnitude
        if abs(avg_ret) < config.min_abs_return:
            continue

        # Stage 5: Direction and win rate
        direction = "long" if avg_ret > 0 else "short"
        effective_wr = wr if direction == "long" else (1.0 - wr)
        if effective_wr < config.min_win_rate:
            continue

        # Stage 6: Risk/reward from MAE/MFE
        if stats.avg_mae == 0.0:
            rr = 0.0
        else:
            rr = abs(stats.avg_mfe / stats.avg_mae)
        if rr < config.min_risk_reward:
            continue

        # Stage 7: Walk-forward stability (3-fold expanding, then 50/50 fallback)
        fold_wrs = stats.fold_win_rates.get(best_horizon)
        if fold_wrs and len(fold_wrs) >= 2:
            # 3-fold: at least min_walk_forward_folds must be profitable
            if direction == "long":
                profitable_folds = sum(1 for fw in fold_wrs if fw > 0.50)
            else:
                profitable_folds = sum(1 for fw in fold_wrs if fw < 0.50)
            if profitable_folds < config.min_walk_forward_folds:
                continue
            wf_stable = (max(fold_wrs) - min(fold_wrs)) <= config.walk_forward_tolerance
        else:
            # Fallback: 50/50 split
            fh_wr = stats.first_half_wr.get(best_horizon, 0.0)
            sh_wr = stats.second_half_wr.get(best_horizon, 0.0)

            if direction == "long":
                both_halves_positive = fh_wr > 0.50 and sh_wr > 0.50
            else:
                both_halves_positive = fh_wr < 0.50 and sh_wr < 0.50

            if not both_halves_positive:
                continue

            wf_stable = abs(fh_wr - sh_wr) <= config.walk_forward_tolerance

        # Stage 8: Indicator confirmation analysis
        confirmation = _find_best_confirmation(
            effective_outcomes, direction, best_horizon, effective_wr,
        )

        candidates.append(DiscoveredPattern(
            key=key,
            direction=direction,
            best_horizon=best_horizon,
            avg_return=avg_ret,
            win_rate=effective_wr,
            occurrences=effective_count,
            t_stat=best_t,
            p_value=p_val_final,
            avg_mae=stats.avg_mae,
            avg_mfe=stats.avg_mfe,
            risk_reward=rr,
            walk_forward_stable=wf_stable,
            human_label=_generate_label(key, direction, best_horizon),
            confirmation=confirmation,
            timeframe=timeframe,
            effect_size=cohens_d,
            dominant_regime=dominant_regime,
            regime_breakdown=regime_breakdown if regime_breakdown else None,
        ))

    # Rank by edge strength: |t-stat| * sqrt(occurrences)
    candidates.sort(
        key=lambda p: abs(p.t_stat) * math.sqrt(p.occurrences),
        reverse=True,
    )
    return candidates[:config.max_patterns]


def apply_fdr_correction(
    patterns: list[DiscoveredPattern],
    alpha: float = 0.05,
    total_hypotheses: int | None = None,
) -> list[DiscoveredPattern]:
    """Apply Benjamini-Hochberg FDR correction.

    Controls the false discovery rate at *alpha*, meaning at most
    *alpha* fraction of the surviving patterns are expected to be
    spurious.

    Parameters
    ----------
    patterns:
        Pre-filtered patterns with p-values.
    alpha:
        FDR threshold (default 0.05 = 5%).
    total_hypotheses:
        Total number of patterns tested (for correct BH denominator).
        If ``None``, uses ``len(patterns)`` (conservative when
        pre-filtering has already been applied).

    Returns
    -------
    Patterns that survive the BH correction, in original order.
    """
    if not patterns:
        return []

    # m = total hypotheses tested, not just the survivors.
    # Using the full family size makes BH less punishing when strong
    # pre-filtering (effect size, walk-forward) has already been applied.
    m = total_hypotheses if total_hypotheses is not None else len(patterns)

    # Sort by p-value ascending
    indexed = sorted(enumerate(patterns), key=lambda x: x[1].p_value)
    surviving_indices: set[int] = set()

    for rank, (orig_idx, pattern) in enumerate(indexed, 1):
        bh_threshold = (rank / m) * alpha
        if pattern.p_value <= bh_threshold:
            surviving_indices.add(orig_idx)
        else:
            break  # BH is sequential

    # Return in original order
    return [p for i, p in enumerate(patterns) if i in surviving_indices]


def deduplicate_patterns(
    patterns: list[DiscoveredPattern],
) -> list[DiscoveredPattern]:
    """Remove longer patterns that are supersets of shorter ones.

    If a 3-bar pattern ``(A, B, C)`` has similar edge to a 2-bar
    sub-pattern ``(B, C)``, keep the shorter one (more occurrences
    = more reliable).

    A longer pattern is kept only if its |t-stat| is strictly higher
    than all sub-patterns.
    """
    if not patterns:
        return []

    # Sort by pattern length ascending, then by |t-stat| descending
    patterns_sorted = sorted(
        patterns,
        key=lambda p: (len(p.key), -abs(p.t_stat)),
    )

    kept: list[DiscoveredPattern] = []
    kept_keys: set[PatternKey] = set()

    for p in patterns_sorted:
        # Check if any shorter pattern's key is a suffix of this one
        is_superset = False
        for existing in kept:
            n = len(existing.key)
            if n < len(p.key) and p.key[-n:] == existing.key:
                # This is a superset — only keep if strictly better
                if abs(p.t_stat) <= abs(existing.t_stat):
                    is_superset = True
                    break

        if not is_superset:
            kept.append(p)
            kept_keys.add(p.key)

    return kept
