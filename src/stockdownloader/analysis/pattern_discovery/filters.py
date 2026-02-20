"""Statistical filtering, FDR correction, and significance tests.

Provides :func:`filter_patterns`, :func:`apply_fdr_correction`,
:func:`deduplicate_patterns`, and helper utilities for evaluating
the statistical significance of discovered patterns.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from scipy.stats import t as t_dist

from stockdownloader.analysis.pattern_discovery.models import (
    DiscoveredPattern,
    FilterConfig,
    PatternKey,
    PatternOutcome,
    PatternStats,
)

# Context fields eligible for confirmation analysis
_CONFIRMATION_FIELDS = (
    "rsi_zone", "vwap_position", "obv_trend",
    "adx_level", "macd_signal", "trend_dir",
    "htf_trend", "cvd_direction",
)


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
    return f"{len(key)}-bar: {pattern_str} \u2192 {direction.upper()} {horizon}b"


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
    8. Indicator confirmation analysis (optional -- enriches patterns)

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
        # -- Regime breakdown (computed for all patterns) ---------------
        regime_breakdown, dominant_regime = _compute_regime_breakdown(
            stats.outcomes,
        )

        # -- Regime-aware stats ----------------------------------------
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
                # This is a superset -- only keep if strictly better
                if abs(p.t_stat) <= abs(existing.t_stat):
                    is_superset = True
                    break

        if not is_superset:
            kept.append(p)
            kept_keys.add(p.key)

    return kept
