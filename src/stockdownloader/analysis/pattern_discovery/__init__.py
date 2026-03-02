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

# Re-export everything from sub-modules for full backward compatibility.
# Existing imports like ``from stockdownloader.analysis.pattern_discovery import X``
# continue to work unchanged.

from stockdownloader.analysis.pattern_discovery.catalog import PatternCatalog
from stockdownloader.analysis.pattern_discovery.filters import (
    _CONFIRMATION_FIELDS,
    _compute_regime_breakdown,
    _find_best_confirmation,
    _generate_label,
    apply_fdr_correction,
    compute_pattern_stats,
    deduplicate_patterns,
    filter_patterns,
)
from stockdownloader.analysis.pattern_discovery.miner import PatternMiner
from stockdownloader.analysis.pattern_discovery.models import (
    DiscoveredPattern,
    FilterConfig,
    PatternKey,
    PatternOutcome,
    PatternStats,
    _TF_OCCURRENCE_SCALE,
)

__all__ = [
    # models
    "PatternKey",
    "PatternOutcome",
    "PatternStats",
    "DiscoveredPattern",
    "FilterConfig",
    # miner
    "PatternMiner",
    # filters
    "filter_patterns",
    "apply_fdr_correction",
    "deduplicate_patterns",
    "compute_pattern_stats",
    # catalog
    "PatternCatalog",
    # semi-private (used by tests)
    "_generate_label",
    "_find_best_confirmation",
    "_compute_regime_breakdown",
    "_CONFIRMATION_FIELDS",
    "_TF_OCCURRENCE_SCALE",
]
