"""GMEDailyScorecard -- Frozen dataclass for daily signal fusion output."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class GMEDailyScorecard:
    """Immutable daily scorecard produced by :class:`GMESignalFusion`.

    Attributes
    ----------
    date:
        Trading date for this scorecard.
    pillar_scores:
        Per-pillar z-score-like values keyed by pillar name
        (``options_flow``, ``volume_premium``, ``cycle_timing``,
        ``momentum``).
    composite:
        Weighted composite score across all pillars.
    regime:
        Detected market regime: ``"squeeze"``, ``"gamma_ramp"``,
        ``"cycle_hot"``, or ``"neutral"``.
    anomaly_flags:
        Human-readable descriptions of any pillar exceeding the
        anomaly threshold.
    """

    date: date
    pillar_scores: dict[str, float]
    composite: float
    regime: str
    anomaly_flags: list[str] = field(default_factory=list)
