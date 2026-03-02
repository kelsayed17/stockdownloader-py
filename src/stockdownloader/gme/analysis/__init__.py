"""GME deep analysis -- modular analysis package.

Re-exports all public dataclasses and analysis functions so callers
can use ``from stockdownloader.gme.analysis import run_event_study``.
"""

from stockdownloader.gme.analysis.distribution import (
    analyze_return_distribution,
    compute_price_statistics,
    compute_volume_profile,
)
from stockdownloader.gme.analysis.event_study import (
    correlate_filings_with_price,
    run_event_study,
)
from stockdownloader.gme.analysis.models import (
    EventStudyResult,
    FilingImpact,
    KeyPeriod,
    OptionsAnalysis,
    PriceStatistics,
    ReturnDistribution,
    StructuralBreak,
    VolatilityRegime,
    VolumeProfile,
)
from stockdownloader.gme.analysis.options import analyze_options_chain
from stockdownloader.gme.analysis.regime import (
    detect_key_periods,
    detect_structural_breaks,
    detect_volatility_regimes,
)

__all__ = [
    # Models
    "EventStudyResult",
    "FilingImpact",
    "KeyPeriod",
    "OptionsAnalysis",
    "PriceStatistics",
    "ReturnDistribution",
    "StructuralBreak",
    "VolatilityRegime",
    "VolumeProfile",
    # Event study
    "run_event_study",
    # Regime / structural
    "detect_volatility_regimes",
    "detect_structural_breaks",
    "detect_key_periods",
    # Distribution / volume / stats
    "analyze_return_distribution",
    "compute_volume_profile",
    "compute_price_statistics",
    # Options
    "analyze_options_chain",
    # Filings
    "correlate_filings_with_price",
]
