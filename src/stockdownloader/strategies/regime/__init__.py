"""Market regime detection and regime-aware strategy selection.

- :class:`MarketRegimeDetector` -- classify bars into market regimes
- :class:`RegimeStrategyMapper` -- track strategy performance per regime
- :class:`EnsembleIntradayStrategy` -- adaptive meta-strategy using regime mapping
"""

from stockdownloader.strategies.regime.ensemble import (
    DrawdownPositionScaler,
    EnsembleIntradayStrategy,
)
from stockdownloader.strategies.regime.detector import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeClassification,
    RegimeDetectorConfig,
)
from stockdownloader.strategies.regime.strategy_map import (
    RegimePerformance,
    RegimeStrategyMapper,
)

__all__ = [
    "DrawdownPositionScaler",
    "EnsembleIntradayStrategy",
    "MarketRegime",
    "MarketRegimeDetector",
    "RegimeClassification",
    "RegimeDetectorConfig",
    "RegimePerformance",
    "RegimeStrategyMapper",
]
