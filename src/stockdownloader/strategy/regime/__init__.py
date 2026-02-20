"""Market regime detection and regime-aware strategy selection.

- :class:`MarketRegimeDetector` — classify bars into market regimes
- :class:`RegimeStrategyMapper` — track strategy performance per regime
- :class:`EnsembleIntradayStrategy` — adaptive meta-strategy using regime mapping
"""

from stockdownloader.strategy.regime.ensemble_strategy import (
    DrawdownPositionScaler,
    EnsembleIntradayStrategy,
)
from stockdownloader.strategy.regime.regime_detector import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeClassification,
    RegimeDetectorConfig,
)
from stockdownloader.strategy.regime.regime_strategy_map import (
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
