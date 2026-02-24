"""Composable signal stacking and multi-timeframe confluence detection.

This package provides atomic signal generators that can be freely combined
into weighted stacks, aligned across multiple timeframes, and backtested
through the existing engine infrastructure.

Submodules
----------
generator
    ABC and ``SignalResult`` dataclass for atomic signal generators.
signal_registry
    Registry for discovering and creating signal generators.
timeframe_aligner
    Cross-timeframe signal alignment using :class:`TimeframeAggregator`.
engine
    Weighted combination of aligned signals with configurable thresholds.
daily_adapter / intraday_adapter
    Strategy wrappers bridging signal stacks to backtest engines.
generators
    17 atomic signal generator implementations.
"""

from stockdownloader.signals.generator import (
    AtomicSignalGenerator,
    SignalDirection,
    SignalResult,
)
from stockdownloader.strategies.registry import SignalGeneratorRegistry
from stockdownloader.signals.timeframe_aligner import (
    AlignedSignal,
    MultiTimeframeAligner,
    TimeframeSignalSpec,
)
from stockdownloader.signals.engine import (
    AggregationMode,
    StackConfig,
    StackedSignalEngine,
    StackResult,
)
from stockdownloader.signals.daily_adapter import StackedDailyStrategy
from stockdownloader.signals.intraday_adapter import StackedIntradayStrategy

__all__ = [
    # Core abstractions
    "AtomicSignalGenerator",
    "SignalDirection",
    "SignalResult",
    # Registry
    "SignalGeneratorRegistry",
    # Multi-timeframe
    "AlignedSignal",
    "MultiTimeframeAligner",
    "TimeframeSignalSpec",
    # Stacked engine
    "AggregationMode",
    "StackConfig",
    "StackedSignalEngine",
    "StackResult",
    # Strategy wrappers
    "StackedDailyStrategy",
    "StackedIntradayStrategy",
]
