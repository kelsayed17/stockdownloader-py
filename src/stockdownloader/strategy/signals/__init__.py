"""Composable signal stacking and multi-timeframe confluence detection.

This package provides atomic signal generators that can be freely combined
into weighted stacks, aligned across multiple timeframes, and backtested
through the existing engine infrastructure.

Submodules
----------
signal_generator
    ABC and ``SignalResult`` dataclass for atomic signal generators.
signal_registry
    Registry for discovering and creating signal generators.
multi_timeframe_aligner
    Cross-timeframe signal alignment using :class:`TimeframeAggregator`.
stacked_signal_engine
    Weighted combination of aligned signals with configurable thresholds.
stacked_signal_strategy
    Strategy wrappers bridging signal stacks to backtest engines.
generators
    17 atomic signal generator implementations.
"""
