"""Intraday strategies and shared infrastructure.

Standalone strategies
---------------------
- :class:`PatternScalpStrategy` — opening-range manipulation fade
- :class:`ORBreakoutStrategy` — OR level breakout continuation
- :class:`ORReversalStrategy` — OR level retest fade
- :class:`PullbackStrategy` — trending VWAP pullback
- :class:`ReversalStrategy` — sideways band mean-reversion

Infrastructure
--------------
- :class:`IntradayInfra` — shared per-session infrastructure (compose, don't inherit)
- :class:`BarContext` — computed values for the current bar
- :class:`SessionState` — per-session mutable state
- :class:`IntradayExitManager` — exit evaluation with pluggable trail strategies
- :class:`TrailStrategy` — ABC for trailing stop strategies
"""

from stockdownloader.strategy.intraday.entry_helpers import (
    clamp_sl_dist,
    detect_reversal_patterns,
    directional_sl_tp,
    make_entry_signal,
    reversal_pattern_label,
)
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.infra import BarContext, IntradayInfra
from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
from stockdownloader.strategy.intraday.or_reversal_strategy import ORReversalStrategy
from stockdownloader.strategy.intraday.pattern_scalp_strategy import PatternScalpStrategy
from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategy
from stockdownloader.strategy.intraday.reversal_strategy import ReversalStrategy
from stockdownloader.strategy.intraday.session_state import SessionState
from stockdownloader.strategy.intraday.trail_strategy import (
    AtrChandelierTrail,
    BreakevenTrail,
    TrailStrategy,
    VwapRatchetTrail,
)

__all__ = [
    # Standalone strategies
    "ORBreakoutStrategy",
    "ORReversalStrategy",
    "PatternScalpStrategy",
    "PullbackStrategy",
    "ReversalStrategy",
    # Infrastructure
    "AtrChandelierTrail",
    "BarContext",
    "BreakevenTrail",
    "IntradayExitManager",
    "IntradayInfra",
    "SessionState",
    "TrailStrategy",
    "VwapRatchetTrail",
    # Entry helpers
    "clamp_sl_dist",
    "detect_reversal_patterns",
    "directional_sl_tp",
    "make_entry_signal",
    "reversal_pattern_label",
]
