"""Intraday strategies and shared infrastructure.

Standalone strategies
---------------------
- :class:`AVWAPPullbackStrategy` — anchored VWAP pullback
- :class:`DmiVwapStrategy` — DMI + session-anchored VWAP
- :class:`ORBreakoutStrategy` — OR level breakout continuation
- :class:`ORReversalStrategy` — OR level retest fade
- :class:`PatternScalpStrategy` — opening-range manipulation fade
- :class:`PullbackStrategy` — trending VWAP pullback
- :class:`ReversalStrategy` — sideways band mean-reversion
- :class:`MLOversoldStrategy` — ML-driven mean-reversion
- :class:`SMCStructureStrategy` — Smart Money Concepts structure

Adapters
--------
- :class:`DailyToIntradayAdapter` — wraps daily strategies for the intraday engine

Infrastructure
--------------
- :class:`BaseIntradayStrategy` — template base with boilerplate delegation
- :class:`IntradayInfra` — shared per-session infrastructure (compose, don't inherit)
- :class:`BarContext` — computed values for the current bar
- :class:`DayTracker` — day-boundary transitions (daily bars, OR, trend)
- :class:`SessionState` — per-session mutable state
- :class:`IntradayExitManager` — exit evaluation with pluggable trail strategies (in ``trade_management``)
- :class:`TrailStrategy` — ABC for trailing stop strategies
"""

from stockdownloader.strategy.intraday.avwap_pullback_strategy import AVWAPPullbackStrategy
from stockdownloader.strategy.intraday.session_state import BarContext
from stockdownloader.strategy.intraday.base_strategy import BaseIntradayStrategy
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.intraday.infra import DayTracker
from stockdownloader.strategy.intraday.dmi_vwap_strategy import DmiVwapStrategy
from stockdownloader.strategy.intraday.trade_management import (
    IntradayExitManager,
    clamp_sl_dist,
    detect_reversal_patterns,
    directional_sl_tp,
    make_entry_signal,
    reversal_pattern_label,
)
from stockdownloader.strategy.intraday.infra import IntradayInfra
from stockdownloader.strategy.intraday.ml_oversold_strategy import MLOversoldConfig
from stockdownloader.strategy.intraday.ml_oversold_strategy import MLOversoldStrategy
from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
from stockdownloader.strategy.intraday.or_reversal_strategy import ORReversalStrategy
from stockdownloader.strategy.intraday.pattern_scalp_strategy import PatternScalpStrategy
from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategy
from stockdownloader.strategy.intraday.reversal_strategy import ReversalStrategy
from stockdownloader.strategy.intraday.session_state import SessionState
from stockdownloader.strategy.intraday.smc_structure_strategy import SMCStructureStrategy
from stockdownloader.strategy.intraday.trail_strategy import (
    AtrChandelierTrail,
    BreakevenTrail,
    TrailStrategy,
    VwapRatchetTrail,
)

__all__ = [
    # Standalone strategies
    "AVWAPPullbackStrategy",
    "DmiVwapStrategy",
    "MLOversoldConfig",
    "MLOversoldStrategy",
    "ORBreakoutStrategy",
    "ORReversalStrategy",
    "PatternScalpStrategy",
    "PullbackStrategy",
    "ReversalStrategy",
    "SMCStructureStrategy",
    # Adapters
    "DailyToIntradayAdapter",
    # Infrastructure
    "AtrChandelierTrail",
    "BarContext",
    "BaseIntradayStrategy",
    "BreakevenTrail",
    "DayTracker",
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
