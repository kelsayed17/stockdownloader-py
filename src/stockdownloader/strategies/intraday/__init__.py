"""Intraday strategies and shared infrastructure.

Standalone strategies
---------------------
- :class:`AVWAPPullbackStrategy` -- anchored VWAP pullback
- :class:`DmiVwapStrategy` -- DMI + session-anchored VWAP
- :class:`ORBreakoutStrategy` -- OR level breakout continuation
- :class:`ORReversalStrategy` -- OR level retest fade
- :class:`PatternScalpStrategy` -- opening-range manipulation fade
- :class:`PullbackStrategy` -- trending VWAP pullback
- :class:`ReversalStrategy` -- sideways band mean-reversion
- :class:`MLOversoldStrategy` -- ML-driven mean-reversion
- :class:`SMCStructureStrategy` -- Smart Money Concepts structure

Adapters
--------
- :class:`DailyToIntradayAdapter` -- wraps daily strategies for the intraday engine

Infrastructure
--------------
- :class:`BaseIntradayStrategy` -- template base with boilerplate delegation
- :class:`IntradayInfra` -- shared per-session infrastructure (compose, don't inherit)
- :class:`BarContext` -- computed values for the current bar
- :class:`DayTracker` -- day-boundary transitions (daily bars, OR, trend)
- :class:`SessionState` -- per-session mutable state
- :class:`IntradayExitManager` -- exit evaluation with pluggable trail strategies (in ``trade_mgmt``)
- :class:`TrailStrategy` -- ABC for trailing stop strategies
"""

from stockdownloader.strategies.intraday.avwap_pullback import AVWAPPullbackStrategy
from stockdownloader.strategies.intraday.session import BarContext
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy
from stockdownloader.strategies.intraday.daily_adapter import DailyToIntradayAdapter
from stockdownloader.strategies.intraday.day_tracker import DayTracker
from stockdownloader.strategies.intraday.dmi_vwap import DmiVwapStrategy
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    clamp_sl_dist,
    detect_reversal_patterns,
    directional_sl_tp,
    make_entry_signal,
    reversal_pattern_label,
)
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.ml_oversold import MLOversoldConfig
from stockdownloader.strategies.intraday.ml_oversold import MLOversoldStrategy
from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy
from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategy
from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy
from stockdownloader.strategies.intraday.pullback import PullbackStrategy
from stockdownloader.strategies.intraday.reversal import ReversalStrategy
from stockdownloader.strategies.intraday.session import SessionState
from stockdownloader.strategies.intraday.smc_structure import SMCStructureStrategy
from stockdownloader.strategies.intraday.trail import (
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
