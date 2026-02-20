"""Concrete exit mechanism implementations for the exit tournament.

All six exit mechanisms extend :class:`TrailingExitBase` and implement
``evaluate_bar`` to decide when to close a position.
"""

from stockdownloader.strategy.exit_mechanisms.composite_exits import HybridExit, TimeDecayExit
from stockdownloader.strategy.exit_mechanisms.trail_exits import AtrTrailExit, TrailingStopExit
from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import TrailingExitBase
from stockdownloader.strategy.exit_mechanisms.vwap_exits import VwapBandExit, VwapCrossExit

__all__ = [
    "AtrTrailExit",
    "HybridExit",
    "TimeDecayExit",
    "TrailingExitBase",
    "TrailingStopExit",
    "VwapBandExit",
    "VwapCrossExit",
]
