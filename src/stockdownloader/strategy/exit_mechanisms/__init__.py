"""Exit mechanism interface and concrete implementations for the exit tournament.

All six exit mechanisms extend :class:`TrailingExitBase` (which extends
:class:`ExitMechanism`) and implement ``evaluate_bar`` to decide when to
close a position.
"""

from stockdownloader.strategy.exit_mechanisms.composite_exits import HybridExit, TimeDecayExit
from stockdownloader.strategy.exit_mechanisms.trail_exits import AtrTrailExit, TrailingStopExit
from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import ExitMechanism, TrailingExitBase
from stockdownloader.strategy.exit_mechanisms.vwap_exits import VwapBandExit, VwapCrossExit

__all__ = [
    "AtrTrailExit",
    "ExitMechanism",
    "HybridExit",
    "TimeDecayExit",
    "TrailingExitBase",
    "TrailingStopExit",
    "VwapBandExit",
    "VwapCrossExit",
]
