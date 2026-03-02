"""Exit mechanism interface and concrete implementations for the exit tournament.

All six exit mechanisms extend :class:`TrailingExitBase` (which extends
:class:`ExitMechanism`) and implement ``evaluate_bar`` to decide when to
close a position.
"""

from stockdownloader.strategies.exits.composite import HybridExit, TimeDecayExit
from stockdownloader.strategies.exits.trailing import AtrTrailExit, TrailingStopExit
from stockdownloader.strategies.exits.base import ExitMechanism, TrailingExitBase
from stockdownloader.strategies.exits.vwap import VwapBandExit, VwapCrossExit

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
