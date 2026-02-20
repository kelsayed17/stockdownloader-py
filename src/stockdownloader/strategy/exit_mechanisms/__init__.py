"""Concrete exit mechanism implementations for the exit tournament.

All six exit mechanisms extend :class:`TrailingExitBase` and implement
``evaluate_bar`` to decide when to close a position.
"""

from stockdownloader.strategy.exit_mechanisms.atr_trail_exit import AtrTrailExit
from stockdownloader.strategy.exit_mechanisms.hybrid_exit import HybridExit
from stockdownloader.strategy.exit_mechanisms.time_decay_exit import TimeDecayExit
from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import TrailingExitBase
from stockdownloader.strategy.exit_mechanisms.trailing_stop_exit import TrailingStopExit
from stockdownloader.strategy.exit_mechanisms.vwap_band_exit import VwapBandExit
from stockdownloader.strategy.exit_mechanisms.vwap_cross_exit import VwapCrossExit

__all__ = [
    "AtrTrailExit",
    "HybridExit",
    "TimeDecayExit",
    "TrailingExitBase",
    "TrailingStopExit",
    "VwapBandExit",
    "VwapCrossExit",
]
