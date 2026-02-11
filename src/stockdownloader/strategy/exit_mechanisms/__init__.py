"""Concrete exit mechanism implementations for the exit tournament."""

from stockdownloader.strategy.exit_mechanisms.trailing_stop_exit import TrailingStopExit
from stockdownloader.strategy.exit_mechanisms.vwap_cross_exit import VwapCrossExit
from stockdownloader.strategy.exit_mechanisms.atr_trail_exit import AtrTrailExit
from stockdownloader.strategy.exit_mechanisms.hybrid_exit import HybridExit
from stockdownloader.strategy.exit_mechanisms.vwap_band_exit import VwapBandExit
from stockdownloader.strategy.exit_mechanisms.time_decay_exit import TimeDecayExit

__all__ = [
    "TrailingStopExit",
    "VwapCrossExit",
    "AtrTrailExit",
    "HybridExit",
    "VwapBandExit",
    "TimeDecayExit",
]
