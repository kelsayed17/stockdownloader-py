"""Streaming (incremental) indicators organized by type.

Re-exports all public names for backward compatibility with
:mod:`stockdownloader.util.incremental_indicators`.
"""

from stockdownloader.util.streaming._base import SCALE, _quantize, _true_range
from stockdownloader.util.streaming.accumulators import StreamingEMA, StreamingOBV
from stockdownloader.util.streaming.oscillators import StreamingMACD, StreamingRSI
from stockdownloader.util.streaming.resampling import StreamingHTFResample
from stockdownloader.util.streaming.trend import ADXState, StreamingADX, StreamingSAR
from stockdownloader.util.streaming.volatility import StreamingATR
from stockdownloader.util.streaming.volume import (
    StreamingAnchoredVWAP,
    StreamingCVD,
    StreamingSessionVWAP,
)

__all__ = [
    # Base helpers
    "SCALE",
    "_quantize",
    "_true_range",
    # Accumulators
    "StreamingOBV",
    "StreamingEMA",
    # Oscillators
    "StreamingRSI",
    "StreamingMACD",
    # Trend
    "ADXState",
    "StreamingADX",
    "StreamingSAR",
    # Volatility
    "StreamingATR",
    # Volume
    "StreamingSessionVWAP",
    "StreamingAnchoredVWAP",
    "StreamingCVD",
    # Resampling
    "StreamingHTFResample",
]
