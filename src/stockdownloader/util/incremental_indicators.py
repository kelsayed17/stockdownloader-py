"""Streaming (incremental) indicator computation for backtesting.

This module is a **backward-compatibility shim**.  The actual
implementations now live in :mod:`stockdownloader.util.streaming`,
organized by indicator type:

- ``streaming._base``        -- shared helpers (``SCALE``, ``_quantize``, ``_true_range``)
- ``streaming.accumulators`` -- ``StreamingOBV``, ``StreamingEMA``
- ``streaming.oscillators``  -- ``StreamingRSI``, ``StreamingMACD``
- ``streaming.trend``        -- ``ADXState``, ``StreamingADX``, ``StreamingSAR``
- ``streaming.volatility``   -- ``StreamingATR``
- ``streaming.volume``       -- ``StreamingSessionVWAP``, ``StreamingAnchoredVWAP``, ``StreamingCVD``
- ``streaming.resampling``   -- ``StreamingHTFResample``

All public names are re-exported here so that existing ``from
stockdownloader.util.incremental_indicators import ...`` statements
continue to work without modification.
"""

from stockdownloader.util.streaming import (  # noqa: F401 -- re-export
    SCALE,
    ADXState,
    StreamingADX,
    StreamingAnchoredVWAP,
    StreamingATR,
    StreamingCVD,
    StreamingEMA,
    StreamingHTFResample,
    StreamingMACD,
    StreamingOBV,
    StreamingRSI,
    StreamingSAR,
    StreamingSessionVWAP,
    _quantize,
    _true_range,
)

__all__ = [
    "SCALE",
    "_quantize",
    "_true_range",
    "ADXState",
    "StreamingADX",
    "StreamingAnchoredVWAP",
    "StreamingATR",
    "StreamingCVD",
    "StreamingEMA",
    "StreamingHTFResample",
    "StreamingMACD",
    "StreamingOBV",
    "StreamingRSI",
    "StreamingSAR",
    "StreamingSessionVWAP",
]
