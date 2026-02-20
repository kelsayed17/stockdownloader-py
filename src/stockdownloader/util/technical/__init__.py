"""Technical indicators organized by category.

Sub-modules:
- volatility: Bollinger Bands, ATR, standard deviation
- momentum: RSI, MACD, Stochastic, MFI, CCI, Williams %R, ROC
- trend: ADX/DMI, Parabolic SAR, Ichimoku, VWAP, Fibonacci, Support/Resistance
- volume: OBV, average volume

All public symbols are re-exported here so that
``from stockdownloader.util.technical import <name>`` works for every indicator.
"""

from stockdownloader.util.technical._helpers import (
    SCALE,
    _deduplicate_levels,
    _period_midpoint,
    _quantize,
    standard_deviation,
    true_range,
)
from stockdownloader.util.technical.volatility import (
    BollingerBands,
    _bollinger_percent_b_from_bands,
    atr,
    bollinger_bands,
    bollinger_percent_b,
)
from stockdownloader.util.technical.momentum import (
    Stochastic,
    _macd_signal_from_lines,
    cci,
    macd_histogram,
    macd_line,
    macd_signal,
    mfi,
    roc,
    rsi,
    stochastic,
    williams_r,
)
from stockdownloader.util.technical.trend import (
    ADXResult,
    FibonacciLevels,
    IchimokuCloud,
    SessionVWAP,
    SupportResistance,
    _compute_session_vwap_core,
    _find_session_start,
    adx,
    fibonacci_retracement,
    ichimoku,
    is_sar_bullish,
    parabolic_sar,
    session_vwap,
    session_vwap_bands,
    support_resistance,
    vwap,
)
from stockdownloader.util.technical.volume import (
    average_volume,
    is_obv_rising,
    obv,
)

__all__ = [
    # _helpers
    "SCALE",
    "_quantize",
    "true_range",
    "standard_deviation",
    "_period_midpoint",
    "_deduplicate_levels",
    # volatility
    "BollingerBands",
    "bollinger_bands",
    "bollinger_percent_b",
    "_bollinger_percent_b_from_bands",
    "atr",
    # momentum
    "Stochastic",
    "stochastic",
    "rsi",
    "macd_line",
    "macd_signal",
    "_macd_signal_from_lines",
    "macd_histogram",
    "roc",
    "mfi",
    "williams_r",
    "cci",
    # trend
    "ADXResult",
    "IchimokuCloud",
    "FibonacciLevels",
    "SupportResistance",
    "SessionVWAP",
    "adx",
    "parabolic_sar",
    "is_sar_bullish",
    "ichimoku",
    "fibonacci_retracement",
    "support_resistance",
    "vwap",
    "session_vwap",
    "session_vwap_bands",
    "_find_session_start",
    "_compute_session_vwap_core",
    # volume
    "obv",
    "is_obv_rising",
    "average_volume",
]
