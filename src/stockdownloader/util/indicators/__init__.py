"""Unified technical and streaming indicators.

Re-exports all public names from sub-modules so that callers can use::

    from stockdownloader.util.indicators import rsi, BollingerBands, IndicatorHub

Sub-modules:
- ``_core``: Moving averages, true range, standard deviation, crossover detection
- ``momentum``: RSI, MACD, Stochastic, MFI, CCI, Williams %R, ROC, OBV
- ``volatility``: Bollinger Bands, ATR, StreamingEMA, StreamingATR
- ``trend``: ADX, Parabolic SAR, Ichimoku, VWAP (lookback), Fibonacci, S/R
- ``volume``: Session VWAP, Extended VWAP, Anchored VWAP, CVD
- ``intraday``: Time-of-day RVOL, LRS, VWAP slope, candle patterns, S/R proximity
- ``smc``: Smart Money Concepts (swing detection, structure tracking)
- ``htf``: Higher-timeframe resampling
- ``hub``: Cached indicator calculator (IndicatorHub)
"""

from __future__ import annotations

# -- _core ------------------------------------------------------------------
from stockdownloader.util.indicators._core import (
    sma,
    ema,
    true_range,
    standard_deviation,
    crossed_above,
    crossed_below,
    crossed_above_series,
    crossed_below_series,
)

# -- momentum ---------------------------------------------------------------
from stockdownloader.util.indicators.momentum import (
    Stochastic,
    stochastic,
    rsi,
    macd_line,
    macd_signal,
    _macd_signal_from_lines,
    macd_histogram,
    roc,
    mfi,
    williams_r,
    cci,
    obv,
    is_obv_rising,
    average_volume,
    StreamingRSI,
    StreamingMACD,
    StreamingOBV,
)

# -- volatility -------------------------------------------------------------
from stockdownloader.util.indicators.volatility import (
    BollingerBands,
    bollinger_bands,
    bollinger_percent_b,
    _bollinger_percent_b_from_bands,
    atr,
    StreamingEMA,
    StreamingATR,
)

# -- trend ------------------------------------------------------------------
from stockdownloader.util.indicators.trend import (
    ADXResult,
    IchimokuCloud,
    FibonacciLevels,
    SupportResistance,
    adx,
    parabolic_sar,
    is_sar_bullish,
    vwap,
    fibonacci_retracement,
    ichimoku,
    support_resistance,
    ADXState,
    StreamingADX,
    StreamingSAR,
)

# -- volume -----------------------------------------------------------------
from stockdownloader.util.indicators.volume import (
    SessionVWAP,
    ExtendedSessionVWAP,
    AnchoredVWAPBands,
    session_vwap,
    session_vwap_bands,
    extended_session_vwap_bands,
    cvd_session,
    cvd_normalized,
    _EMPTY_VWAP,
    _EMPTY_AVWAP,
    StreamingSessionVWAP,
    StreamingAnchoredVWAP,
    StreamingCVD,
)

# -- intraday ---------------------------------------------------------------
from stockdownloader.util.indicators.intraday import (
    tod_rvol,
    linear_regression_slope,
    lrs_normalized,
    vwap_slope,
    vwap_acceleration,
    _session_vwap_at,
    aggregate_to_daily,
    daily_atr_prior,
    resample_to_htf,
    htf_ema_trend,
    CandleStrength,
    candle_strength,
    is_hammer,
    is_inv_hammer,
    is_bull_engulfing,
    is_bear_engulfing,
    near_level,
    compute_sr_score,
    _round_to_5,
    rel_vol,
)

# -- smc --------------------------------------------------------------------
from stockdownloader.util.indicators.smc import (
    SwingPoint,
    StructureState,
    _EMPTY_STRUCTURE,
    is_swing_high,
    is_swing_low,
    is_liquidity_sweep_high,
    is_liquidity_sweep_low,
    impulse_strength,
    zone_from_impulse_origin,
    StreamingStructureTracker,
)

# -- htf --------------------------------------------------------------------
from stockdownloader.util.indicators.htf import StreamingHTFResample

# -- hub --------------------------------------------------------------------
from stockdownloader.util.indicators.hub import IndicatorHub


__all__ = [
    # _core
    "sma",
    "ema",
    "true_range",
    "standard_deviation",
    "crossed_above",
    "crossed_below",
    "crossed_above_series",
    "crossed_below_series",
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
    "obv",
    "is_obv_rising",
    "average_volume",
    "StreamingRSI",
    "StreamingMACD",
    "StreamingOBV",
    # volatility
    "BollingerBands",
    "bollinger_bands",
    "bollinger_percent_b",
    "_bollinger_percent_b_from_bands",
    "atr",
    "StreamingEMA",
    "StreamingATR",
    # trend
    "ADXResult",
    "IchimokuCloud",
    "FibonacciLevels",
    "SupportResistance",
    "adx",
    "parabolic_sar",
    "is_sar_bullish",
    "vwap",
    "fibonacci_retracement",
    "ichimoku",
    "support_resistance",
    "ADXState",
    "StreamingADX",
    "StreamingSAR",
    # volume
    "SessionVWAP",
    "ExtendedSessionVWAP",
    "AnchoredVWAPBands",
    "session_vwap",
    "session_vwap_bands",
    "extended_session_vwap_bands",
    "cvd_session",
    "cvd_normalized",
    "_EMPTY_VWAP",
    "_EMPTY_AVWAP",
    "StreamingSessionVWAP",
    "StreamingAnchoredVWAP",
    "StreamingCVD",
    # intraday
    "tod_rvol",
    "linear_regression_slope",
    "lrs_normalized",
    "vwap_slope",
    "vwap_acceleration",
    "_session_vwap_at",
    "aggregate_to_daily",
    "daily_atr_prior",
    "resample_to_htf",
    "htf_ema_trend",
    "CandleStrength",
    "candle_strength",
    "is_hammer",
    "is_inv_hammer",
    "is_bull_engulfing",
    "is_bear_engulfing",
    "near_level",
    "compute_sr_score",
    "_round_to_5",
    "rel_vol",
    # smc
    "SwingPoint",
    "StructureState",
    "_EMPTY_STRUCTURE",
    "is_swing_high",
    "is_swing_low",
    "is_liquidity_sweep_high",
    "is_liquidity_sweep_low",
    "impulse_strength",
    "zone_from_impulse_origin",
    "StreamingStructureTracker",
    # htf
    "StreamingHTFResample",
    # hub
    "IndicatorHub",
]
