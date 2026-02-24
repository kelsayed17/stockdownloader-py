"""Atomic signal generators and their registry registrations.

Generators are organised by category across four submodules:
- :mod:`.momentum` -- RSI, MACD, Stochastic, CCI, Williams %R
- :mod:`.trend` -- SMA Crossover, ADX, EMA, Ichimoku, SAR, VWAP, SMA Position
- :mod:`.volatility` -- BB Touch, BB Squeeze
- :mod:`.volume` -- OBV, MFI, Volume Surge

All classes are re-exported here for convenience.
"""

from stockdownloader.signals.generators.momentum import (
    CCIGenerator,
    MACDCrossoverGenerator,
    RSISignalGenerator,
    StochasticGenerator,
    WilliamsRGenerator,
)
from stockdownloader.signals.generators.trend import (
    ADXTrendGenerator,
    EMATrendGenerator,
    IchimokuGenerator,
    SARGenerator,
    SMACrossoverGenerator,
    SMAPositionGenerator,
    VWAPGenerator,
)
from stockdownloader.signals.generators.volatility import (
    BBSqueezeGenerator,
    BBTouchGenerator,
)
from stockdownloader.signals.generators.volume import (
    MFIGenerator,
    OBVGenerator,
    VolumeSurgeGenerator,
)
from stockdownloader.strategies.registry import SignalGeneratorRegistry

__all__ = [
    # Momentum
    "RSISignalGenerator",
    "MACDCrossoverGenerator",
    "StochasticGenerator",
    "CCIGenerator",
    "WilliamsRGenerator",
    # Trend
    "SMACrossoverGenerator",
    "ADXTrendGenerator",
    "EMATrendGenerator",
    "IchimokuGenerator",
    "SARGenerator",
    "VWAPGenerator",
    "SMAPositionGenerator",
    # Volatility
    "BBTouchGenerator",
    "BBSqueezeGenerator",
    # Volume
    "OBVGenerator",
    "MFIGenerator",
    "VolumeSurgeGenerator",
]


# =====================================================================
# Registry registrations
# =====================================================================

# --- Momentum ---

SignalGeneratorRegistry.register(
    name="rsi",
    display_name="RSI",
    category="momentum",
    factory=RSISignalGenerator,
    default_kwargs={"period": 14, "oversold": 30.0, "overbought": 70.0},
    param_space={"period": [7, 10, 14, 21], "oversold": [20.0, 25.0, 30.0, 35.0],
                 "overbought": [65.0, 70.0, 75.0, 80.0]},
)

SignalGeneratorRegistry.register(
    name="macd",
    display_name="MACD Crossover",
    category="momentum",
    factory=MACDCrossoverGenerator,
    default_kwargs={"fast": 12, "slow": 26, "signal": 9},
    param_space={"fast": [8, 10, 12, 15], "slow": [20, 26, 30, 35],
                 "signal": [5, 7, 9, 12]},
)

SignalGeneratorRegistry.register(
    name="stochastic",
    display_name="Stochastic",
    category="momentum",
    factory=StochasticGenerator,
    default_kwargs={"oversold": 20.0, "overbought": 80.0},
    param_space={"oversold": [15.0, 20.0, 25.0], "overbought": [75.0, 80.0, 85.0]},
)

SignalGeneratorRegistry.register(
    name="cci",
    display_name="CCI",
    category="momentum",
    factory=CCIGenerator,
    default_kwargs={"period": 20, "oversold": -100.0, "overbought": 100.0},
    param_space={"period": [14, 20, 30], "oversold": [-150.0, -100.0, -50.0],
                 "overbought": [50.0, 100.0, 150.0]},
)

SignalGeneratorRegistry.register(
    name="williams_r",
    display_name="Williams %R",
    category="momentum",
    factory=WilliamsRGenerator,
    default_kwargs={"period": 14, "oversold": -80.0, "overbought": -20.0},
    param_space={"period": [10, 14, 21], "oversold": [-85.0, -80.0, -75.0],
                 "overbought": [-25.0, -20.0, -15.0]},
)

# --- Trend ---

SignalGeneratorRegistry.register(
    name="sma_cross",
    display_name="SMA Crossover",
    category="trend",
    factory=SMACrossoverGenerator,
    default_kwargs={"short_period": 9, "long_period": 21},
    param_space={"short_period": [5, 9, 12, 15, 20],
                 "long_period": [21, 30, 50, 100, 200]},
)

SignalGeneratorRegistry.register(
    name="adx",
    display_name="ADX Trend",
    category="trend",
    factory=ADXTrendGenerator,
    default_kwargs={"period": 14, "strength": 25.0, "weak": 20.0},
    param_space={"strength": [20.0, 25.0, 30.0], "weak": [15.0, 20.0]},
)

SignalGeneratorRegistry.register(
    name="ema_trend",
    display_name="EMA Trend",
    category="trend",
    factory=EMATrendGenerator,
    default_kwargs={"period": 200},
    param_space={"period": [50, 100, 150, 200]},
)

SignalGeneratorRegistry.register(
    name="ichimoku",
    display_name="Ichimoku Cloud",
    category="trend",
    factory=IchimokuGenerator,
    default_kwargs={},
    param_space={},
)

SignalGeneratorRegistry.register(
    name="sar",
    display_name="Parabolic SAR",
    category="trend",
    factory=SARGenerator,
    default_kwargs={},
    param_space={},
)

SignalGeneratorRegistry.register(
    name="vwap",
    display_name="VWAP",
    category="trend",
    factory=VWAPGenerator,
    default_kwargs={"period": 20},
    param_space={"period": [10, 20, 30]},
)

SignalGeneratorRegistry.register(
    name="sma_position",
    display_name="SMA Position",
    category="trend",
    factory=SMAPositionGenerator,
    default_kwargs={"period": 200},
    param_space={"period": [50, 100, 200]},
)

# --- Volatility ---

SignalGeneratorRegistry.register(
    name="bb_touch",
    display_name="BB Touch",
    category="volatility",
    factory=BBTouchGenerator,
    default_kwargs={"period": 20, "std_dev": 2.0},
    param_space={"period": [15, 20, 25], "std_dev": [1.5, 2.0, 2.5]},
)

SignalGeneratorRegistry.register(
    name="bb_squeeze",
    display_name="BB Squeeze",
    category="volatility",
    factory=BBSqueezeGenerator,
    default_kwargs={"period": 20, "squeeze_lookback": 120},
    param_space={"period": [15, 20, 25], "squeeze_lookback": [60, 90, 120, 150]},
)

# --- Volume ---

SignalGeneratorRegistry.register(
    name="obv",
    display_name="OBV",
    category="volume",
    factory=OBVGenerator,
    default_kwargs={"lookback": 5},
    param_space={"lookback": [3, 5, 10]},
)

SignalGeneratorRegistry.register(
    name="mfi",
    display_name="MFI",
    category="volume",
    factory=MFIGenerator,
    default_kwargs={"period": 14, "oversold": 20.0, "overbought": 80.0},
    param_space={"period": [10, 14, 21], "oversold": [15.0, 20.0, 25.0],
                 "overbought": [75.0, 80.0, 85.0]},
)

SignalGeneratorRegistry.register(
    name="volume_surge",
    display_name="Volume Surge",
    category="volume",
    factory=VolumeSurgeGenerator,
    default_kwargs={"period": 20, "multiplier": 1.5},
    param_space={"period": [10, 20, 30], "multiplier": [1.2, 1.5, 2.0, 2.5]},
)
