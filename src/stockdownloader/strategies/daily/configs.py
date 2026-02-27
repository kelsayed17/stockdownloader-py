"""Config dataclasses for daily strategies — enables JSON serialization.

Each config mirrors the constructor parameters of the corresponding strategy
class.  All fields have sensible defaults so that ``MyConfig()`` produces the
same strategy as calling the constructor with no arguments.

Usage::

    from stockdownloader.strategies.daily.configs import MACDConfig

    cfg = MACDConfig(fast_period=8)
    cfg.save("optimized_macd.json")

    cfg2 = MACDConfig.load("optimized_macd.json")
    assert cfg2 == cfg
"""

from __future__ import annotations

from dataclasses import dataclass

from stockdownloader.strategies.config_base import StrategyConfigMixin


@dataclass(frozen=True, slots=True)
class MACDConfig(StrategyConfigMixin):
    """Config for :class:`~stockdownloader.strategies.daily.simple.MACDStrategy`."""

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9


@dataclass(frozen=True, slots=True)
class SMACrossoverConfig(StrategyConfigMixin):
    """Config for :class:`~stockdownloader.strategies.daily.simple.SMACrossoverStrategy`."""

    short_period: int = 9
    long_period: int = 21


@dataclass(frozen=True, slots=True)
class RSIConfig(StrategyConfigMixin):
    """Config for :class:`~stockdownloader.strategies.daily.simple.RSIStrategy`."""

    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0


@dataclass(frozen=True, slots=True)
class BollingerRSIConfig(StrategyConfigMixin):
    """Config for :class:`~stockdownloader.strategies.daily.bollinger_rsi.BollingerBandRSIStrategy`."""

    bb_period: int = 20
    bb_std_dev: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    adx_threshold: float = 25.0


@dataclass(frozen=True, slots=True)
class BreakoutConfig(StrategyConfigMixin):
    """Config for :class:`~stockdownloader.strategies.daily.breakout.BreakoutStrategy`."""

    bb_period: int = 20
    squeeze_lookback: int = 120
    volume_multiplier: float = 1.5


@dataclass(frozen=True, slots=True)
class MomentumConfig(StrategyConfigMixin):
    """Config for :class:`~stockdownloader.strategies.daily.momentum.MomentumConfluenceStrategy`."""

    fast_ema: int = 12
    slow_ema: int = 26
    signal_period: int = 9
    ema_trend_filter: int = 200
    adx_strength_threshold: float = 25.0
    adx_weak_threshold: float = 20.0


@dataclass(frozen=True, slots=True)
class MultiIndicatorConfig(StrategyConfigMixin):
    """Config for :class:`~stockdownloader.strategies.daily.multi_indicator.MultiIndicatorStrategy`."""

    buy_threshold: int = 4
    sell_threshold: int = 4
