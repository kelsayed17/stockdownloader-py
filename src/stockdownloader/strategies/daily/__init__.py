"""Daily trading strategy implementations.

All strategies extend :class:`~stockdownloader.strategies.base.TradingStrategy`
and return :class:`~stockdownloader.strategies.base.Signal` (BUY/SELL/HOLD).
"""
from stockdownloader.strategies.daily.simple import SMACrossoverStrategy
from stockdownloader.strategies.daily.simple import RSIStrategy
from stockdownloader.strategies.daily.simple import MACDStrategy
from stockdownloader.strategies.daily.bollinger_rsi import BollingerBandRSIStrategy
from stockdownloader.strategies.daily.breakout import BreakoutStrategy
from stockdownloader.strategies.daily.momentum import MomentumConfluenceStrategy
from stockdownloader.strategies.daily.multi_indicator import MultiIndicatorStrategy

__all__ = [
    "SMACrossoverStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "BollingerBandRSIStrategy",
    "BreakoutStrategy",
    "MomentumConfluenceStrategy",
    "MultiIndicatorStrategy",
]
