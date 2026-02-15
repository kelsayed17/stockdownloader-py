"""Daily trading strategy implementations.

All strategies extend :class:`~stockdownloader.strategy.trading_strategy.TradingStrategy`
and return :class:`~stockdownloader.strategy.trading_strategy.Signal` (BUY/SELL/HOLD).
"""
from stockdownloader.strategy.daily.sma_crossover_strategy import SMACrossoverStrategy
from stockdownloader.strategy.daily.rsi_strategy import RSIStrategy
from stockdownloader.strategy.daily.macd_strategy import MACDStrategy
from stockdownloader.strategy.daily.bollinger_band_rsi_strategy import BollingerBandRSIStrategy
from stockdownloader.strategy.daily.breakout_strategy import BreakoutStrategy
from stockdownloader.strategy.daily.momentum_confluence_strategy import MomentumConfluenceStrategy
from stockdownloader.strategy.daily.multi_indicator_strategy import MultiIndicatorStrategy

__all__ = [
    "SMACrossoverStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "BollingerBandRSIStrategy",
    "BreakoutStrategy",
    "MomentumConfluenceStrategy",
    "MultiIndicatorStrategy",
]
