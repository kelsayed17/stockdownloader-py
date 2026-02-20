"""Trading strategy implementations.

Subpackages
-----------
daily : Daily equity strategies (SMA, RSI, MACD, etc.)
options : Options strategies (covered call, protective put)
intraday : Standalone intraday strategies and shared infrastructure
regime : Market regime detection and adaptive ensemble meta-strategy
exit_mechanisms : Exit mechanism implementations for tournaments
signals : Signal filtering and routing
"""

from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import ExitMechanism
from stockdownloader.strategy.trading_strategy import IntradayTradingStrategy
from stockdownloader.strategy.base_registry import StrategyRegistry

__all__ = [
    "ExitMechanism",
    "IntradayTradingStrategy",
    "StrategyRegistry",
]
