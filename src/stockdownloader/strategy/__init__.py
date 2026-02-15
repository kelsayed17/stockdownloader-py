"""Trading strategy implementations.

Subpackages
-----------
daily : Daily equity strategies (SMA, RSI, MACD, etc.)
options : Options strategies (covered call, protective put)
intraday : Standalone intraday strategies and shared infrastructure
exit_mechanisms : Exit mechanism implementations for tournaments
"""

from stockdownloader.strategy.exit_mechanism import ExitMechanism
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.strategy.registry import StrategyRegistry

__all__ = [
    "ExitMechanism",
    "IntradayTradingStrategy",
    "StrategyRegistry",
]
