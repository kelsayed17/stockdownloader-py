"""Stock analysis, backtesting, and options trading platform."""

from stockdownloader.backtesting.engines.daily import BacktestEngine
from stockdownloader.backtesting.results.result import BacktestResult
from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.backtesting.engines.options import OptionsBacktestEngine
from stockdownloader.data.market.yahoo_data_client import YahooDataClient
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import Trade
from stockdownloader.strategies.registry import StrategyRegistry

__all__ = [
    # Engines
    "BacktestEngine",
    "IntradayBacktestEngine",
    "OptionsBacktestEngine",
    # Results
    "BacktestResult",
    # Data
    "YahooDataClient",
    # Models
    "PriceData",
    "IntradayPriceData",
    "Trade",
    # Registry
    "StrategyRegistry",
]
