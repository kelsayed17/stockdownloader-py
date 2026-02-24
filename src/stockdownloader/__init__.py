"""Stock analysis, backtesting, and options trading platform."""

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.options_backtest_engine import OptionsBacktestEngine
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
