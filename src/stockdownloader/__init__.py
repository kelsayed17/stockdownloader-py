"""Stock analysis, backtesting, and options trading platform."""

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.options_backtest_engine import OptionsBacktestEngine
from stockdownloader.data.yahoo_data_client import YahooDataClient
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.model.trade import Trade
from stockdownloader.strategy.base_registry import StrategyRegistry

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
