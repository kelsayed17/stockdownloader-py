"""Backtesting engines and reporting."""

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.backtest_result import (
    BaseBacktestResult,
    BacktestResult,
    OptionsBacktestResult,
)
from stockdownloader.backtest import report_formatter
from stockdownloader.backtest.options_backtest_engine import OptionsBacktestEngine
from stockdownloader.backtest.exit_tournament_engine import ExitTournamentEngine
from stockdownloader.backtest.exit_tournament_result import ExitTournamentResult
from stockdownloader.backtest import exit_tournament_report_formatter
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_base import OptimizerBase
from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.backtest.strategy_optimizer import StrategyOptimizer
from stockdownloader.backtest.daily_strategy_optimizer import DailyStrategyOptimizer
from stockdownloader.backtest.walk_forward import WalkForwardValidator, WalkForwardResult
from stockdownloader.backtest.walk_forward_optimizer import WalkForwardOptimizer, WFOptResult
from stockdownloader.backtest.combinatorial_tester import CombinatorialTester, CombinatorialConfig
from stockdownloader.backtest.tournament_models import (
    ComboKey,
    ComboResult,
    MatchResult,
    TournamentResult,
    RegimeTradeStats,
    RegimeAnalysis,
    MonteCarloPercentiles,
    MonteCarloResult,
)
from stockdownloader.backtest import tournament_engine

__all__ = [
    "BacktestEngine",
    "BaseBacktestResult",
    "BacktestResult",
    "report_formatter",
    "OptionsBacktestEngine",
    "OptionsBacktestResult",
    "ExitTournamentEngine",
    "ExitTournamentResult",
    "exit_tournament_report_formatter",
    "IntradayBacktestEngine",
    "OptimizerBase",
    "score_v2",
    "StrategyOptimizer",
    "DailyStrategyOptimizer",
    "WalkForwardValidator",
    "WalkForwardResult",
    "WalkForwardOptimizer",
    "WFOptResult",
    "CombinatorialTester",
    "CombinatorialConfig",
    "ComboKey",
    "ComboResult",
    "MatchResult",
    "TournamentResult",
    "RegimeTradeStats",
    "RegimeAnalysis",
    "MonteCarloPercentiles",
    "MonteCarloResult",
    "tournament_engine",
]
