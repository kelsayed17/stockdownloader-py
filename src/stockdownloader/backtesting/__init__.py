"""Backtesting engines and reporting."""

from stockdownloader.backtesting.engines.daily import BacktestEngine
from stockdownloader.backtesting.results.result import (
    BaseBacktestResult,
    BacktestResult,
    OptionsBacktestResult,
)
from stockdownloader.backtesting.results import formatter as report_formatter
from stockdownloader.backtesting.engines.options import OptionsBacktestEngine
from stockdownloader.backtesting.tournament.exit_engine import ExitTournamentEngine
from stockdownloader.backtesting.tournament.exit_result import ExitTournamentResult
from stockdownloader.backtesting.tournament import exit_report as exit_tournament_report_formatter
from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.backtesting.optimization.base import OptimizerBase
from stockdownloader.backtesting.optimization.scoring import score_v2
from stockdownloader.backtesting.optimization.strategy import StrategyOptimizer
from stockdownloader.backtesting.optimization.daily import DailyStrategyOptimizer
from stockdownloader.backtesting.optimization.walk_forward import WalkForwardValidator, WalkForwardResult
from stockdownloader.backtesting.optimization.wf_optimizer import WalkForwardOptimizer, WFOptResult
from stockdownloader.backtesting.combinatorial import CombinatorialTester, CombinatorialConfig
from stockdownloader.backtesting.tournament.models import (
    ComboKey,
    ComboResult,
    MatchResult,
    TournamentResult,
    RegimeTradeStats,
    RegimeAnalysis,
    MonteCarloPercentiles,
    MonteCarloResult,
)
from stockdownloader.backtesting.tournament import engine as tournament_engine

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
