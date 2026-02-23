"""Trading strategy implementations.

Subpackages
-----------
daily : Daily equity strategies (SMA, RSI, MACD, etc.)
options : Options strategies (covered call, protective put)
intraday : Standalone intraday strategies and shared infrastructure
regime : Market regime detection and adaptive ensemble meta-strategy
exit_mechanisms : Exit mechanism implementations for tournaments
signals : Signal filtering and routing

Core ABCs & enums
-----------------
- :class:`TradingStrategy` -- ABC for daily strategies
- :class:`IntradayTradingStrategy` -- ABC for intraday strategies
- :class:`Signal` -- BUY / SELL / HOLD enum for daily strategies

Registries
----------
- :class:`StrategyRegistry` -- auto-discovery for strategies
- :class:`SignalGeneratorRegistry` -- auto-discovery for signal generators
- :class:`RegistryEntry` -- dataclass entry stored in both registries
- :func:`ensure_registered` -- idempotent loader for built-in strategies

Intraday infrastructure
-----------------------
- :class:`BaseIntradayStrategy` -- template base with boilerplate delegation
- :class:`IntradayInfra` -- shared per-session infrastructure (compose, don't inherit)
- :class:`DailyToIntradayAdapter` -- wraps daily strategies for the intraday engine

Signal stack
------------
- :class:`AtomicSignalGenerator` -- ABC for single-indicator generators
- :class:`StackedSignalEngine` -- composable multi-signal aggregator
- :class:`StackedDailyStrategy` -- daily wrapper for signal stacks
- :class:`StackedIntradayStrategy` -- intraday wrapper for signal stacks

Regime
------
- :class:`EnsembleIntradayStrategy` -- adaptive regime-aware meta-strategy

Exit mechanisms
---------------
- :class:`ExitMechanism` -- ABC for tournament exit mechanisms
"""

from stockdownloader.strategy.trading_strategy import (
    IntradayTradingStrategy,
    Signal,
    TradingStrategy,
)
from stockdownloader.strategy.base_registry import (
    RegistryEntry,
    SignalGeneratorRegistry,
    StrategyRegistry,
)
from stockdownloader.strategy.registration_loader import ensure_registered
from stockdownloader.strategy.intraday.base_strategy import BaseIntradayStrategy
from stockdownloader.strategy.intraday.infra import IntradayInfra
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import (
    DailyToIntradayAdapter,
)
from stockdownloader.strategy.signals.signal_generator import AtomicSignalGenerator
from stockdownloader.strategy.signals.stacked_signal_engine import StackedSignalEngine
from stockdownloader.strategy.signals.stacked_daily_strategy import StackedDailyStrategy
from stockdownloader.strategy.signals.stacked_intraday_strategy import StackedIntradayStrategy
from stockdownloader.strategy.regime.ensemble_strategy import EnsembleIntradayStrategy
from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import ExitMechanism

__all__ = [
    # Core ABCs & enums
    "IntradayTradingStrategy",
    "Signal",
    "TradingStrategy",
    # Registries
    "RegistryEntry",
    "SignalGeneratorRegistry",
    "StrategyRegistry",
    "ensure_registered",
    # Intraday infrastructure
    "BaseIntradayStrategy",
    "DailyToIntradayAdapter",
    "IntradayInfra",
    # Signal stack
    "AtomicSignalGenerator",
    "StackedDailyStrategy",
    "StackedIntradayStrategy",
    "StackedSignalEngine",
    # Regime
    "EnsembleIntradayStrategy",
    # Exit mechanisms
    "ExitMechanism",
]
