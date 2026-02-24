"""Trading strategy implementations.

Subpackages
-----------
daily : Daily equity strategies (SMA, RSI, MACD, etc.)
options : Options strategies (covered call, protective put)
intraday : Standalone intraday strategies and shared infrastructure
regime : Market regime detection and adaptive ensemble meta-strategy
exits : Exit mechanism implementations for tournaments

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

from stockdownloader.strategies.base import (
    IntradayTradingStrategy,
    Signal,
    TradingStrategy,
)
from stockdownloader.strategies.registry import (
    RegistryEntry,
    SignalGeneratorRegistry,
    StrategyRegistry,
)
from stockdownloader.strategies.loader import ensure_registered
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.daily_adapter import (
    DailyToIntradayAdapter,
)
from stockdownloader.signals.generator import AtomicSignalGenerator
from stockdownloader.signals.engine import StackedSignalEngine
from stockdownloader.signals.daily_adapter import StackedDailyStrategy
from stockdownloader.signals.intraday_adapter import StackedIntradayStrategy
from stockdownloader.strategies.regime.ensemble import EnsembleIntradayStrategy
from stockdownloader.strategies.exits.base import ExitMechanism

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
