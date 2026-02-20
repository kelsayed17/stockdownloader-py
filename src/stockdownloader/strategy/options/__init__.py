"""Options trading strategy interface and implementations.

All strategies extend :class:`~stockdownloader.strategy.options.options_strategies.OptionsStrategy`
and return :class:`~stockdownloader.strategy.options.options_strategies.OptionsSignal` (OPEN/CLOSE/HOLD).
"""
from stockdownloader.strategy.options.options_strategies import (
    CoveredCallStrategy,
    OptionsSignal,
    OptionsStrategy,
    ProtectivePutStrategy,
)

__all__ = [
    "CoveredCallStrategy",
    "OptionsSignal",
    "OptionsStrategy",
    "ProtectivePutStrategy",
]
