"""Options trading strategy interface and implementations.

All strategies extend :class:`~stockdownloader.strategies.options.strategies.OptionsStrategy`
and return :class:`~stockdownloader.strategies.options.strategies.OptionsSignal` (OPEN/CLOSE/HOLD).
"""
from stockdownloader.strategies.options.strategies import (
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
