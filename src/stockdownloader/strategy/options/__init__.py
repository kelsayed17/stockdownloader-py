"""Options trading strategy implementations.

All strategies extend :class:`~stockdownloader.strategy.options_strategy.OptionsStrategy`
and return :class:`~stockdownloader.strategy.options_strategy.OptionsSignal` (OPEN/CLOSE/HOLD).
"""
from stockdownloader.strategy.options.options_strategies import CoveredCallStrategy
from stockdownloader.strategy.options.options_strategies import ProtectivePutStrategy

__all__ = ["CoveredCallStrategy", "ProtectivePutStrategy"]
