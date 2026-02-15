"""Options trading strategy implementations.

All strategies extend :class:`~stockdownloader.strategy.options_strategy.OptionsStrategy`
and return :class:`~stockdownloader.strategy.options_strategy.OptionsSignal` (OPEN/CLOSE/HOLD).
"""
from stockdownloader.strategy.options.covered_call_strategy import CoveredCallStrategy
from stockdownloader.strategy.options.protective_put_strategy import ProtectivePutStrategy

__all__ = ["CoveredCallStrategy", "ProtectivePutStrategy"]
