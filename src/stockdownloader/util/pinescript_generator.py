"""PineScript generation -- re-exports from pinescript package.

All functionality has been moved to :mod:`stockdownloader.util.pinescript`.
This module re-exports the public API for backward compatibility.
"""

from __future__ import annotations

# Re-export all models so existing imports keep working.
from stockdownloader.util.pinescript_models import (  # noqa: F401
    CompositeStrategyDefinition,
    Condition,
    Indicator,
    Input,
    InputType,
    InputValue,
    ModeDefinition,
    SharedInfrastructure,
    StrategyDefinition,
)

from stockdownloader.util.pinescript.generator import PineScriptGenerator  # noqa: F401
from stockdownloader.util.pinescript.converters import (  # noqa: F401
    _indicators_as_lines,
    mode_to_strategy,
    strategy_to_mode,
)

__all__ = [
    "CompositeStrategyDefinition",
    "Condition",
    "Indicator",
    "Input",
    "InputType",
    "InputValue",
    "ModeDefinition",
    "PineScriptGenerator",
    "SharedInfrastructure",
    "StrategyDefinition",
    "mode_to_strategy",
    "strategy_to_mode",
]
