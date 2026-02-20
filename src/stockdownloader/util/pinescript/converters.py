"""Converter functions between strategy/mode definitions and Pine Script.

Standalone helpers for converting between ``StrategyDefinition`` and
``ModeDefinition`` objects, plus indicator-to-code-line utilities.
"""

from __future__ import annotations

from stockdownloader.util.pinescript_models import (
    Indicator,
    ModeDefinition,
    SharedInfrastructure,
    StrategyDefinition,
)
from stockdownloader.util.pinescript.generator import PineScriptGenerator


# ======================================================================
# Strategy-to-Mode converter
# ======================================================================


def _indicators_as_lines(indicators: list[Indicator]) -> list[str]:
    """Convert Indicator objects to raw Pine Script lines for extra_code."""
    lines: list[str] = []
    for ind in indicators:
        if ind.var_name.startswith("["):
            lines.append(f"{ind.var_name} = {ind.code}")
        else:
            tp = PineScriptGenerator._infer_type(ind.code)
            lines.append(f"{tp} {ind.var_name} = {ind.code}")
    return lines


def mode_to_strategy(
    mode: ModeDefinition,
    *,
    shared: SharedInfrastructure | None = None,
    session: str = "0930-1600",
    timezone: str = "America/New_York",
    use_session_filter: bool = True,
    name_override: str = "",
    short_name_override: str = "",
) -> StrategyDefinition:
    """Convert a ``ModeDefinition`` into a standalone ``StrategyDefinition``."""
    infra = shared or SharedInfrastructure()

    return StrategyDefinition(
        name=name_override or mode.name,
        short_name=short_name_override or mode.short_name.upper(),
        description=mode.description,
        session=session,
        timezone=timezone,
        use_session_filter=use_session_filter,
        inputs=[*infra.inputs, *mode.inputs],
        indicators=[*infra.indicators],
        extra_code=[
            *infra.code,
            *_indicators_as_lines(mode.indicators),
            *mode.extra_code,
        ],
        long_entry=mode.long_entry,
        short_entry=mode.short_entry,
        long_exit=mode.long_exit,
        short_exit=mode.short_exit,
        exit_on_reverse=mode.exit_on_reverse,
    )


def strategy_to_mode(
    strategy: StrategyDefinition,
    short_name: str,
    group: str,
    enabled_default: bool = True,
    label_color_long: str = "color.green",
    label_color_short: str = "color.red",
) -> ModeDefinition:
    """Convert a standalone ``StrategyDefinition`` into a ``ModeDefinition``."""
    return ModeDefinition(
        name=strategy.name,
        short_name=short_name,
        group=group,
        enabled_default=enabled_default,
        inputs=strategy.inputs,
        indicators=strategy.indicators,
        long_entry=strategy.long_entry,
        short_entry=strategy.short_entry,
        long_exit=strategy.long_exit,
        short_exit=strategy.short_exit,
        exit_on_reverse=strategy.exit_on_reverse,
        extra_code=strategy.extra_code,
        description=strategy.description,
        label_color_long=label_color_long,
        label_color_short=label_color_short,
    )
