"""Composite (multi-mode) rendering for PineScriptGenerator.

Provides the header, inputs, session handling, mode sections,
signal aggregation, labels, background, and alerts for
composite Pine Script v6 indicators with per-mode toggles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stockdownloader.pinescript.models import (
    CompositeStrategyDefinition,
    Input,
    ModeDefinition,
)

if TYPE_CHECKING:
    from stockdownloader.pinescript.generator import PineScriptGenerator


def composite_header(gen: PineScriptGenerator, d: CompositeStrategyDefinition) -> str:
    desc = ""
    if d.description:
        for line in d.description.split("\n"):
            desc += f"// {line}\n"

    mode_list = ", ".join(m.short_name.upper() for m in d.modes)
    desc += f"//\n// Modes: {mode_list}\n"
    desc += f"// Aggregation: {d.aggregation}\n"

    return gen._render_header(d.name, d.short_name, desc)


def composite_inputs(
    gen: PineScriptGenerator,
    d: CompositeStrategyDefinition,
    *,
    strategy_mode: bool = False,
) -> str:
    lines: list[str] = []

    lines.append(gen._section_header("INPUTS"))
    if not strategy_mode:
        lines.append('bool   showLabels    = input.bool(true, '
                      'title="Show Signal Labels", group="Display")')
        lines.append('bool   showBgColor   = input.bool(true, '
                      'title="Show Background Coloring", group="Display")')

    if d.shared_inputs:
        lines.append("")
        lines.append("// --- Shared ---")
        for inp in d.shared_inputs:
            lines.append(gen._input_line(inp))

    for mode in d.modes:
        lines.append("")
        lines.append(f"// --- {mode.group} ---")
        toggle_name = f"enable{mode.short_name.upper()}"
        default_val = "true" if mode.enabled_default else "false"
        lines.append(
            f'bool   {toggle_name:16s} = input.bool({default_val}, '
            f'title="Enable {mode.name}", group="{mode.group}")'
        )
        for inp in mode.inputs:
            forced = Input(inp.name, inp.input_type, inp.default,
                           inp.title, inp.min_val, inp.max_val,
                           inp.step, inp.options, mode.group)
            lines.append(gen._input_line(forced))

    return "\n".join(lines)


def composite_session(gen: PineScriptGenerator, d: CompositeStrategyDefinition) -> str:
    if not d.use_session_filter:
        return ""
    return gen._render_session(d.session, d.timezone)


def composite_shared_indicators(
    gen: PineScriptGenerator, d: CompositeStrategyDefinition,
) -> str:
    lines: list[str] = []
    lines.append(gen._section_header("SHARED INDICATORS"))
    gen._render_indicators(d.shared_indicators, lines)
    return "\n".join(lines)


def composite_mode_section(gen: PineScriptGenerator, mode: ModeDefinition) -> str:
    lines: list[str] = []
    sn = mode.short_name
    toggle = f"enable{sn.upper()}"

    lines.append(gen._section_header(f"MODE: {mode.name.upper()}"))

    if mode.description:
        for dline in mode.description.split("\n"):
            lines.append(f"// {dline}")
        lines.append("")

    gen._render_indicators(mode.indicators, lines, render_plots=False)

    for ec in mode.extra_code:
        lines.append(ec)

    if mode.long_entry:
        lines.append(
            f"bool {sn}Long = {toggle} and ({mode.long_entry.expr})"
        )
    else:
        lines.append(f"bool {sn}Long = false")

    if mode.short_entry:
        lines.append(
            f"bool {sn}Short = {toggle} and ({mode.short_entry.expr})"
        )
    else:
        lines.append(f"bool {sn}Short = false")

    return "\n".join(lines)


def composite_aggregation(
    gen: PineScriptGenerator, d: CompositeStrategyDefinition,
) -> str:
    lines: list[str] = []
    lines.append(gen._section_header("SIGNAL AGGREGATION"))

    lines.append("bool longCondition = false")
    lines.append("bool shortCondition = false")
    lines.append('var string activeMode = ""')
    lines.append("")

    if d.aggregation == "first_to_fire":
        order = d.priority_order or [m.short_name for m in d.modes]
        first = True
        for sn in order:
            kw = "if" if first else "else if"
            lines.append(f"{kw} {sn}Long")
            lines.append("    longCondition := true")
            lines.append(f'    activeMode := "{sn.upper()}"')
            first = False
        first = True
        lines.append("")
        for sn in order:
            kw = "if" if first else "else if"
            lines.append(f"{kw} {sn}Short")
            lines.append("    shortCondition := true")
            lines.append(f'    activeMode := "{sn.upper()}"')
            first = False
    else:
        # "any" aggregation
        long_parts = [f"{m.short_name}Long" for m in d.modes]
        short_parts = [f"{m.short_name}Short" for m in d.modes]
        lines.append(
            f"longCondition := {' or '.join(long_parts)}"
        )
        lines.append(
            f"shortCondition := {' or '.join(short_parts)}"
        )
        for m in d.modes:
            sn = m.short_name
            lines.append(f'if {sn}Long or {sn}Short')
            lines.append(f'    activeMode := "{sn.upper()}"')

    gen._emit_state_machine(
        lines,
        has_long=True,
        has_short=True,
        long_exit_expr="shortCondition",
        short_exit_expr="longCondition",
        use_session_filter=d.use_session_filter,
    )
    return "\n".join(lines)


def composite_labels(gen: PineScriptGenerator, d: CompositeStrategyDefinition) -> str:
    lines: list[str] = []
    lines.append(gen._section_header("LABELS"))

    guard = "showLabels"
    if d.use_session_filter:
        guard += " and inSession"

    lines.append(f"if {guard}")
    gen._emit_label(lines, "buySignal", "low",
                    'activeMode + " Buy"',
                    "label.style_label_up", "color.green")
    gen._emit_label(lines, "sellSignal", "high",
                    'activeMode + " Sell"',
                    "label.style_label_down", "color.red")
    gen._emit_label(lines, "exitLongSignal", "high",
                    '"Exit Long"',
                    "label.style_label_down", "color.orange")
    gen._emit_label(lines, "exitShortSignal", "low",
                    '"Exit Short"',
                    "label.style_label_up", "color.orange")
    return "\n".join(lines)


def composite_background(
    gen: PineScriptGenerator, d: CompositeStrategyDefinition,
) -> str:
    guard = "showBgColor"
    if d.use_session_filter:
        guard += " and inSession"
    return gen._render_background(guard, has_short=True)


def composite_alerts(gen: PineScriptGenerator, d: CompositeStrategyDefinition) -> str:
    lines: list[str] = []
    lines.append(gen._section_header("ALERTS"))

    sg = " and inSession" if d.use_session_filter else ""

    gen._emit_alert(lines, f"buySignal{sg}",
                    "Buy", f"{d.name}: Buy signal triggered.")
    gen._emit_alert(lines, f"sellSignal{sg}",
                    "Sell", f"{d.name}: Sell signal triggered.")
    gen._emit_alert(lines, f"exitLongSignal{sg}",
                    "Exit Long", f"{d.name}: Exit Long signal.")
    gen._emit_alert(lines, f"exitShortSignal{sg}",
                    "Exit Short", f"{d.name}: Exit Short signal.")

    # Per-mode alerts
    for mode in d.modes:
        sn = mode.short_name
        gen._emit_alert(
            lines, f"({sn}Long or {sn}Short){sg}",
            f"{mode.name} Signal",
            f"{d.name}: {mode.name} signal triggered.",
        )

    # Combined alert
    gen._emit_alert(
        lines,
        f"(buySignal or sellSignal or exitLongSignal or exitShortSignal){sg}",
        "Any Signal",
        f"{d.name}: Signal triggered. Check chart.",
    )
    return "\n".join(lines)


def composite_signal_aggregation(
    gen: PineScriptGenerator, d: CompositeStrategyDefinition,
) -> str:
    """Signal aggregation for composite strategy mode.

    Same as :func:`composite_aggregation` but without the indicator-mode
    state machine.  Sets ``longCondition``, ``shortCondition``, and
    ``activeMode`` for consumption by :func:`emit_strategy_logic`.
    """
    lines: list[str] = []
    lines.append(gen._section_header("SIGNAL AGGREGATION"))

    lines.append("bool longCondition = false")
    lines.append("bool shortCondition = false")
    lines.append('var string activeMode = ""')
    lines.append("")

    if d.aggregation == "first_to_fire":
        order = d.priority_order or [m.short_name for m in d.modes]
        first = True
        for sn in order:
            kw = "if" if first else "else if"
            lines.append(f"{kw} {sn}Long")
            lines.append("    longCondition := true")
            lines.append(f'    activeMode := "{sn.upper()}"')
            first = False
        first = True
        lines.append("")
        for sn in order:
            kw = "if" if first else "else if"
            lines.append(f"{kw} {sn}Short")
            lines.append("    shortCondition := true")
            lines.append(f'    activeMode := "{sn.upper()}"')
            first = False
    else:
        # "any" aggregation
        long_parts = [f"{m.short_name}Long" for m in d.modes]
        short_parts = [f"{m.short_name}Short" for m in d.modes]
        lines.append(
            f"longCondition := {' or '.join(long_parts)}"
        )
        lines.append(
            f"shortCondition := {' or '.join(short_parts)}"
        )
        for m in d.modes:
            sn = m.short_name
            lines.append(f'if {sn}Long or {sn}Short')
            lines.append(f'    activeMode := "{sn.upper()}"')

    return "\n".join(lines)
