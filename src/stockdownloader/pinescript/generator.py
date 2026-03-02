"""Automated Pine Script v6 generator for trading strategies.

Converts Python strategy definitions into production-ready TradingView
Pine Script v6 indicators with:
- Configurable inputs for all parameters
- Signal labels (Buy/Sell/Exit Long/Exit Short)
- Alert conditions for each signal type
- Background coloring for position state
- Optional session-anchored VWAP overlay

Pre-built strategies live in :mod:`pinescript_strategies` (standalone)
and :mod:`pinescript_composites` (multi-mode).

Usage::

    from stockdownloader.pinescript.generator import PineScriptGenerator
    from stockdownloader.pinescript.models import (
        StrategyDefinition, Indicator, Condition, Input,
    )

    strategy = StrategyDefinition(
        name="SMA Crossover",
        short_name="SMA-X",
        indicators=[
            Indicator.sma("smaFast", "close", "fastPeriod"),
            Indicator.sma("smaSlow", "close", "slowPeriod"),
        ],
        inputs=[
            Input.int_("fastPeriod", 10, "Fast SMA Period"),
            Input.int_("slowPeriod", 20, "Slow SMA Period"),
        ],
        long_entry=Condition("ta.crossover(smaFast, smaSlow)"),
        long_exit=Condition("ta.crossunder(smaFast, smaSlow)"),
    )

    gen = PineScriptGenerator()
    pine = gen.generate(strategy)
"""

from __future__ import annotations

from stockdownloader.pinescript.models import (
    CompositeStrategyDefinition,
    Condition,
    Indicator,
    Input,
    InputType,
    ModeDefinition,
    SharedInfrastructure,
    StrategyDefinition,
)
from stockdownloader.pinescript.composite_renderer import (
    composite_aggregation,
    composite_alerts,
    composite_background,
    composite_header,
    composite_inputs,
    composite_labels,
    composite_mode_section,
    composite_session,
    composite_shared_indicators,
    composite_signal_aggregation,
)
from stockdownloader.pinescript.strategy_renderer import (
    emit_state_machine,
    emit_strategy_logic,
    render_strategy_background,
    render_strategy_header,
)


# ======================================================================
# Generator
# ======================================================================


class PineScriptGenerator:
    """Generates Pine Script v6 indicators from strategy definitions."""

    # ------------------------------------------------------------------
    # Standalone strategy generation
    # ------------------------------------------------------------------

    def generate(self, strategy: StrategyDefinition) -> str:
        """Generate complete Pine Script v6 code for *strategy*."""
        parts: list[str] = []
        parts.append(self._header(strategy))
        parts.append(self._inputs(strategy))

        if strategy.use_session_filter:
            parts.append(self._render_session(strategy.session, strategy.timezone))

        parts.append(self._indicators(strategy))

        if strategy.extra_code:
            parts.append(self._section("EXTRA COMPUTATIONS",
                                       "\n".join(strategy.extra_code)))

        parts.append(self._signals(strategy))
        parts.append(self._labels(strategy))
        parts.append(self._background(strategy))
        parts.append(self._alerts(strategy))

        if strategy.extra_plots:
            parts.append(self._section("ADDITIONAL PLOTS",
                                       "\n".join(strategy.extra_plots)))

        return "\n".join(parts) + "\n"

    def _header(self, s: StrategyDefinition) -> str:
        desc_lines = ""
        if s.description:
            for line in s.description.split("\n"):
                desc_lines += f"// {line}\n"

        long_desc = s.long_entry.description if s.long_entry else ""
        short_desc = s.short_entry.description if s.short_entry else ""
        entry_block = ""
        if long_desc or short_desc:
            entry_block = "//\n// Entry Logic:\n"
            if long_desc:
                entry_block += f"//   Long  - {long_desc}\n"
            if short_desc:
                entry_block += f"//   Short - {short_desc}\n"

        if s.strategy_mode:
            return self._render_strategy_header(
                s.name, s.short_name, f"{desc_lines}{entry_block}", s,
            )
        return self._render_header(s.name, s.short_name, f"{desc_lines}{entry_block}")

    def _inputs(self, s: StrategyDefinition) -> str:
        if not s.inputs:
            return ""
        lines: list[str] = []
        lines.append(self._section_header("INPUTS"))

        current_group = ""
        for inp in s.inputs:
            if inp.group and inp.group != current_group:
                current_group = inp.group
                lines.append(f"\n// --- {current_group} ---")
            lines.append(self._input_line(inp))

        if s.use_session_filter:
            lines.append("")
            lines.append('bool   showLabels    = input.bool(true, title="Show Signal Labels")')
            lines.append('bool   showBgColor   = input.bool(true, title="Show Background Coloring")')

        return "\n".join(lines)

    def _indicators(self, s: StrategyDefinition) -> str:
        lines: list[str] = []
        lines.append(self._section_header("INDICATORS"))
        self._render_indicators(s.indicators, lines)
        return "\n".join(lines)

    def _signals(self, s: StrategyDefinition) -> str:
        if s.strategy_mode:
            return self._emit_strategy_logic(s)

        lines: list[str] = []
        lines.append(self._section_header("SIGNAL LOGIC"))

        has_long = s.long_entry is not None
        has_short = s.short_entry is not None

        if has_long:
            lines.append(f"bool longCondition = {s.long_entry.expr}")
        if has_short:
            lines.append(f"bool shortCondition = {s.short_entry.expr}")

        # Build exit expressions
        if s.long_exit:
            long_exit_expr = s.long_exit.expr
        elif s.exit_on_reverse and has_short:
            long_exit_expr = "shortCondition"
        else:
            long_exit_expr = f"not longCondition" if has_long else "false"

        if s.short_exit:
            short_exit_expr = s.short_exit.expr
        elif s.exit_on_reverse and has_long:
            short_exit_expr = "longCondition"
        else:
            short_exit_expr = f"not shortCondition" if has_short else "false"

        self._emit_state_machine(
            lines,
            has_long=has_long,
            has_short=has_short,
            long_exit_expr=long_exit_expr,
            short_exit_expr=short_exit_expr,
            use_session_filter=s.use_session_filter,
        )
        return "\n".join(lines)

    def _labels(self, s: StrategyDefinition) -> str:
        if s.strategy_mode:
            # In strategy mode, labels are emitted inline with execution.
            return ""

        lines: list[str] = []
        lines.append(self._section_header("LABELS"))

        has_long = s.long_entry is not None
        has_short = s.short_entry is not None

        guard = "showLabels" if s.use_session_filter else "true"
        if s.use_session_filter:
            guard += " and inSession"

        lines.append(f"if {guard}")
        if has_long:
            self._emit_label(lines, "buySignal", "low", f'"{s.long_label}"',
                             "label.style_label_up", "color.green")
        if has_short:
            self._emit_label(lines, "sellSignal", "high", f'"{s.short_label}"',
                             "label.style_label_down", "color.red")
        self._emit_label(lines, "exitLongSignal", "high", '"Exit Long"',
                         "label.style_label_down", "color.orange")
        if has_short:
            self._emit_label(lines, "exitShortSignal", "low", '"Exit Short"',
                             "label.style_label_up", "color.orange")
        return "\n".join(lines)

    def _background(self, s: StrategyDefinition) -> str:
        if s.strategy_mode:
            return self._render_strategy_background(
                has_short=s.short_entry is not None,
            )
        guard = "showBgColor" if s.use_session_filter else "true"
        if s.use_session_filter:
            guard += " and inSession"
        return self._render_background(guard, has_short=s.short_entry is not None)

    def _alerts(self, s: StrategyDefinition) -> str:
        if s.strategy_mode:
            # In strategy mode, alerts are inline with entry/exit execution.
            return ""

        lines: list[str] = []
        lines.append(self._section_header("ALERTS"))

        has_long = s.long_entry is not None
        has_short = s.short_entry is not None
        sg = " and inSession" if s.use_session_filter else ""

        if has_long:
            self._emit_alert(lines, f"buySignal{sg}",
                             s.long_label, f"{s.name}: {s.long_label} signal triggered.")
        if has_short:
            self._emit_alert(lines, f"sellSignal{sg}",
                             s.short_label, f"{s.name}: {s.short_label} signal triggered.")
        self._emit_alert(lines, f"exitLongSignal{sg}",
                         "Exit Long", f"{s.name}: Exit Long signal.")
        if has_short:
            self._emit_alert(lines, f"exitShortSignal{sg}",
                             "Exit Short", f"{s.name}: Exit Short signal.")

        # Combined alert
        all_sigs = []
        if has_long:
            all_sigs.append("buySignal")
        if has_short:
            all_sigs.append("sellSignal")
        all_sigs.append("exitLongSignal")
        if has_short:
            all_sigs.append("exitShortSignal")

        combined = " or ".join(all_sigs)
        self._emit_alert(lines, f"({combined}){sg}",
                         "Any Signal", f"{s.name}: Signal triggered. Check chart.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Composite strategy generation
    # ------------------------------------------------------------------

    def generate_composite(self, defn: CompositeStrategyDefinition) -> str:
        """Generate Pine Script v6 with per-mode toggle inputs."""
        if defn.strategy_mode:
            return self._generate_composite_strategy(defn)

        parts: list[str] = []
        parts.append(self._composite_header(defn))
        parts.append(self._composite_inputs(defn))
        parts.append(self._composite_session(defn))
        parts.append(self._composite_shared_indicators(defn))

        if defn.shared_code:
            parts.append(self._section("SHARED COMPUTATIONS",
                                       "\n".join(defn.shared_code)))

        for mode in defn.modes:
            parts.append(self._composite_mode_section(mode))

        parts.append(self._composite_aggregation(defn))
        parts.append(self._composite_labels(defn))
        parts.append(self._composite_background(defn))
        parts.append(self._composite_alerts(defn))

        if defn.extra_plots:
            parts.append(self._section("ADDITIONAL PLOTS",
                                       "\n".join(defn.extra_plots)))

        return "\n".join(parts) + "\n"

    def _generate_composite_strategy(
        self, defn: CompositeStrategyDefinition,
    ) -> str:
        """Generate composite Pine Script in strategy() mode."""
        # Build a temporary StrategyDefinition for the strategy renderer
        temp = StrategyDefinition(
            name=defn.name,
            short_name=defn.short_name,
            strategy_mode=True,
            initial_capital=defn.initial_capital,
            commission_per_order=defn.commission_per_order,
            slippage=defn.slippage,
            risk_per_trade_pct=defn.risk_per_trade_pct,
            use_fixed_capital=defn.use_fixed_capital,
            sl_atr_mult=defn.sl_atr_mult,
            sl_cap_dollars=defn.sl_cap_dollars,
            rr_ratio=defn.rr_ratio,
            be_trigger=defn.be_trigger,
            max_trades_per_day=defn.max_trades_per_day,
            min_bars_between=defn.min_bars_between,
            circuit_breaker_losses=defn.circuit_breaker_losses,
            daily_loss_limit_pct=defn.daily_loss_limit_pct,
            close_eod=defn.close_eod,
            barstate_confirmed=defn.barstate_confirmed,
            # Composites always support both directions
            long_entry=Condition("longCondition"),
            short_entry=Condition("shortCondition"),
            long_exit=Condition("shortCondition"),
            short_exit=Condition("longCondition"),
            exit_on_reverse=False,
        )

        # Build description block
        desc = ""
        if defn.description:
            for line in defn.description.split("\n"):
                desc += f"// {line}\n"
        mode_list = ", ".join(m.short_name.upper() for m in defn.modes)
        desc += f"//\n// Modes: {mode_list}\n"
        desc += f"// Aggregation: {defn.aggregation}\n"

        parts: list[str] = []
        parts.append(render_strategy_header(
            defn.name, defn.short_name, desc, temp,
        ))
        parts.append(composite_inputs(self, defn, strategy_mode=True))
        parts.append(composite_session(self, defn))
        parts.append(composite_shared_indicators(self, defn))

        if defn.shared_code:
            parts.append(self._section("SHARED COMPUTATIONS",
                                       "\n".join(defn.shared_code)))

        for mode in defn.modes:
            parts.append(composite_mode_section(self, mode))

        # Signal aggregation (no state machine — strategy handles state)
        parts.append(composite_signal_aggregation(self, defn))

        # Strategy execution (skip signal declarations — already aggregated)
        parts.append(emit_strategy_logic(
            temp, self._section_header,
            skip_signals=True, mode_var="activeMode",
        ))

        # Background coloring
        parts.append(render_strategy_background(True, self._section_header))

        if defn.extra_plots:
            parts.append(self._section("ADDITIONAL PLOTS",
                                       "\n".join(defn.extra_plots)))

        return "\n".join(parts) + "\n"

    def _composite_header(self, d: CompositeStrategyDefinition) -> str:
        return composite_header(self, d)

    def _composite_inputs(self, d: CompositeStrategyDefinition) -> str:
        return composite_inputs(self, d)

    def _composite_session(self, d: CompositeStrategyDefinition) -> str:
        return composite_session(self, d)

    def _composite_shared_indicators(
        self, d: CompositeStrategyDefinition,
    ) -> str:
        return composite_shared_indicators(self, d)

    def _composite_mode_section(self, mode: ModeDefinition) -> str:
        return composite_mode_section(self, mode)

    def _composite_aggregation(
        self, d: CompositeStrategyDefinition,
    ) -> str:
        return composite_aggregation(self, d)

    def _composite_labels(self, d: CompositeStrategyDefinition) -> str:
        return composite_labels(self, d)

    def _composite_background(
        self, d: CompositeStrategyDefinition,
    ) -> str:
        return composite_background(self, d)

    def _composite_alerts(self, d: CompositeStrategyDefinition) -> str:
        return composite_alerts(self, d)

    # ------------------------------------------------------------------
    # Shared helpers (eliminate duplication between standalone/composite)
    # ------------------------------------------------------------------

    def _emit_label(
        self,
        lines: list[str],
        condition: str,
        position: str,
        text: str,
        style: str,
        color: str,
    ) -> None:
        """Append a Pine Script label block for a signal condition."""
        lines.append(f'    if {condition}')
        lines.append(f'        label.new(bar_index, {position}, {text},')
        lines.append(f'             style={style},')
        lines.append(f'             color={color},')
        lines.append(f'             textcolor=color.white,')
        lines.append(f'             size=size.small)')

    def _emit_alert(
        self,
        lines: list[str],
        expr: str,
        title: str,
        message: str,
    ) -> None:
        """Append a Pine Script alertcondition block."""
        lines.append(
            f'alertcondition({expr},\n'
            f'     title="{title}",\n'
            f'     message="{message}")'
        )

    # ------------------------------------------------------------------
    # Strategy mode rendering
    # ------------------------------------------------------------------

    def _render_strategy_header(
        self, name: str, short_name: str, desc_block: str,
        s: StrategyDefinition,
    ) -> str:
        """Generate a strategy() header instead of indicator()."""
        return render_strategy_header(name, short_name, desc_block, s)

    def _emit_strategy_logic(self, s: StrategyDefinition) -> str:
        """Emit strategy entry/exit logic with position sizing & risk management."""
        return emit_strategy_logic(s, self._section_header)

    def _render_strategy_background(self, has_short: bool) -> str:
        """Background coloring using strategy.position_size."""
        return render_strategy_background(has_short, self._section_header)

    def _emit_state_machine(
        self,
        lines: list[str],
        *,
        has_long: bool,
        has_short: bool,
        long_exit_expr: str,
        short_exit_expr: str,
        use_session_filter: bool,
    ) -> None:
        """Append the position state machine (posState, buy/sell/exit signals)."""
        emit_state_machine(
            lines,
            has_long=has_long,
            has_short=has_short,
            long_exit_expr=long_exit_expr,
            short_exit_expr=short_exit_expr,
            use_session_filter=use_session_filter,
        )

    # ------------------------------------------------------------------
    # Rendering primitives
    # ------------------------------------------------------------------

    def _input_line(self, inp: Input) -> str:
        """Generate a single input declaration."""
        parts: list[str] = []
        if inp.input_type == InputType.INT:
            parts.append(f'int    {inp.name:16s} = input.int({inp.default}')
            parts.append(f'title="{inp.title}"')
            if inp.min_val is not None:
                parts.append(f'minval={inp.min_val}')
            if inp.max_val is not None:
                parts.append(f'maxval={inp.max_val}')
        elif inp.input_type == InputType.FLOAT:
            parts.append(f'float  {inp.name:16s} = input.float({inp.default}')
            parts.append(f'title="{inp.title}"')
            if inp.min_val is not None:
                parts.append(f'minval={inp.min_val}')
            if inp.step is not None:
                parts.append(f'step={inp.step}')
        elif inp.input_type == InputType.BOOL:
            val = "true" if inp.default else "false"
            parts.append(f'bool   {inp.name:16s} = input.bool({val}')
            parts.append(f'title="{inp.title}"')
        elif inp.input_type == InputType.STRING:
            parts.append(f'string {inp.name:16s} = input.string("{inp.default}"')
            parts.append(f'title="{inp.title}"')
        return ", ".join(parts) + ")"

    def _render_indicators(
        self,
        indicators: list[Indicator],
        lines: list[str],
        *,
        render_plots: bool = True,
    ) -> None:
        """Append Pine Script lines for a list of indicators."""
        for ind in indicators:
            if ind.code == "__SESSION_VWAP__":
                lines.append(self._session_vwap_code(ind.var_name))
                continue

            if ind.var_name.startswith("["):
                lines.append(f"{ind.var_name} = {ind.code}")
            else:
                tp = self._infer_type(ind.code)
                lines.append(f"{tp} {ind.var_name} = {ind.code}")

            if render_plots and ind.plot:
                color = ind.plot_color or "color.blue"
                title = ind.plot_title or ind.var_name
                disp = ""
                if ind.display == "data_window":
                    disp = ", display=display.data_window"
                    color = f"color.new({color}, 100)"
                lines.append(
                    f'plot({ind.var_name}, title="{title}", '
                    f'color={color}, linewidth={ind.linewidth}{disp})'
                )

            if render_plots and ind.display == "data_window" and not ind.plot:
                color = ind.plot_color or "color.gray"
                title = ind.plot_title or ind.var_name
                lines.append(
                    f'plot({ind.var_name}, title="{title}", '
                    f'color=color.new({color}, 100), '
                    f'display=display.data_window)'
                )

    def _session_vwap_code(self, var_name: str) -> str:
        return f"""\

// Session-Anchored VWAP
float tp = (high + low + close) / 3.0
var float cumTPV = 0.0
var float cumVol = 0.0

if isNewSession
    cumTPV := tp * volume
    cumVol := volume
else
    cumTPV := cumTPV + tp * volume
    cumVol := cumVol + volume

float {var_name} = cumVol != 0 ? cumTPV / cumVol : close
plot(inSession ? {var_name} : na, title="VWAP", color=color.blue, linewidth=2)
"""

    def _render_session(self, session: str, timezone: str) -> str:
        """Generate the session-detection block."""
        return f"""\

{self._section_header("SESSION DETECTION")}

string SESSION  = "{session}"
string TIMEZONE = "{timezone}"

bool inSession = not na(time(timeframe.period, SESSION, TIMEZONE))
bool isNewSession = inSession and not inSession[1]"""

    def _render_background(self, guard: str, has_short: bool) -> str:
        """Generate background coloring block."""
        lines: list[str] = []
        lines.append(self._section_header("BACKGROUND"))
        lines.append("color bgColor = na")
        lines.append("")
        lines.append(f"if {guard}")
        lines.append("    if posState == 1")
        lines.append("        bgColor := color.new(color.green, 90)")
        if has_short:
            lines.append("    else if posState == -1")
            lines.append("        bgColor := color.new(color.red, 90)")
        lines.append("")
        lines.append('bgcolor(bgColor, title="Position Zone")')
        return "\n".join(lines)

    def _render_header(self, name: str, short_name: str, desc_block: str) -> str:
        """Generate the script header."""
        return f"""\
// This Pine Script(tm) v6 indicator is subject to the terms of the Mozilla Public License 2.0
// https://mozilla.org/MPL/2.0/
//
// {name}
//
{desc_block}
//@version=6
indicator("{name}", shorttitle="{short_name}", overlay=true, max_labels_count=500)"""

    @staticmethod
    def _infer_type(code: str) -> str:
        """Infer Pine Script type prefix from an expression."""
        bool_ops = (" > ", " < ", " >= ", " <= ", " == ", " != ",
                     " and ", " or ", "not ")
        if any(op in code for op in bool_ops):
            return "bool"
        return "float"

    @staticmethod
    def _section_header(title: str) -> str:
        bar = "\u2500" * 77
        return f"\n// {bar}\n// {title}\n// {bar}\n"

    @staticmethod
    def _section(title: str, body: str) -> str:
        bar = "\u2500" * 77
        return f"\n// {bar}\n// {title}\n// {bar}\n\n{body}"


# ======================================================================
# Strategy/Mode converters (formerly converters.py)
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
    strategy_mode: bool = False,
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
        strategy_mode=strategy_mode,
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
