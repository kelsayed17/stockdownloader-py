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

    from stockdownloader.util.pinescript.generator import PineScriptGenerator
    from stockdownloader.util.pinescript_models import (
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

from stockdownloader.util.pinescript_models import (
    CompositeStrategyDefinition,
    Indicator,
    Input,
    InputType,
    ModeDefinition,
    SharedInfrastructure,
    StrategyDefinition,
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

    def _composite_header(self, d: CompositeStrategyDefinition) -> str:
        desc = ""
        if d.description:
            for line in d.description.split("\n"):
                desc += f"// {line}\n"

        mode_list = ", ".join(m.short_name.upper() for m in d.modes)
        desc += f"//\n// Modes: {mode_list}\n"
        desc += f"// Aggregation: {d.aggregation}\n"

        return self._render_header(d.name, d.short_name, desc)

    def _composite_inputs(self, d: CompositeStrategyDefinition) -> str:
        lines: list[str] = []

        lines.append(self._section_header("INPUTS"))
        lines.append('bool   showLabels    = input.bool(true, '
                      'title="Show Signal Labels", group="Display")')
        lines.append('bool   showBgColor   = input.bool(true, '
                      'title="Show Background Coloring", group="Display")')

        if d.shared_inputs:
            lines.append("")
            lines.append("// --- Shared ---")
            for inp in d.shared_inputs:
                lines.append(self._input_line(inp))

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
                lines.append(self._input_line(forced))

        return "\n".join(lines)

    def _composite_session(self, d: CompositeStrategyDefinition) -> str:
        if not d.use_session_filter:
            return ""
        return self._render_session(d.session, d.timezone)

    def _composite_shared_indicators(
        self, d: CompositeStrategyDefinition
    ) -> str:
        lines: list[str] = []
        lines.append(self._section_header("SHARED INDICATORS"))
        self._render_indicators(d.shared_indicators, lines)
        return "\n".join(lines)

    def _composite_mode_section(self, mode: ModeDefinition) -> str:
        lines: list[str] = []
        sn = mode.short_name
        toggle = f"enable{sn.upper()}"

        lines.append(self._section_header(f"MODE: {mode.name.upper()}"))

        if mode.description:
            for dline in mode.description.split("\n"):
                lines.append(f"// {dline}")
            lines.append("")

        self._render_indicators(mode.indicators, lines, render_plots=False)

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

    def _composite_aggregation(
        self, d: CompositeStrategyDefinition
    ) -> str:
        lines: list[str] = []
        lines.append(self._section_header("SIGNAL AGGREGATION"))

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

        self._emit_state_machine(
            lines,
            has_long=True,
            has_short=True,
            long_exit_expr="shortCondition",
            short_exit_expr="longCondition",
            use_session_filter=d.use_session_filter,
        )
        return "\n".join(lines)

    def _composite_labels(self, d: CompositeStrategyDefinition) -> str:
        lines: list[str] = []
        lines.append(self._section_header("LABELS"))

        guard = "showLabels"
        if d.use_session_filter:
            guard += " and inSession"

        lines.append(f"if {guard}")
        self._emit_label(lines, "buySignal", "low",
                         'activeMode + " Buy"',
                         "label.style_label_up", "color.green")
        self._emit_label(lines, "sellSignal", "high",
                         'activeMode + " Sell"',
                         "label.style_label_down", "color.red")
        self._emit_label(lines, "exitLongSignal", "high",
                         '"Exit Long"',
                         "label.style_label_down", "color.orange")
        self._emit_label(lines, "exitShortSignal", "low",
                         '"Exit Short"',
                         "label.style_label_up", "color.orange")
        return "\n".join(lines)

    def _composite_background(
        self, d: CompositeStrategyDefinition
    ) -> str:
        guard = "showBgColor"
        if d.use_session_filter:
            guard += " and inSession"
        return self._render_background(guard, has_short=True)

    def _composite_alerts(self, d: CompositeStrategyDefinition) -> str:
        lines: list[str] = []
        lines.append(self._section_header("ALERTS"))

        sg = " and inSession" if d.use_session_filter else ""

        self._emit_alert(lines, f"buySignal{sg}",
                         "Buy", f"{d.name}: Buy signal triggered.")
        self._emit_alert(lines, f"sellSignal{sg}",
                         "Sell", f"{d.name}: Sell signal triggered.")
        self._emit_alert(lines, f"exitLongSignal{sg}",
                         "Exit Long", f"{d.name}: Exit Long signal.")
        self._emit_alert(lines, f"exitShortSignal{sg}",
                         "Exit Short", f"{d.name}: Exit Short signal.")

        # Per-mode alerts
        for mode in d.modes:
            sn = mode.short_name
            self._emit_alert(
                lines, f"({sn}Long or {sn}Short){sg}",
                f"{mode.name} Signal",
                f"{d.name}: {mode.name} signal triggered.",
            )

        # Combined alert
        self._emit_alert(
            lines,
            f"(buySignal or sellSignal or exitLongSignal or exitShortSignal){sg}",
            "Any Signal",
            f"{d.name}: Signal triggered. Check chart.",
        )
        return "\n".join(lines)

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
        cap = int(s.initial_capital)
        comm = s.commission_per_order
        slip = s.slippage
        return f"""\
// This Pine Script(tm) v6 strategy is subject to the terms of the Mozilla Public License 2.0
// https://mozilla.org/MPL/2.0/
//
// {name}
//
{desc_block}
//@version=6
strategy("{name}", shorttitle="{short_name}", overlay=true,
     initial_capital={cap}, default_qty_type=strategy.fixed, default_qty_value=1,
     commission_type=strategy.commission.cash_per_order, commission_value={comm},
     slippage={slip}, process_orders_on_close=false, calc_on_every_tick=false,
     margin_long=100, margin_short=100)"""

    def _emit_strategy_logic(self, s: StrategyDefinition) -> str:
        """Emit strategy entry/exit logic with position sizing & risk management."""
        has_long = s.long_entry is not None
        has_short = s.short_entry is not None

        # Build exit expressions
        if s.long_exit:
            long_exit_expr = s.long_exit.expr
        elif s.exit_on_reverse and has_short:
            long_exit_expr = "shortCondition"
        else:
            long_exit_expr = "false"

        if s.short_exit:
            short_exit_expr = s.short_exit.expr
        elif s.exit_on_reverse and has_long:
            short_exit_expr = "longCondition"
        else:
            short_exit_expr = "false"

        cap_val = int(s.initial_capital)
        sn = s.short_name

        lines: list[str] = []

        # --- Session ---
        lines.append(self._section_header("SESSION"))
        lines.append("bool isMarket = session.ismarket")
        lines.append("bool isNewDay = timeframe.change(\"D\")")
        lines.append("int  barOfDay = ta.barssince(isNewDay) + 1")

        # --- Position sizing function ---
        lines.append(self._section_header("POSITION SIZING"))
        cap_expr = (f"{cap_val}.0"
                    if s.use_fixed_capital
                    else "strategy.equity")
        lines.append(f"""\
f_qty(float entry, float stop) =>
    float cap    = {cap_expr}
    float risk   = cap * ({s.risk_per_trade_pct} / 100.0)
    float perSh  = math.abs(entry - stop)
    float raw    = perSh > 0 ? risk / perSh : 1.0
    float maxQty = entry > 0 ? cap / entry * 0.95 : 1.0
    math.max(math.min(raw, maxQty), 1.0)""")

        # --- Risk management state ---
        lines.append(self._section_header("RISK MANAGEMENT"))
        lines.append("var int   dayTrades  = 0")
        lines.append(f"var int   lastBarEntry = 0")
        lines.append("var int   consecLoss = 0")
        lines.append("var bool  tripped    = false")
        lines.append("var float dayEquity  = 0.0")
        lines.append("var bool  dayLimited = false")
        lines.append("")
        lines.append("if isNewDay")
        lines.append("    dayTrades  := 0")
        lines.append("    lastBarEntry := 0")
        lines.append("    consecLoss := 0")
        lines.append("    tripped    := false")
        lines.append("    dayEquity  := strategy.equity")
        lines.append("    dayLimited := false")
        lines.append("")
        lines.append(
            f"if dayEquity > 0 and "
            f"(strategy.equity - dayEquity) / dayEquity * 100"
            f" <= -{s.daily_loss_limit_pct}"
        )
        lines.append("    dayLimited := true")
        lines.append("")
        lines.append(
            f"bool ready  = isMarket and dayTrades < {s.max_trades_per_day}"
            f" and not tripped and not dayLimited"
        )
        lines.append(
            f"bool spaced = (bar_index - lastBarEntry) >= {s.min_bars_between}"
            f" or lastBarEntry == 0"
        )

        # --- Signal conditions ---
        lines.append(self._section_header("SIGNAL LOGIC"))
        if has_long:
            lines.append(f"bool longCondition  = {s.long_entry.expr}")
        if has_short:
            lines.append(f"bool shortCondition = {s.short_entry.expr}")
        lines.append("")
        lines.append(
            f"bool exitLongCond  = strategy.position_size > 0"
            f" and ({long_exit_expr})"
        )
        if has_short:
            lines.append(
                f"bool exitShortCond = strategy.position_size < 0"
                f" and ({short_exit_expr})"
            )

        # --- SL / TP ---
        lines.append(self._section_header("SL / TP CALCULATION"))
        lines.append("float atrVal   = ta.atr(14)")
        lines.append(
            f"float slRisk   = math.min(atrVal * {s.sl_atr_mult},"
            f" {s.sl_cap_dollars})"
        )
        lines.append("float longSL   = close - slRisk")
        lines.append(f"float longTP   = close + slRisk * {s.rr_ratio}")
        if has_short:
            lines.append("float shortSL  = close + slRisk")
            lines.append(f"float shortTP  = close - slRisk * {s.rr_ratio}")

        # --- Execution ---
        lines.append(self._section_header("EXECUTION"))

        confirmed = " and barstate.isconfirmed" if s.barstate_confirmed else ""
        long_guard = f"longCondition and ready and spaced{confirmed}"
        short_guard = f"shortCondition and ready and spaced{confirmed}"

        if has_long:
            lines.append(f"bool goLong  = {long_guard}")
        if has_short:
            lines.append(f"bool goShort = {short_guard}")
        lines.append("")

        # Long entry
        if has_long:
            lines.append("if goLong")
            if has_short:
                lines.append("    if strategy.position_size < 0")
                lines.append('        strategy.close("S", comment="Flip")')
            lines.append("    float qty = f_qty(close, longSL)")
            lines.append(
                f'    strategy.entry("L", strategy.long, qty=qty,'
                f' comment="{sn}|L")'
            )
            lines.append(
                '    strategy.exit("LX", "L", stop=longSL, limit=longTP)'
            )
            lines.append("    dayTrades    += 1")
            lines.append("    lastBarEntry := bar_index")
            lines.append(
                f'    alert("{s.name}: LONG'
                f' | E " + str.tostring(close, format.mintick)'
                f' + " | SL " + str.tostring(longSL, format.mintick)'
                f' + " | TP " + str.tostring(longTP, format.mintick)'
                f' + " | Qty " + str.tostring(qty, "#")'
                f', alert.freq_once_per_bar_close)'
            )
            # Label
            lines.append('    label.new(bar_index, low, "Buy",')
            lines.append("         style=label.style_label_up,")
            lines.append("         color=color.green,")
            lines.append("         textcolor=color.white,")
            lines.append("         size=size.small)")
            lines.append("")

        # Short entry
        if has_short:
            lines.append("if goShort")
            if has_long:
                lines.append("    if strategy.position_size > 0")
                lines.append('        strategy.close("L", comment="Flip")')
            lines.append("    float qty = f_qty(close, shortSL)")
            lines.append(
                f'    strategy.entry("S", strategy.short, qty=qty,'
                f' comment="{sn}|S")'
            )
            lines.append(
                '    strategy.exit("SX", "S", stop=shortSL, limit=shortTP)'
            )
            lines.append("    dayTrades    += 1")
            lines.append("    lastBarEntry := bar_index")
            lines.append(
                f'    alert("{s.name}: SHORT'
                f' | E " + str.tostring(close, format.mintick)'
                f' + " | SL " + str.tostring(shortSL, format.mintick)'
                f' + " | TP " + str.tostring(shortTP, format.mintick)'
                f' + " | Qty " + str.tostring(qty, "#")'
                f', alert.freq_once_per_bar_close)'
            )
            lines.append('    label.new(bar_index, high, "Sell",')
            lines.append("         style=label.style_label_down,")
            lines.append("         color=color.red,")
            lines.append("         textcolor=color.white,")
            lines.append("         size=size.small)")
            lines.append("")

        # Strategy exits (condition-based, not TP/SL)
        lines.append("// Condition-based exits (strategy SL/TP handled by strategy.exit)")
        lines.append("if exitLongCond")
        lines.append('    strategy.close("L", comment="ExitLong")')
        lines.append('    label.new(bar_index, high, "Exit Long",')
        lines.append("         style=label.style_label_down,")
        lines.append("         color=color.orange,")
        lines.append("         textcolor=color.white,")
        lines.append("         size=size.small)")
        if has_short:
            lines.append("if exitShortCond")
            lines.append('    strategy.close("S", comment="ExitShort")')
            lines.append('    label.new(bar_index, low, "Exit Short",')
            lines.append("         style=label.style_label_up,")
            lines.append("         color=color.orange,")
            lines.append("         textcolor=color.white,")
            lines.append("         size=size.small)")

        # --- Exit management: BE + fill alignment ---
        lines.append(self._section_header(
            "EXIT MANAGEMENT \u2014 BREAKEVEN + FILL ALIGNMENT"
        ))
        lines.append("var float entryPrice = na")
        lines.append("var float origSL     = na")
        lines.append("var float pendingTP  = na")
        lines.append("var bool  movedBE    = false")
        lines.append("float beBuf = 0.05")
        lines.append("")
        lines.append("// Position open \u2014 capture fill price and realign exits")
        lines.append(
            "if strategy.position_size != 0 and strategy.position_size[1] == 0"
        )
        lines.append("    entryPrice := strategy.position_avg_price")
        lines.append("    movedBE    := false")
        lines.append("    if strategy.position_size > 0")
        lines.append(
            f"        float _slDist = slRisk"
        )
        lines.append("        origSL    := entryPrice - _slDist")
        lines.append(
            f"        pendingTP := entryPrice + _slDist * {s.rr_ratio}"
        )
        lines.append(
            '        strategy.exit("LX", "L", stop=origSL, limit=pendingTP)'
        )
        if has_short:
            lines.append("    else")
            lines.append(
                f"        float _slDist = slRisk"
            )
            lines.append("        origSL    := entryPrice + _slDist")
            lines.append(
                f"        pendingTP := entryPrice - _slDist * {s.rr_ratio}"
            )
            lines.append(
                '        strategy.exit("SX", "S", stop=origSL, limit=pendingTP)'
            )
        lines.append("")

        # BE trigger -- long
        lines.append(f"// Breakeven trigger at {s.be_trigger}R")
        lines.append(
            "if strategy.position_size > 0 and not movedBE"
            " and not na(entryPrice) and not na(origSL)"
        )
        lines.append("    float riskAmt = entryPrice - origSL")
        lines.append(
            f"    if riskAmt > 0 and close >= entryPrice + riskAmt * {s.be_trigger}"
        )
        lines.append(
            '        strategy.exit("LX", "L",'
            " stop=entryPrice + beBuf, limit=pendingTP,"
            ' comment="BE")'
        )
        lines.append("        movedBE := true")
        lines.append(
            f'        alert("{s.name}: BE LONG'
            f' | SL \u2192 " + str.tostring(entryPrice + beBuf, format.mintick)'
            f', alert.freq_once_per_bar_close)'
        )

        # BE trigger -- short
        if has_short:
            lines.append("")
            lines.append(
                "if strategy.position_size < 0 and not movedBE"
                " and not na(entryPrice) and not na(origSL)"
            )
            lines.append("    float riskAmt = origSL - entryPrice")
            lines.append(
                f"    if riskAmt > 0 and close <= entryPrice - riskAmt * {s.be_trigger}"
            )
            lines.append(
                '        strategy.exit("SX", "S",'
                " stop=entryPrice - beBuf, limit=pendingTP,"
                ' comment="BE")'
            )
            lines.append("        movedBE := true")
            lines.append(
                f'        alert("{s.name}: BE SHORT'
                f' | SL \u2192 " + str.tostring(entryPrice - beBuf, format.mintick)'
                f', alert.freq_once_per_bar_close)'
            )

        # Position closed -- reset
        lines.append("")
        lines.append("if strategy.position_size == 0")
        lines.append("    movedBE := false")

        # EOD close
        if s.close_eod:
            lines.append(self._section_header("EOD CLOSE"))
            lines.append(
                "if not isMarket and isMarket[1]"
                " and strategy.position_size != 0"
            )
            lines.append('    strategy.close_all(comment="EOD")')
            lines.append(
                f'    alert("{s.name}: EOD CLOSE'
                f' | " + str.tostring(close, format.mintick)'
                f', alert.freq_once_per_bar_close)'
            )

        # Loss tracking + circuit breaker
        lines.append(self._section_header("LOSS TRACKING"))
        lines.append(
            "if strategy.position_size == 0 and strategy.position_size[1] != 0"
        )
        lines.append(
            "    float pnl = strategy.netprofit - nz(strategy.netprofit[1])"
        )
        lines.append("    if pnl > 0")
        lines.append("        consecLoss := 0")
        lines.append("    else")
        lines.append("        consecLoss += 1")
        lines.append(f"        if consecLoss >= {s.circuit_breaker_losses}")
        lines.append("            tripped := true")
        lines.append(
            f'            alert("{s.name}: CIRCUIT BREAKER'
            f' | " + str.tostring(consecLoss) + " losses"'
            f', alert.freq_once_per_bar_close)'
        )

        # Visual SL/TP/Entry lines
        lines.append(self._section_header("LEVEL PLOTS"))
        lines.append(
            "var float plotEntry = na\n"
            "var float plotSL    = na\n"
            "var float plotTP    = na"
        )
        lines.append("")
        lines.append(
            "if strategy.position_size != 0 and strategy.position_size[1] == 0"
        )
        lines.append("    plotEntry := entryPrice")
        lines.append("    plotSL    := origSL")
        lines.append("    plotTP    := pendingTP")
        lines.append("if strategy.position_size == 0")
        lines.append("    plotEntry := na")
        lines.append("    plotSL    := na")
        lines.append("    plotTP    := na")
        lines.append("")
        lines.append(
            'plot(plotEntry, title="Entry Price",'
            " color=color.white, style=plot.style_linebr, linewidth=2)"
        )
        lines.append(
            'plot(plotTP, title="Take Profit",'
            " color=color.new(color.lime, 20),"
            " style=plot.style_linebr, linewidth=2)"
        )
        lines.append(
            'plot(plotSL, title="Stop Loss",'
            " color=color.new(color.red, 20),"
            " style=plot.style_linebr, linewidth=2)"
        )

        return "\n".join(lines)

    def _render_strategy_background(self, has_short: bool) -> str:
        """Background coloring using strategy.position_size."""
        lines: list[str] = []
        lines.append(self._section_header("BACKGROUND"))
        lines.append("color bgColor = na")
        lines.append("")
        lines.append("if strategy.position_size > 0")
        lines.append("    bgColor := color.new(color.green, 90)")
        if has_short:
            lines.append("else if strategy.position_size < 0")
            lines.append("    bgColor := color.new(color.red, 90)")
        lines.append("")
        lines.append('bgcolor(bgColor, title="Position Zone")')
        return "\n".join(lines)

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
        lines.append("")
        lines.append("// Position state: +1 = long, -1 = short, 0 = flat")
        lines.append("var int posState = 0")
        lines.append("")

        if has_long:
            lines.append("bool buySignal      = longCondition and posState != 1")
        if has_short:
            lines.append(
                "bool sellSignal     = shortCondition and posState != -1"
            )
        lines.append(
            f"bool exitLongSignal  = posState == 1 and ({long_exit_expr})"
        )
        if has_short:
            lines.append(
                f"bool exitShortSignal = posState == -1 and ({short_exit_expr})"
            )

        lines.append("")
        lines.append("// Update position state")
        if has_long:
            lines.append("if buySignal")
            lines.append("    posState := 1")
        if has_short:
            joiner = "else if" if has_long else "if"
            lines.append(f"{joiner} sellSignal")
            lines.append("    posState := -1")

        exit_cond = "exitLongSignal"
        if has_short:
            exit_cond += " or exitShortSignal"
        joiner = "else if" if (has_long or has_short) else "if"
        lines.append(f"{joiner} {exit_cond}")
        lines.append("    posState := 0")

        if use_session_filter:
            lines.append("")
            lines.append("if isNewSession")
            lines.append("    posState := 0")

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
