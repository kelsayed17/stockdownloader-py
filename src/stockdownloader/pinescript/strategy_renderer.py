"""Strategy-mode rendering for PineScriptGenerator.

Provides the strategy() header, execution logic, risk management,
position sizing, breakeven triggers, and state machine that power
backtesting mode in generated Pine Script v6 strategies.
"""

from __future__ import annotations

from collections.abc import Callable

from stockdownloader.pinescript.models import StrategyDefinition


def render_strategy_header(
    name: str, short_name: str, desc_block: str, s: StrategyDefinition,
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


def emit_strategy_logic(
    s: StrategyDefinition, section_header_fn: Callable[[str], str],
) -> str:
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
    lines.append(section_header_fn("SESSION"))
    lines.append("bool isMarket = session.ismarket")
    lines.append("bool isNewDay = timeframe.change(\"D\")")
    lines.append("int  barOfDay = ta.barssince(isNewDay) + 1")

    # --- Position sizing function ---
    lines.append(section_header_fn("POSITION SIZING"))
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
    lines.append(section_header_fn("RISK MANAGEMENT"))
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
    lines.append(section_header_fn("SIGNAL LOGIC"))
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
    lines.append(section_header_fn("SL / TP CALCULATION"))
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
    lines.append(section_header_fn("EXECUTION"))

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
    lines.append(section_header_fn(
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
        lines.append(section_header_fn("EOD CLOSE"))
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
    lines.append(section_header_fn("LOSS TRACKING"))
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
    lines.append(section_header_fn("LEVEL PLOTS"))
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


def render_strategy_background(
    has_short: bool, section_header_fn: Callable[[str], str],
) -> str:
    """Background coloring using strategy.position_size."""
    lines: list[str] = []
    lines.append(section_header_fn("BACKGROUND"))
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


def emit_state_machine(
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
