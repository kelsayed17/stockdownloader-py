"""Pine Script v6 data models for strategy definitions.

These frozen dataclasses describe the building blocks used by
:class:`~stockdownloader.util.pinescript_generator.PineScriptGenerator`
to render Pine Script v6 code.

Typical usage::

    from stockdownloader.util.pinescript_models import (
        Condition, Indicator, Input, StrategyDefinition,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ======================================================================
# Building blocks
# ======================================================================

#: Union of types valid for Pine Script input defaults and bounds.
InputValue = int | float | bool | str


class InputType(Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"


@dataclass(frozen=True, slots=True)
class Input:
    """A user-configurable input parameter."""

    name: str
    input_type: InputType
    default: InputValue
    title: str
    min_val: int | float | None = None
    max_val: int | float | None = None
    step: int | float | None = None
    options: list[str] | None = None
    group: str = ""

    @staticmethod
    def int_(name: str, default: int, title: str,
             min_val: int = 1, group: str = "") -> Input:
        return Input(name, InputType.INT, default, title,
                     min_val=min_val, group=group)

    @staticmethod
    def float_(name: str, default: float, title: str,
               min_val: float = 0.0, step: float = 0.5,
               group: str = "") -> Input:
        return Input(name, InputType.FLOAT, default, title,
                     min_val=min_val, step=step, group=group)

    @staticmethod
    def bool_(name: str, default: bool, title: str,
              group: str = "") -> Input:
        return Input(name, InputType.BOOL, default, title, group=group)

    @staticmethod
    def string_(name: str, default: str, title: str,
                options: list[str] | None = None,
                group: str = "") -> Input:
        return Input(name, InputType.STRING, default, title,
                     options=options, group=group)


@dataclass(frozen=True, slots=True)
class Indicator:
    """A Pine Script indicator computation.

    ``code`` is the Pine Script expression that computes the value.
    ``var_name`` is the variable name assigned to the result.
    ``plot`` controls whether the indicator is plotted on the chart.
    """

    var_name: str
    code: str
    plot: bool = False
    plot_color: str = ""
    plot_title: str = ""
    linewidth: int = 1
    display: str = ""  # "data_window" to hide from chart

    # -- Convenience constructors --

    @staticmethod
    def sma(var: str, src: str, period: str, plot: bool = True,
            color: str = "color.blue") -> Indicator:
        return Indicator(var, f"ta.sma({src}, {period})",
                         plot=plot, plot_color=color,
                         plot_title=f"SMA({period})")

    @staticmethod
    def ema(var: str, src: str, period: str, plot: bool = True,
            color: str = "color.orange") -> Indicator:
        return Indicator(var, f"ta.ema({src}, {period})",
                         plot=plot, plot_color=color,
                         plot_title=f"EMA({period})")

    @staticmethod
    def rsi(var: str, src: str, period: str) -> Indicator:
        return Indicator(var, f"ta.rsi({src}, {period})",
                         display="data_window",
                         plot_title="RSI")

    @staticmethod
    def macd(fast: str, slow: str, signal: str,
             var_line: str = "macdLine",
             var_signal: str = "macdSignal",
             var_hist: str = "macdHist") -> list[Indicator]:
        """Return the three MACD indicators as a group."""
        return [
            Indicator(
                var_name=f"[{var_line}, {var_signal}, {var_hist}]",
                code=f"ta.macd(close, {fast}, {slow}, {signal})",
            ),
        ]

    @staticmethod
    def bbands(var_mid: str, var_upper: str, var_lower: str,
               src: str, period: str, mult: str) -> Indicator:
        return Indicator(
            var_name=f"[{var_mid}, {var_upper}, {var_lower}]",
            code=f"ta.bb({src}, {period}, {mult})",
            plot=True,
            plot_title="BB",
        )

    @staticmethod
    def atr(var: str, period: str) -> Indicator:
        return Indicator(var, f"ta.atr({period})",
                         display="data_window",
                         plot_title="ATR")

    @staticmethod
    def dmi(period: str, smoothing: str,
            var_plus: str = "plusDI",
            var_minus: str = "minusDI",
            var_adx: str = "adxValue") -> Indicator:
        return Indicator(
            var_name=f"[{var_plus}, {var_minus}, {var_adx}]",
            code=f"ta.dmi({period}, {smoothing})",
        )

    @staticmethod
    def stochastic(k_period: str = "14", d_period: str = "3",
                   smooth: str = "3",
                   var_k: str = "stochK",
                   var_d: str = "stochD") -> list[Indicator]:
        return [
            Indicator(var_k, f"ta.stoch(close, high, low, {k_period})"),
            Indicator(var_d, f"ta.sma({var_k}, {d_period})"),
        ]

    @staticmethod
    def obv(var: str = "obvValue") -> Indicator:
        return Indicator(var, "ta.obv")

    @staticmethod
    def mfi(var: str, period: str) -> Indicator:
        return Indicator(var, f"ta.mfi(hlc3, {period})",
                         display="data_window",
                         plot_title="MFI")

    @staticmethod
    def session_vwap(var: str = "vwapValue") -> Indicator:
        """Session-anchored VWAP (handled specially in generator)."""
        return Indicator(var, "__SESSION_VWAP__")

    @staticmethod
    def ichimoku(
        var_tenkan: str = "tenkanSen",
        var_kijun: str = "kijunSen",
        var_span_a: str = "senkouA",
        var_span_b: str = "senkouB",
        var_above: str = "aboveCloud",
    ) -> list[Indicator]:
        """Return Ichimoku Cloud components.

        Returns 5 indicators: Tenkan-sen (9), Kijun-sen (26),
        Senkou Span A, Senkou Span B (52), and ``aboveCloud`` boolean.
        """
        return [
            Indicator(var_tenkan,
                      "(ta.highest(high, 9) + ta.lowest(low, 9)) / 2"),
            Indicator(var_kijun,
                      "(ta.highest(high, 26) + ta.lowest(low, 26)) / 2"),
            Indicator(var_span_a,
                      f"({var_tenkan} + {var_kijun}) / 2"),
            Indicator(var_span_b,
                      "(ta.highest(high, 52) + ta.lowest(low, 52)) / 2"),
            Indicator(var_above,
                      f"close > math.max({var_span_a}, {var_span_b})"),
        ]

    @staticmethod
    def raw(var: str, code: str, plot: bool = False,
            color: str = "", title: str = "") -> Indicator:
        """Raw Pine Script expression."""
        return Indicator(var, code, plot=plot, plot_color=color,
                         plot_title=title)


@dataclass(frozen=True, slots=True)
class Condition:
    """A Pine Script boolean condition expression."""

    expr: str
    description: str = ""

    def __and__(self, other: Condition) -> Condition:
        return Condition(f"({self.expr} and {other.expr})")

    def __or__(self, other: Condition) -> Condition:
        return Condition(f"({self.expr} or {other.expr})")

    def negate(self) -> Condition:
        return Condition(f"not ({self.expr})")


# ======================================================================
# Strategy / Mode / Composite definitions
# ======================================================================


@dataclass(slots=True)
class ModeDefinition:
    """A single toggleable entry mode within a composite strategy."""

    name: str                # "VWAP Pullback"
    short_name: str          # "pb" — used in Pine variable prefixes
    group: str               # TradingView input group label
    enabled_default: bool = True

    inputs: list[Input] = field(default_factory=list)
    indicators: list[Indicator] = field(default_factory=list)

    long_entry: Condition | None = None
    short_entry: Condition | None = None
    long_exit: Condition | None = None
    short_exit: Condition | None = None
    exit_on_reverse: bool = True

    extra_code: list[str] = field(default_factory=list)
    description: str = ""
    label_color_long: str = "color.green"
    label_color_short: str = "color.red"


@dataclass(slots=True)
class CompositeStrategyDefinition:
    """Strategy composed of multiple toggleable modes."""

    name: str
    short_name: str
    description: str = ""

    # Shared across all modes
    shared_inputs: list[Input] = field(default_factory=list)
    shared_indicators: list[Indicator] = field(default_factory=list)
    shared_code: list[str] = field(default_factory=list)

    # Individual modes
    modes: list[ModeDefinition] = field(default_factory=list)

    # Signal aggregation: "first_to_fire" or "any"
    aggregation: str = "first_to_fire"
    priority_order: list[str] = field(default_factory=list)  # short_names

    # Session
    session: str = "0930-1600"
    timezone: str = "America/New_York"
    use_session_filter: bool = True

    extra_plots: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StrategyDefinition:
    """Complete definition of a strategy for Pine Script generation."""

    name: str
    short_name: str

    indicators: list[Indicator] = field(default_factory=list)
    inputs: list[Input] = field(default_factory=list)

    long_entry: Condition | None = None
    short_entry: Condition | None = None
    long_exit: Condition | None = None
    short_exit: Condition | None = None

    # If True, long_exit defaults to short_entry conditions and vice versa
    exit_on_reverse: bool = True

    # Extra Pine Script lines injected verbatim
    extra_code: list[str] = field(default_factory=list)
    extra_plots: list[str] = field(default_factory=list)

    # Session management
    session: str = "0930-1600"
    timezone: str = "America/New_York"
    use_session_filter: bool = False

    # Labels
    long_label: str = "Buy"
    short_label: str = "Sell"

    # Description comment
    description: str = ""


@dataclass(slots=True)
class SharedInfrastructure:
    """Shared inputs, indicators, and code for a group of related modes."""

    inputs: list[Input] = field(default_factory=list)
    indicators: list[Indicator] = field(default_factory=list)
    code: list[str] = field(default_factory=list)
