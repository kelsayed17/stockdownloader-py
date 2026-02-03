"""Represents a trading alert generated from multi-indicator analysis."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

from stockdownloader.model.indicator_values import IndicatorValues
from stockdownloader.model.option_type import OptionType


class Direction(Enum):
    """Signal direction for trading alerts."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class Action(Enum):
    """Recommended action for options."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class OptionsRecommendation:
    """Options recommendation with strike, expiration, and rationale."""

    type: OptionType
    action: Action
    suggested_strike: Decimal
    suggested_dte: int
    estimated_premium: Decimal
    target_delta: Decimal
    rationale: str

    def __str__(self) -> str:
        if self.action == Action.HOLD:
            return f"{self.type.value}: No action recommended"
        return (
            f"{self.action.value} {self.type.value} "
            f"${self.suggested_strike.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} strike, "
            f"{self.suggested_dte}DTE, "
            f"est. premium ${self.estimated_premium.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}, "
            f"delta {float(self.target_delta):.2f} - "
            f"{self.rationale}"
        )


@dataclass(frozen=True)
class AlertResult:
    """Represents a trading alert generated from multi-indicator analysis.

    Contains the signal direction, confidence level, contributing indicators,
    and specific options recommendations (call/put with strike prices).
    """

    symbol: str
    date: str
    current_price: Decimal
    direction: Direction
    confluence_score: float
    total_indicators: int
    bullish_indicators: list[str]
    bearish_indicators: list[str]
    call_recommendation: OptionsRecommendation
    put_recommendation: OptionsRecommendation
    support_levels: list[Decimal]
    resistance_levels: list[Decimal]
    indicators: IndicatorValues

    def get_signal_strength(self) -> str:
        """Return a formatted string describing signal strength."""
        return (
            f"{self.confluence_score * 100:.0f}% "
            f"({int(self.confluence_score * self.total_indicators)}/{self.total_indicators} indicators)"
        )

    def __str__(self) -> str:
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append(f"  TRADING ALERT: {self.symbol} - {self.date}")
        lines.append("=" * 80)
        lines.append("")

        lines.append(
            f"  Current Price:     ${self.current_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
        )
        lines.append(f"  Signal:            {self.direction.value}")
        lines.append(f"  Confluence:        {self.get_signal_strength()}")
        lines.append("")

        lines.append("-" * 80)
        lines.append(f"  BULLISH INDICATORS ({len(self.bullish_indicators)})")
        lines.append("-" * 80)
        for ind in self.bullish_indicators:
            lines.append(f"    + {ind}")
        lines.append("")

        lines.append("-" * 80)
        lines.append(f"  BEARISH INDICATORS ({len(self.bearish_indicators)})")
        lines.append("-" * 80)
        for ind in self.bearish_indicators:
            lines.append(f"    - {ind}")
        lines.append("")

        lines.append("-" * 80)
        lines.append("  OPTIONS RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append(f"    CALL: {self.call_recommendation}")
        lines.append(f"    PUT:  {self.put_recommendation}")
        lines.append("")

        lines.append("-" * 80)
        lines.append("  SUPPORT & RESISTANCE")
        lines.append("-" * 80)
        if self.support_levels:
            support_str = "  ".join(
                f"${level.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
                for level in self.support_levels[:3]
            )
            lines.append(f"    Support:    {support_str}")
        if self.resistance_levels:
            resistance_str = "  ".join(
                f"${level.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
                for level in self.resistance_levels[:3]
            )
            lines.append(f"    Resistance: {resistance_str}")
        lines.append("")

        lines.append("-" * 80)
        lines.append("  INDICATOR SNAPSHOT")
        lines.append("-" * 80)
        lines.append(self.indicators.summary())
        lines.append("")

        lines.append("=" * 80)
        lines.append("  DISCLAIMER: This is for educational purposes only.")
        lines.append("  Not financial advice. Always do your own research.")
        lines.append("=" * 80)

        return "\n".join(lines)
