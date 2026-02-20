"""Structured advisory model for the Intelligent Signal Advisory System.

Combines signal confluence, market regime classification, walk-forward
validation confidence, and ATR-based stop/target levels into a single
JSON-serializable recommendation.

The primary output type is :class:`SignalAdvisory` — a frozen dataclass
returned by :class:`~stockdownloader.analysis.signal_advisor.SignalAdvisor`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
from typing import Any


class AdvisoryAction(Enum):
    """Directional recommendation after regime filtering."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass(frozen=True, slots=True)
class AdvisoryReasoning:
    """Human-readable explanation of how the advisory was derived."""

    signal_confluence: str
    regime_alignment: str
    walk_forward_validated: str
    key_bullish: list[str] = field(default_factory=list)
    key_bearish: list[str] = field(default_factory=list)
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OptionsAdvisory:
    """Condensed options recommendation (call or put)."""

    action: str  # "BUY", "SELL", "HOLD"
    strike: float
    dte: int
    delta: float
    est_premium: float


@dataclass(frozen=True, slots=True)
class SignalAdvisory:
    """Primary output of the advisory system.

    Immutable, JSON-serializable recommendation combining signal analysis,
    regime awareness, and walk-forward-validated confidence.

    Attributes
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    timestamp:
        ISO-8601 datetime string of the advisory.
    action:
        Regime-filtered directional recommendation.
    confidence:
        Composite confidence in ``[0, 1]``.
    regime:
        Current market regime label.
    regime_confidence:
        Regime classification confidence in ``[0, 1]``.
    entry_price:
        Suggested entry price.
    stop_loss:
        ATR-based stop-loss level.
    take_profit:
        ATR-based take-profit level.
    risk_reward:
        Risk-reward ratio (take_profit distance / stop_loss distance).
    position_size_pct:
        Suggested position size as a percentage of capital.
    reasoning:
        Structured explanation of the advisory derivation.
    call_advisory:
        Options call recommendation (or ``None``).
    put_advisory:
        Options put recommendation (or ``None``).
    """

    symbol: str
    timestamp: str
    action: AdvisoryAction
    confidence: float
    regime: str
    regime_confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    position_size_pct: float
    reasoning: AdvisoryReasoning
    call_advisory: OptionsAdvisory | None = None
    put_advisory: OptionsAdvisory | None = None

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @property
    def dedup_key(self) -> str:
        """Key for deduplication: same symbol + action + date = duplicate.

        The date portion is extracted from *timestamp* (first 10 chars,
        i.e. ``YYYY-MM-DD``).  A direction change on the same day produces
        a different key and is therefore **not** a duplicate.
        """
        date_part = self.timestamp[:10] if len(self.timestamp) >= 10 else self.timestamp
        return f"{self.symbol}:{self.action.value}:{date_part}"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for JSON output."""
        d = asdict(self)
        # Convert AdvisoryAction enum → string value
        d["action"] = self.action.value
        return d

    def to_json(self, *, indent: int = 2) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Fallback serializer for :func:`json.dumps`."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
