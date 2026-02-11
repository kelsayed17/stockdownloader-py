"""Signal model for intraday trading strategies."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

_ZERO = Decimal("0")


class IntradayAction(Enum):
    """Action produced by an intraday strategy evaluation."""

    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass(frozen=True)
class IntradaySignal:
    """Signal produced by an intraday strategy on each bar.

    Carries all the information an execution layer needs: direction,
    stop/target levels, confluence score, and a human-readable reason.
    """

    action: IntradayAction
    mode: str = ""
    stop_loss: Decimal = _ZERO
    take_profit: Decimal = _ZERO
    confluence_score: int = 0
    max_score: int = 0
    risk_per_share: Decimal = _ZERO
    reason: str = ""


# Convenience singleton for the common "do nothing" case.
HOLD = IntradaySignal(action=IntradayAction.HOLD)
