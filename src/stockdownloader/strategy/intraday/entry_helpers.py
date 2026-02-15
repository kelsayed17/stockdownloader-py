"""Shared helpers for intraday entry logic.

Centralises the SL/TP directional pricing, signal construction,
and pattern detection patterns shared across intraday strategies.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import IntradayAction, IntradaySignal
from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.intraday_indicators import (
    is_bear_engulfing,
    is_bull_engulfing,
    is_hammer,
    is_inv_hammer,
)

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData

# ------------------------------------------------------------------
# SL / TP helpers
# ------------------------------------------------------------------

def directional_sl_tp(
    go_long: bool,
    close: Decimal,
    sl_dist: Decimal,
    tp_dist: Decimal,
) -> tuple[Decimal, Decimal]:
    """Compute SL and TP prices from distance values.

    Parameters
    ----------
    go_long:
        ``True`` for long entries, ``False`` for short.
    close:
        Current bar close price.
    sl_dist:
        Stop-loss distance (positive).
    tp_dist:
        Take-profit distance (positive).  Pass ``Decimal("0")`` for
        no fixed TP (e.g. trail-only modes).

    Returns
    -------
    (sl_price, tp_price)
    """
    if go_long:
        return close - sl_dist, close + tp_dist
    return close + sl_dist, close - tp_dist

def clamp_sl_dist(
    raw: Decimal,
    cap: Decimal,
) -> Decimal | None:
    """Clamp *raw* SL distance to *cap* and reject if non-positive.

    Returns ``None`` when the clamped value is <= 0 (caller should
    return ``None`` / skip entry).
    """
    dist = min(raw, cap)
    if dist <= ZERO:
        return None
    return dist

# ------------------------------------------------------------------
# Entry signal construction
# ------------------------------------------------------------------

def make_entry_signal(
    *,
    go_long: bool,
    mode: str,
    sl_price: Decimal,
    tp_price: Decimal,
    score: int,
    max_score: int,
    risk_per_share: Decimal,
    reason: str,
) -> IntradaySignal:
    """Build an ``IntradaySignal`` for a long or short entry.

    Consolidates the identical ``IntradaySignal(...)`` construction
    that appears at the end of every strategy's entry method.
    """
    return IntradaySignal(
        action=IntradayAction.ENTER_LONG if go_long else IntradayAction.ENTER_SHORT,
        mode=mode,
        stop_loss=sl_price,
        take_profit=tp_price,
        confluence_score=score,
        max_score=max_score,
        risk_per_share=risk_per_share,
        reason=reason,
    )

# ------------------------------------------------------------------
# Pattern detection helpers (shared by ORR + PS)
# ------------------------------------------------------------------

def detect_reversal_patterns(
    bar: IntradayPriceData,
    prev_bar: IntradayPriceData | None,
    atr_val: Decimal,
    engulf_ratio: Decimal,
) -> tuple[bool, bool, bool, bool]:
    """Detect hammer and engulfing reversal patterns.

    Returns
    -------
    (bull_hammer, bear_hammer, bull_engulf, bear_engulf)
    """
    bull_hammer = is_hammer(bar, atr_val)
    bear_hammer = is_inv_hammer(bar, atr_val)
    bull_engulf = prev_bar is not None and is_bull_engulfing(bar, prev_bar, engulf_ratio)
    bear_engulf = prev_bar is not None and is_bear_engulfing(bar, prev_bar, engulf_ratio)
    return bull_hammer, bear_hammer, bull_engulf, bear_engulf

def reversal_pattern_label(
    bull_hammer: bool,
    bear_hammer: bool,
    bull_engulf: bool,
    bear_engulf: bool,
) -> str:
    """Return a human-readable label for the matched reversal pattern.

    Must be called only when at least one of the flags is ``True``.
    """
    if bull_hammer:
        return "Hammer"
    if bear_hammer:
        return "Inv Hammer"
    if bull_engulf:
        return "Engulfing"
    return "Bear Engulf"
