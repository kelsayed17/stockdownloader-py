"""Crossover detection utilities for indicator comparisons.

Provides reusable helpers for the common pattern of detecting when a value
crosses above or below a threshold (or another series) between two
consecutive observations.

These are pure comparison functions — they do not compute indicator values
themselves.  Pair them with :class:`~stockdownloader.util.indicator_hub.IndicatorHub`
calls for a clean, composable signal detection pattern::

    from stockdownloader.util.crossover import crossed_above

    current_rsi = hub.rsi(data, i, period)
    prev_rsi    = hub.rsi(data, i - 1, period)
    if crossed_above(current_rsi, prev_rsi, oversold_threshold):
        return Signal.BUY
"""
from __future__ import annotations

from decimal import Decimal


def crossed_above(
    current: Decimal,
    previous: Decimal,
    threshold: Decimal,
) -> bool:
    """Return ``True`` if *current* is above *threshold* while *previous* was at or below it.

    This detects the exact bar where an upward crossing occurs.
    """
    return current > threshold and previous <= threshold


def crossed_below(
    current: Decimal,
    previous: Decimal,
    threshold: Decimal,
) -> bool:
    """Return ``True`` if *current* is below *threshold* while *previous* was at or above it.

    This detects the exact bar where a downward crossing occurs.
    """
    return current < threshold and previous >= threshold


def crossed_above_series(
    current_a: Decimal,
    prev_a: Decimal,
    current_b: Decimal,
    prev_b: Decimal,
) -> bool:
    """Return ``True`` if series *A* crossed above series *B*.

    Useful for MACD-over-signal, SMA-fast-over-slow, and similar
    two-series crossover patterns.
    """
    return current_a > current_b and prev_a <= prev_b


def crossed_below_series(
    current_a: Decimal,
    prev_a: Decimal,
    current_b: Decimal,
    prev_b: Decimal,
) -> bool:
    """Return ``True`` if series *A* crossed below series *B*.

    Useful for MACD-under-signal, SMA-fast-under-slow, and similar
    two-series crossover patterns.
    """
    return current_a < current_b and prev_a >= prev_b
