"""Smart Money Concepts (SMC) indicator primitives.

Provides market structure detection: swing highs/lows, break of structure
(BoS), supply/demand zone identification from impulse origins, and
liquidity sweep detection.

All functions work with ``Sequence[PriceData]`` (bar-indexed) and use
``Decimal`` arithmetic for consistency with the rest of the indicator stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stockdownloader.model.price_data import PriceData


# =========================================================================
# DATA STRUCTURES
# =========================================================================


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """A confirmed swing high or low."""

    index: int
    price: Decimal
    kind: str  # "high" | "low"


@dataclass(frozen=True, slots=True)
class StructureState:
    """Snapshot of market structure at a given bar.

    Captures swing levels, HH/HL/LH/LL relationships, break-of-structure
    flags, and supply/demand zones with validity tracking.
    """

    trend: int  # +1 uptrend, -1 downtrend, 0 undefined

    # Most recent confirmed swings
    last_swing_high: Decimal
    last_swing_low: Decimal
    last_swing_high_idx: int
    last_swing_low_idx: int

    # Previous swings (for HH/LL comparison)
    prev_swing_high: Decimal
    prev_swing_low: Decimal

    # Structure relationships
    higher_high: bool
    higher_low: bool
    lower_high: bool
    lower_low: bool

    # Break of structure flags (set on the bar that breaks)
    bos_up: bool  # Close broke above last swing high
    bos_down: bool  # Close broke below last swing low

    # BoS tracking (bar index of last BoS, for age calculation)
    last_bos_idx: int

    # Supply / demand zones
    demand_zone_top: Decimal
    demand_zone_bot: Decimal
    supply_zone_top: Decimal
    supply_zone_bot: Decimal
    demand_zone_valid: bool
    supply_zone_valid: bool


_EMPTY_STRUCTURE = StructureState(
    trend=0,
    last_swing_high=ZERO,
    last_swing_low=ZERO,
    last_swing_high_idx=-1,
    last_swing_low_idx=-1,
    prev_swing_high=ZERO,
    prev_swing_low=ZERO,
    higher_high=False,
    higher_low=False,
    lower_high=False,
    lower_low=False,
    bos_up=False,
    bos_down=False,
    last_bos_idx=-1,
    demand_zone_top=ZERO,
    demand_zone_bot=ZERO,
    supply_zone_top=ZERO,
    supply_zone_bot=ZERO,
    demand_zone_valid=False,
    supply_zone_valid=False,
)


# =========================================================================
# SWING DETECTION
# =========================================================================


def is_swing_high(
    data: Sequence[PriceData],
    index: int,
    lookback: int = 5,
) -> bool:
    """True if bar at *index* has the highest high within *lookback* bars each side.

    Requires ``lookback`` bars before AND after *index* in the data.
    """
    if index < lookback or index + lookback >= len(data):
        return False

    pivot_high = data[index].high

    for i in range(index - lookback, index):
        if data[i].high >= pivot_high:
            return False
    for i in range(index + 1, index + lookback + 1):
        if data[i].high >= pivot_high:
            return False
    return True


def is_swing_low(
    data: Sequence[PriceData],
    index: int,
    lookback: int = 5,
) -> bool:
    """True if bar at *index* has the lowest low within *lookback* bars each side.

    Requires ``lookback`` bars before AND after *index* in the data.
    """
    if index < lookback or index + lookback >= len(data):
        return False

    pivot_low = data[index].low

    for i in range(index - lookback, index):
        if data[i].low <= pivot_low:
            return False
    for i in range(index + 1, index + lookback + 1):
        if data[i].low <= pivot_low:
            return False
    return True


# =========================================================================
# LIQUIDITY SWEEP
# =========================================================================


def is_liquidity_sweep_high(
    bar: PriceData,
    swing_high: Decimal,
    atr_val: Decimal,
    tolerance: Decimal = Decimal("0.3"),
) -> bool:
    """True if bar wicked above *swing_high* but closed below it.

    A liquidity sweep occurs when price briefly pierces a swing level
    (grabbing stop-loss liquidity) but reverses, indicating potential
    institutional selling/buying.

    Parameters
    ----------
    tolerance:
        Maximum distance above swing level (in ATR) for sweep to be valid.
        Prevents matching bars that blast far through the level.
    """
    if swing_high <= ZERO or atr_val <= ZERO:
        return False
    # Wick above the level
    if bar.high <= swing_high:
        return False
    # Body closed below the level
    body_top = max(bar.open, bar.close)
    if body_top >= swing_high:
        return False
    # Within tolerance
    overshoot = bar.high - swing_high
    return overshoot <= atr_val * tolerance


def is_liquidity_sweep_low(
    bar: PriceData,
    swing_low: Decimal,
    atr_val: Decimal,
    tolerance: Decimal = Decimal("0.3"),
) -> bool:
    """True if bar wicked below *swing_low* but closed above it."""
    if swing_low <= ZERO or atr_val <= ZERO:
        return False
    # Wick below the level
    if bar.low >= swing_low:
        return False
    # Body closed above the level
    body_bot = min(bar.open, bar.close)
    if body_bot <= swing_low:
        return False
    # Within tolerance
    overshoot = swing_low - bar.low
    return overshoot <= atr_val * tolerance


# =========================================================================
# IMPULSE STRENGTH
# =========================================================================


def impulse_strength(
    data: Sequence[PriceData],
    start_idx: int,
    end_idx: int,
    atr_val: Decimal,
) -> Decimal:
    """Measure impulse move size from *start_idx* to *end_idx* in ATR units.

    Returns ``|close[end] - close[start]| / ATR``.  Zero if ATR is zero.
    """
    if atr_val <= ZERO or start_idx < 0 or end_idx >= len(data):
        return ZERO
    move = abs(data[end_idx].close - data[start_idx].close)
    return move / atr_val


# =========================================================================
# ZONE IDENTIFICATION
# =========================================================================


def zone_from_impulse_origin(
    data: Sequence[PriceData],
    impulse_start_idx: int,
    zone_bars: int = 2,
) -> tuple[Decimal, Decimal]:
    """Identify supply/demand zone from the candles at the impulse origin.

    The zone is the high-low range of the ``zone_bars`` candles starting
    at ``impulse_start_idx`` (the base before the impulse move).

    Returns ``(zone_top, zone_bot)``.
    """
    if impulse_start_idx < 0:
        return ZERO, ZERO

    end = min(impulse_start_idx + zone_bars, len(data))
    if end <= impulse_start_idx:
        return ZERO, ZERO

    zone_top = max(data[i].high for i in range(impulse_start_idx, end))
    zone_bot = min(data[i].low for i in range(impulse_start_idx, end))
    return zone_top, zone_bot
