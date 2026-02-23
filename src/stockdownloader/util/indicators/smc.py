"""Smart Money Concepts (SMC) indicator primitives and streaming tracker.

Section 1 — Data structures (SwingPoint, StructureState, _EMPTY_STRUCTURE)
Section 2 — Primitives (swing detection, liquidity sweep, impulse strength, zone identification)
Section 3 — Streaming tracker (StreamingStructureTracker)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.math import ZERO

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stockdownloader.model.price_data import PriceData

__all__ = [
    # Data structures
    "SwingPoint",
    "StructureState",
    "_EMPTY_STRUCTURE",
    # Primitives
    "is_swing_high",
    "is_swing_low",
    "is_liquidity_sweep_high",
    "is_liquidity_sweep_low",
    "impulse_strength",
    "zone_from_impulse_origin",
    # Streaming
    "StreamingStructureTracker",
]


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


# =========================================================================
# STREAMING — STRUCTURE TRACKER
# =========================================================================


class StreamingStructureTracker:
    """Incremental market structure state machine.

    Tracks swing highs/lows with configurable lookback, detects BoS/ChoCh,
    identifies supply/demand zones from impulse origins, and maintains
    trend state across bars.

    Parameters
    ----------
    lookback:
        Number of bars each side for swing point confirmation.
    min_impulse_atr:
        Minimum impulse move (in ATR units) to create a supply/demand zone.
    zone_bars:
        Number of candles at impulse origin that define the zone width.
    """

    __slots__ = (
        "_lookback",
        "_min_impulse_atr",
        "_zone_bars",
        # Confirmed swing points (most recent at end)
        "_swing_highs",
        "_swing_lows",
        # Structure state
        "_trend",
        "_last_bos_idx",
        # Supply / demand zones: (top, bot, valid)
        "_demand_zone_top",
        "_demand_zone_bot",
        "_demand_zone_valid",
        "_supply_zone_top",
        "_supply_zone_bot",
        "_supply_zone_valid",
        # Tracking
        "_last_index",
        "_last_checked_swing",  # last index checked for swing confirmation
        "_history",
    )

    def __init__(
        self,
        lookback: int = 5,
        min_impulse_atr: float = 2.0,
        zone_bars: int = 2,
    ) -> None:
        self._lookback = lookback
        self._min_impulse_atr = Decimal(str(min_impulse_atr))
        self._zone_bars = zone_bars

        self._swing_highs: list[SwingPoint] = []
        self._swing_lows: list[SwingPoint] = []

        self._trend: int = 0
        self._last_bos_idx: int = -1

        self._demand_zone_top: Decimal = ZERO
        self._demand_zone_bot: Decimal = ZERO
        self._demand_zone_valid: bool = False
        self._supply_zone_top: Decimal = ZERO
        self._supply_zone_bot: Decimal = ZERO
        self._supply_zone_valid: bool = False

        self._last_index: int = -1
        self._last_checked_swing: int = -1
        self._history: list[StructureState] = []

    def reset(self) -> None:
        """Clear all state."""
        self._swing_highs.clear()
        self._swing_lows.clear()
        self._trend = 0
        self._last_bos_idx = -1
        self._demand_zone_top = ZERO
        self._demand_zone_bot = ZERO
        self._demand_zone_valid = False
        self._supply_zone_top = ZERO
        self._supply_zone_bot = ZERO
        self._supply_zone_valid = False
        self._last_index = -1
        self._last_checked_swing = -1
        self._history.clear()

    def update(
        self,
        data: Sequence[PriceData],
        index: int,
        atr_val: Decimal,
    ) -> StructureState:
        """Process bars up to *index* and return the structure state.

        Idempotent for previously-processed indices (returns cached result).
        """
        # History lookup for already-processed indices
        if index <= self._last_index:
            if 0 <= index < len(self._history):
                return self._history[index]
            return _EMPTY_STRUCTURE

        start = self._last_index + 1
        for i in range(start, index + 1):
            state = self._process_bar(data, i, atr_val)
            self._history.append(state)

        self._last_index = index
        return self._history[index]

    def _process_bar(
        self,
        data: Sequence[PriceData],
        index: int,
        atr_val: Decimal,
    ) -> StructureState:
        """Process a single bar and update internal state."""
        bar = data[index]
        lookback = self._lookback
        bos_up = False
        bos_down = False

        # -- Check for newly confirmed swings --
        # A swing at bar (index - lookback) is confirmed now because we
        # have enough right-side bars.
        check_idx = index - lookback
        if check_idx > self._last_checked_swing and check_idx >= 0:
            self._last_checked_swing = check_idx

            if is_swing_high(data, check_idx, lookback):
                new_sh = SwingPoint(
                    index=check_idx,
                    price=data[check_idx].high,
                    kind="high",
                )
                self._swing_highs.append(new_sh)

                # Detect BoS up: this swing high broke above the previous one
                if len(self._swing_highs) >= 2:
                    prev = self._swing_highs[-2]
                    if new_sh.price > prev.price:
                        bos_up = True
                        self._last_bos_idx = index
                        # Create demand zone from the lowest swing low
                        # between the two swing highs (impulse origin)
                        self._create_demand_zone(data, prev.index, new_sh.index, atr_val)

            if is_swing_low(data, check_idx, lookback):
                new_sl = SwingPoint(
                    index=check_idx,
                    price=data[check_idx].low,
                    kind="low",
                )
                self._swing_lows.append(new_sl)

                # Detect BoS down: this swing low broke below the previous one
                if len(self._swing_lows) >= 2:
                    prev = self._swing_lows[-2]
                    if new_sl.price < prev.price:
                        bos_down = True
                        self._last_bos_idx = index
                        # Create supply zone from the highest swing high
                        # between the two swing lows (impulse origin)
                        self._create_supply_zone(data, prev.index, new_sl.index, atr_val)

        # -- Also check real-time BoS on current bar close --
        # Even before a new swing is confirmed, if current close breaks
        # the last confirmed swing high/low, that's a live BoS.
        if not bos_up and self._swing_highs:
            last_sh = self._swing_highs[-1]
            if bar.close > last_sh.price:
                bos_up = True
                self._last_bos_idx = index

        if not bos_down and self._swing_lows:
            last_sl = self._swing_lows[-1]
            if bar.close < last_sl.price:
                bos_down = True
                self._last_bos_idx = index

        # -- Update trend --
        if bos_up and not bos_down:
            self._trend = 1
        elif bos_down and not bos_up:
            self._trend = -1
        # Both BoS in same bar: conflict, keep current trend

        # -- Zone invalidation --
        # Demand zone invalidated if price closes below zone bottom
        if self._demand_zone_valid and bar.close < self._demand_zone_bot:
            self._demand_zone_valid = False

        # Supply zone invalidated if price closes above zone top
        if self._supply_zone_valid and bar.close > self._supply_zone_top:
            self._supply_zone_valid = False

        # -- Build structure relationships --
        last_sh_price = self._swing_highs[-1].price if self._swing_highs else ZERO
        last_sl_price = self._swing_lows[-1].price if self._swing_lows else ZERO
        last_sh_idx = self._swing_highs[-1].index if self._swing_highs else -1
        last_sl_idx = self._swing_lows[-1].index if self._swing_lows else -1

        prev_sh_price = self._swing_highs[-2].price if len(self._swing_highs) >= 2 else ZERO
        prev_sl_price = self._swing_lows[-2].price if len(self._swing_lows) >= 2 else ZERO

        higher_high = last_sh_price > prev_sh_price > ZERO
        higher_low = last_sl_price > prev_sl_price > ZERO
        lower_high = ZERO < last_sh_price < prev_sh_price
        lower_low = ZERO < last_sl_price < prev_sl_price

        return StructureState(
            trend=self._trend,
            last_swing_high=last_sh_price,
            last_swing_low=last_sl_price,
            last_swing_high_idx=last_sh_idx,
            last_swing_low_idx=last_sl_idx,
            prev_swing_high=prev_sh_price,
            prev_swing_low=prev_sl_price,
            higher_high=higher_high,
            higher_low=higher_low,
            lower_high=lower_high,
            lower_low=lower_low,
            bos_up=bos_up,
            bos_down=bos_down,
            last_bos_idx=self._last_bos_idx,
            demand_zone_top=self._demand_zone_top,
            demand_zone_bot=self._demand_zone_bot,
            supply_zone_top=self._supply_zone_top,
            supply_zone_bot=self._supply_zone_bot,
            demand_zone_valid=self._demand_zone_valid,
            supply_zone_valid=self._supply_zone_valid,
        )

    def _create_demand_zone(
        self,
        data: Sequence[PriceData],
        prev_sh_idx: int,
        new_sh_idx: int,
        atr_val: Decimal,
    ) -> None:
        """Create demand zone from the lowest point between two swing highs.

        The demand zone is at the origin of the bullish impulse -- the swing
        low between the old and new swing high.
        """
        if atr_val <= ZERO:
            return

        # Find the lowest low between the two swing highs
        search_start = max(prev_sh_idx, 0)
        search_end = min(new_sh_idx + 1, len(data))
        if search_start >= search_end:
            return

        min_low_idx = search_start
        min_low = data[search_start].low
        for i in range(search_start + 1, search_end):
            if data[i].low < min_low:
                min_low = data[i].low
                min_low_idx = i

        # Check impulse strength (from the low to the new high)
        move = data[new_sh_idx].high - min_low
        if atr_val > ZERO and move / atr_val >= self._min_impulse_atr:
            top, bot = zone_from_impulse_origin(data, min_low_idx, self._zone_bars)
            if top > ZERO:
                self._demand_zone_top = top
                self._demand_zone_bot = bot
                self._demand_zone_valid = True

    def _create_supply_zone(
        self,
        data: Sequence[PriceData],
        prev_sl_idx: int,
        new_sl_idx: int,
        atr_val: Decimal,
    ) -> None:
        """Create supply zone from the highest point between two swing lows.

        The supply zone is at the origin of the bearish impulse -- the swing
        high between the old and new swing low.
        """
        if atr_val <= ZERO:
            return

        # Find the highest high between the two swing lows
        search_start = max(prev_sl_idx, 0)
        search_end = min(new_sl_idx + 1, len(data))
        if search_start >= search_end:
            return

        max_high_idx = search_start
        max_high = data[search_start].high
        for i in range(search_start + 1, search_end):
            if data[i].high > max_high:
                max_high = data[i].high
                max_high_idx = i

        # Check impulse strength (from the high to the new low)
        move = max_high - data[new_sl_idx].low
        if atr_val > ZERO and move / atr_val >= self._min_impulse_atr:
            top, bot = zone_from_impulse_origin(data, max_high_idx, self._zone_bars)
            if top > ZERO:
                self._supply_zone_top = top
                self._supply_zone_bot = bot
                self._supply_zone_valid = True
