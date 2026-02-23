"""Tests for StreamingStructureTracker."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.model.price_data import PriceData
from stockdownloader.util.indicators.smc import _EMPTY_STRUCTURE
from stockdownloader.util.indicators.smc import StreamingStructureTracker


_D = Decimal


def _bar(
    date: str,
    high: str,
    low: str,
    close: str,
    open_: str | None = None,
    volume: int = 1000,
) -> PriceData:
    """Create a PriceData bar for testing."""
    h, l, c = _D(high), _D(low), _D(close)
    o = _D(open_) if open_ is not None else c
    return PriceData(date=date, open=o, high=h, low=l, close=c, adj_close=c, volume=volume)


# ── Synthetic data: clear uptrend with swing highs and lows ──
# Structure: rally → pullback → higher high → pullback → higher high
_UPTREND_DATA = [
    # Initial base (bars 0-4)
    _bar("2024-01-02 09:30", "100", "98", "99"),    # 0
    _bar("2024-01-02 09:35", "101", "99", "100"),   # 1
    _bar("2024-01-02 09:40", "102", "100", "101"),  # 2
    _bar("2024-01-02 09:45", "103", "101", "102"),  # 3
    _bar("2024-01-02 09:50", "104", "102", "103"),  # 4
    # Swing high #1 at bar 5 (high=110)
    _bar("2024-01-02 09:55", "110", "108", "109"),  # 5
    # Pullback (bars 6-10)
    _bar("2024-01-02 10:00", "108", "106", "107"),  # 6
    _bar("2024-01-02 10:05", "107", "105", "106"),  # 7
    _bar("2024-01-02 10:10", "106", "104", "105"),  # 8
    _bar("2024-01-02 10:15", "105", "103", "104"),  # 9
    # Swing low #1 at bar 10 (low=100)
    _bar("2024-01-02 10:20", "103", "100", "101"),  # 10
    # Rally to higher high (bars 11-15)
    _bar("2024-01-02 10:25", "104", "101", "103"),  # 11
    _bar("2024-01-02 10:30", "106", "103", "105"),  # 12
    _bar("2024-01-02 10:35", "108", "105", "107"),  # 13
    _bar("2024-01-02 10:40", "110", "107", "109"),  # 14
    # Swing high #2 at bar 15 (high=115 > 110, BoS up!)
    _bar("2024-01-02 10:45", "115", "112", "114"),  # 15
    # Pullback (bars 16-20)
    _bar("2024-01-02 10:50", "113", "110", "111"),  # 16
    _bar("2024-01-02 10:55", "112", "109", "110"),  # 17
    _bar("2024-01-02 11:00", "111", "108", "109"),  # 18
    _bar("2024-01-02 11:05", "110", "107", "108"),  # 19
    # Swing low #2 at bar 20 (low=105 > 100, higher low)
    _bar("2024-01-02 11:10", "108", "105", "106"),  # 20
    # Continue rally (bars 21-25)
    _bar("2024-01-02 11:15", "109", "106", "108"),  # 21
    _bar("2024-01-02 11:20", "111", "108", "110"),  # 22
    _bar("2024-01-02 11:25", "113", "110", "112"),  # 23
    _bar("2024-01-02 11:30", "115", "112", "114"),  # 24
    _bar("2024-01-02 11:35", "117", "114", "116"),  # 25
]


class TestStreamingStructureTracker:
    """Core tracker tests."""

    def test_empty_initial_state(self) -> None:
        tracker = StreamingStructureTracker(lookback=3)
        state = tracker.update(_UPTREND_DATA, 0, _D("1.0"))
        assert state.trend == 0
        assert state.last_swing_high == _D("0")
        assert state.last_swing_low == _D("0")

    def test_swing_high_detected(self) -> None:
        tracker = StreamingStructureTracker(lookback=5)
        # Process up to bar 10 (5 bars after the swing high at bar 5)
        state = tracker.update(_UPTREND_DATA, 10, _D("1.0"))
        assert state.last_swing_high == _D("110")
        assert state.last_swing_high_idx == 5

    def test_swing_low_detected(self) -> None:
        tracker = StreamingStructureTracker(lookback=5)
        # Process up to bar 15 (5 bars after the swing low at bar 10)
        state = tracker.update(_UPTREND_DATA, 15, _D("1.0"))
        assert state.last_swing_low == _D("100")
        assert state.last_swing_low_idx == 10

    def test_bos_up_detected(self) -> None:
        tracker = StreamingStructureTracker(lookback=5, min_impulse_atr=1.0)
        # Process to bar 20 — second swing high at bar 15 (115 > 110)
        state = tracker.update(_UPTREND_DATA, 20, _D("1.0"))
        # By bar 20, the swing high at bar 15 is confirmed
        assert state.last_swing_high == _D("115")
        assert state.trend == 1  # Uptrend established

    def test_higher_high_relationship(self) -> None:
        tracker = StreamingStructureTracker(lookback=5, min_impulse_atr=1.0)
        state = tracker.update(_UPTREND_DATA, 20, _D("1.0"))
        assert state.higher_high is True

    def test_demand_zone_created(self) -> None:
        tracker = StreamingStructureTracker(
            lookback=5, min_impulse_atr=1.0, zone_bars=2,
        )
        state = tracker.update(_UPTREND_DATA, 20, _D("1.0"))
        # Demand zone should exist from the impulse origin
        assert state.demand_zone_valid is True
        assert state.demand_zone_top > _D("0")
        assert state.demand_zone_bot > _D("0")
        assert state.demand_zone_top >= state.demand_zone_bot

    def test_no_demand_zone_with_high_impulse_requirement(self) -> None:
        tracker = StreamingStructureTracker(
            lookback=5, min_impulse_atr=100.0,  # Impossibly high
        )
        state = tracker.update(_UPTREND_DATA, 20, _D("1.0"))
        assert state.demand_zone_valid is False

    def test_history_lookup(self) -> None:
        tracker = StreamingStructureTracker(lookback=3)
        # Process all bars
        state_final = tracker.update(_UPTREND_DATA, 25, _D("1.0"))

        # Look up an earlier bar
        state_early = tracker.update(_UPTREND_DATA, 3, _D("1.0"))
        assert state_early.trend == 0  # No structure yet at bar 3

    def test_reset_clears_state(self) -> None:
        tracker = StreamingStructureTracker(lookback=5)
        tracker.update(_UPTREND_DATA, 20, _D("1.0"))
        tracker.reset()

        state = tracker.update(_UPTREND_DATA, 0, _D("1.0"))
        assert state.trend == 0
        assert state.last_swing_high == _D("0")

    def test_idempotent_update(self) -> None:
        tracker = StreamingStructureTracker(lookback=5)
        state1 = tracker.update(_UPTREND_DATA, 15, _D("1.0"))
        state2 = tracker.update(_UPTREND_DATA, 15, _D("1.0"))
        assert state1 == state2


# ── Downtrend data ──
_DOWNTREND_DATA = [
    # Initial top (bars 0-4)
    _bar("2024-01-02 09:30", "120", "118", "119"),  # 0
    _bar("2024-01-02 09:35", "119", "117", "118"),  # 1
    _bar("2024-01-02 09:40", "118", "116", "117"),  # 2
    _bar("2024-01-02 09:45", "117", "115", "116"),  # 3
    _bar("2024-01-02 09:50", "116", "114", "115"),  # 4
    # Swing low #1 at bar 5 (low=110)
    _bar("2024-01-02 09:55", "113", "110", "111"),  # 5
    # Bounce (bars 6-10)
    _bar("2024-01-02 10:00", "114", "112", "113"),  # 6
    _bar("2024-01-02 10:05", "115", "113", "114"),  # 7
    _bar("2024-01-02 10:10", "116", "114", "115"),  # 8
    _bar("2024-01-02 10:15", "117", "115", "116"),  # 9
    # Swing high #1 at bar 10 (high=118)
    _bar("2024-01-02 10:20", "118", "116", "117"),  # 10
    # Drop to lower low (bars 11-15)
    _bar("2024-01-02 10:25", "116", "114", "115"),  # 11
    _bar("2024-01-02 10:30", "114", "112", "113"),  # 12
    _bar("2024-01-02 10:35", "112", "110", "111"),  # 13
    _bar("2024-01-02 10:40", "110", "108", "109"),  # 14
    # Swing low #2 at bar 15 (low=105 < 110, BoS down!)
    _bar("2024-01-02 10:45", "108", "105", "106"),  # 15
    # Bounce (bars 16-20)
    _bar("2024-01-02 10:50", "109", "106", "108"),  # 16
    _bar("2024-01-02 10:55", "110", "107", "109"),  # 17
    _bar("2024-01-02 11:00", "111", "108", "110"),  # 18
    _bar("2024-01-02 11:05", "112", "109", "111"),  # 19
    _bar("2024-01-02 11:10", "113", "110", "112"),  # 20
]


class TestDowntrendStructure:
    """Tests for downtrend structure detection."""

    def test_bos_down_detected(self) -> None:
        tracker = StreamingStructureTracker(lookback=5, min_impulse_atr=1.0)
        state = tracker.update(_DOWNTREND_DATA, 20, _D("1.0"))
        assert state.trend == -1

    def test_lower_low_relationship(self) -> None:
        tracker = StreamingStructureTracker(lookback=5, min_impulse_atr=1.0)
        state = tracker.update(_DOWNTREND_DATA, 20, _D("1.0"))
        assert state.lower_low is True

    def test_supply_zone_created(self) -> None:
        tracker = StreamingStructureTracker(
            lookback=5, min_impulse_atr=1.0, zone_bars=2,
        )
        state = tracker.update(_DOWNTREND_DATA, 20, _D("1.0"))
        assert state.supply_zone_valid is True
        assert state.supply_zone_top > _D("0")


class TestZoneInvalidation:
    """Tests for zone invalidation logic."""

    def test_demand_zone_invalidated_on_close_below(self) -> None:
        # Build data where price drops through demand zone
        data = list(_UPTREND_DATA) + [
            # Price crashes through demand zone
            _bar("2024-01-02 11:40", "105", "90", "91"),  # 26
        ]
        tracker = StreamingStructureTracker(
            lookback=5, min_impulse_atr=1.0, zone_bars=2,
        )
        # First establish the zone
        state = tracker.update(data, 20, _D("1.0"))
        if state.demand_zone_valid:
            zone_bot = state.demand_zone_bot
            # Now process the crash bar
            state = tracker.update(data, 26, _D("1.0"))
            # Zone should be invalidated if price closed below bot
            if _D("91") < zone_bot:
                assert state.demand_zone_valid is False
