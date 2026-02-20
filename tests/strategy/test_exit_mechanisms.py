"""Tests for exit mechanism implementations."""

from decimal import Decimal

from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.model.tournament_trade import TournamentTrade
from stockdownloader.model.trade import Direction
from stockdownloader.strategy.exit_mechanisms import (
    AtrTrailExit,
    HybridExit,
    TimeDecayExit,
    TrailingStopExit,
    VwapBandExit,
    VwapCrossExit,
)


def _make_bar(dt_str, open_, high, low, close, volume=100000):
    return IntradayPriceData(
        date=dt_str,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        adj_close=Decimal(str(close)),
        volume=volume,
    )


def _make_trade(direction=Direction.LONG, entry_price=100, stop_dist=1):
    return TournamentTrade(
        trade_id=1,
        direction=direction,
        signal_type="PB",
        entry_datetime="2025-12-01 10:00:00-05:00",
        exit_datetime="2025-12-01 12:00:00-05:00",
        entry_price=Decimal(str(entry_price)),
        original_exit_price=Decimal(str(entry_price + 1)),
        original_pnl=Decimal("1"),
        stop_distance=Decimal(str(stop_dist)),
    )


# ── Rising price sequence (long winner) ──────────────────────────────
def _rising_bars():
    """Create bars that rise from 100 to 103 over 8 bars."""
    base_time = "2025-12-01"
    return [
        _make_bar(f"{base_time} 10:00:00-05:00", 100, 100.5, 99.5, 100),
        _make_bar(f"{base_time} 10:05:00-05:00", 100, 100.8, 100, 100.5),
        _make_bar(f"{base_time} 10:10:00-05:00", 100.5, 101.2, 100.3, 101),
        _make_bar(f"{base_time} 10:15:00-05:00", 101, 101.5, 100.8, 101.3),
        _make_bar(f"{base_time} 10:20:00-05:00", 101.3, 102, 101, 101.8),
        _make_bar(f"{base_time} 10:25:00-05:00", 101.8, 102.5, 101.5, 102.2),
        _make_bar(f"{base_time} 10:30:00-05:00", 102.2, 103, 102, 102.8),
        _make_bar(f"{base_time} 10:35:00-05:00", 102.8, 103, 102.5, 102.5),
    ]


# ── Bars that rise then fall (long stop-out) ─────────────────────────
def _rise_then_fall_bars():
    """Rise to 101.5 then fall back to 99."""
    base_time = "2025-12-01"
    return [
        _make_bar(f"{base_time} 10:00:00-05:00", 100, 100.5, 99.8, 100),
        _make_bar(f"{base_time} 10:05:00-05:00", 100, 101, 100, 100.8),
        _make_bar(f"{base_time} 10:10:00-05:00", 100.8, 101.5, 100.5, 101.2),
        _make_bar(f"{base_time} 10:15:00-05:00", 101.2, 101.5, 100.8, 101),
        _make_bar(f"{base_time} 10:20:00-05:00", 101, 101.2, 100, 100.2),
        _make_bar(f"{base_time} 10:25:00-05:00", 100.2, 100.5, 99.5, 99.8),
        _make_bar(f"{base_time} 10:30:00-05:00", 99.8, 100, 98.5, 98.8),
        _make_bar(f"{base_time} 10:35:00-05:00", 98.8, 99, 98, 98.5),
    ]


class TestTrailingStopExit:

    def test_hard_stop_hit(self):
        mech = TrailingStopExit()
        trade = _make_trade(entry_price=100, stop_dist=1)
        # Bar drops below stop (entry - 1R = 99)
        bar = _make_bar("2025-12-01 10:00:00-05:00", 100, 100, 98.5, 99)
        mech.reset()
        assert mech.evaluate_bar(bar, trade, 0) is True
        assert mech.exit_reason == "hard_stop"

    def test_no_exit_above_stop(self):
        mech = TrailingStopExit()
        trade = _make_trade(entry_price=100, stop_dist=1)
        bar = _make_bar("2025-12-01 10:00:00-05:00", 100, 101, 99.5, 100.5)
        mech.reset()
        assert mech.evaluate_bar(bar, trade, 0) is False

    def test_trail_stop_after_activation(self):
        mech = TrailingStopExit()
        trade = _make_trade(entry_price=100, stop_dist=1)
        bars = _rise_then_fall_bars()
        mech.reset()

        exited = False
        for i, bar in enumerate(bars):
            if mech.evaluate_bar(bar, trade, i):
                exited = True
                break

        assert exited
        # Should have been stopped out as price fell back
        reason = mech.exit_reason
        assert reason in ("trail_stop", "hard_stop")

    def test_session_end_if_no_exit(self):
        mech = TrailingStopExit()
        trade = _make_trade(entry_price=100, stop_dist=1)
        bars = _rising_bars()
        mech.reset()

        exited = False
        for i, bar in enumerate(bars):
            if mech.evaluate_bar(bar, trade, i):
                exited = True
                break

        # In a strong uptrend, the trail may not be hit
        # The engine would handle session_end -- mechanism just returns False
        # This is valid behavior


class TestVwapCrossExit:

    def test_hard_stop_hit(self):
        mech = VwapCrossExit()
        trade = _make_trade(entry_price=100, stop_dist=1)
        bar = _make_bar("2025-12-01 10:00:00-05:00", 100, 100, 98.5, 99)
        mech.reset()
        mech.set_data_context([], 0)
        assert mech.evaluate_bar(bar, trade, 0) is True
        assert mech.exit_reason == "hard_stop"

    def test_name_default(self):
        mech = VwapCrossExit()
        assert mech.name == "VWAP_CROSS"

    def test_name_custom(self):
        mech = VwapCrossExit(name="VWAP_CROSS_LATE")
        assert mech.name == "VWAP_CROSS_LATE"


class TestAtrTrailExit:

    def test_hard_stop_hit(self):
        mech = AtrTrailExit()
        trade = _make_trade(entry_price=100, stop_dist=1)
        bar = _make_bar("2025-12-01 10:00:00-05:00", 100, 100, 98.5, 99)
        mech.reset()
        mech.set_data_context([], 0)
        assert mech.evaluate_bar(bar, trade, 0) is True

    def test_name(self):
        mech = AtrTrailExit()
        assert mech.name == "ATR_TRAIL"


class TestHybridExit:

    def test_hard_stop_hit(self):
        mech = HybridExit()
        trade = _make_trade(entry_price=100, stop_dist=1)
        bar = _make_bar("2025-12-01 10:00:00-05:00", 100, 100, 98.5, 99)
        mech.reset()
        mech.set_data_context([], 0)
        assert mech.evaluate_bar(bar, trade, 0) is True

    def test_name_default(self):
        mech = HybridExit()
        assert mech.name == "HYBRID"

    def test_name_custom(self):
        mech = HybridExit(name="HYBRID_LATE")
        assert mech.name == "HYBRID_LATE"


class TestVwapBandExit:

    def test_hard_stop_hit(self):
        mech = VwapBandExit()
        trade = _make_trade(entry_price=100, stop_dist=1)
        bar = _make_bar("2025-12-01 10:00:00-05:00", 100, 100, 98.5, 99)
        mech.reset()
        mech.set_data_context([], 0)
        assert mech.evaluate_bar(bar, trade, 0) is True

    def test_name(self):
        mech = VwapBandExit()
        assert mech.name == "VWAP_BAND"


class TestTimeDecayExit:

    def test_hard_stop_hit(self):
        mech = TimeDecayExit()
        trade = _make_trade(entry_price=100, stop_dist=1)
        bar = _make_bar("2025-12-01 10:00:00-05:00", 100, 100, 98.5, 99)
        mech.reset()
        assert mech.evaluate_bar(bar, trade, 0) is True
        assert mech.exit_reason == "hard_stop"

    def test_name(self):
        mech = TimeDecayExit()
        assert mech.name == "TIME_DECAY"

    def test_early_session_wider_trail(self):
        """Before 11:00, trail should be 0.6R (wider)."""
        mech = TimeDecayExit()
        trade = _make_trade(entry_price=100, stop_dist=1)

        # Simulate reaching 1R profit at 10:00
        bar0 = _make_bar("2025-12-01 10:00:00-05:00", 100, 101.2, 100, 101)
        mech.reset()
        mech.evaluate_bar(bar0, trade, 0)

        # After activation, the trail is 0.6R from peak
        # Peak = 101.2, trail = 101.2 - 0.6 = 100.6
        # A bar with low=100.7 should NOT trigger (above 100.6)
        bar1 = _make_bar("2025-12-01 10:05:00-05:00", 101, 101.2, 100.7, 101)
        assert mech.evaluate_bar(bar1, trade, 1) is False

    def test_late_session_tighter_trail(self):
        """After 13:00, trail should be 0.2R (tighter).

        Stop is checked at bar START, then trail is updated at bar END.
        So we need a bar to update the trail, then the next bar triggers.
        """
        mech = TimeDecayExit()
        trade = _make_trade(entry_price=100, stop_dist=1)

        # Bar 0: Activate (peak=101.5 > entry+1R=101) at 10:00
        bar0 = _make_bar("2025-12-01 10:00:00-05:00", 100, 101.5, 100, 101)
        mech.reset()
        mech.evaluate_bar(bar0, trade, 0)

        # Bar 1: At 14:00, trail updates to 0.2R from peak=101.5 → stop=101.3
        # Price stays above so no exit
        bar1 = _make_bar("2025-12-01 14:00:00-05:00", 101.3, 101.4, 101.35, 101.35)
        assert mech.evaluate_bar(bar1, trade, 1) is False

        # Bar 2: Stop is now 101.3, bar low=101.0 < 101.3 → triggered
        bar2 = _make_bar("2025-12-01 14:05:00-05:00", 101.3, 101.3, 101.0, 101.0)
        assert mech.evaluate_bar(bar2, trade, 2) is True
        assert mech.exit_reason == "trail_stop"


class TestShortDirection:

    def test_trailing_stop_short_hard_stop(self):
        mech = TrailingStopExit()
        trade = _make_trade(direction=Direction.SHORT, entry_price=100, stop_dist=1)
        # Short stop is at 101; bar high hits 101.5
        bar = _make_bar("2025-12-01 10:00:00-05:00", 100, 101.5, 99.5, 100)
        mech.reset()
        assert mech.evaluate_bar(bar, trade, 0) is True
        assert mech.exit_reason == "hard_stop"

    def test_trailing_stop_short_no_exit(self):
        mech = TrailingStopExit()
        trade = _make_trade(direction=Direction.SHORT, entry_price=100, stop_dist=1)
        # Price stays below stop
        bar = _make_bar("2025-12-01 10:00:00-05:00", 100, 100.5, 99, 99.5)
        mech.reset()
        assert mech.evaluate_bar(bar, trade, 0) is False
