"""Tests for SessionState."""

from decimal import Decimal

from stockdownloader.core.models.trade import Direction
from stockdownloader.strategies.intraday.session import SessionState


def test_initial_state():
    state = SessionState()
    assert state.trading_date == ""
    assert state.bar_count == 0
    assert state.in_position is False
    assert state.position_direction is None


def test_reset_clears_all_fields():
    state = SessionState()

    # Mutate some state
    state.bar_count = 42
    state.or_high = Decimal("500")
    state.or_done = True
    state.day_hod = Decimal("510")
    state.in_position = True
    state.position_direction = Direction.LONG
    state.entry_price = Decimal("450")
    state.day_trades = 3
    state.consec_losses = 2
    state.tripped = True
    state.vwap_crosses = 5

    state.reset("2025-01-15")

    assert state.trading_date == "2025-01-15"
    assert state.bar_count == 0
    assert state.or_high == Decimal("0")
    assert state.or_low == Decimal("999999")
    assert state.or_done is False
    assert state.day_hod == Decimal("0")
    assert state.day_lod == Decimal("999999")
    assert state.in_position is False
    assert state.position_direction is None
    assert state.entry_price == Decimal("0")
    assert state.day_trades == 0
    assert state.consec_losses == 0
    assert state.tripped is False
    assert state.vwap_crosses == 0


def test_reset_clears_position_tracking():
    state = SessionState()
    state.in_position = True
    state.position_direction = Direction.SHORT
    state.stop_loss = Decimal("100")
    state.take_profit = Decimal("90")
    state.entry_mode = "PB"
    state.entry_bar = 15
    state.orig_sl = Decimal("100")
    state.risk_amount = Decimal("1.50")
    state.be_triggered = True
    state.trailing_vwap = True
    state.trailing_atr = True
    state.trail_level = Decimal("95")

    state.reset("2025-02-01")

    assert state.in_position is False
    assert state.position_direction is None
    assert state.stop_loss == Decimal("0")
    assert state.take_profit == Decimal("0")
    assert state.entry_mode == ""
    assert state.entry_bar == 0
    assert state.be_triggered is False
    assert state.trailing_vwap is False
    assert state.trailing_atr is False
    assert state.trail_level == Decimal("0")


def test_reset_clears_risk_management():
    state = SessionState()
    state.day_trades = 5
    state.last_entry_bar = 30
    state.consec_losses = 3
    state.tripped = True
    state.day_limited = True
    state.session_start_equity = Decimal("100000")
    state.session_pnl = Decimal("-500")

    state.reset("2025-03-01")

    assert state.day_trades == 0
    assert state.last_entry_bar == -100
    assert state.consec_losses == 0
    assert state.tripped is False
    assert state.day_limited is False
    assert state.session_start_equity == Decimal("0")
    assert state.session_pnl == Decimal("0")


def test_reset_clears_trend_tracking():
    state = SessionState()
    state.bull_bars = 10
    state.bear_bars = 5
    state.trend_age = 20
    state.vwap_crosses = 8
    state.cum_vd = Decimal("1500")
    state.bars_above_vwap = 12
    state.bars_below_vwap = 3
    state.prev_close_vs_vwap = 1

    state.reset("2025-04-01")

    assert state.bull_bars == 0
    assert state.bear_bars == 0
    assert state.trend_age == 0
    assert state.vwap_crosses == 0
    assert state.cum_vd == Decimal("0")
    assert state.bars_above_vwap == 0
    assert state.bars_below_vwap == 0
    assert state.prev_close_vs_vwap is None


def test_reset_clears_previous_day_data():
    state = SessionState()
    state.pd_high = Decimal("510")
    state.pd_low = Decimal("490")
    state.pd_close = Decimal("505")
    state.pw_high = Decimal("520")
    state.pw_low = Decimal("480")
    state.daily_atr = Decimal("8.5")
    state.daily_sma = Decimal("500")
    state.prev_vwap_close = Decimal("502")

    state.reset("2025-05-01")

    assert state.pd_high == Decimal("0")
    assert state.pd_low == Decimal("0")
    assert state.pd_close == Decimal("0")
    assert state.pw_high == Decimal("0")
    assert state.pw_low == Decimal("0")
    assert state.daily_atr == Decimal("0")
    assert state.daily_sma == Decimal("0")
    assert state.prev_vwap_close == Decimal("0")


def test_reset_clears_fired_today():
    """Regression test: fired_today must reset each session.

    Previously ``fired_today`` lived on the strategy instance as
    ``_fired_today`` and was only reset in ``on_session_start()``
    which the engine never called.  Moving it to SessionState
    ensures it resets via ``state.reset()`` at each new trading day.
    """
    state = SessionState()
    assert state.fired_today is False

    state.fired_today = True
    assert state.fired_today is True

    state.reset("2025-06-01")
    assert state.fired_today is False


def test_opening_range_defaults():
    state = SessionState()
    assert state.or_high == Decimal("0")
    assert state.or_low == Decimal("999999")
    assert state.or_open == Decimal("0")
    assert state.or_close == Decimal("0")
    assert state.or_done is False
    assert state.or_range == Decimal("0")
    assert state.or_dir == 0
    assert state.is_manip is False
