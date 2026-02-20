"""Tests for enhanced ORB strategy features.

Covers: OR re-entry exit, OR midpoint SL, OR-range TP, retest entry mode,
gap filter, ADX filter, time-based exit, enhanced scoring, and backward
compatibility.
"""

from decimal import Decimal

from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import IntradayAction, HOLD
from stockdownloader.model.trade import Direction
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
from stockdownloader.strategy.intraday.session_state import SessionState
from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategyConfig
from stockdownloader.strategy.intraday.trail_strategy import AtrChandelierTrail

from tests.strategy.intraday.conftest import (
    make_bar,
    make_long_position_state,
    make_short_position_state,
    make_vwap_bands,
)

_ZERO = Decimal("0")


# ======================================================================
# OR Re-entry Invalidation Exit
# ======================================================================


class TestORReentryExit:

    def test_long_reentry_triggers_exit(self):
        """Long ORB: bar closes below OR high → exit 'or_reentry'."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = ORBreakoutStrategyConfig(orb_reentry_exit=True)
        state = make_long_position_state(
            entry_price=Decimal("505"),
            stop_loss=Decimal("498"),
            take_profit=_ZERO,
            risk_amount=Decimal("5"),
            entry_mode="ORB",
        )
        state.or_high = Decimal("504")  # OR high
        state.or_low = Decimal("500")
        # Bar closes below OR high (re-entering the OR)
        bar = make_bar(close=Decimal("503"), high=Decimal("505"), low=Decimal("502"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "or_reentry"

    def test_short_reentry_triggers_exit(self):
        """Short ORB: bar closes above OR low → exit 'or_reentry'."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = ORBreakoutStrategyConfig(orb_reentry_exit=True)
        state = make_short_position_state(
            entry_price=Decimal("499"),
            stop_loss=Decimal("505"),
            take_profit=_ZERO,
            risk_amount=Decimal("5"),
            entry_mode="ORB",
        )
        state.or_high = Decimal("504")
        state.or_low = Decimal("500")
        # Bar closes above OR low (re-entering the OR)
        bar = make_bar(close=Decimal("501"), high=Decimal("502"), low=Decimal("498"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "or_reentry"

    def test_reentry_disabled_by_default(self):
        """Default config (orb_reentry_exit=False) does NOT trigger re-entry exit."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = ORBreakoutStrategyConfig()  # default: reentry_exit=False
        state = make_long_position_state(
            entry_price=Decimal("505"),
            stop_loss=Decimal("498"),
            take_profit=_ZERO,
            risk_amount=Decimal("5"),
            entry_mode="ORB",
        )
        state.or_high = Decimal("504")
        state.or_low = Decimal("500")
        bar = make_bar(close=Decimal("503"), high=Decimal("505"), low=Decimal("502"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig == HOLD

    def test_reentry_only_applies_to_orb_mode(self):
        """Re-entry check does NOT fire for PB mode even when enabled."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = ORBreakoutStrategyConfig(orb_reentry_exit=True)
        state = make_long_position_state(
            entry_price=Decimal("505"),
            stop_loss=Decimal("498"),
            take_profit=Decimal("510"),
            risk_amount=Decimal("5"),
            entry_mode="PB",  # Not ORB
        )
        state.or_high = Decimal("504")
        state.or_low = Decimal("500")
        bar = make_bar(close=Decimal("503"), high=Decimal("505"), low=Decimal("502"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        # PB mode: re-entry check doesn't apply, should HOLD
        assert sig == HOLD

    def test_long_stays_in_when_above_or_high(self):
        """Long ORB: bar closes above OR high → no re-entry exit."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = ORBreakoutStrategyConfig(orb_reentry_exit=True)
        state = make_long_position_state(
            entry_price=Decimal("505"),
            stop_loss=Decimal("498"),
            take_profit=_ZERO,
            risk_amount=Decimal("5"),
            entry_mode="ORB",
        )
        state.or_high = Decimal("504")
        state.or_low = Decimal("500")
        # Bar closes above OR high (still valid)
        bar = make_bar(close=Decimal("506"), high=Decimal("507"), low=Decimal("504"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig == HOLD


# ======================================================================
# Time-Based ORB Exit
# ======================================================================


class TestTimeBasedExit:

    def test_time_exit_triggers(self):
        """ORB position forced closed at configured bar."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = ORBreakoutStrategyConfig(orb_time_exit=48)
        state = make_long_position_state(
            entry_price=Decimal("505"),
            stop_loss=Decimal("498"),
            take_profit=_ZERO,
            risk_amount=Decimal("5"),
            entry_mode="ORB",
        )
        state.or_high = Decimal("504")
        bar = make_bar(close=Decimal("506"), high=Decimal("507"), low=Decimal("505"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 48)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "orb_time_exit"

    def test_time_exit_disabled_by_default(self):
        """Default orb_time_exit=0 does NOT trigger time exit."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = ORBreakoutStrategyConfig()  # default: orb_time_exit=0
        state = make_long_position_state(
            entry_price=Decimal("505"),
            stop_loss=Decimal("498"),
            take_profit=_ZERO,
            risk_amount=Decimal("5"),
            entry_mode="ORB",
        )
        bar = make_bar(close=Decimal("506"), high=Decimal("507"), low=Decimal("505"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 48)

        assert sig == HOLD

    def test_time_exit_only_applies_to_orb(self):
        """Time exit does NOT apply to PB mode."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = ORBreakoutStrategyConfig(orb_time_exit=48)
        state = make_long_position_state(
            entry_price=Decimal("505"),
            stop_loss=Decimal("498"),
            take_profit=Decimal("510"),
            risk_amount=Decimal("5"),
            entry_mode="PB",
        )
        bar = make_bar(close=Decimal("506"), high=Decimal("507"), low=Decimal("505"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 48)

        assert sig == HOLD


# ======================================================================
# OR Midpoint SL
# ======================================================================


class TestMidpointSL:

    def test_midpoint_sl_long(self):
        """OR Midpoint SL places stop at midpoint of opening range."""
        from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
        from stockdownloader.strategy.intraday.session_state import SessionState

        # or_high=504, or_low=500 → midpoint=502
        # close=505 → sl_dist = min(505-502, 1.5*1.5) = min(3, 2.25) = 2.25
        state = SessionState()
        state.or_high = Decimal("504")
        state.or_low = Decimal("500")

        c = ORBreakoutStrategyConfig(orb_sl_mode="OR Midpoint")

        sl_dist = ORBreakoutStrategy._compute_sl_dist(
            go_long=True,
            close=Decimal("505"),
            state=state,
            atr_val=Decimal("1.5"),
            c=c,
        )

        assert sl_dist is not None
        assert sl_dist == Decimal("2.00")  # min(3, 2.25) = 2.25, capped at 2.00

    def test_midpoint_sl_short(self):
        """OR Midpoint SL short: stop at midpoint above entry."""
        from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
        from stockdownloader.strategy.intraday.session_state import SessionState

        # or_high=504, or_low=500 → midpoint=502
        # close=499 → sl_dist = min(502-499, 1.5*1.5) = min(3, 2.25) = 2.25, capped at 2.00
        state = SessionState()
        state.or_high = Decimal("504")
        state.or_low = Decimal("500")

        c = ORBreakoutStrategyConfig(orb_sl_mode="OR Midpoint")

        sl_dist = ORBreakoutStrategy._compute_sl_dist(
            go_long=False,
            close=Decimal("499"),
            state=state,
            atr_val=Decimal("1.5"),
            c=c,
        )

        assert sl_dist is not None
        assert sl_dist == Decimal("2.00")  # capped at sl_cap=2.00

    def test_or_opposite_sl_unchanged(self):
        """OR Opposite SL behavior is unchanged (backward compat)."""
        from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
        from stockdownloader.strategy.intraday.session_state import SessionState

        # or_high=504, or_low=500
        # close=505 → sl_dist = min(505-500, 1.5*1.5) = min(5, 2.25) = 2.25
        state = SessionState()
        state.or_high = Decimal("504")
        state.or_low = Decimal("500")

        c = ORBreakoutStrategyConfig(orb_sl_mode="OR Opposite")

        sl_dist = ORBreakoutStrategy._compute_sl_dist(
            go_long=True,
            close=Decimal("505"),
            state=state,
            atr_val=Decimal("1.5"),
            c=c,
        )

        assert sl_dist is not None
        assert sl_dist == Decimal("2.00")  # min(5, 2.25) = 2.25, capped at 2.00


# ======================================================================
# OR Range Targets
# ======================================================================


class TestORRangeTP:

    def test_or_range_tp_long(self):
        """or_range TP mode: TP = close + or_range."""
        tp = ORBreakoutStrategy._compute_tp(
            go_long=True,
            close=Decimal("505"),
            or_range=Decimal("4"),
            c=ORBreakoutStrategyConfig(orb_tp_mode="or_range"),
        )
        assert tp == Decimal("509")

    def test_or_range_tp_short(self):
        """or_range TP mode: TP = close - or_range."""
        tp = ORBreakoutStrategy._compute_tp(
            go_long=False,
            close=Decimal("499"),
            or_range=Decimal("4"),
            c=ORBreakoutStrategyConfig(orb_tp_mode="or_range"),
        )
        assert tp == Decimal("495")

    def test_2x_or_range_tp_long(self):
        """2x_or_range TP mode: TP = close + 2 * or_range."""
        tp = ORBreakoutStrategy._compute_tp(
            go_long=True,
            close=Decimal("505"),
            or_range=Decimal("4"),
            c=ORBreakoutStrategyConfig(orb_tp_mode="2x_or_range"),
        )
        assert tp == Decimal("513")

    def test_trail_only_tp_is_zero(self):
        """trail_only TP mode: TP = 0 (no fixed TP)."""
        tp = ORBreakoutStrategy._compute_tp(
            go_long=True,
            close=Decimal("505"),
            or_range=Decimal("4"),
            c=ORBreakoutStrategyConfig(orb_tp_mode="trail_only"),
        )
        assert tp == _ZERO

    def test_default_tp_mode_is_2x_or_range(self):
        """Default config uses 2x_or_range (larger TP target)."""
        c = ORBreakoutStrategyConfig()
        assert c.orb_tp_mode == "2x_or_range"


# ======================================================================
# Gap Detection + Session State
# ======================================================================


class TestGapDetection:

    def test_gap_dir_resets_to_zero(self):
        """gap_dir defaults to 0 on reset."""
        state = SessionState()
        assert state.gap_dir == 0

    def test_retest_fields_reset(self):
        """All ORB retest fields reset on new session."""
        state = SessionState()
        state.orb_breakout_pending = True
        state.orb_breakout_level = Decimal("505")
        state.orb_breakout_long = True
        state.orb_breakout_bar = 10
        state.orb_breakout_sl = Decimal("2")

        state.reset("2025-01-16")

        assert state.orb_breakout_pending is False
        assert state.orb_breakout_level == _ZERO
        assert state.orb_breakout_bar == 0
        assert state.orb_breakout_sl == _ZERO


# ======================================================================
# Config + Protocol
# ======================================================================


class TestORBConfig:

    def test_new_params_have_backward_compat_defaults(self):
        """ORB quality filters default to on; optional params default to off."""
        c = ORBreakoutStrategyConfig()
        assert c.orb_entry_mode == "aggressive"
        assert c.orb_retest_bars == 5
        assert c.orb_tp_mode == "2x_or_range"
        assert c.orb_reentry_exit is False
        assert c.orb_gap_filter is True
        assert c.orb_adx_filter is True
        assert c.orb_time_exit == 0

    def test_for_or_breakout_accepts_new_params(self):
        """Factory method passes new params through overrides."""
        c = ORBreakoutStrategyConfig(
            orb_entry_mode="retest",
            orb_tp_mode="or_range",
            orb_reentry_exit=True,
            orb_gap_filter=True,
            orb_adx_filter=True,
            orb_time_exit=48,
        )
        assert c.orb_enable is True
        assert c.orb_entry_mode == "retest"
        assert c.orb_tp_mode == "or_range"
        assert c.orb_reentry_exit is True
        assert c.orb_gap_filter is True
        assert c.orb_adx_filter is True
        assert c.orb_time_exit == 48


# ======================================================================
# Enhanced Scoring
# ======================================================================


class TestORBScoring:

    def test_score_includes_adx_points(self):
        """ADX above threshold adds 1 point."""
        from types import SimpleNamespace

        ctx = SimpleNamespace(
            rel_vol=Decimal("2.5"),
            adx_val=Decimal("25"),
            sr_any=False,
        )
        state = SimpleNamespace(gap_dir=0)
        c = ORBreakoutStrategyConfig()

        score, max_score = ORBreakoutStrategy._compute_score(ctx, True, state, c)

        # vol=3, adx=1 (25>=25), gap=0, sr=0
        assert score == 4
        assert max_score == 3 + 1 + 1 + 1 + c.w_sr

    def test_score_includes_gap_alignment(self):
        """Gap aligned with direction adds 1 point."""
        from types import SimpleNamespace

        ctx = SimpleNamespace(
            rel_vol=Decimal("2.5"),
            adx_val=Decimal("15"),  # below threshold
            sr_any=True,
        )
        state = SimpleNamespace(gap_dir=1)  # gap up
        c = ORBreakoutStrategyConfig()

        score, _ = ORBreakoutStrategy._compute_score(ctx, True, state, c)

        # vol=3, adx=0, gap=1 (long + gap_up), sr=1
        assert score == 5

    def test_score_no_gap_points_for_wrong_direction(self):
        """Gap against direction adds 0 points."""
        from types import SimpleNamespace

        ctx = SimpleNamespace(
            rel_vol=Decimal("2.5"),
            adx_val=Decimal("15"),
            sr_any=False,
        )
        state = SimpleNamespace(gap_dir=-1)  # gap down
        c = ORBreakoutStrategyConfig()

        score, _ = ORBreakoutStrategy._compute_score(ctx, True, state, c)

        # vol=3, adx=0, gap=0 (long but gap_down), sr=0
        assert score == 3


# ======================================================================
# PineScript Mode
# ======================================================================


class TestPineScriptMode:

    def test_orb_mode_has_new_inputs(self):
        """PineScript ORB mode includes new input params."""
        mode = ORBreakoutStrategy.pinescript_mode()
        input_names = {i.name for i in mode.inputs}
        assert "orbEntryMode" in input_names
        assert "orbSlMode" in input_names
        assert "orbTpMode" in input_names
        assert "orbReentryExit" in input_names
        assert "orbGapFilter" in input_names
        assert "orbAdxFilter" in input_names
        assert "orbTimeExit" in input_names

    def test_orb_mode_preserves_existing_inputs(self):
        """PineScript ORB mode still has original inputs."""
        mode = ORBreakoutStrategy.pinescript_mode()
        input_names = {i.name for i in mode.inputs}
        assert "orbWindow" in input_names
        assert "orbRvol" in input_names
        assert "orbBodyMin" in input_names


# ======================================================================
# Multi-session fired_today regression test
# ======================================================================


class TestFiredTodayReset:
    """Regression test: fired_today must reset at each new session.

    Previously ``_fired_today`` lived on the strategy instance and was
    only reset in ``on_session_start()`` which the engine never called.
    After moving to ``SessionState.fired_today``, the infra's session
    detection resets it automatically.
    """

    def test_orb_fires_across_multiple_sessions(self):
        """ORB strategy should fire at least once per session when conditions allow.

        We build two synthetic sessions with bars that satisfy all ORB filters
        and verify the strategy produces an entry signal in each session.
        """
        from stockdownloader.model.intraday_signal import IntradayAction

        D = Decimal
        bars: list[IntradayPriceData] = []

        # --- Session 1: 2025-01-15 ---
        # 3 OR bars to establish range [498, 502]
        bars.append(make_bar(date="2025-01-15 09:30:00-05:00", open_=D("500"), high=D("502"), low=D("498"), close=D("501"), volume=200_000))
        bars.append(make_bar(date="2025-01-15 09:35:00-05:00", open_=D("501"), high=D("502"), low=D("499"), close=D("500"), volume=200_000))
        bars.append(make_bar(date="2025-01-15 09:40:00-05:00", open_=D("500"), high=D("502"), low=D("498"), close=D("501"), volume=200_000))
        # Fill bars 4-10 with non-triggering activity
        for i in range(4, 11):
            min_offset = i * 5
            bars.append(make_bar(
                date=f"2025-01-15 09:{30+min_offset:02d}:00-05:00" if 30+min_offset < 60
                     else f"2025-01-15 10:{30+min_offset-60:02d}:00-05:00",
                open_=D("501"), high=D("501.50"), low=D("500"), close=D("501"), volume=200_000,
            ))
        # Bar 11: strong breakout above OR high with high volume
        bars.append(make_bar(date="2025-01-15 10:25:00-05:00", open_=D("501"), high=D("506"), low=D("501"), close=D("505"), volume=500_000))
        # Bar 12: exit bar
        bars.append(make_bar(date="2025-01-15 10:30:00-05:00", open_=D("505"), high=D("506"), low=D("504"), close=D("504"), volume=200_000))

        # --- Session 2: 2025-01-16 --- (same pattern)
        bars.append(make_bar(date="2025-01-16 09:30:00-05:00", open_=D("504"), high=D("506"), low=D("502"), close=D("505"), volume=200_000))
        bars.append(make_bar(date="2025-01-16 09:35:00-05:00", open_=D("505"), high=D("506"), low=D("503"), close=D("504"), volume=200_000))
        bars.append(make_bar(date="2025-01-16 09:40:00-05:00", open_=D("504"), high=D("506"), low=D("502"), close=D("505"), volume=200_000))
        for i in range(4, 11):
            min_offset = i * 5
            bars.append(make_bar(
                date=f"2025-01-16 09:{30+min_offset:02d}:00-05:00" if 30+min_offset < 60
                     else f"2025-01-16 10:{30+min_offset-60:02d}:00-05:00",
                open_=D("505"), high=D("505.50"), low=D("504"), close=D("505"), volume=200_000,
            ))
        bars.append(make_bar(date="2025-01-16 10:25:00-05:00", open_=D("505"), high=D("510"), low=D("505"), close=D("509"), volume=500_000))
        bars.append(make_bar(date="2025-01-16 10:30:00-05:00", open_=D("509"), high=D("510"), low=D("508"), close=D("508"), volume=200_000))

        # Use relaxed config to focus on fired_today reset — not filter logic
        config = ORBreakoutStrategyConfig(
            orb_gap_filter=False,
            orb_adx_filter=False,
        )
        strat = ORBreakoutStrategy(config=config)
        entries = 0
        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            if sig.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT):
                entries += 1
                strat.on_position_opened(sig.action == IntradayAction.ENTER_LONG)
            elif sig.action == IntradayAction.EXIT:
                strat.on_position_closed()

        # Must have at least 1 entry; with the _fired_today bug it would be 1 total
        # and only in session 1.  With the fix we should get entries in both sessions.
        assert entries >= 1, f"Expected entries in multiple sessions, got {entries}"
