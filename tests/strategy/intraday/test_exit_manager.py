"""Tests for IntradayExitManager."""

from decimal import Decimal

from stockdownloader.model.intraday_signal import IntradayAction, HOLD
from stockdownloader.model.trade import Direction
from stockdownloader.strategy.intraday.base_config import InfraExitConfig
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.trail_strategy import AtrChandelierTrail, BreakevenTrail, VwapRatchetTrail

from tests.strategy.intraday.conftest import (
    make_bar,
    make_long_position_state,
    make_short_position_state,
    make_vwap_bands,
)

_ZERO = Decimal("0")


class TestHardStop:

    def test_long_hard_stop_triggered(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig()
        state = make_long_position_state(
            entry_price=Decimal("500"), stop_loss=Decimal("498"),
        )
        bar = make_bar(low=Decimal("497.50"), high=Decimal("500"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "hard_stop"
        assert state.in_position is False

    def test_short_hard_stop_triggered(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig()
        state = make_short_position_state(
            entry_price=Decimal("500"), stop_loss=Decimal("502"),
        )
        bar = make_bar(high=Decimal("502.50"), low=Decimal("500"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "hard_stop"
        assert state.in_position is False

    def test_long_stop_not_hit(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig()
        state = make_long_position_state(
            entry_price=Decimal("500"), stop_loss=Decimal("498"),
            take_profit=Decimal("510"),
        )
        # Bar stays safely above SL and below TP
        bar = make_bar(
            close=Decimal("500.50"), high=Decimal("501"), low=Decimal("498.50"),
        )
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig == HOLD
        assert state.in_position is True


class TestFixedTP:

    def test_long_take_profit_hit(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig()
        state = make_long_position_state(
            entry_price=Decimal("500"), stop_loss=Decimal("495"),
            take_profit=Decimal("504"),
        )
        bar = make_bar(high=Decimal("504.50"), low=Decimal("500"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "take_profit"

    def test_short_take_profit_hit(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig()
        state = make_short_position_state(
            entry_price=Decimal("500"), stop_loss=Decimal("505"),
            take_profit=Decimal("496"),
        )
        bar = make_bar(low=Decimal("495.50"), high=Decimal("500"))
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "take_profit"

    def test_orb_trail_only_no_tp(self):
        """ORB default (trail_only) has tp=ZERO, so fixed TP never fires."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = InfraExitConfig()
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("495"),
            take_profit=_ZERO,  # trail_only mode → tp is ZERO
            risk_amount=Decimal("5"),
            entry_mode="ORB",
        )
        # High far above entry, but no fixed TP set
        bar = make_bar(
            close=Decimal("501"), high=Decimal("504.50"), low=Decimal("499"),
        )
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        # No fixed TP → should HOLD (trail hasn't triggered yet)
        assert sig == HOLD

    def test_orb_or_range_tp_fires(self):
        """ORB with or_range TP mode fires fixed TP when price hits target."""
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = InfraExitConfig()  # exit manager doesn't read orb_tp_mode
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("495"),
            take_profit=Decimal("504"),  # or_range target
            risk_amount=Decimal("5"),
            entry_mode="ORB",
        )
        bar = make_bar(
            close=Decimal("503"), high=Decimal("504.50"), low=Decimal("499"),
        )
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "take_profit"


class TestBEAndTrail:

    def test_be_triggers_on_sufficient_unrealized(self):
        mgr = IntradayExitManager(VwapRatchetTrail())
        config = InfraExitConfig(be_trigger=Decimal("0.5"), trail_vwap=True)
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("498"),
            take_profit=Decimal("510"),
            risk_amount=Decimal("2"),
        )
        bar = make_bar(
            close=Decimal("501.50"), high=Decimal("501.50"), low=Decimal("500.50"),
        )
        bands = make_vwap_bands(vwap=Decimal("500"))

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert state.be_triggered is True
        assert state.trailing_vwap is True
        assert sig == HOLD  # Trail not hit yet

    def test_orb_be_activates_atr_trail(self):
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = InfraExitConfig(be_trigger=Decimal("0.5"), orb_trail_atr=Decimal("1.5"))
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("498"),
            take_profit=_ZERO,
            risk_amount=Decimal("2"),
            entry_mode="ORB",
        )
        state.orb_extreme = Decimal("502")
        bar = make_bar(
            close=Decimal("501.50"), high=Decimal("502"), low=Decimal("500.50"),
        )
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert state.be_triggered is True
        assert state.trailing_atr is True
        assert sig == HOLD

    def test_be_not_triggered_when_insufficient(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig(be_trigger=Decimal("0.5"))
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("498"),
            take_profit=Decimal("510"),
            risk_amount=Decimal("2"),
        )
        bar = make_bar(
            close=Decimal("500.50"), high=Decimal("500.50"), low=Decimal("499"),
        )
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert state.be_triggered is False

    def test_simple_be_when_not_pb_not_orb(self):
        """Non-PB, non-ORB trade gets simple BE (SL moved to entry + buffer)."""
        mgr = IntradayExitManager(BreakevenTrail())
        config = InfraExitConfig(be_trigger=Decimal("0.5"), trail_vwap=True)
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("498"),
            take_profit=Decimal("510"),
            risk_amount=Decimal("2"),
        )
        bar = make_bar(
            close=Decimal("501.50"), high=Decimal("501.50"), low=Decimal("499"),
        )
        bands = make_vwap_bands()

        mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert state.be_triggered is True
        assert state.trailing_vwap is False
        assert state.trailing_atr is False
        assert state.stop_loss == Decimal("500.05")  # entry + BE_BUF


class TestVwapTrailStop:

    def test_vwap_trail_exit_long(self):
        mgr = IntradayExitManager(VwapRatchetTrail())
        config = InfraExitConfig(trail_buf=Decimal("0.15"))
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("495"),
            take_profit=Decimal("510"),
            risk_amount=Decimal("2"),
        )
        state.be_triggered = True
        state.trailing_vwap = True
        state.trail_level = Decimal("500.10")

        bar = make_bar(
            close=Decimal("500.05"), high=Decimal("500.50"), low=Decimal("500.00"),
        )
        bands = make_vwap_bands(vwap=Decimal("501"))

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "vwap_trail"

    def test_atr_trail_exit_long(self):
        mgr = IntradayExitManager(AtrChandelierTrail())
        config = InfraExitConfig(orb_trail_atr=Decimal("1.5"))
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("495"),
            take_profit=_ZERO,
            entry_mode="ORB",
        )
        state.be_triggered = True
        state.trailing_atr = True
        state.trail_level = Decimal("501")
        state.orb_extreme = Decimal("503")

        bar = make_bar(
            close=Decimal("500.80"), high=Decimal("501.50"), low=Decimal("500.50"),
        )
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "atr_trail"


class TestEOD:

    def test_eod_flattens_position(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig(close_eod=True, bars_per_day=78)
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("490"),
            take_profit=Decimal("520"),
        )
        bar = make_bar(
            close=Decimal("505"), high=Decimal("505"), low=Decimal("499"),
        )
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 78)

        assert sig.action == IntradayAction.EXIT
        assert sig.reason == "eod"

    def test_eod_disabled_does_not_flatten(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig(close_eod=False, bars_per_day=78)
        state = make_long_position_state(
            entry_price=Decimal("500"),
            stop_loss=Decimal("490"),
            take_profit=Decimal("520"),
        )
        bar = make_bar(
            close=Decimal("505"), high=Decimal("505"), low=Decimal("499"),
        )
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 78)

        assert sig == HOLD


class TestExitPnLTracking:

    def test_winning_long_resets_consec_losses(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig()
        state = make_long_position_state(
            entry_price=Decimal("500"), stop_loss=Decimal("495"),
            take_profit=Decimal("504"),
        )
        state.consec_losses = 2
        state.session_pnl = Decimal("-3.00")
        bar = make_bar(high=Decimal("504.50"), low=Decimal("500"))
        bands = make_vwap_bands()

        mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert state.consec_losses == 0
        assert state.session_pnl == Decimal("1.00")  # -3 + 4 = 1

    def test_losing_long_increments_consec_losses(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig()
        state = make_long_position_state(
            entry_price=Decimal("500"), stop_loss=Decimal("498"),
        )
        state.consec_losses = 1
        state.session_pnl = Decimal("0")
        bar = make_bar(low=Decimal("497"), high=Decimal("500"))
        bands = make_vwap_bands()

        mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert state.consec_losses == 2
        assert state.session_pnl == Decimal("-2")  # 498 - 500

    def test_not_in_position_returns_hold(self):
        mgr = IntradayExitManager()
        config = InfraExitConfig()
        state = make_long_position_state()
        state.in_position = False
        bar = make_bar()
        bands = make_vwap_bands()

        sig = mgr.evaluate(bar, state, config, bands, Decimal("1.5"), 20)

        assert sig == HOLD
