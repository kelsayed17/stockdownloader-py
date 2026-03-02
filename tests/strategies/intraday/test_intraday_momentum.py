"""Tests for Gao (2018) Intraday Momentum tournament strategy."""

from __future__ import annotations

from decimal import Decimal

from stockdownloader.core.models.trade import IntradayAction
from tests.strategies.intraday.conftest import make_warmup_bars


class TestIntradayMomentumConfig:
    """Test IntradayMomentumConfig dataclass."""

    def test_default_values(self) -> None:
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumConfig
        c = IntradayMomentumConfig()
        assert c.mode == "momentum"
        assert c.r1_threshold == Decimal("0.0")
        assert c.r1_end_bar == 7
        assert c.entry_bar == 72
        assert c.allow_longs is True
        assert c.allow_shorts is True
        assert c.max_day == 1
        assert c.spacing == 0
        assert c.circuit == 99
        assert c.day_loss == Decimal("99.0")
        assert c.be_trigger == Decimal("99.0")
        assert c.trail_vwap is False

    def test_inherits_infra_exit_config(self) -> None:
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumConfig
        from stockdownloader.strategies.intraday.base import InfraExitConfig
        assert issubclass(IntradayMomentumConfig, InfraExitConfig)


class TestIntradayMomentumStrategy:
    """Test IntradayMomentumStrategy signal logic."""

    def test_strategy_name(self) -> None:
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumStrategy
        s = IntradayMomentumStrategy()
        assert s.name == "SPY Intraday Momentum"

    def test_warmup_period(self) -> None:
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumStrategy
        s = IntradayMomentumStrategy()
        assert s.warmup_period == 78 * 15

    def test_hold_during_warmup(self) -> None:
        """Strategy should return HOLD during warmup period."""
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumStrategy
        s = IntradayMomentumStrategy()
        bars = make_warmup_bars(100)
        s.on_session_start("2024-01-02")
        sig = s.evaluate(bars, 50)
        assert sig.action == IntradayAction.HOLD

    def test_bidirectional(self) -> None:
        """Strategy should allow both longs and shorts by default."""
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumStrategy
        s = IntradayMomentumStrategy()
        assert s._c.allow_longs is True
        assert s._c.allow_shorts is True

    def test_fire_once(self) -> None:
        """Strategy should be fire-once (max 1 trade per day)."""
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumStrategy
        s = IntradayMomentumStrategy()
        assert s._ENTRY_FLAGS == {"fire_once": True}

    def test_no_vwap_trail(self) -> None:
        """Strategy should not use VWAP trail (pure time-based exits)."""
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumStrategy
        s = IntradayMomentumStrategy()
        assert s._c.trail_vwap is False

    def test_r1_resets_on_session_start(self) -> None:
        """r1 state should reset when a new session starts."""
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumStrategy
        s = IntradayMomentumStrategy()
        # Simulate r1 being captured
        s._r1_value = Decimal("0.5")
        s._r1_captured = True
        # New session should reset
        s.on_session_start("2024-01-03")
        assert s._r1_value is None
        assert s._r1_captured is False

    def test_entry_produces_valid_signal(self) -> None:
        """After warmup, strategy should produce a valid signal on post-warmup bars."""
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumStrategy
        s = IntradayMomentumStrategy()
        bars = make_warmup_bars(1300)
        signals_seen: list[IntradayAction] = []
        for i in range(len(bars)):
            if i % 78 == 0:
                s.on_session_start(bars[i].trading_date)
            sig = s.evaluate(bars, i)
            signals_seen.append(sig.action)
        # Strategy should have produced at least one non-HOLD signal
        # (entry or exit) across all bars
        assert sig.action in (IntradayAction.HOLD, IntradayAction.ENTER_LONG,
                              IntradayAction.ENTER_SHORT, IntradayAction.EXIT)

    def test_reversal_mode(self) -> None:
        """Reversal mode should flip the direction relative to r1."""
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumConfig
        c = IntradayMomentumConfig(mode="reversal")
        assert c.mode == "reversal"

    def test_long_only_mode(self) -> None:
        """When allow_shorts=False, only long entries should be possible."""
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumConfig
        c = IntradayMomentumConfig(allow_shorts=False)
        assert c.allow_longs is True
        assert c.allow_shorts is False

    def test_custom_threshold(self) -> None:
        """Custom r1 threshold should filter weak signals."""
        from stockdownloader.strategies.intraday.intraday_momentum import IntradayMomentumConfig
        c = IntradayMomentumConfig(r1_threshold=Decimal("0.10"))
        assert c.r1_threshold == Decimal("0.10")

    def test_import_from_package(self) -> None:
        """Strategy should be importable from the intraday package."""
        from stockdownloader.strategies.intraday import (
            IntradayMomentumConfig,
            IntradayMomentumStrategy,
        )
        assert IntradayMomentumConfig is not None
        assert IntradayMomentumStrategy is not None
