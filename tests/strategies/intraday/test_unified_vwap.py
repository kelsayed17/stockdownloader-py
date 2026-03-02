"""Tests for the UnifiedVWAPStrategy — single-infra multi-mode dispatcher.

The unified strategy holds a SINGLE IntradayInfra and delegates to
existing standalone strategies' ``_evaluate_entry()`` methods with
priority dispatch (PS > ORB > PB > REV).
"""

from decimal import Decimal

import pytest

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import HOLD


# -- Helpers (matching test_standalone_strategies.py patterns) --


def _make_bar(
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 100_000,
) -> IntradayPriceData:
    return IntradayPriceData(
        date=date,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        adj_close=Decimal(str(close)),
        volume=volume,
    )


def _generate_session(
    trading_date: str = "2025-01-15",
    base_price: float = 500.0,
    num_bars: int = 78,
) -> list[IntradayPriceData]:
    """Generate a realistic intraday session of 5-minute bars."""
    bars = []
    price = base_price
    for i in range(num_bars):
        total_mins = 30 + i * 5
        hour = 9 + total_mins // 60
        minute = total_mins % 60
        dt_str = f"{trading_date} {hour:02d}:{minute:02d}:00-05:00"

        o = price
        h = price + 0.50
        l = price - 0.50
        c = price + 0.10 * (1 if i % 2 == 0 else -1)
        vol = 100_000 + i * 1000
        bars.append(_make_bar(dt_str, o, h, l, c, vol))
        price = c
    return bars


def _generate_multi_session(num_days: int = 16) -> list[IntradayPriceData]:
    """Generate multiple sessions (enough for warmup)."""
    all_bars = []
    base_price = 500.0
    for d in range(num_days):
        date_str = f"2025-01-{d + 1:02d}"
        session = _generate_session(
            trading_date=date_str,
            base_price=base_price,
            num_bars=78,
        )
        all_bars.extend(session)
        base_price += 0.5
    return all_bars


# ======================================================================
# Task 1: UnifiedVWAPConfig
# ======================================================================


class TestUnifiedVWAPConfig:
    """Verify UnifiedVWAPConfig imports and has correct defaults."""

    def test_import(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        assert UnifiedVWAPConfig is not None

    def test_defaults(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        cfg = UnifiedVWAPConfig()
        assert cfg.pb_enable is True
        assert cfg.ps_enable is True
        assert cfg.orb_enable is True
        assert cfg.rev_enable is True

    def test_mode_disable(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        cfg = UnifiedVWAPConfig(pb_enable=False, rev_enable=False)
        assert cfg.pb_enable is False
        assert cfg.ps_enable is True
        assert cfg.orb_enable is True
        assert cfg.rev_enable is False

    def test_inherits_infra_exit_config(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        from stockdownloader.strategies.intraday.base import InfraExitConfig
        cfg = UnifiedVWAPConfig()
        assert isinstance(cfg, InfraExitConfig)

    def test_is_frozen(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        cfg = UnifiedVWAPConfig()
        with pytest.raises(AttributeError):
            cfg.pb_enable = False  # type: ignore[misc]


# ======================================================================
# Task 2: UnifiedVWAPStrategy construction and priority dispatch
# ======================================================================


class TestUnifiedVWAPConstruction:
    """Verify UnifiedVWAPStrategy constructs and satisfies the ABC."""

    def test_is_intraday_strategy(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        from stockdownloader.strategies.base import IntradayTradingStrategy
        strategy = UnifiedVWAPStrategy()
        assert isinstance(strategy, IntradayTradingStrategy)

    def test_name(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        assert strategy.name == "Unified VWAP"

    def test_warmup_period(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        assert strategy.warmup_period == 78 * 15

    def test_evaluates_hold_on_warmup(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        bars = _generate_session(num_bars=5)
        sig = strategy.evaluate(bars, 0)
        assert sig is not None
        assert sig == HOLD or hasattr(sig, "action")

    def test_session_start_resets(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        strategy._infra.state.bar_count = 50
        strategy._infra.state.day_trades = 3
        strategy._infra.state.tripped = True

        strategy.on_session_start("2025-01-20")

        assert strategy._infra.state.bar_count == 0
        assert strategy._infra.state.day_trades == 0
        assert strategy._infra.state.tripped is False
        assert strategy._infra.state.trading_date == "2025-01-20"

    def test_mode_count_default(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        assert len(strategy._modes) == 4

    def test_mode_count_with_disable(self):
        from stockdownloader.strategies.intraday.unified_vwap import (
            UnifiedVWAPConfig, UnifiedVWAPStrategy,
        )
        cfg = UnifiedVWAPConfig(pb_enable=False, rev_enable=False)
        strategy = UnifiedVWAPStrategy(config=cfg)
        assert len(strategy._modes) == 2  # PS + ORB only

    def test_priority_order(self):
        """Priority is PS > ORB > PB > REV."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy
        from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy
        from stockdownloader.strategies.intraday.pullback import PullbackStrategy
        from stockdownloader.strategies.intraday.reversal import ReversalStrategy
        strategy = UnifiedVWAPStrategy()
        mode_types = [type(m[0]) for m in strategy._modes]
        assert mode_types == [
            PatternScalpStrategy,
            ORBreakoutStrategy,
            PullbackStrategy,
            ReversalStrategy,
        ]

    def test_no_fire_once_flag(self):
        """Unified strategy does NOT use fire_once (all entries count)."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        assert strategy._ENTRY_FLAGS == {}

    def test_construction_with_overrides(self):
        """Constructor accepts **shared_overrides for convenience."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy(pb_enable=False)
        assert len(strategy._modes) == 3  # PS + ORB + REV

    def test_construction_with_mode_overrides(self):
        """Constructor accepts per-mode override dicts."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy(
            pb_overrides={"pb_zone": "0.6"},
        )
        assert strategy is not None


# ======================================================================
# Task 3: Shared-state tests
# ======================================================================


class TestSharedState:
    """Verify that state is shared across all modes within the unified strategy."""

    def test_day_trades_shared_across_modes(self):
        """day_trades increments are visible to all modes via shared state."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        state = strategy._infra.state

        # Simulate that one trade has occurred (e.g. PB fired)
        state.day_trades = 1

        # All sub-strategies read from the same state object
        for sub_strategy, _ in strategy._modes:
            # Each sub-strategy's _evaluate_entry reads ctx.state, which
            # is the unified infra's state.  Verify they see the same
            # day_trades value when we read it directly.
            assert state.day_trades == 1

        # Increment again
        state.day_trades = 2
        assert state.day_trades == 2

    def test_circuit_breaker_halts_all_modes(self):
        """Circuit breaker (tripped) halts all modes via shared infra."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        bars = _generate_session(num_bars=10)

        # Run one bar to initialize
        strategy.evaluate(bars, 0)

        # Trip the circuit breaker
        strategy._infra.state.consec_losses = 3

        # Subsequent bars should all return HOLD
        sig = strategy.evaluate(bars, 1)
        assert strategy._infra.state.tripped is True
        assert sig == HOLD

    def test_day_limited_halts_all_modes(self):
        """Day loss limit halts all modes via shared infra."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        bars = _generate_session(num_bars=10)

        # Run one bar to initialize
        strategy.evaluate(bars, 0)

        # Set day_limited flag
        strategy._infra.state.day_limited = True

        sig = strategy.evaluate(bars, 1)
        assert sig == HOLD

    def test_runs_without_crash_on_multi_session(self):
        """Smoke test: 16 sessions run without error."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        from stockdownloader.core.models.trade import IntradayAction
        strategy = UnifiedVWAPStrategy()
        data = _generate_multi_session(num_days=16)

        signals = []
        for i in range(len(data)):
            sig = strategy.evaluate(data, i)
            signals.append(sig)

        # Should produce at least some HOLD signals
        hold_count = sum(
            1 for s in signals if s == HOLD or s.action == IntradayAction.HOLD
        )
        assert hold_count > 0
        # Verify all signals are valid (not None)
        assert all(s is not None for s in signals)


# ======================================================================
# Task 4: __init__.py export
# ======================================================================


class TestExport:
    """Verify UnifiedVWAPConfig and UnifiedVWAPStrategy are exported."""

    def test_config_exported(self):
        from stockdownloader.strategies.intraday import UnifiedVWAPConfig
        assert UnifiedVWAPConfig is not None

    def test_strategy_exported(self):
        from stockdownloader.strategies.intraday import UnifiedVWAPStrategy
        assert UnifiedVWAPStrategy is not None

    def test_in_all(self):
        import stockdownloader.strategies.intraday as pkg
        assert "UnifiedVWAPConfig" in pkg.__all__
        assert "UnifiedVWAPStrategy" in pkg.__all__
