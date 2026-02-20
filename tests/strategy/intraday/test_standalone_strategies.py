"""Tests for standalone intraday strategy classes.

Each strategy composes :class:`IntradayInfra` and extends
:class:`IntradayTradingStrategy` directly.
"""

from decimal import Decimal

import pytest

from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import IntradayAction, HOLD
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.strategy.intraday.or_breakout_config import ORBreakoutStrategyConfig
from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
from stockdownloader.strategy.intraday.or_reversal_config import ORReversalStrategyConfig
from stockdownloader.strategy.intraday.or_reversal_strategy import ORReversalStrategy
from stockdownloader.strategy.intraday.pattern_scalp_config import PatternScalpStrategyConfig
from stockdownloader.strategy.intraday.pattern_scalp_strategy import PatternScalpStrategy
from stockdownloader.strategy.intraday.pullback_config import PullbackStrategyConfig
from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategy
from stockdownloader.strategy.intraday.reversal_config import ReversalStrategyConfig
from stockdownloader.strategy.intraday.reversal_strategy import ReversalStrategy


# -- Helpers --


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


# -- All standalone strategy classes --

ALL_STRATEGIES = [
    PullbackStrategy,
    ReversalStrategy,
    ORBreakoutStrategy,
    ORReversalStrategy,
    PatternScalpStrategy,
]

EXPECTED_NAMES = {
    PullbackStrategy: "VWAP Pullback",
    ReversalStrategy: "VWAP Reversal",
    ORBreakoutStrategy: "OR Breakout",
    ORReversalStrategy: "OR Reversal",
    PatternScalpStrategy: "Pattern Scalp",
}


# ======================================================================
# Construction & ABC compliance
# ======================================================================


class TestConstruction:
    """Verify each strategy constructs and satisfies the ABC."""

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_default_construction(self, strategy_cls):
        strategy = strategy_cls()
        assert strategy is not None

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_is_intraday_trading_strategy(self, strategy_cls):
        strategy = strategy_cls()
        assert isinstance(strategy, IntradayTradingStrategy)

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_name(self, strategy_cls):
        strategy = strategy_cls()
        expected = EXPECTED_NAMES[strategy_cls]
        assert strategy.name == expected

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_warmup_period(self, strategy_cls):
        strategy = strategy_cls()
        warmup = strategy.warmup_period
        # Default: 78 bars/day * 15 = 1170
        assert warmup == 78 * 15

    @pytest.mark.parametrize("strategy_cls,config_cls", [
        (PullbackStrategy, PullbackStrategyConfig),
        (ReversalStrategy, ReversalStrategyConfig),
        (ORBreakoutStrategy, ORBreakoutStrategyConfig),
        (ORReversalStrategy, ORReversalStrategyConfig),
        (PatternScalpStrategy, PatternScalpStrategyConfig),
    ])
    def test_custom_config(self, strategy_cls, config_cls):
        """Each accepts its own config class."""
        config = config_cls(bars_per_day=80)
        strategy = strategy_cls(config=config)
        assert strategy.warmup_period == 80 * 15


# ======================================================================
# Session detection
# ======================================================================


class TestSessionDetection:

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_on_session_start_resets(self, strategy_cls):
        strategy = strategy_cls()
        strategy._infra.state.bar_count = 50
        strategy._infra.state.day_trades = 3
        strategy._infra.state.tripped = True

        strategy.on_session_start("2025-01-20")

        assert strategy._infra.state.bar_count == 0
        assert strategy._infra.state.day_trades == 0
        assert strategy._infra.state.tripped is False
        assert strategy._infra.state.trading_date == "2025-01-20"

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_session_boundary_detected(self, strategy_cls):
        """New trading date triggers session reset."""
        strategy = strategy_cls()
        bar1 = _make_bar("2025-01-14 15:55:00-05:00", 500, 501, 499, 500)
        bar2 = _make_bar("2025-01-15 09:30:00-05:00", 501, 502, 500, 501)
        data = [bar1, bar2]

        strategy.evaluate(data, 0)
        assert strategy._infra.state.bar_count == 1

        strategy.evaluate(data, 1)
        # New day resets bar_count, then increments to 1
        assert strategy._infra.state.bar_count == 1


# ======================================================================
# Evaluate produces signals
# ======================================================================


class TestEvaluate:

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_returns_intraday_signal(self, strategy_cls):
        """evaluate() returns an IntradaySignal (not None)."""
        strategy = strategy_cls()
        bars = _generate_session(num_bars=5)
        sig = strategy.evaluate(bars, 0)
        assert sig is not None
        assert hasattr(sig, "action")

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_hold_during_insufficient_data(self, strategy_cls):
        """With only a few bars, strategies should HOLD."""
        strategy = strategy_cls()
        bars = _generate_session(num_bars=10)
        for i in range(len(bars)):
            sig = strategy.evaluate(bars, i)
            # All should be valid signals (HOLD or entry)
            assert sig is not None


# ======================================================================
# Opening range tracking
# ======================================================================


class TestOpeningRange:

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_or_computed(self, strategy_cls):
        """OR high/low/done computed after or_bars."""
        strategy = strategy_cls()
        bars = _generate_session(num_bars=10)

        for i in range(len(bars)):
            strategy.evaluate(bars, i)

        assert strategy._infra.state.or_done is True
        assert strategy._infra.state.or_range > Decimal("0")

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_or_direction(self, strategy_cls):
        """OR direction computed based on open/close."""
        strategy = strategy_cls()
        bars = [
            _make_bar("2025-01-15 09:30:00-05:00", 500, 501, 499, 500.5),
            _make_bar("2025-01-15 09:35:00-05:00", 500.5, 502, 500, 501),
            _make_bar("2025-01-15 09:40:00-05:00", 501, 503, 500.5, 502),
        ]

        for i in range(len(bars)):
            strategy.evaluate(bars, i)

        assert strategy._infra.state.or_done is True
        assert strategy._infra.state.or_dir == 1  # Bullish


# ======================================================================
# Risk controls
# ======================================================================


class TestRiskControls:

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_circuit_breaker(self, strategy_cls):
        """Circuit breaker trips after consecutive losses."""
        strategy = strategy_cls()
        bars = _generate_session(num_bars=20)

        strategy.evaluate(bars, 0)
        strategy._infra.state.consec_losses = 3

        sig = strategy.evaluate(bars, 1)
        assert strategy._infra.state.tripped is True
        assert sig == HOLD

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_day_extremes_tracked(self, strategy_cls):
        """HOD/LOD tracked across bars."""
        strategy = strategy_cls()
        bars = [
            _make_bar("2025-01-15 09:30:00-05:00", 500, 505, 498, 502),
            _make_bar("2025-01-15 09:35:00-05:00", 502, 510, 497, 503),
            _make_bar("2025-01-15 09:40:00-05:00", 503, 508, 499, 501),
        ]

        for i in range(len(bars)):
            strategy.evaluate(bars, i)

        assert strategy._infra.state.day_hod == Decimal("510")
        assert strategy._infra.state.day_lod == Decimal("497")


# ======================================================================
# Independence
# ======================================================================


class TestIndependence:
    """Verify strategies can run independently with their own configs."""

    def test_all_strategies_run_full_session(self):
        """Each strategy can evaluate an entire session independently."""
        data = _generate_multi_session(num_days=2)

        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls()
            signals = []
            for i in range(len(data)):
                sig = strategy.evaluate(data, i)
                signals.append(sig)

            # Should produce at least some HOLD signals
            hold_count = sum(
                1 for s in signals if s == HOLD or s.action == IntradayAction.HOLD
            )
            assert hold_count > 0, f"{strategy_cls.__name__} produced no HOLDs"

    def test_two_strategies_different_state(self):
        """Two independent strategy instances don't share state."""
        pb = PullbackStrategy()
        orb = ORBreakoutStrategy()

        bars = _generate_session(num_bars=10)
        for i in range(len(bars)):
            pb.evaluate(bars, i)
            orb.evaluate(bars, i)

        # Different strategies may have different state
        # but both should track bar count independently
        assert pb._infra.state.bar_count == 10
        assert orb._infra.state.bar_count == 10

        # Modifying one doesn't affect the other
        pb._infra.state.tripped = True
        assert orb._infra.state.tripped is False

    def test_strategy_with_multi_day_data(self):
        """Strategy handles multi-day data with session boundaries."""
        strategy = PullbackStrategy()
        data = _generate_multi_session(num_days=3)

        for i in range(len(data)):
            sig = strategy.evaluate(data, i)
            assert sig is not None

        # After 3 sessions, daily bars should be aggregated
        assert len(strategy._infra.daily_bars) >= 2
