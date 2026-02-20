"""Tests for DailyToIntradayAdapter.

Verifies that daily strategies (BUY/SELL/HOLD) are correctly mapped
to intraday signals (ENTER_LONG/ENTER_SHORT/EXIT/HOLD) with proper
ATR-based stop/target levels and position state tracking.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import IntradayAction
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData

# Real data file for integration tests
_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "spy" / "5m_bars.csv"


# ---------------------------------------------------------------------------
# Stub strategy for controlled signal testing
# ---------------------------------------------------------------------------


class _StubStrategy(TradingStrategy):
    """Strategy that returns a pre-programmed sequence of signals."""

    def __init__(self, signals: list[Signal] | None = None) -> None:
        self._signals = signals or []
        self._call_count = 0

    @property
    def name(self) -> str:
        return "Stub"

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if self._call_count < len(self._signals):
            sig = self._signals[self._call_count]
            self._call_count += 1
            return sig
        return Signal.HOLD

    @property
    def warmup_period(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_bars(n: int = 50) -> list[IntradayPriceData]:
    """Create n synthetic 5-minute bars with predictable prices."""
    bars = []
    base = Decimal("500")
    for i in range(n):
        price = base + Decimal(str(i)) * Decimal("0.10")
        bars.append(IntradayPriceData(
            date=f"2025-01-15 09:{30 + (i * 5) // 60:02d}:{(i * 5) % 60:02d}-05:00",
            open=price,
            high=price + Decimal("0.50"),
            low=price - Decimal("0.50"),
            close=price,
            volume=1000,
            adj_close=price,
        ))
    return bars


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdapterBasics:

    def test_name_includes_adapted(self):
        stub = _StubStrategy()
        adapter = DailyToIntradayAdapter(stub)
        assert "adapted" in adapter.name
        assert "Stub" in adapter.name

    def test_warmup_at_least_atr_period(self):
        stub = _StubStrategy()
        adapter = DailyToIntradayAdapter(stub, atr_period=14)
        assert adapter.warmup_period >= 15  # atr_period + 1

    def test_warmup_uses_max_of_strategy_and_atr(self):
        class _LongWarmup(TradingStrategy):
            @property

            def name(self): return "Long"
            def evaluate(self, data, idx): return Signal.HOLD
            @property
            def warmup_period(self): return 50

        adapter = DailyToIntradayAdapter(_LongWarmup(), atr_period=14)
        assert adapter.warmup_period == 50

    def test_hold_during_warmup(self):
        stub = _StubStrategy([Signal.BUY])
        adapter = DailyToIntradayAdapter(stub, atr_period=14)
        bars = _make_bars(10)  # Not enough for warmup
        signal = adapter.evaluate(bars, 5)
        assert signal.action == IntradayAction.HOLD


class TestSignalMapping:

    def test_buy_enters_long(self):
        stub = _StubStrategy([Signal.BUY])
        adapter = DailyToIntradayAdapter(stub, atr_period=5)
        bars = _make_bars(30)
        signal = adapter.evaluate(bars, 20)
        assert signal.action == IntradayAction.ENTER_LONG

    def test_buy_provides_stop_loss_and_take_profit(self):
        stub = _StubStrategy([Signal.BUY])
        adapter = DailyToIntradayAdapter(stub, atr_period=5)
        bars = _make_bars(30)
        signal = adapter.evaluate(bars, 20)
        assert signal.stop_loss > Decimal("0")
        assert signal.take_profit > Decimal("0")
        assert signal.risk_per_share > Decimal("0")
        # Stop loss should be below entry
        assert signal.stop_loss < bars[20].close
        # Take profit should be above entry
        assert signal.take_profit > bars[20].close

    def test_sell_after_buy_exits_long(self):
        stub = _StubStrategy([Signal.BUY, Signal.SELL])
        adapter = DailyToIntradayAdapter(stub, atr_period=5)
        bars = _make_bars(30)

        # First: BUY → ENTER_LONG
        sig1 = adapter.evaluate(bars, 20)
        assert sig1.action == IntradayAction.ENTER_LONG
        # Simulate engine confirming the position was opened
        adapter.on_position_opened(is_long=True)

        # Second: SELL → EXIT
        sig2 = adapter.evaluate(bars, 21)
        assert sig2.action == IntradayAction.EXIT

    def test_sell_no_position_shorts_disabled(self):
        stub = _StubStrategy([Signal.SELL])
        adapter = DailyToIntradayAdapter(stub, atr_period=5, allow_shorts=False)
        bars = _make_bars(30)
        signal = adapter.evaluate(bars, 20)
        assert signal.action == IntradayAction.HOLD

    def test_sell_no_position_shorts_enabled(self):
        stub = _StubStrategy([Signal.SELL])
        adapter = DailyToIntradayAdapter(stub, atr_period=5, allow_shorts=True)
        bars = _make_bars(30)
        signal = adapter.evaluate(bars, 20)
        assert signal.action == IntradayAction.ENTER_SHORT

    def test_short_entry_has_stop_above_and_target_below(self):
        stub = _StubStrategy([Signal.SELL])
        adapter = DailyToIntradayAdapter(stub, atr_period=5, allow_shorts=True)
        bars = _make_bars(30)
        signal = adapter.evaluate(bars, 20)
        assert signal.action == IntradayAction.ENTER_SHORT
        # Stop above entry for short
        assert signal.stop_loss > bars[20].close
        # Target below entry for short
        assert signal.take_profit < bars[20].close

    def test_buy_closes_short_first(self):
        stub = _StubStrategy([Signal.SELL, Signal.BUY])
        adapter = DailyToIntradayAdapter(stub, atr_period=5, allow_shorts=True)
        bars = _make_bars(30)

        # Enter short
        sig1 = adapter.evaluate(bars, 20)
        assert sig1.action == IntradayAction.ENTER_SHORT
        # Simulate engine confirming the position was opened
        adapter.on_position_opened(is_long=False)

        # BUY while short → EXIT (not ENTER_LONG)
        sig2 = adapter.evaluate(bars, 21)
        assert sig2.action == IntradayAction.EXIT

    def test_hold_returns_hold(self):
        stub = _StubStrategy([Signal.HOLD])
        adapter = DailyToIntradayAdapter(stub, atr_period=5)
        bars = _make_bars(30)
        signal = adapter.evaluate(bars, 20)
        assert signal.action == IntradayAction.HOLD

    def test_duplicate_buy_while_long_holds(self):
        stub = _StubStrategy([Signal.BUY, Signal.BUY])
        adapter = DailyToIntradayAdapter(stub, atr_period=5)
        bars = _make_bars(30)

        sig1 = adapter.evaluate(bars, 20)
        assert sig1.action == IntradayAction.ENTER_LONG
        # Simulate engine confirming the position was opened
        adapter.on_position_opened(is_long=True)

        # Already long, another BUY → HOLD
        sig2 = adapter.evaluate(bars, 21)
        assert sig2.action == IntradayAction.HOLD


class TestStopLossComputation:

    def test_stop_loss_capped(self):
        stub = _StubStrategy([Signal.BUY])
        # Very high ATR multiplier but low cap
        adapter = DailyToIntradayAdapter(
            stub, sl_atr_mult=Decimal("100"), sl_cap=Decimal("0.50"), atr_period=5
        )
        bars = _make_bars(30)
        signal = adapter.evaluate(bars, 20)
        entry = bars[20].close
        sl_distance = entry - signal.stop_loss
        assert sl_distance <= Decimal("0.50")

    def test_risk_per_share_equals_sl_distance(self):
        stub = _StubStrategy([Signal.BUY])
        adapter = DailyToIntradayAdapter(stub, atr_period=5)
        bars = _make_bars(30)
        signal = adapter.evaluate(bars, 20)
        entry = bars[20].close
        sl_distance = (entry - signal.stop_loss).quantize(Decimal("0.01"))
        assert signal.risk_per_share == sl_distance


class TestSessionBoundary:

    def test_session_change_detected(self):
        bars = []
        base = Decimal("500")
        # Day 1: bars 0-9
        for i in range(10):
            bars.append(IntradayPriceData(
                date=f"2025-01-15 09:{30 + i * 5}:00-05:00",
                open=base, high=base + Decimal("1"),
                low=base - Decimal("1"), close=base,
                volume=1000, adj_close=base,
            ))
        # Day 2: bars 10-19
        for i in range(10):
            bars.append(IntradayPriceData(
                date=f"2025-01-16 09:{30 + i * 5}:00-05:00",
                open=base, high=base + Decimal("1"),
                low=base - Decimal("1"), close=base,
                volume=1000, adj_close=base,
            ))

        stub = _StubStrategy([Signal.HOLD] * 20)
        adapter = DailyToIntradayAdapter(stub, atr_period=5)

        # Process a bar from day 1
        adapter.evaluate(bars, 8)
        assert adapter._current_session == "2025-01-15"

        # Process a bar from day 2 → session resets
        adapter.evaluate(bars, 15)
        assert adapter._current_session == "2025-01-16"


class TestWrappedDailyStrategies:
    """Verify adapter wraps each real daily strategy without error."""

    def test_sma_crossover_wraps(self):
        from stockdownloader.strategy.daily.sma_crossover_strategy import SMACrossoverStrategy
        strat = SMACrossoverStrategy(short_period=9, long_period=21)
        adapter = DailyToIntradayAdapter(strat)
        assert "SMA" in adapter.name
        assert adapter.warmup_period >= 21

    def test_rsi_wraps(self):
        from stockdownloader.strategy.daily.rsi_strategy import RSIStrategy
        strat = RSIStrategy(period=14, oversold=30.0, overbought=70.0)
        adapter = DailyToIntradayAdapter(strat)
        assert "RSI" in adapter.name
        assert adapter.warmup_period >= 15

    def test_macd_wraps(self):
        from stockdownloader.strategy.daily.macd_strategy import MACDStrategy
        strat = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        adapter = DailyToIntradayAdapter(strat)
        assert "MACD" in adapter.name

    def test_bollinger_rsi_wraps(self):
        from stockdownloader.strategy.daily.bollinger_band_rsi_strategy import BollingerBandRSIStrategy
        strat = BollingerBandRSIStrategy()
        adapter = DailyToIntradayAdapter(strat)
        assert adapter.warmup_period > 0

    def test_breakout_wraps(self):
        from stockdownloader.strategy.daily.breakout_strategy import BreakoutStrategy
        strat = BreakoutStrategy()
        adapter = DailyToIntradayAdapter(strat)
        assert adapter.warmup_period > 0

    def test_momentum_confluence_wraps(self):
        from stockdownloader.strategy.daily.momentum_confluence_strategy import MomentumConfluenceStrategy
        strat = MomentumConfluenceStrategy()
        adapter = DailyToIntradayAdapter(strat)
        assert adapter.warmup_period >= 200

    def test_multi_indicator_wraps(self):
        from stockdownloader.strategy.daily.multi_indicator_strategy import MultiIndicatorStrategy
        strat = MultiIndicatorStrategy()
        adapter = DailyToIntradayAdapter(strat)
        assert adapter.warmup_period > 0


class TestCircuitBreaker:
    """Tests for the consecutive-loss circuit-breaker mechanism."""

    def _make_multi_day_bars(self, n_days: int = 3, bars_per_day: int = 20):
        """Create bars spanning multiple trading days with realistic ATR."""
        bars: list[IntradayPriceData] = []
        base = Decimal("500")
        for day in range(n_days):
            date = f"2025-01-{15 + day:02d}"
            for i in range(bars_per_day):
                price = base + Decimal(str(i)) * Decimal("0.10")
                bars.append(IntradayPriceData(
                    date=f"{date} 09:{30 + (i * 5) // 60:02d}:{(i * 5) % 60:02d}-05:00",
                    open=price,
                    high=price + Decimal("0.50"),
                    low=price - Decimal("0.50"),
                    close=price,
                    volume=1000,
                    adj_close=price,
                ))
        return bars

    def test_circuit_breaker_trips_after_consecutive_losses(self):
        """After max_consecutive_losses, new entries are blocked."""
        adapter = DailyToIntradayAdapter(
            _StubStrategy([Signal.BUY] * 20),
            atr_period=5,
            max_consecutive_losses=2,
        )
        bars = _make_bars(50)

        # Trade 1: enter long
        sig = adapter.evaluate(bars, 20)
        assert sig.action == IntradayAction.ENTER_LONG
        adapter.on_position_opened(is_long=True)

        # Close trade 1 as a loss (simulate price dropped)
        adapter._last_bar_price = bars[20].close - Decimal("5.00")
        adapter.on_position_closed()
        assert adapter._consecutive_losses == 1
        assert not adapter._circuit_tripped

        # Trade 2: enter long
        sig = adapter.evaluate(bars, 22)
        assert sig.action == IntradayAction.ENTER_LONG
        adapter.on_position_opened(is_long=True)

        # Close trade 2 as a loss
        adapter._last_bar_price = bars[22].close - Decimal("5.00")
        adapter.on_position_closed()
        assert adapter._consecutive_losses == 2
        assert adapter._circuit_tripped  # Should be tripped now

        # Trade 3: should be blocked
        sig = adapter.evaluate(bars, 24)
        assert sig.action == IntradayAction.HOLD

    def test_circuit_breaker_resets_on_win(self):
        """A winning trade resets the consecutive loss counter."""
        adapter = DailyToIntradayAdapter(
            _StubStrategy([Signal.BUY] * 20),
            atr_period=5,
            max_consecutive_losses=3,
        )
        bars = _make_bars(50)

        # Trade 1: loss
        sig = adapter.evaluate(bars, 20)
        adapter.on_position_opened(is_long=True)
        adapter._last_bar_price = bars[20].close - Decimal("5.00")
        adapter.on_position_closed()
        assert adapter._consecutive_losses == 1

        # Trade 2: win
        sig = adapter.evaluate(bars, 22)
        adapter.on_position_opened(is_long=True)
        adapter._last_bar_price = bars[22].close + Decimal("5.00")
        adapter.on_position_closed()
        assert adapter._consecutive_losses == 0
        assert not adapter._circuit_tripped

    def test_circuit_breaker_resets_on_new_session(self):
        """Circuit-breaker resets on a new trading day."""
        bars = self._make_multi_day_bars(n_days=2, bars_per_day=30)
        adapter = DailyToIntradayAdapter(
            _StubStrategy([Signal.BUY] * 50),
            atr_period=5,
            max_consecutive_losses=2,
        )

        # Trip the circuit breaker in day 1
        sig = adapter.evaluate(bars, 20)
        adapter.on_position_opened(is_long=True)
        adapter._last_bar_price = bars[20].close - Decimal("5.00")
        adapter.on_position_closed()

        sig = adapter.evaluate(bars, 22)
        adapter.on_position_opened(is_long=True)
        adapter._last_bar_price = bars[22].close - Decimal("5.00")
        adapter.on_position_closed()
        assert adapter._circuit_tripped

        # Move to day 2 — circuit should reset
        day2_idx = 30  # First bar of day 2
        sig = adapter.evaluate(bars, day2_idx + 20)
        # Session change triggers reset, so new entries should be allowed
        assert adapter._consecutive_losses == 0
        assert not adapter._circuit_tripped

    def test_circuit_breaker_disabled_with_zero(self):
        """Setting max_consecutive_losses=0 disables the circuit-breaker."""
        adapter = DailyToIntradayAdapter(
            _StubStrategy([Signal.BUY] * 20),
            atr_period=5,
            max_consecutive_losses=0,
        )
        bars = _make_bars(50)

        # 5 consecutive losses
        for i in range(5):
            sig = adapter.evaluate(bars, 20 + i * 2)
            if sig.action == IntradayAction.ENTER_LONG:
                adapter.on_position_opened(is_long=True)
                adapter._last_bar_price = bars[20 + i * 2].close - Decimal("5.00")
                adapter.on_position_closed()

        # Should NOT be tripped
        assert not adapter._circuit_tripped

    def test_circuit_breaker_short_loss_detection(self):
        """Loss detection works correctly for short positions."""
        adapter = DailyToIntradayAdapter(
            _StubStrategy([Signal.SELL] * 10),
            atr_period=5,
            allow_shorts=True,
            max_consecutive_losses=2,
        )
        bars = _make_bars(50)

        # Short trade: loss occurs when price goes UP
        sig = adapter.evaluate(bars, 20)
        assert sig.action == IntradayAction.ENTER_SHORT
        adapter.on_position_opened(is_long=False)

        # Price went up = loss for short
        adapter._last_bar_price = bars[20].close + Decimal("5.00")
        adapter.on_position_closed()
        assert adapter._consecutive_losses == 1

    def test_circuit_allows_exit_of_existing_position(self):
        """Even when tripped, EXIT signals should still work."""
        adapter = DailyToIntradayAdapter(
            _StubStrategy([Signal.BUY, Signal.BUY, Signal.SELL]),
            atr_period=5,
            max_consecutive_losses=1,
        )
        bars = _make_bars(50)

        # Trade 1: enter, lose, trip circuit
        sig = adapter.evaluate(bars, 20)
        adapter.on_position_opened(is_long=True)
        adapter._last_bar_price = bars[20].close - Decimal("5.00")
        adapter.on_position_closed()
        assert adapter._circuit_tripped

        # BUY blocked by circuit
        sig = adapter.evaluate(bars, 22)
        assert sig.action == IntradayAction.HOLD


class TestIntegration:
    """Integration tests using real data file."""

    @pytest.fixture(scope="class")
    def real_data(self):
        if not _DATA_FILE.exists():
            pytest.skip("data/spy/5m_bars.csv not found")
        from stockdownloader.data.intraday_csv import IntradayCsvLoader
        data = IntradayCsvLoader.load_from_file(_DATA_FILE)
        if len(data) < 500:
            pytest.skip("Not enough data for integration tests")
        return data[:2000]

    def test_full_backtest_with_sma_adapter(self, real_data):
        """Run a full backtest with an adapted SMA strategy."""
        from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
        from stockdownloader.backtest.backtest_result import BacktestResult
        from stockdownloader.strategy.daily.sma_crossover_strategy import SMACrossoverStrategy

        strat = SMACrossoverStrategy(short_period=9, long_period=21)
        adapter = DailyToIntradayAdapter(strat)

        engine = IntradayBacktestEngine(Decimal("100000"), Decimal("0.01"))
        result = engine.run(adapter, real_data)

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "SMA Crossover (9/21) (adapted)"

    def test_full_backtest_with_rsi_adapter(self, real_data):
        """Run a full backtest with an adapted RSI strategy."""
        from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
        from stockdownloader.backtest.backtest_result import BacktestResult
        from stockdownloader.strategy.daily.rsi_strategy import RSIStrategy

        strat = RSIStrategy(period=14, oversold=30.0, overbought=70.0)
        adapter = DailyToIntradayAdapter(strat)

        engine = IntradayBacktestEngine(Decimal("100000"), Decimal("0.01"))
        result = engine.run(adapter, real_data)

        assert isinstance(result, BacktestResult)
        assert "RSI" in result.strategy_name
        assert "(adapted)" in result.strategy_name
