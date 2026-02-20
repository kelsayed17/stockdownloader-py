"""Tests for the ML Oversold strategy.

Uses mocking to avoid requiring a trained model file on disk.
``MLPredictor.from_path`` is patched at module level so the strategy
can be constructed with a fake predictor that returns a fixed probability.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import HOLD, IntradayAction

_D = Decimal


# ======================================================================
# Helpers
# ======================================================================


def _bar(
    dt: str,
    close: float,
    high: float | None = None,
    low: float | None = None,
    vol: int = 100_000,
) -> IntradayPriceData:
    """Create a single synthetic intraday bar."""
    c = _D(str(close))
    h = _D(str(high if high is not None else close + 0.5))
    l = _D(str(low if low is not None else close - 0.5))
    o = c
    return IntradayPriceData(
        date=dt, open=o, high=h, low=l, close=c, adj_close=c, volume=vol,
    )


def _make_data(
    n: int = 300,
    base_price: float = 550.0,
    dt_base: str = "2025-01-15",
) -> list[IntradayPriceData]:
    """Create *n* bars of synthetic 5-minute data spanning multiple days.

    Generates enough bars to clear typical warmup requirements when
    *n* >= 250.  Each day contains 78 bars (09:30 - 16:00 ET).
    """
    bars: list[IntradayPriceData] = []
    bars_per_day = 78
    price = base_price

    for i in range(n):
        day_offset = i // bars_per_day
        bar_in_day = i % bars_per_day

        day_num = 15 + day_offset
        month = 1
        if day_num > 28:
            month = 2
            day_num -= 28

        total_mins = 30 + bar_in_day * 5
        hour = 9 + total_mins // 60
        minute = total_mins % 60

        dt = f"2025-{month:02d}-{day_num:02d} {hour:02d}:{minute:02d}:00-05:00"
        price = base_price + (i % 10) * 0.1
        bars.append(_bar(dt, price))

    return bars


def _make_strategy(prob: float = 0.7, config=None):
    """Create an ``MLOversoldStrategy`` with a mocked predictor.

    Patches ``MLPredictor.from_path`` so the strategy never touches disk.
    The returned mock predictor always returns *prob* for ``predict_proba``.
    """
    with patch(
        "stockdownloader.strategy.intraday.ml_oversold_strategy.MLPredictor",
    ) as MockPredictor:
        mock_pred = MagicMock()
        mock_pred.predict_proba.return_value = prob
        mock_pred.is_available = True
        MockPredictor.from_path.return_value = mock_pred

        from stockdownloader.strategy.intraday.ml_oversold_strategy import (
            MLOversoldStrategy,
        )

        strategy = MLOversoldStrategy(config=config)

    # Ensure the mock predictor survives beyond the patch context manager.
    strategy._predictor = mock_pred
    return strategy


# ======================================================================
# Tests
# ======================================================================


class TestInstantiation:
    """Verify the strategy can be constructed with a mocked predictor."""

    def test_instantiation(self) -> None:
        strategy = _make_strategy()
        assert strategy is not None
        assert strategy.name == "ML Oversold"

    def test_has_required_methods(self) -> None:
        strategy = _make_strategy()
        assert callable(strategy.evaluate)
        assert callable(strategy.on_session_start)
        assert callable(strategy.on_position_opened)
        assert callable(strategy.on_position_closed)

    def test_warmup_period_positive(self) -> None:
        strategy = _make_strategy()
        assert strategy.warmup_period > 0

    def test_custom_config(self) -> None:
        from stockdownloader.strategy.intraday.ml_oversold_config import (
            MLOversoldConfig,
        )

        cfg = MLOversoldConfig(ml_threshold=_D("0.80"))
        strategy = _make_strategy(config=cfg)
        assert strategy._c.ml_threshold == _D("0.80")


class TestEntryThreshold:
    """Verify ML probability threshold gates entry decisions."""

    def test_entry_below_threshold_returns_hold(self) -> None:
        """Probability below threshold should produce HOLD."""
        strategy = _make_strategy(prob=0.40)
        data = _make_data(n=300)
        sig = strategy.evaluate(data, 250)
        assert sig.action == IntradayAction.HOLD

    def test_entry_above_threshold_returns_long(self) -> None:
        """Probability above threshold should produce ENTER_LONG (when
        all other guards pass).
        """
        strategy = _make_strategy(prob=0.75)
        data = _make_data(n=300)

        # Walk through enough bars for the infra to build state, then
        # check the last bar for an entry signal.
        last_sig = HOLD
        for i in range(len(data)):
            sig = strategy.evaluate(data, i)
            if sig.action == IntradayAction.ENTER_LONG:
                last_sig = sig
                break

        assert last_sig.action == IntradayAction.ENTER_LONG


class TestWarmupGuard:
    """Infra warmup period must be respected."""

    def test_warmup_guard(self) -> None:
        """Evaluating at an index below the warmup period returns HOLD,
        regardless of how high the predicted probability is.
        """
        strategy = _make_strategy(prob=0.99)
        data = _make_data(n=300)
        sig = strategy.evaluate(data, 100)
        assert sig.action == IntradayAction.HOLD


class TestDayTradeLimit:
    """Session-level day trade cap must prevent excess entries."""

    def test_day_trade_limit(self) -> None:
        """After exhausting max_day entries, strategy should HOLD."""
        from stockdownloader.strategy.intraday.ml_oversold_config import (
            MLOversoldConfig,
        )

        cfg = MLOversoldConfig(max_day=2)
        strategy = _make_strategy(prob=0.85, config=cfg)

        data = _make_data(n=300)

        # Walk through bars to let infra establish session state.
        # Stop at a point safely inside the last session.
        bars_per_day = 78
        last_session_start = (len(data) // bars_per_day) * bars_per_day
        mid_session = last_session_start + 20  # well inside the session

        for i in range(mid_session):
            strategy.evaluate(data, i)

        # Now artificially exhaust the day trade limit for this session.
        strategy._infra.state.day_trades = cfg.max_day

        # Remaining bars in this session should never produce entries.
        session_end = min(last_session_start + bars_per_day, len(data))
        for i in range(mid_session, session_end):
            sig = strategy.evaluate(data, i)
            assert sig.action in (IntradayAction.HOLD, IntradayAction.EXIT), (
                f"Expected HOLD or EXIT at idx={i}, got {sig.action}"
            )


class TestStopLossAndTakeProfit:
    """Verify SL/TP levels are set on entry signals."""

    def test_sl_tp_vwap_mode(self) -> None:
        """When an ENTER_LONG signal fires, SL should be below close
        and TP should be set (non-zero).
        """
        strategy = _make_strategy(prob=0.85)
        data = _make_data(n=300)

        for i in range(len(data)):
            sig = strategy.evaluate(data, i)
            if sig.action == IntradayAction.ENTER_LONG:
                close = data[i].close
                assert sig.stop_loss < close, "SL must be below entry for longs"
                assert sig.take_profit != _D("0"), "TP must be non-zero"
                assert sig.risk_per_share > _D("0"), "Risk per share must be positive"
                break
        else:
            pytest.skip("No ENTER_LONG produced in synthetic data")


class TestRiskRewardFilter:
    """R:R filter should reject entries with poor reward/risk."""

    def test_rr_filter(self) -> None:
        """Strategy should not produce entries where TP is effectively
        at entry price (bad R:R).  We verify by checking that any entry
        signal that does appear has a reasonable TP distance.
        """
        from stockdownloader.strategy.intraday.ml_oversold_config import (
            MLOversoldConfig,
        )

        cfg = MLOversoldConfig(ml_min_rr=_D("0.5"))
        strategy = _make_strategy(prob=0.90, config=cfg)
        data = _make_data(n=300)

        for i in range(len(data)):
            sig = strategy.evaluate(data, i)
            if sig.action == IntradayAction.ENTER_LONG:
                close = data[i].close
                tp_dist = abs(sig.take_profit - close)
                sl_dist = abs(close - sig.stop_loss)
                if sl_dist > _D("0"):
                    rr = tp_dist / sl_dist
                    assert rr >= _D("0.3"), (
                        f"R:R too low: {rr} (TP dist={tp_dist}, SL dist={sl_dist})"
                    )
                break


class TestLongOnly:
    """Direction filter: allow_shorts=False should block shorts."""

    def test_long_only(self) -> None:
        """With ``allow_shorts=False`` (default), no ENTER_SHORT signals
        should be produced regardless of probability.
        """
        strategy = _make_strategy(prob=0.95)
        data = _make_data(n=300)

        for i in range(len(data)):
            sig = strategy.evaluate(data, i)
            assert sig.action != IntradayAction.ENTER_SHORT, (
                f"ENTER_SHORT at idx={i} with allow_shorts=False"
            )
