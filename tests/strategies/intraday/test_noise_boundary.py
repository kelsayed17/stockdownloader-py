"""Tests for NoiseBoundaryStrategy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.core.models.trade import IntradayAction
from stockdownloader.strategies.intraday.noise_boundary import (
    NoiseBoundaryConfig,
    NoiseBoundaryStrategy,
)
from tests.strategies.intraday.conftest import make_warmup_bars, make_tournament_bar


def _make_session_bars(
    date: str,
    open_price: float,
    trend: float,
    n: int = 78,
) -> list:
    """Build bars for a single session with a given trend."""
    bars = []
    price = open_price
    for i in range(n):
        hour = 9 + (i * 5 + 30) // 60
        minute = (i * 5 + 30) % 60
        dt = f"{date} {hour:02d}:{minute:02d}:00-05:00"
        price += trend
        bars.append(make_tournament_bar(
            dt, price - 0.05, price + 0.10, price - 0.10, price, 5000,
        ))
    return bars


class TestNoiseBoundaryConfig:
    def test_defaults(self):
        c = NoiseBoundaryConfig()
        assert c.lookback_days == 14
        assert c.vol_multiplier == Decimal("1.0")
        assert c.checkpoint_interval == 6

    def test_json_round_trip(self):
        c = NoiseBoundaryConfig(lookback_days=10, vol_multiplier=Decimal("1.5"))
        c2 = NoiseBoundaryConfig.from_json(c.to_json())
        assert c == c2


class TestNoiseBoundaryEntry:
    def test_no_signal_before_warmup(self):
        """Needs lookback_days of daily data before generating signals."""
        strat = NoiseBoundaryStrategy(lookback_days=14)
        # Only 1 day of bars (78 bars) — not enough history
        bars = _make_session_bars(
            "2024-06-15", 500.0, trend=0.10,
        )
        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            assert sig.action == IntradayAction.HOLD

    def test_long_on_upper_breakout(self):
        """Price above upper noise boundary -> long entry."""
        strat = NoiseBoundaryStrategy(
            lookback_days=3,  # Short lookback for testing
        )
        # Build 4 days: 3 tight-range warmup + 1 strong breakout
        all_bars = []
        for d in range(3):
            date = f"2024-06-{15+d:02d}"
            bars = _make_session_bars(date, 500.0, trend=0.01)
            all_bars.extend(bars)
        # Day 4: strong breakout above noise boundary
        breakout_bars = _make_session_bars("2024-06-18", 510.0, trend=0.50)
        all_bars.extend(breakout_bars)

        # Need enough total data for infra warmup too (need 78*15 bars)
        warmup = make_warmup_bars(1200)
        full_data = warmup + all_bars

        entry_found = False
        for i in range(len(full_data)):
            sig = strat.evaluate(full_data, i)
            if sig.action == IntradayAction.ENTER_LONG:
                entry_found = True
                break
        assert entry_found, "Expected long entry on upper noise breakout"

    def test_fire_once_per_session(self):
        """Only one entry per session."""
        strat = NoiseBoundaryStrategy(lookback_days=3)
        all_bars = make_warmup_bars(1200)
        for d in range(3):
            date = f"2024-01-{18+d:02d}"
            bars = _make_session_bars(date, 455.0, trend=0.01)
            all_bars.extend(bars)
        # Last day: breakout
        breakout = _make_session_bars("2024-01-21", 460.0, trend=0.50)
        all_bars.extend(breakout)

        entries = 0
        for i in range(len(all_bars)):
            sig = strat.evaluate(all_bars, i)
            if sig.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT):
                entries += 1
                strat.on_position_opened(True)
                strat.on_position_closed()
        assert entries <= 1

    def test_warmup_period(self):
        s = NoiseBoundaryStrategy()
        assert s.warmup_period == 78 * 15
