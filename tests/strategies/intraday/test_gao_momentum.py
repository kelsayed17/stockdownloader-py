"""Tests for GaoMomentumStrategy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.core.models.trade import IntradayAction
from stockdownloader.strategies.intraday.gao_momentum import (
    GaoMomentumConfig,
    GaoMomentumStrategy,
)
from tests.strategies.intraday.conftest import make_warmup_bars, make_tournament_bar


def _make_session_bars(
    date: str,
    open_price: float,
    first_hh_trend: float,
    rest_trend: float,
    n: int = 78,
) -> list:
    """Build 78 bars for a single session with controlled first-half-hour trend.

    Parameters
    ----------
    first_hh_trend:
        Price delta per bar during bars 1-6 (first half hour).
    rest_trend:
        Price delta per bar for remaining bars.
    """
    bars = []
    price = open_price
    for i in range(n):
        bar_of_day = i + 1
        hour = 9 + (i * 5 + 30) // 60
        minute = (i * 5 + 30) % 60
        dt = f"{date} {hour:02d}:{minute:02d}:00-05:00"

        if bar_of_day <= 6:
            delta = first_hh_trend
        else:
            delta = rest_trend
        price += delta

        bars.append(make_tournament_bar(
            dt, price - 0.05, price + 0.10, price - 0.10, price, 5000,
        ))
    return bars


class TestGaoMomentumConfig:
    def test_defaults(self):
        c = GaoMomentumConfig()
        assert c.entry_start_bar == 73
        assert c.first_hh_bars == 6
        assert c.require_dual_signal is True

    def test_json_round_trip(self):
        c = GaoMomentumConfig(min_r1_magnitude=Decimal("0.001"))
        c2 = GaoMomentumConfig.from_json(c.to_json())
        assert c == c2

    def test_overrides(self):
        s = GaoMomentumStrategy(entry_start_bar=70)
        assert s._c.entry_start_bar == 70


class TestGaoMomentumEntry:
    def test_no_signal_before_entry_bar(self):
        """Strategy should not fire in bars before entry_start_bar."""
        strat = GaoMomentumStrategy(require_dual_signal=False, vix_filter=False)
        # Build warmup + test session with positive first half-hour
        bars = make_warmup_bars(1200)
        session = _make_session_bars(
            "2024-01-18", 455.0, first_hh_trend=0.5, rest_trend=0.01,
        )
        bars.extend(session)

        # Evaluate all bars, check no signal before bar 73 of last session
        session_start_idx = len(bars) - 78
        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            if i < session_start_idx + 72:  # bar_of_day < 73
                # Should still be HOLD
                assert sig.action != IntradayAction.ENTER_LONG, (
                    f"Unexpected ENTER_LONG at index {i}"
                )

    def test_long_signal_on_positive_first_hh(self):
        """Positive first half-hour → long in last 30 min (single signal mode)."""
        strat = GaoMomentumStrategy(
            require_dual_signal=False,
            vix_filter=False,
        )
        bars = make_warmup_bars(1200)
        # Strong positive first half-hour, then continued up trend
        session = _make_session_bars(
            "2024-01-18", 455.0, first_hh_trend=0.5, rest_trend=0.05,
        )
        bars.extend(session)

        signal_fired = False
        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            if sig.action == IntradayAction.ENTER_LONG:
                signal_fired = True
                # Should be in last 30 min (bar 73+)
                session_start = len(bars) - 78
                bar_of_session = i - session_start + 1  # 1-indexed
                assert bar_of_session >= 73, (
                    f"Signal at bar {bar_of_session}, expected >= 73"
                )
                break
        assert signal_fired, "Expected long signal in last 30 min"

    def test_no_signal_on_flat_first_hh(self):
        """Flat first half-hour (below min magnitude) → no signal."""
        strat = GaoMomentumStrategy(
            require_dual_signal=False,
            min_r1_magnitude=Decimal("0.005"),  # High threshold
            vix_filter=False,
        )
        bars = make_warmup_bars(1200)
        # Very small first half-hour move
        session = _make_session_bars(
            "2024-01-18", 455.0, first_hh_trend=0.001, rest_trend=0.001,
        )
        bars.extend(session)

        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            if i >= len(bars) - 78:
                assert sig.action == IntradayAction.HOLD, (
                    f"Unexpected signal at index {i}"
                )

    def test_fire_once_per_session(self):
        """Only one entry per session."""
        strat = GaoMomentumStrategy(
            require_dual_signal=False,
            vix_filter=False,
        )
        bars = make_warmup_bars(1200)
        session = _make_session_bars(
            "2024-01-18", 455.0, first_hh_trend=0.5, rest_trend=0.05,
        )
        bars.extend(session)

        entries = 0
        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            if sig.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT):
                entries += 1
                strat.on_position_opened(True)
                strat.on_position_closed()
        assert entries <= 1

    def test_warmup_period(self):
        s = GaoMomentumStrategy()
        assert s.warmup_period == 78 * 15
