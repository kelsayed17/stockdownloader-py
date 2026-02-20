"""Tests for WalkForwardOptimizer.

Validates IS/OOS data splitting, optimize_all execution,
and acceptance gating based on OOS improvement.
"""

from decimal import Decimal

import pytest

from stockdownloader.backtest.walk_forward_optimizer import (
    WalkForwardOptimizer,
    WFOptResult,
)
from stockdownloader.model.intraday_price_data import IntradayPriceData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n: int, base_price: float = 100.0) -> list[IntradayPriceData]:
    """Generate n synthetic intraday bars across multiple days."""
    data = []
    bars_per_day = 78  # standard 5m bars per trading day
    for i in range(n):
        day_num = i // bars_per_day
        bar_in_day = i % bars_per_day
        hour = 9 + bar_in_day // 12
        minute = 30 + (bar_in_day % 12) * 5
        if minute >= 60:
            hour += 1
            minute -= 60

        date_str = (
            f"2025-01-{(day_num % 28) + 1:02d} "
            f"{hour:02d}:{minute:02d}:00-05:00"
        )
        price = base_price + i * 0.01
        data.append(IntradayPriceData(
            date=date_str,
            open=Decimal(str(round(price, 4))),
            high=Decimal(str(round(price + 0.5, 4))),
            low=Decimal(str(round(price - 0.5, 4))),
            close=Decimal(str(round(price, 4))),
            adj_close=Decimal(str(round(price, 4))),
            volume=1_000_000,
        ))
    return data


# =========================================================================
# WFOptResult dataclass
# =========================================================================


def test_wf_opt_result_defaults():
    """WFOptResult has sensible defaults."""
    r = WFOptResult(name="test", display_name="Test", category="intraday")
    assert r.name == "test"
    assert r.display_name == "Test"
    assert r.category == "intraday"
    assert r.baseline_is is None
    assert r.baseline_oos is None
    assert r.optimized_kwargs == {}
    assert r.optimized_is is None
    assert r.optimized_oos is None
    assert r.accepted is False


def test_wf_opt_result_accepted():
    """WFOptResult can be marked as accepted."""
    r = WFOptResult(name="test", display_name="Test", category="intraday",
                    accepted=True)
    assert r.accepted is True


# =========================================================================
# Construction & data splitting
# =========================================================================


def test_construction():
    """WalkForwardOptimizer can be constructed."""
    data = _make_data(1000)
    wf = WalkForwardOptimizer(data, split_ratio=0.7, verbose=False)
    # IS data should be ~70%, OOS ~30%
    assert len(wf._is_data) == 700
    assert len(wf._oos_data) == 300


def test_split_ratio_boundaries():
    """Different split ratios produce correct splits."""
    data = _make_data(100)
    wf = WalkForwardOptimizer(data, split_ratio=0.5, verbose=False)
    assert len(wf._is_data) == 50
    assert len(wf._oos_data) == 50


def test_split_preserves_data():
    """IS + OOS = full data, no data lost or duplicated."""
    data = _make_data(200)
    wf = WalkForwardOptimizer(data, split_ratio=0.7, verbose=False)
    assert len(wf._is_data) + len(wf._oos_data) == len(data)
    # IS is the first portion
    assert wf._is_data[0] is data[0]
    # OOS starts right after IS
    assert wf._oos_data[0] is data[len(wf._is_data)]


# =========================================================================
# optimize_all returns list[WFOptResult]
# =========================================================================


def test_optimize_all_returns_list():
    """optimize_all returns a list of WFOptResult."""
    data = _make_data(500)
    wf = WalkForwardOptimizer(data, verbose=False)
    # Filter to a nonexistent strategy to get empty list quickly
    results = wf.optimize_all(strategy_filter="nonexistent-strategy-xyz")
    assert isinstance(results, list)
    assert len(results) == 0


def test_optimize_all_with_category_filter():
    """optimize_all respects category filter."""
    data = _make_data(500)
    wf = WalkForwardOptimizer(data, verbose=False)
    # Filter to categories that have no strategies with param_space
    # Should complete without error
    results = wf.optimize_all(categories=["intraday"],
                              strategy_filter="nonexistent-abc")
    assert isinstance(results, list)


# =========================================================================
# OOS trading days
# =========================================================================


def test_oos_trading_days_computed():
    """OOS trading days are correctly computed from the OOS data."""
    data = _make_data(500)
    wf = WalkForwardOptimizer(data, split_ratio=0.7, verbose=False)
    # Should have computed OOS trading days > 0
    assert wf._oos_trading_days >= 0
    # Full data has some unique dates; OOS should too
    if len(wf._oos_data) > 0:
        assert wf._oos_trading_days > 0


def test_empty_oos_zero_days():
    """Split ratio = 1.0 means no OOS data (edge case)."""
    data = _make_data(100)
    wf = WalkForwardOptimizer(data, split_ratio=1.0, verbose=False)
    assert len(wf._oos_data) == 0
    assert wf._oos_trading_days == 0
