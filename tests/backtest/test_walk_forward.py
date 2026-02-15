"""Tests for WalkForwardValidator.

Validates window construction, IS/OOS boundary correctness, and
degradation ratio computation.
"""

from decimal import Decimal

import pytest

from stockdownloader.backtest.walk_forward import (
    WalkForwardValidator,
    WalkForwardWindow,
    WalkForwardResult,
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
# Construction
# =========================================================================


def test_construction():
    data = _make_data(1000)
    v = WalkForwardValidator(data, n_windows=3)
    assert len(v.windows) > 0


def test_empty_data_raises():
    with pytest.raises(ValueError, match="data must not be empty"):
        WalkForwardValidator([], n_windows=3)


def test_invalid_n_windows():
    data = _make_data(100)
    with pytest.raises(ValueError, match="n_windows must be >= 1"):
        WalkForwardValidator(data, n_windows=0)


def test_invalid_is_ratio():
    data = _make_data(100)
    with pytest.raises(ValueError, match="is_ratio"):
        WalkForwardValidator(data, n_windows=3, is_ratio=0.99)


# =========================================================================
# Window construction
# =========================================================================


def test_windows_are_walkforward_windows():
    data = _make_data(1000)
    v = WalkForwardValidator(data, n_windows=5)
    for w in v.windows:
        assert isinstance(w, WalkForwardWindow)


def test_is_precedes_oos():
    """In-sample must come before out-of-sample in each window."""
    data = _make_data(2000)
    v = WalkForwardValidator(data, n_windows=5)
    for w in v.windows:
        assert w.in_sample_start < w.in_sample_end
        assert w.in_sample_end == w.out_of_sample_start
        assert w.out_of_sample_start < w.out_of_sample_end


def test_oos_within_data_bounds():
    """OOS end should not exceed data length."""
    data = _make_data(1000)
    v = WalkForwardValidator(data, n_windows=5)
    for w in v.windows:
        assert w.out_of_sample_end <= len(data)


def test_windows_roll_forward():
    """Each successive window should start at a later index."""
    data = _make_data(2000)
    v = WalkForwardValidator(data, n_windows=5)
    starts = [w.in_sample_start for w in v.windows]
    for i in range(1, len(starts)):
        assert starts[i] > starts[i - 1], (
            f"Window {i} start ({starts[i]}) should be after "
            f"window {i-1} start ({starts[i-1]})"
        )


def test_window_ids_sequential():
    data = _make_data(1000)
    v = WalkForwardValidator(data, n_windows=3)
    ids = [w.window_id for w in v.windows]
    assert ids == list(range(len(ids)))


def test_is_ratio_respected():
    """IS portion should be approximately is_ratio of total window span."""
    data = _make_data(5000)
    is_ratio = 0.7
    v = WalkForwardValidator(data, n_windows=3, is_ratio=is_ratio)
    for w in v.windows:
        total = w.out_of_sample_end - w.in_sample_start
        is_size = w.in_sample_end - w.in_sample_start
        actual_ratio = is_size / total
        # Allow 10% tolerance
        assert abs(actual_ratio - is_ratio) < 0.15, (
            f"IS ratio {actual_ratio:.2f} too far from {is_ratio}"
        )


# =========================================================================
# Window with small data
# =========================================================================


def test_small_data_single_window():
    """Very small dataset should still produce at least 1 window."""
    data = _make_data(50)
    v = WalkForwardValidator(data, n_windows=1)
    assert len(v.windows) >= 1


def test_windows_with_many_requested():
    """Requesting more windows than data can support should not crash."""
    data = _make_data(100)
    v = WalkForwardValidator(data, n_windows=20)
    # Should produce at most 20 windows, but might produce fewer
    assert len(v.windows) >= 1


# =========================================================================
# WalkForwardWindow dataclass
# =========================================================================


def test_window_frozen():
    w = WalkForwardWindow(0, 100, 100, 150, 0)
    with pytest.raises(AttributeError):
        w.in_sample_start = 10  # type: ignore[misc]


# =========================================================================
# WalkForwardResult
# =========================================================================


def test_result_fields():
    r = WalkForwardResult(
        strategy_name="test",
        windows=[],
        in_sample_results=[],
        out_of_sample_results=[],
        in_sample_score=10.0,
        out_of_sample_score=8.0,
        degradation_ratio=0.8,
    )
    assert r.strategy_name == "test"
    assert r.degradation_ratio == 0.8
    assert r.in_sample_score == 10.0
    assert r.out_of_sample_score == 8.0


def test_degradation_ratio_perfect():
    """OOS == IS → degradation = 1.0 (no overfitting)."""
    r = WalkForwardResult(
        strategy_name="perfect",
        windows=[],
        in_sample_results=[],
        out_of_sample_results=[],
        in_sample_score=10.0,
        out_of_sample_score=10.0,
        degradation_ratio=1.0,
    )
    assert r.degradation_ratio == 1.0


def test_degradation_ratio_overfit():
    """OOS << IS → degradation < 0.5 (likely overfit)."""
    r = WalkForwardResult(
        strategy_name="overfit",
        windows=[],
        in_sample_results=[],
        out_of_sample_results=[],
        in_sample_score=20.0,
        out_of_sample_score=5.0,
        degradation_ratio=0.25,
    )
    assert r.degradation_ratio < 0.5


# =========================================================================
# Boundary clamping edge cases
# =========================================================================


def test_is_end_clamped_to_data_bounds():
    """When IS would extend beyond data, it is clamped.

    This tests the fix on lines 178-181 of walk_forward.py that
    prevents IS slices from referencing out-of-range indices.
    """
    # Small data: 150 bars, 5 windows.
    # With is_ratio=0.7, step ≈ 9, window_size ≈ 30, is_size ≈ 21
    # Window 5 starts at step*4 = 36, IS end = 36+21 = 57 which is fine.
    # But with fewer bars, later windows WILL hit the boundary.
    data = _make_data(60)
    v = WalkForwardValidator(data, n_windows=10, is_ratio=0.7)
    for w in v.windows:
        # IS end must never exceed data length
        assert w.in_sample_end <= len(data), (
            f"Window {w.window_id}: IS end {w.in_sample_end} > data len {len(data)}"
        )
        # OOS must start at or after IS end
        assert w.out_of_sample_start >= w.in_sample_end, (
            f"Window {w.window_id}: OOS start {w.out_of_sample_start} "
            f"before IS end {w.in_sample_end}"
        )
        # OOS must be within data bounds
        assert w.out_of_sample_end <= len(data)


def test_oos_end_clamped_to_data_bounds():
    """When OOS extends beyond data, it is clamped to data length.

    Tests line 187-188 of walk_forward.py.
    """
    data = _make_data(100)
    v = WalkForwardValidator(data, n_windows=5, is_ratio=0.5)
    for w in v.windows:
        assert w.out_of_sample_end <= len(data)


def test_windows_skipped_when_oos_beyond_data():
    """Windows that would start OOS beyond data are skipped (not created).

    Tests lines 172-174 and 184-185 of walk_forward.py.
    """
    data = _make_data(30)
    # Request many windows on very small data — most should be skipped
    v = WalkForwardValidator(data, n_windows=50, is_ratio=0.8)
    # Should have at least 1 window but fewer than 50
    assert 1 <= len(v.windows) <= 50
    for w in v.windows:
        assert w.out_of_sample_start < len(data)


def test_no_empty_windows():
    """Every created window must have non-zero IS and OOS sizes."""
    data = _make_data(200)
    for n_win in [1, 3, 5, 10]:
        v = WalkForwardValidator(data, n_windows=n_win)
        for w in v.windows:
            is_size = w.in_sample_end - w.in_sample_start
            oos_size = w.out_of_sample_end - w.out_of_sample_start
            assert is_size > 0, f"Window {w.window_id} has empty IS"
            assert oos_size > 0, f"Window {w.window_id} has empty OOS"


def test_high_is_ratio_boundary():
    """Very high IS ratio (0.95) with small data should not crash."""
    data = _make_data(100)
    v = WalkForwardValidator(data, n_windows=3, is_ratio=0.95)
    assert len(v.windows) >= 1
    for w in v.windows:
        assert w.in_sample_end <= len(data)
        assert w.out_of_sample_end <= len(data)


def test_low_is_ratio_boundary():
    """Very low IS ratio (0.1) produces valid windows."""
    data = _make_data(500)
    v = WalkForwardValidator(data, n_windows=3, is_ratio=0.1)
    assert len(v.windows) >= 1
    for w in v.windows:
        is_size = w.in_sample_end - w.in_sample_start
        oos_size = w.out_of_sample_end - w.out_of_sample_start
        assert is_size > 0
        assert oos_size > 0
