"""Tests for crossover detection utilities."""
from decimal import Decimal

import pytest

from stockdownloader.indicators import (
    crossed_above,
    crossed_below,
    crossed_above_series,
    crossed_below_series,
)

D = Decimal


# =========================================================================
# crossed_above
# =========================================================================


class TestCrossedAbove:
    """Tests for crossed_above(current, previous, threshold)."""

    def test_crosses_above(self) -> None:
        """prev <= threshold AND current > threshold → True."""
        assert crossed_above(D("31"), D("29"), D("30")) is True

    def test_prev_exactly_at_threshold(self) -> None:
        """prev == threshold AND current > threshold → True (crosses from boundary)."""
        assert crossed_above(D("31"), D("30"), D("30")) is True

    def test_already_above(self) -> None:
        """Both above threshold → False (no crossing this bar)."""
        assert crossed_above(D("35"), D("31"), D("30")) is False

    def test_still_below(self) -> None:
        """Both below threshold → False."""
        assert crossed_above(D("28"), D("25"), D("30")) is False

    def test_current_equals_threshold(self) -> None:
        """current == threshold (not strictly above) → False."""
        assert crossed_above(D("30"), D("28"), D("30")) is False

    def test_crossing_back_down(self) -> None:
        """prev above, current below → False (this is a downward cross)."""
        assert crossed_above(D("28"), D("32"), D("30")) is False


# =========================================================================
# crossed_below
# =========================================================================


class TestCrossedBelow:
    """Tests for crossed_below(current, previous, threshold)."""

    def test_crosses_below(self) -> None:
        """prev >= threshold AND current < threshold → True."""
        assert crossed_below(D("69"), D("71"), D("70")) is True

    def test_prev_exactly_at_threshold(self) -> None:
        """prev == threshold AND current < threshold → True."""
        assert crossed_below(D("69"), D("70"), D("70")) is True

    def test_already_below(self) -> None:
        """Both below → False."""
        assert crossed_below(D("65"), D("68"), D("70")) is False

    def test_still_above(self) -> None:
        """Both above → False."""
        assert crossed_below(D("75"), D("72"), D("70")) is False

    def test_current_equals_threshold(self) -> None:
        """current == threshold (not strictly below) → False."""
        assert crossed_below(D("70"), D("72"), D("70")) is False

    def test_crossing_back_up(self) -> None:
        """prev below, current above → False (this is an upward cross)."""
        assert crossed_below(D("72"), D("68"), D("70")) is False


# =========================================================================
# crossed_above_series
# =========================================================================


class TestCrossedAboveSeries:
    """Tests for crossed_above_series(cur_a, prev_a, cur_b, prev_b)."""

    def test_a_crosses_above_b(self) -> None:
        """A was below B, now above → True."""
        assert crossed_above_series(D("11"), D("9"), D("10"), D("10")) is True

    def test_a_already_above_b(self) -> None:
        """A was already above B → False."""
        assert crossed_above_series(D("12"), D("11"), D("10"), D("10")) is False

    def test_a_equals_b_now(self) -> None:
        """A == B now (not strictly above) → False."""
        assert crossed_above_series(D("10"), D("9"), D("10"), D("10")) is False

    def test_prev_a_equals_prev_b(self) -> None:
        """A was exactly at B, now above → True (crosses from equal)."""
        assert crossed_above_series(D("11"), D("10"), D("10"), D("10")) is True


# =========================================================================
# crossed_below_series
# =========================================================================


class TestCrossedBelowSeries:
    """Tests for crossed_below_series(cur_a, prev_a, cur_b, prev_b)."""

    def test_a_crosses_below_b(self) -> None:
        """A was above B, now below → True."""
        assert crossed_below_series(D("9"), D("11"), D("10"), D("10")) is True

    def test_a_already_below_b(self) -> None:
        """A was already below B → False."""
        assert crossed_below_series(D("8"), D("9"), D("10"), D("10")) is False

    def test_a_equals_b_now(self) -> None:
        """A == B now (not strictly below) → False."""
        assert crossed_below_series(D("10"), D("11"), D("10"), D("10")) is False

    def test_prev_a_equals_prev_b(self) -> None:
        """A was exactly at B, now below → True (crosses from equal)."""
        assert crossed_below_series(D("9"), D("10"), D("10"), D("10")) is True
