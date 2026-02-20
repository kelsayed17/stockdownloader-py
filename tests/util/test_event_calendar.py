"""Tests for the event calendar module."""

from __future__ import annotations

import pytest

from stockdownloader.util.event_calendar import (
    FOMC_DATES,
    days_since_anchor,
    get_anchor_date,
    is_anchor_date,
)


class TestGetAnchorDate:
    """Tests for get_anchor_date()."""

    def test_returns_most_recent_fomc(self) -> None:
        # 2024-02-12 is after 2024-01-31 FOMC
        assert get_anchor_date("2024-02-12") == "2024-01-31"

    def test_returns_exact_fomc_date(self) -> None:
        # On the FOMC day itself
        assert get_anchor_date("2024-01-31") == "2024-01-31"

    def test_returns_none_before_first_fomc(self) -> None:
        # Before any known anchor
        assert get_anchor_date("2023-12-01") is None

    def test_mid_year_anchor(self) -> None:
        # Between 2024-06-12 and 2024-07-31
        assert get_anchor_date("2024-07-01") == "2024-06-12"

    def test_last_fomc_in_range(self) -> None:
        # After last 2026 FOMC
        assert get_anchor_date("2026-12-31") == "2026-12-16"

    def test_unknown_anchor_type(self) -> None:
        assert get_anchor_date("2024-02-12", "unknown") is None

    def test_day_after_fomc(self) -> None:
        assert get_anchor_date("2024-02-01") == "2024-01-31"


class TestIsAnchorDate:
    """Tests for is_anchor_date()."""

    def test_exact_fomc_date(self) -> None:
        assert is_anchor_date("2024-01-31") is True

    def test_non_fomc_date(self) -> None:
        assert is_anchor_date("2024-02-01") is False

    def test_unknown_anchor_type(self) -> None:
        assert is_anchor_date("2024-01-31", "unknown") is False

    def test_all_fomc_dates_recognized(self) -> None:
        for d in FOMC_DATES:
            assert is_anchor_date(d) is True, f"{d} not recognized"


class TestDaysSinceAnchor:
    """Tests for days_since_anchor()."""

    def test_on_anchor_day(self) -> None:
        assert days_since_anchor("2024-01-31") == 0

    def test_one_day_after(self) -> None:
        assert days_since_anchor("2024-02-01") == 1

    def test_twelve_days_after(self) -> None:
        # 2024-02-12 is 12 days after 2024-01-31
        assert days_since_anchor("2024-02-12") == 12

    def test_returns_none_before_first(self) -> None:
        assert days_since_anchor("2023-01-01") is None

    def test_days_reset_on_new_anchor(self) -> None:
        # 2024-03-20 is a new FOMC date
        assert days_since_anchor("2024-03-20") == 0
        # Day after
        assert days_since_anchor("2024-03-21") == 1


class TestFOMCDates:
    """Verify the FOMC date table is well-formed."""

    def test_dates_sorted(self) -> None:
        assert list(FOMC_DATES) == sorted(FOMC_DATES)

    def test_dates_unique(self) -> None:
        assert len(FOMC_DATES) == len(set(FOMC_DATES))

    def test_expected_count(self) -> None:
        # 8 meetings per year × 3 years = 24
        assert len(FOMC_DATES) == 24

    def test_date_format(self) -> None:
        for d in FOMC_DATES:
            assert len(d) == 10
            assert d[4] == "-" and d[7] == "-"
