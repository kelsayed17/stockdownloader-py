"""Integration test for the refactored DateHelper.
Verifies market-day adjustments and date formatting across various day-of-week scenarios.
"""
from __future__ import annotations

import re
from datetime import date

import pytest

from stockdownloader.util.parsers import DateHelper, adjust_to_market_day


def test_weekday_reference_date():
    # Wednesday 2024-01-10
    helper = DateHelper(date(2024, 1, 10))

    assert helper.today_market == date(2024, 1, 10)
    assert helper.yesterday_market == date(2024, 1, 9)
    assert helper.tomorrow_market == date(2024, 1, 11)

    assert helper.today == "01/10/2024"
    assert helper.yesterday == "01/09/2024"
    assert helper.tomorrow == "01/11/2024"


def test_saturday_reference_date():
    # Saturday 2024-01-13
    helper = DateHelper(date(2024, 1, 13))

    # Today should snap back to Friday
    assert helper.today_market == date(2024, 1, 12)
    # Yesterday (Friday) should be Friday itself
    assert helper.yesterday_market == date(2024, 1, 12)
    # Tomorrow (Sunday) -> next Monday
    assert helper.tomorrow_market == date(2024, 1, 15)


def test_sunday_reference_date():
    # Sunday 2024-01-14
    helper = DateHelper(date(2024, 1, 14))

    # Today should snap back to Friday
    assert helper.today_market == date(2024, 1, 12)
    # Yesterday (Saturday) -> snaps back to Friday
    assert helper.yesterday_market == date(2024, 1, 12)
    # Tomorrow (Monday) -> Monday
    assert helper.tomorrow_market == date(2024, 1, 15)


def test_monday_reference_date():
    # Monday 2024-01-15
    helper = DateHelper(date(2024, 1, 15))

    assert helper.today_market == date(2024, 1, 15)
    # Yesterday (Sunday) -> snaps back to Friday
    assert helper.yesterday_market == date(2024, 1, 12)
    assert helper.tomorrow_market == date(2024, 1, 16)


def test_friday_reference_date():
    # Friday 2024-01-12
    helper = DateHelper(date(2024, 1, 12))

    assert helper.today_market == date(2024, 1, 12)
    assert helper.yesterday_market == date(2024, 1, 11)
    # Tomorrow (Saturday) -> next Monday
    assert helper.tomorrow_market == date(2024, 1, 15)


def test_six_months_ago_calculation():
    helper = DateHelper(date(2024, 7, 15))

    assert helper.six_months_ago == "01/15/2024"
    assert helper.from_month == "01"
    assert helper.from_day == "15"
    assert helper.from_year == "2024"


def test_current_date_components():
    helper = DateHelper(date(2024, 3, 5))

    assert helper.current_month == "03"
    assert helper.current_day == "05"
    assert helper.current_year == "2024"


def test_adjust_to_market_day_static():
    # Weekdays stay the same
    assert adjust_to_market_day(date(2024, 1, 10)) == date(2024, 1, 10)
    assert adjust_to_market_day(date(2024, 1, 12)) == date(2024, 1, 12)

    # Saturday -> Friday
    assert adjust_to_market_day(date(2024, 1, 13)) == date(2024, 1, 12)

    # Sunday -> Friday
    assert adjust_to_market_day(date(2024, 1, 14)) == date(2024, 1, 12)


def test_all_market_days_are_weekdays():
    # Test across a full week
    for day in range(8, 15):
        helper = DateHelper(date(2024, 1, day))

        today_dow = helper.today_market.weekday()
        assert today_dow != 5  # Not Saturday
        assert today_dow != 6  # Not Sunday

        yesterday_dow = helper.yesterday_market.weekday()
        assert yesterday_dow != 5
        assert yesterday_dow != 6

        tomorrow_dow = helper.tomorrow_market.weekday()
        assert tomorrow_dow != 5
        assert tomorrow_dow != 6


def test_default_constructor_does_not_throw():
    # The refactored DateHelper constructor no longer throws ParseException
    helper = DateHelper()
    assert helper is not None


def test_date_formatting_consistency():
    helper = DateHelper(date(2024, 12, 25))

    today = helper.today
    # Should match MM/dd/yyyy format
    assert re.match(r"\d{2}/\d{2}/\d{4}", today), \
        f"Date should match MM/dd/yyyy format: {today}"
    assert today == "12/25/2024"
