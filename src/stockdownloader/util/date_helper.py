"""Date utility providing market-aware date calculations and multiple format support.

Uses :mod:`datetime` exclusively for all date operations.
"""
from __future__ import annotations

from datetime import date, timedelta

STANDARD_FORMAT = '%m/%d/%Y'
YAHOO_FORMAT = '%Y-%m-%d'
YAHOO_EARNINGS_FORMAT = '%Y%m%d'
MORNINGSTAR_FORMAT = '%Y-%m'

_MONTH_FMT = '%m'
_DAY_FMT = '%d'
_YEAR_FMT = '%Y'


class DateHelper:
    """Market-aware date helper.

    Provides today/yesterday/tomorrow adjusted to market days (Mon-Fri)
    and formatted strings in multiple date formats.

    Args:
        reference_date: The reference date to base calculations on.
            Defaults to today.
    """

    def __init__(self, reference_date: date | None = None) -> None:
        self._reference_date = reference_date or date.today()
        self._today_market = adjust_to_market_day(self._reference_date)
        self._yesterday_market = adjust_to_market_day(
            self._reference_date - timedelta(days=1)
        )
        self._tomorrow_market = _adjust_to_next_market_day(
            self._reference_date + timedelta(days=1)
        )
        self._six_months_ago = _subtract_months(self._reference_date, 6)

    # ------------------------------------------------------------------
    # Market-adjusted date objects
    # ------------------------------------------------------------------

    @property
    def yesterday_market(self) -> date:
        """Previous market day."""
        return self._yesterday_market

    @property
    def today_market(self) -> date:
        """Current (or most recent) market day."""
        return self._today_market

    @property
    def tomorrow_market(self) -> date:
        """Next market day."""
        return self._tomorrow_market

    # ------------------------------------------------------------------
    # Formatted date strings
    # ------------------------------------------------------------------

    @property
    def today(self) -> str:
        """Today's market date in MM/DD/YYYY format."""
        return self._today_market.strftime(STANDARD_FORMAT)

    @property
    def tomorrow(self) -> str:
        """Tomorrow's market date in MM/DD/YYYY format."""
        return self._tomorrow_market.strftime(STANDARD_FORMAT)

    @property
    def yesterday(self) -> str:
        """Yesterday's market date in MM/DD/YYYY format."""
        return self._yesterday_market.strftime(STANDARD_FORMAT)

    @property
    def six_months_ago(self) -> str:
        """Date six months ago in MM/DD/YYYY format."""
        return self._six_months_ago.strftime(STANDARD_FORMAT)

    @property
    def current_month(self) -> str:
        """Two-digit month of the reference date."""
        return self._reference_date.strftime(_MONTH_FMT)

    @property
    def current_day(self) -> str:
        """Two-digit day of the reference date."""
        return self._reference_date.strftime(_DAY_FMT)

    @property
    def current_year(self) -> str:
        """Four-digit year of the reference date."""
        return self._reference_date.strftime(_YEAR_FMT)

    @property
    def from_month(self) -> str:
        """Two-digit month of the six-months-ago date."""
        return self._six_months_ago.strftime(_MONTH_FMT)

    @property
    def from_day(self) -> str:
        """Two-digit day of the six-months-ago date."""
        return self._six_months_ago.strftime(_DAY_FMT)

    @property
    def from_year(self) -> str:
        """Four-digit year of the six-months-ago date."""
        return self._six_months_ago.strftime(_YEAR_FMT)


# =========================================================================
# Module-level helpers
# =========================================================================


def adjust_to_market_day(d: date) -> date:
    """Adjust *d* backward to the nearest weekday (Mon-Fri).

    Saturday becomes Friday, Sunday becomes Friday.
    """
    weekday = d.weekday()  # Mon=0 .. Sun=6
    if weekday == 5:  # Saturday
        return d - timedelta(days=1)
    if weekday == 6:  # Sunday
        return d - timedelta(days=2)
    return d


def _adjust_to_next_market_day(d: date) -> date:
    """Adjust *d* forward to the nearest weekday (Mon-Fri).

    Saturday becomes Monday, Sunday becomes Monday.
    """
    weekday = d.weekday()
    if weekday == 5:  # Saturday
        return d + timedelta(days=2)
    if weekday == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def _subtract_months(d: date, months: int) -> date:
    """Subtract *months* from *d*, clamping the day to the valid range."""
    month = d.month - months
    year = d.year
    while month <= 0:
        month += 12
        year -= 1
    # Clamp day (e.g. Mar 31 - 1 month = Feb 28/29)
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)
