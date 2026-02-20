"""Parsing utilities: CSV parsing and market-aware date calculations.

CSV parsing provides a context-manager-compatible parser for reading
delimited data, with support for quoted fields and configurable separators.

Date helpers provide market-aware date calculations (Mon-Fri adjustments)
and multiple format support using :mod:`datetime` exclusively.
"""
from __future__ import annotations

import csv
import io
from datetime import date, timedelta
from typing import TextIO

# =========================================================================
# Date formats and constants (formerly date_helper.py)
# =========================================================================

STANDARD_FORMAT = '%m/%d/%Y'
YAHOO_FORMAT = '%Y-%m-%d'
YAHOO_EARNINGS_FORMAT = '%Y%m%d'
MORNINGSTAR_FORMAT = '%Y-%m'

_MONTH_FMT = '%m'
_DAY_FMT = '%d'
_YEAR_FMT = '%Y'


# =========================================================================
# DateHelper class
# =========================================================================


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
# Date module-level helpers
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


# =========================================================================
# CsvParser class (formerly csv_parser.py)
# =========================================================================


class CsvParser:
    """A thin wrapper around Python's :mod:`csv` reader that mirrors the Java
    ``CsvParser`` API.

    Supports reading from any text stream (file, ``StringIO``, etc.) and can
    be used as a context manager.

    Args:
        source: A file-like text stream to read from.
        separator: The field delimiter character (default ``','``).
    """

    def __init__(self, source: TextIO, separator: str = ',') -> None:
        self._source = source
        self._reader = csv.reader(source, delimiter=separator, quotechar='"')

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bytes(cls, data: bytes, separator: str = ',') -> CsvParser:
        """Create a parser from raw bytes (decoded as UTF-8)."""
        return cls(io.TextIOWrapper(io.BytesIO(data), encoding='utf-8'), separator)

    @classmethod
    def from_string(cls, text: str, separator: str = ',') -> CsvParser:
        """Create a parser from an in-memory string."""
        return cls(io.StringIO(text), separator)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_next(self) -> list[str] | None:
        """Read and return the next row as a list of strings.

        Returns ``None`` when there are no more rows.
        """
        try:
            row = next(self._reader)
            return [field.strip() for field in row]
        except StopIteration:
            return None

    def read_all(self) -> list[list[str]]:
        """Read all remaining rows and return them as a list of lists."""
        result: list[list[str]] = []
        while True:
            row = self.read_next()
            if row is None:
                break
            result.append(row)
        return result

    def skip_lines(self, count: int) -> None:
        """Skip *count* lines from the underlying stream."""
        for _ in range(count):
            try:
                next(self._reader)
            except StopIteration:
                break

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying source stream."""
        self._source.close()

    def __enter__(self) -> CsvParser:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
