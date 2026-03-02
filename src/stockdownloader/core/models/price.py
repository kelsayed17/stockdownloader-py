"""Immutable OHLCV price-data models (daily and intraday)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class PriceData:
    """Immutable representation of a single day's OHLCV price data."""

    date: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adj_close: Decimal
    volume: int

    def __post_init__(self) -> None:
        if self.date is None:
            raise ValueError("date must not be null")
        if self.open is None:
            raise ValueError("open must not be null")
        if self.high is None:
            raise ValueError("high must not be null")
        if self.low is None:
            raise ValueError("low must not be null")
        if self.close is None:
            raise ValueError("close must not be null")
        if self.adj_close is None:
            raise ValueError("adj_close must not be null")
        if self.volume < 0:
            raise ValueError("volume must not be negative")

    def __str__(self) -> str:
        return f"{self.date} O:{self.open} H:{self.high} L:{self.low} C:{self.close} V:{self.volume}"


@dataclass(frozen=True, slots=True)
class IntradayPriceData(PriceData):
    """Intraday OHLCV price data with datetime precision.

    Extends PriceData to support datetime strings like
    ``'2025-11-13 09:30:00-05:00'``.  The inherited *date* field stores the
    full datetime string.  Use :pyattr:`datetime_parsed` for a proper
    :class:`~datetime.datetime` and :pyattr:`trading_date` for the calendar
    date portion (``'2025-11-13'``).

    Because ``PriceData`` is a frozen dataclass and all existing indicator
    functions accept ``Sequence[PriceData]``, ``IntradayPriceData`` instances
    can be used transparently wherever ``PriceData`` is expected.
    """

    @property
    def datetime_parsed(self) -> datetime:
        """Parse the *date* string into a timezone-aware datetime."""
        return datetime.fromisoformat(self.date)

    @property
    def trading_date(self) -> str:
        """Extract the calendar date portion (``YYYY-MM-DD``)."""
        # Fast path: first 10 characters of an ISO datetime string.
        return self.date[:10]

    @property
    def time_str(self) -> str:
        """Extract the time portion (``HH:MM:SS``)."""
        return self.date[11:19]
