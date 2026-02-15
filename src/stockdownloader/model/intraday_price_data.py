"""Intraday OHLCV price data with datetime precision for 5-minute bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from stockdownloader.model.price_data import PriceData


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
