"""Immutable representation of a single day's OHLCV price data."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
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
