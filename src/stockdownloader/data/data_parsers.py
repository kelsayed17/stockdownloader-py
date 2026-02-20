"""Data parsing utilities for Yahoo Finance API responses and CSV price files.

This module consolidates two categories of data-parsing helpers:

1. **JSON field-extraction functions** -- safely extract values from Yahoo
   Finance JSON payloads, handling missing keys and type-conversion errors
   gracefully (``get_decimal``, ``get_long``, ``get_string``, ``get_boolean``,
   ``get_decimal_at``, ``get_long_at``, ``get_raw_long``, ``get_raw_string``,
   ``get_raw_decimal``, ``format_market_cap``).

2. **CSV price-data loader** -- :class:`CsvPriceDataLoader` reads daily OHLCV
   data from CSV files or binary streams in Yahoo Finance historical format
   (expected columns: Date, Open, High, Low, Close, Adj Close, Volume).
"""

from __future__ import annotations

import csv
import io
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO

from stockdownloader.model import PriceData

logger = logging.getLogger(__name__)


# ===================================================================
# JSON field-extraction helpers
# ===================================================================


def get_decimal(obj: dict, field: str) -> Decimal:
    """Extract a Decimal from *field* in *obj*, returning ``Decimal(0)`` on failure."""
    val = obj.get(field)
    if val is None:
        return Decimal(0)
    try:
        return Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError) as exc:
        logger.debug("Cannot convert field '%s' value %r to Decimal: %s", field, val, exc)
        return Decimal(0)


def get_long(obj: dict, field: str) -> int:
    """Extract an integer from *field* in *obj*, returning ``0`` on failure."""
    val = obj.get(field)
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def get_string(obj: dict, field: str) -> str:
    """Extract a string from *field* in *obj*, returning ``""`` on failure."""
    val = obj.get(field)
    if val is None:
        return ""
    return str(val)


def get_boolean(obj: dict, field: str) -> bool:
    """Extract a boolean from *field* in *obj*, returning ``False`` on failure."""
    val = obj.get(field)
    if val is None:
        return False
    return bool(val)


def get_decimal_at(arr: list | None, index: int) -> Decimal:
    """Extract a Decimal from *arr[index]*, returning ``Decimal(0)`` on failure."""
    if arr is None or index >= len(arr):
        return Decimal(0)
    val = arr[index]
    if val is None:
        return Decimal(0)
    return Decimal(str(val))


def get_long_at(arr: list | None, index: int) -> int:
    """Extract an integer from *arr[index]*, returning ``0`` on failure."""
    if arr is None or index >= len(arr):
        return 0
    val = arr[index]
    if val is None:
        return 0
    return int(val)


def get_raw_long(obj: dict, field: str) -> int:
    """Extract a long integer from a Yahoo Finance JSON field.

    Yahoo wraps numeric values as ``{"raw": 123, "fmt": "123"}``.
    This helper handles both wrapped and direct numeric values.
    """
    val = obj.get(field)
    if val is None:
        return 0

    # Yahoo Finance wraps numeric values in {"raw": ..., "fmt": ...}
    if isinstance(val, dict):
        raw = val.get("raw")
        if raw is not None:
            try:
                return int(raw)
            except (ValueError, TypeError):
                return 0

    # Direct numeric value
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def get_raw_string(obj: dict, field: str) -> str:
    """Extract a string from a Yahoo Finance JSON field.

    Handles the ``{"raw": ..., "fmt": "..."}`` wrapper format.
    """
    val = obj.get(field)
    if val is None:
        return ""

    if isinstance(val, dict):
        fmt = val.get("fmt")
        if fmt is not None:
            return str(fmt)

    return str(val)


def get_raw_decimal(obj: dict, field: str) -> Decimal:
    """Extract a Decimal from a Yahoo Finance JSON field.

    Handles the ``{"raw": ..., "fmt": "..."}`` wrapper format as well
    as direct numeric values.  Returns ``Decimal(0)`` on failure.
    """
    val = obj.get(field)
    if val is None:
        return Decimal(0)

    # Yahoo Finance wraps numeric values in {"raw": ..., "fmt": ...}
    if isinstance(val, dict):
        raw = val.get("raw")
        if raw is not None:
            try:
                return Decimal(str(raw))
            except (InvalidOperation, ValueError, TypeError):
                return Decimal(0)

    # Direct numeric value
    try:
        return Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)


def format_market_cap(market_cap: int) -> str:
    """Format a market cap integer as a human-readable string."""
    if market_cap >= 1_000_000_000:
        return f"{market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"{market_cap / 1_000_000:.2f}M"
    return str(market_cap)


# ===================================================================
# CSV price-data loader
# ===================================================================


class CsvPriceDataLoader:
    """Utility class for loading :class:`PriceData` from CSV files or streams.

    All methods are static; the class is not intended to be instantiated.
    """

    def __init__(self) -> None:  # pragma: no cover
        raise TypeError("CsvPriceDataLoader should not be instantiated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_file(filename: str | Path) -> list[PriceData]:
        """Load price data from *filename* (path to a CSV file).

        Returns an empty list if the file cannot be read.
        """
        try:
            with open(filename, newline="", encoding="utf-8") as fh:
                return _parse_records(fh)
        except (OSError, csv.Error, InvalidOperation, ValueError) as exc:
            logger.warning("Error loading CSV file %s: %s", filename, exc)
            return []

    @staticmethod
    def load_from_stream(stream: BinaryIO) -> list[PriceData]:
        """Load price data from a binary *stream* (e.g. an HTTP response body).

        Returns an empty list if the stream cannot be parsed.
        """
        try:
            text_stream = io.TextIOWrapper(stream, encoding="utf-8")
            return _parse_records(text_stream)
        except (OSError, csv.Error, InvalidOperation, ValueError) as exc:
            logger.warning("Error loading CSV from stream: %s", exc)
            return []


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_records(text_io: io.TextIOBase | io.TextIOWrapper) -> list[PriceData]:
    """Parse CSV rows into :class:`PriceData` instances.

    The first line is assumed to be a header and is skipped.
    """
    reader = csv.reader(text_io)
    next(reader, None)  # skip header

    data: list[PriceData] = []
    for line in reader:
        try:
            date = line[0]
            open_ = Decimal(line[1])
            high = Decimal(line[2])
            low = Decimal(line[3])
            close = Decimal(line[4])
            adj_close = Decimal(line[5]) if len(line) > 5 else close
            volume = int(line[6]) if len(line) > 6 else 0

            data.append(
                PriceData(
                    date=date,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    adj_close=adj_close,
                    volume=volume,
                )
            )
        except (InvalidOperation, ValueError, IndexError) as exc:
            logger.debug("Skipping invalid CSV row: %s (%s)", line, exc)
            continue

    return data
