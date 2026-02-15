"""Shared JSON field-extraction helpers for Yahoo Finance API responses.

These functions safely extract values from Yahoo Finance JSON payloads,
handling missing keys and type-conversion errors gracefully.
"""

from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


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


def format_market_cap(market_cap: int) -> str:
    """Format a market cap integer as a human-readable string."""
    if market_cap >= 1_000_000_000:
        return f"{market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"{market_cap / 1_000_000:.2f}M"
    return str(market_cap)
