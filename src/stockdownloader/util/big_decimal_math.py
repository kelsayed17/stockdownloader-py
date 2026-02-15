"""Utility functions for common Decimal arithmetic with safe division handling.

Public constants
----------------
``ZERO``, ``ONE``, ``TWO``, ``HUNDRED`` — pre-constructed :class:`Decimal`
sentinels used across the codebase to avoid repeated ``Decimal("0")`` allocations.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

DEFAULT_SCALE = 10

# Pre-constructed Decimal constants used across the codebase.
ZERO = Decimal("0")
ONE = Decimal("1")
TWO = Decimal("2")
HUNDRED = Decimal("100")


def divide(dividend: Decimal, divisor: Decimal, scale: int = DEFAULT_SCALE) -> Decimal:
    """Divide two Decimals safely, returning zero when the divisor is zero.

    Args:
        dividend: The numerator.
        divisor: The denominator.
        scale: Number of decimal places (default 10).

    Returns:
        The quotient rounded to *scale* decimal places, or ``Decimal('0')`` when
        *divisor* is zero.
    """
    if divisor == Decimal('0'):
        return Decimal('0')
    result = dividend / divisor
    return _quantize(result, scale)



def quantize_decimal(value: Decimal, places: str = "0.01") -> Decimal:
    """Quantize *value* to the precision specified by *places*.

    Example::

        quantize_decimal(Decimal("1.2345"))           # -> Decimal("1.23")
        quantize_decimal(Decimal("1.2345"), "0.0001") # -> Decimal("1.2345")
    """
    return value.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def scale2(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places using ROUND_HALF_UP."""
    return _quantize(value, 2)


def average(*values: Decimal | None) -> Decimal:
    """Return the average of non-zero Decimal values.

    Zero-valued and ``None`` arguments are excluded.  Returns ``Decimal('0')``
    when no non-zero values are provided or when called with ``None``.
    """
    if not values or (len(values) == 1 and values[0] is None):
        return Decimal('0')
    total = Decimal('0')
    count = 0
    for v in values:
        if v is not None and v != Decimal('0'):
            total += v
            count += 1
    if count == 0:
        return Decimal('0')
    return _quantize(total / Decimal(str(count)), DEFAULT_SCALE)


def percent_change(from_val: Decimal, to_val: Decimal) -> Decimal:
    """Calculate percentage change from *from_val* to *to_val*.

    Returns ``Decimal('0')`` when *from_val* is zero.
    """
    if from_val == Decimal('0'):
        return Decimal('0')
    change = to_val - from_val
    return _quantize(change / from_val, 6) * Decimal('100')


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _quantize(value: Decimal, scale: int) -> Decimal:
    """Quantize *value* to the given number of decimal places."""
    return value.quantize(Decimal(10) ** -scale, rounding=ROUND_HALF_UP)
