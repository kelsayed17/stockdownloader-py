"""Utility functions for common Decimal arithmetic with safe division handling.

Public constants
----------------
``ZERO``, ``ONE``, ``TWO``, ``THREE``, ``HALF``, ``TEN``, ``HUNDRED`` —
pre-constructed :class:`Decimal` sentinels used across the codebase to avoid
repeated ``Decimal("0")`` allocations.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

DEFAULT_SCALE = 10

# Pre-constructed Decimal constants used across the codebase.
ZERO = Decimal("0")
ONE = Decimal("1")
TWO = Decimal("2")
HUNDRED = Decimal("100")
HALF = Decimal("0.5")
THREE = Decimal("3")
TEN = Decimal("10")


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
    return quantize(result, scale)



def quantize_decimal(value: Decimal, places: str = "0.01") -> Decimal:
    """Quantize *value* to the precision specified by *places*.

    Example::

        quantize_decimal(Decimal("1.2345"))           # -> Decimal("1.23")
        quantize_decimal(Decimal("1.2345"), "0.0001") # -> Decimal("1.2345")
    """
    return value.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def scale2(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places using ROUND_HALF_UP."""
    return quantize(value, 2)


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
    return quantize(total / Decimal(str(count)), DEFAULT_SCALE)


def percent_change(from_val: Decimal, to_val: Decimal) -> Decimal:
    """Calculate percentage change from *from_val* to *to_val*.

    Returns ``Decimal('0')`` when *from_val* is zero.
    """
    if from_val == Decimal('0'):
        return Decimal('0')
    change = to_val - from_val
    return quantize(change / from_val, 6) * Decimal('100')


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def quantize(value: Decimal, scale: int = DEFAULT_SCALE) -> Decimal:
    """Quantize *value* to the given number of decimal places."""
    return value.quantize(Decimal(10) ** -scale, rounding=ROUND_HALF_UP)


# Backward-compatible alias for internal util callers.
_quantize = quantize


__all__ = [
    "DEFAULT_SCALE",
    "ZERO",
    "ONE",
    "TWO",
    "HUNDRED",
    "HALF",
    "THREE",
    "TEN",
    "divide",
    "quantize_decimal",
    "scale2",
    "average",
    "percent_change",
    "quantize",
]
