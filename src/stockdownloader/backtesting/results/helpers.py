"""Shared rounding utilities for report formatters.

Provides :func:`scale2` and :func:`scale3` — previously duplicated
as ``scale2()`` in :mod:`report_formatter` and ``_s2()`` / ``_s3()``
in :mod:`exit_tournament_report_formatter`.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def scale2(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def scale3(value: Decimal) -> Decimal:
    """Round a Decimal to 3 decimal places."""
    return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
