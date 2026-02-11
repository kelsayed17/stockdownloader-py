"""Holds the result of a pattern frequency analysis."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import FrozenSet


@dataclass(frozen=True)
class PatternResult:
    """Holds the result of a pattern frequency analysis including the pattern,
    its inverse, offset, frequencies, and associated symbols.

    Ordered by descending pattern_freq (higher frequency first).
    """

    pattern: str
    similar: str
    offset: str
    pattern_freq: Decimal
    similar_freq: Decimal
    offset_freq: Decimal
    accuracy: Decimal
    pattern_symbols: frozenset[str]
    similar_symbols: frozenset[str]
    offset_symbols: frozenset[str]

    def __lt__(self, other: PatternResult) -> bool:
        """Sort by descending pattern_freq (higher frequency comes first)."""
        return other.pattern_freq < self.pattern_freq
