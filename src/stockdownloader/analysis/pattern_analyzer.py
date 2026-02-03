"""Analyzes stock price movement patterns, computing frequency,
accuracy, and offset-by-one-day variations.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import divide, multiply, scale2

if TYPE_CHECKING:
    from stockdownloader.model.pattern_result import PatternResult

_TOP_N = 10


def analyze(frequency: dict[str, set[str]]) -> list[PatternResult]:
    """Analyze pattern frequency data and return results sorted by frequency descending.

    Args:
        frequency: A mapping of pattern key strings to sets of symbol strings.
                   In Java this was a ``HashMultimap<String, String>``;
                   in Python use ``defaultdict(set)``.

    Returns:
        A list of ``PatternResult`` objects sorted by pattern frequency (descending).
    """
    from stockdownloader.model.pattern_result import PatternResult

    results: list[PatternResult] = []

    for pattern_key in frequency:
        inner = pattern_key[1:-1]  # strip surrounding brackets
        up_down_list = inner.split(", ")
        up_down_list = list(up_down_list)

        element = int(up_down_list[0])
        if element == 1:
            up_down_list[0] = "-1"
        elif element == -1:
            up_down_list[0] = "1"
        else:
            up_down_list[0] = "0"

        similar_key = "[" + ", ".join(up_down_list) + "]"
        offset_list = up_down_list[1:]
        offset_key = "[" + ", ".join(offset_list) + "]"

        pattern_set = frequency.get(pattern_key, set())
        similar_set = frequency.get(similar_key, set())
        offset_set = frequency.get(offset_key, set())

        pattern_freq = Decimal(str(len(pattern_set)))
        similar_freq = Decimal(str(len(similar_set)))
        offset_freq = Decimal(str(len(offset_set)))

        total = pattern_freq + similar_freq
        accuracy = scale2(divide(multiply(pattern_freq, Decimal("100")), total))

        results.append(
            PatternResult(
                pattern=pattern_key,
                similar=similar_key,
                offset=offset_key,
                pattern_freq=pattern_freq,
                similar_freq=similar_freq,
                offset_freq=offset_freq,
                accuracy=accuracy,
                pattern_symbols=pattern_set,
                similar_symbols=similar_set,
                offset_symbols=offset_set,
            )
        )

    # Sort by pattern_freq descending (matches Java TreeSet with compareTo)
    results.sort(key=lambda r: r.pattern_freq, reverse=True)
    return results


def print_results(results: list[PatternResult]) -> None:
    """Print the top-10 patterns and their offset-by-one-day counterparts.

    Args:
        results: Sorted list of ``PatternResult`` objects.
    """
    print("Top 10 patterns")
    print(f"{'Frequency:':<16}{'Percentage:':<16}{'Pattern:':<64}{'Stocks:'}")

    for i, entry in enumerate(results):
        if i >= _TOP_N:
            break
        print(
            f"{entry.pattern_freq!s:<16}"
            f"{entry.accuracy!s}%{'':.<15}"[: 16]
            + f"{entry.pattern:<64}"
            f"{entry.pattern_symbols}"
        )

    print()
    print("Top 10 patterns offset by one day")
    print(f"{'Frequency:':<16}{'Percentage:':<16}{'Pattern:':<64}{'Stocks:'}")

    for i, entry in enumerate(results):
        if i >= _TOP_N:
            break
        print(
            f"{entry.offset_freq!s:<16}"
            f"{entry.accuracy!s}%{'':.<15}"[: 16]
            + f"{entry.offset:<64}"
            f"{entry.offset_symbols}"
        )
