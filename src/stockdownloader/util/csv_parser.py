"""Lightweight CSV parser that handles standard CSV formatting.

Provides a context-manager-compatible parser for reading delimited data,
with support for quoted fields and configurable separators.
"""
from __future__ import annotations

import csv
import io
from typing import TextIO


class CsvParser:
    """A thin wrapper around Python's :mod:`csv` reader that mirrors the Java
    ``CsvParser`` API.

    Supports reading from any text stream (file, ``StringIO``, etc.) and can
    be used as a context manager.

    Args:
        source: A file-like text stream to read from.
        separator: The field delimiter character (default ``','``).
    """

    def __init__(self, source: TextIO, separator: str = ',') -> None:
        self._source = source
        self._reader = csv.reader(source, delimiter=separator, quotechar='"')

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bytes(cls, data: bytes, separator: str = ',') -> CsvParser:
        """Create a parser from raw bytes (decoded as UTF-8)."""
        return cls(io.TextIOWrapper(io.BytesIO(data), encoding='utf-8'), separator)

    @classmethod
    def from_string(cls, text: str, separator: str = ',') -> CsvParser:
        """Create a parser from an in-memory string."""
        return cls(io.StringIO(text), separator)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_next(self) -> list[str] | None:
        """Read and return the next row as a list of strings.

        Returns ``None`` when there are no more rows.
        """
        try:
            row = next(self._reader)
            return [field.strip() for field in row]
        except StopIteration:
            return None

    def read_all(self) -> list[list[str]]:
        """Read all remaining rows and return them as a list of lists."""
        result: list[list[str]] = []
        while True:
            row = self.read_next()
            if row is None:
                break
            result.append(row)
        return result

    def skip_lines(self, count: int) -> None:
        """Skip *count* lines from the underlying stream."""
        for _ in range(count):
            try:
                next(self._reader)
            except StopIteration:
                break

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying source stream."""
        self._source.close()

    def __enter__(self) -> CsvParser:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
