"""Centralized file I/O operations for reading, writing, appending, and deleting files.

Uses :class:`pathlib.Path` for all filesystem interactions.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def write_lines(lines: Iterable[str], filename: str) -> None:
    """Write *lines* joined by the platform line separator to *filename*.

    Creates or truncates the file.
    """
    try:
        Path(filename).write_text(
            os.linesep.join(lines), encoding='utf-8'
        )
    except OSError as e:
        logger.warning('Error writing file %s: %s', filename, e)


def write_content(content: str, filename: str) -> None:
    """Write *content* to *filename*, stripping bracket characters ``[`` and ``]``.

    Creates or truncates the file.
    """
    try:
        cleaned = content.replace('[', '').replace(']', '')
        Path(filename).write_text(cleaned, encoding='utf-8')
    except OSError as e:
        logger.warning('Error writing file %s: %s', filename, e)


def read_lines(filename: str) -> set[str]:
    """Read non-blank lines from *filename* into a sorted set.

    Returns an empty set if the file does not exist or an error occurs.
    """
    result: set[str] = set()
    path = Path(filename)
    try:
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                stripped = line.strip()
                if stripped:
                    result.add(stripped)
    except OSError as e:
        logger.warning('Error reading file %s: %s', filename, e)
    return result


def read_csv_lines(filename: str) -> set[str]:
    """Read a file, split each line by commas, and return a sorted set of values."""
    result: set[str] = set()
    try:
        for line in Path(filename).read_text(encoding='utf-8').splitlines():
            for token in line.split(','):
                stripped = token.strip()
                if stripped:
                    result.add(stripped)
    except OSError as e:
        logger.warning('Error reading CSV file %s: %s', filename, e)
    return result


def append_line(line: str, filename: str) -> None:
    """Append *line* followed by a newline to *filename*.

    Creates the file if it does not exist.
    """
    try:
        with Path(filename).open('a', encoding='utf-8') as f:
            f.write(line + os.linesep)
    except OSError as e:
        logger.warning('Error appending to %s: %s', filename, e)


def delete_file(filename: str) -> bool:
    """Delete *filename* if it exists.

    Returns ``True`` if the file was deleted, ``False`` otherwise.
    """
    try:
        path = Path(filename)
        if path.exists():
            path.unlink()
            return True
        return False
    except OSError as e:
        logger.warning('Error deleting %s: %s', filename, e)
        return False
