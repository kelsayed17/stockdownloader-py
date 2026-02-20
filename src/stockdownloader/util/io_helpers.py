"""Centralized I/O operations and retry logic.

File helpers use :class:`pathlib.Path` for all filesystem interactions.
Retry helpers provide both a fire-and-forget :func:`execute` for callables
that return nothing, and a :func:`execute_with_result` variant for callables
that produce a value.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, TextIO, TypeVar

T = TypeVar('T')

logger = logging.getLogger(__name__)

# =========================================================================
# File I/O helpers (formerly file_helper.py)
# =========================================================================


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


class TeeWriter:
    """Write to both stdout and a file simultaneously."""

    def __init__(self, file: TextIO) -> None:
        self._file = file

    def write(self, msg: str) -> None:
        """Write *msg* to both stdout and the backing file."""
        sys.stdout.write(msg)
        sys.stdout.flush()
        self._file.write(msg)
        self._file.flush()

    def print(self, msg: str = "") -> None:
        """Write *msg* followed by a newline (convenience wrapper)."""
        self.write(msg + "\n")


# =========================================================================
# Retry logic (formerly retry_executor.py)
# =========================================================================

DEFAULT_MAX_RETRIES = 3


def execute(
    action: Callable[[], None],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    logger: logging.Logger | None = None,
    context: str = '',
    delay: float = 0.0,
) -> None:
    """Execute *action*, retrying up to *max_retries* times on exception.

    Args:
        action: A no-argument callable to execute.
        max_retries: Maximum number of retry attempts (default 3).
        logger: Logger for retry/failure messages. Falls back to module logger.
        context: Descriptive label included in log messages.
        delay: Seconds to sleep between retries (default 0).
    """
    log = logger or logging.getLogger(__name__)
    for attempt in range(max_retries + 1):
        try:
            action()
            return
        except Exception as e:
            if attempt < max_retries:
                log.debug('Retrying %s, attempt %d', context, attempt + 1)
                if delay > 0:
                    time.sleep(delay)
            else:
                log.warning(
                    'Failed %s after %d retries: %s', context, max_retries, e
                )


def execute_with_result(
    action: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    logger: logging.Logger | None = None,
    context: str = '',
    delay: float = 0.0,
) -> T | None:
    """Execute *action* and return its result, retrying on failure.

    Returns ``None`` if all attempts are exhausted.

    Args:
        action: A no-argument callable whose return value is desired.
        max_retries: Maximum number of retry attempts (default 3).
        logger: Logger for retry/failure messages. Falls back to module logger.
        context: Descriptive label included in log messages.
        delay: Seconds to sleep between retries (default 0).
    """
    log = logger or logging.getLogger(__name__)
    for attempt in range(max_retries + 1):
        try:
            return action()
        except Exception as e:
            if attempt < max_retries:
                log.debug('Retrying %s, attempt %d', context, attempt + 1)
                if delay > 0:
                    time.sleep(delay)
            else:
                log.warning(
                    'Failed %s after %d retries: %s', context, max_retries, e
                )
    return None
