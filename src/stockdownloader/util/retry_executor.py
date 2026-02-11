"""Execute an action with configurable retry logic.

Provides both a fire-and-forget :func:`execute` for callables that return
nothing, and a :func:`execute_with_result` variant for callables that produce
a value.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

T = TypeVar('T')

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
