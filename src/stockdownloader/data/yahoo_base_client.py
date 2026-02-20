"""Abstract base for all Yahoo Finance API clients.

Extracts the shared boilerplate that every Yahoo client duplicates:

* Constructor with optional :class:`YahooAuthHelper`
* Lazy ``_ensure_authenticated()`` crumb check
* Generic ``_fetch_with_retry()`` loop with configurable parser callback
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, TypeVar

import requests

from stockdownloader.data.yahoo_auth_helper import YahooAuthHelper

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    requests.RequestException,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
)

T = TypeVar("T")


class YahooBaseClient:
    """Shared foundation for Yahoo Finance HTTP clients."""

    def __init__(self, auth: YahooAuthHelper | None = None) -> None:
        self._auth = auth or YahooAuthHelper()

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _ensure_authenticated(self) -> None:
        """Authenticate lazily — only if a crumb has not yet been obtained."""
        if self._auth.crumb is None:
            self._auth.authenticate()

    # ------------------------------------------------------------------
    # Retry infrastructure
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self,
        url: str,
        parser: Callable[[str], T],
        context: str,
        *,
        max_retries: int = _MAX_RETRIES,
        timeout: int = 15,
        extra_exceptions: tuple[type[Exception], ...] = (),
    ) -> T | None:
        """GET *url* and parse the response text with *parser*.

        Retries up to *max_retries* times on transient errors.  Returns
        ``None`` only when all retries are exhausted (and *parser* would
        normally return a value).
        """
        caught = _RETRY_EXCEPTIONS + extra_exceptions
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = self._auth.session.get(url, timeout=timeout)
                return parser(resp.text)
            except caught as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.debug(
                        "Retrying %s, attempt %d", context, attempt + 1
                    )
                else:
                    logger.warning(
                        "Failed %s after %d retries: %s",
                        context,
                        max_retries,
                        last_exc,
                    )

        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def auth(self) -> YahooAuthHelper:
        """Return the shared :class:`YahooAuthHelper`."""
        return self._auth
