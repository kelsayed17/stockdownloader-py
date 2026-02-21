"""Base data client with shared rate-limiting, session, and retry logic.

Provides :class:`BaseDataClient` as a foundation for HTTP-based data
clients that need:

* A ``requests.Session`` with configurable default headers
* Rate limiting between requests
* Retry logic for transient failures
* Cache directory setup

Subclasses override or extend these primitives while keeping
client-specific parsing, authentication, and business logic in the
concrete class.

Usage::

    class MyClient(BaseDataClient):
        def __init__(self) -> None:
            super().__init__(
                rate_limit_delay=0.5,
                max_retries=3,
                data_dir="data",
                default_headers={
                    "User-Agent": "MyApp",
                    "Accept": "application/json",
                },
            )
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class BaseDataClient:
    """Shared infrastructure for HTTP data clients.

    Parameters
    ----------
    rate_limit_delay:
        Minimum seconds between consecutive requests.
    max_retries:
        Maximum number of consecutive failures before giving up
        in :meth:`_fetch_with_retry`.
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/``.  Created automatically.
    default_headers:
        Headers applied to every request via the session.
    """

    def __init__(
        self,
        *,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
        data_dir: str = "data",
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._rate_limit_delay = rate_limit_delay
        self._max_retries = max_retries

        self._session = requests.Session()
        if default_headers:
            self._session.headers.update(default_headers)

        self._last_request_time: float = 0.0

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if needed to maintain the configured request rate."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response | None:
        """Send an HTTP request with automatic retry on transient errors.

        Parameters
        ----------
        method:
            HTTP method (``"GET"`` or ``"POST"``).
        url:
            Request URL.
        **kwargs:
            Passed through to ``self._session.request()``.  Common
            keys: ``json``, ``params``, ``timeout``.

        Returns
        -------
        The :class:`requests.Response` on success (any 2xx), or
        ``None`` after exhausting retries.
        """
        kwargs.setdefault("timeout", 30)

        for attempt in range(self._max_retries):
            try:
                self._rate_limit()
                resp = self._session.request(method, url, **kwargs)
                if resp.ok:
                    return resp
                if resp.status_code == 404:
                    # Resource does not exist -- don't retry
                    return resp
                logger.debug(
                    "%s %s returned %d (attempt %d/%d)",
                    method.upper(), url, resp.status_code,
                    attempt + 1, self._max_retries,
                )
            except (requests.RequestException, OSError) as exc:
                logger.debug(
                    "%s %s failed: %s (attempt %d/%d)",
                    method.upper(), url, exc,
                    attempt + 1, self._max_retries,
                )

            # Back off slightly on retries
            if attempt < self._max_retries - 1:
                time.sleep(0.5 * (attempt + 1))

        return None

    # ------------------------------------------------------------------
    # JSON caching helpers
    # ------------------------------------------------------------------

    def _symbol_dir(self, symbol: str) -> Path:
        """Return per-symbol data directory, creating it if needed."""
        d = self._data_dir / symbol.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_cached_json(self, key: str) -> Any | None:
        """Load a JSON cache file by *key* (filename without extension).

        Returns the parsed JSON data, or ``None`` if no cache exists
        or the cache is corrupt.
        """
        cache_file = self._data_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cache %s: %s", cache_file, exc)
            return None

    def _save_cached_json(self, key: str, data: Any) -> None:
        """Persist JSON-serializable *data* to cache under *key*."""
        cache_file = self._data_dir / f"{key}.json"
        try:
            cache_file.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save cache %s: %s", cache_file, exc)
