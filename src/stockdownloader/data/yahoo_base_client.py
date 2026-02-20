"""Abstract base for all Yahoo Finance API clients.

Includes :class:`YahooAuthHelper` which handles cookie/crumb authentication,
and :class:`YahooBaseClient` which extracts the shared boilerplate that every
Yahoo client duplicates:

* Constructor with optional :class:`YahooAuthHelper`
* Lazy ``_ensure_authenticated()`` crumb check
* Generic ``_fetch_with_retry()`` loop with configurable parser callback

Authentication flow (via :class:`YahooAuthHelper`):

1. GET https://fc.yahoo.com to obtain session cookie
2. GET https://query2.finance.yahoo.com/v1/test/getcrumb with cookie to get crumb
3. Reuse cookie + crumb for all subsequent API calls
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, TypeVar

import requests

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Auth constants
# ------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_COOKIE_URL = "https://fc.yahoo.com"
_CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"

# ------------------------------------------------------------------
# Retry constants
# ------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    requests.RequestException,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
)

T = TypeVar("T")


# ===================================================================
# YahooAuthHelper
# ===================================================================


class YahooAuthHelper:
    """Handles cookie/crumb authentication for Yahoo Finance API."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._crumb: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """Initialise authentication by obtaining cookie and crumb.

        Returns ``True`` if successful.
        """
        try:
            # Step 1: Get cookie from fc.yahoo.com
            self._session.get(_COOKIE_URL, timeout=15)

            # Step 2: Get crumb using the cookie
            resp = self._session.get(_CRUMB_URL, timeout=15)
            crumb_text = resp.text.strip()

            if crumb_text and "Too Many Requests" not in crumb_text:
                self._crumb = crumb_text
                logger.debug("Yahoo Finance authentication successful")
                return True

            logger.warning("Failed to obtain valid crumb from Yahoo Finance")
            return False
        except (requests.RequestException, OSError) as exc:
            logger.warning("Yahoo Finance authentication failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def crumb(self) -> str | None:
        """Return the cached crumb, or ``None`` if not yet authenticated."""
        return self._crumb

    @property
    def session(self) -> requests.Session:
        """Return the underlying :class:`requests.Session`."""
        return self._session

    @property
    def user_agent(self) -> str:
        """Return the User-Agent header string."""
        return _USER_AGENT


# ===================================================================
# YahooBaseClient
# ===================================================================


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
