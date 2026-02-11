"""Shared Yahoo Finance authentication helper that obtains and caches
the cookie/crumb pair required by Yahoo Finance API endpoints.

Flow:
1. GET https://fc.yahoo.com to obtain session cookie
2. GET https://query2.finance.yahoo.com/v1/test/getcrumb with cookie to get crumb
3. Reuse cookie + crumb for all subsequent API calls
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_COOKIE_URL = "https://fc.yahoo.com"
_CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"


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
        except Exception as exc:
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
