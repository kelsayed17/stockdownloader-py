"""Shared base class for all FINRA data clients.

Extracts the duplicated OAuth2 authentication, constructor logic,
constants, and date normalization that were previously copied across
:mod:`finra_dark_pool_client`, :mod:`finra_short_interest_client`,
and :mod:`finra_short_volume_client`.

Subclasses only need to implement their specific API query, parsing,
caching, and public fetch methods.

Usage::

    class MyFinraClient(FinraBaseClient):
        def fetch_data(self, symbol: str) -> list:
            if self._access_token is None:
                self._authenticate()
            ...
"""

from __future__ import annotations

import base64
import logging
import os

import requests

from stockdownloader.data.base_client import BaseDataClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants shared across all FINRA clients
# ------------------------------------------------------------------

_FINRA_TOKEN_URL = (
    "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
    "?grant_type=client_credentials"
)
_MAX_RETRIES = 3
_RATE_LIMIT_DELAY = 0.5  # conservative rate for FINRA API


# ------------------------------------------------------------------
# Module-level date normalization
# ------------------------------------------------------------------


def normalize_finra_date(raw: str) -> str:
    """Normalize a date string to ``YYYY-MM-DD`` format.

    Handles ``YYYY-MM-DD``, ``MM/DD/YYYY``, and ``YYYYMMDD`` formats.
    Returns an empty string for unrecognizable dates.
    """
    if not raw:
        return ""

    # Already in ISO format
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return raw

    # Try MM/DD/YYYY
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3:
            try:
                month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass

    # Try YYYYMMDD
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    return ""


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------


class FinraBaseClient(BaseDataClient):
    """Shared base for FINRA data clients with OAuth2 authentication.

    Provides:

    * Constructor that configures the ``requests.Session`` with FINRA
      headers and stores OAuth2 credentials.
    * :meth:`_authenticate` to obtain an access token via the FINRA
      FIP OAuth2 endpoint.

    Rate limiting is inherited from :class:`BaseDataClient._rate_limit`.

    Parameters
    ----------
    client_id:
        FINRA API client ID.  Falls back to ``FINRA_CLIENT_ID`` env var.
    client_secret:
        FINRA API client secret.  Falls back to ``FINRA_CLIENT_SECRET`` env var.
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/``.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        data_dir: str = "data",
    ) -> None:
        super().__init__(
            rate_limit_delay=_RATE_LIMIT_DELAY,
            max_retries=_MAX_RETRIES,
            data_dir=data_dir,
            default_headers={
                "User-Agent": "StockDownloader admin@example.com",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        self._client_id = client_id or os.environ.get("FINRA_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get(
            "FINRA_CLIENT_SECRET", ""
        )
        self._access_token: str | None = None

    # ------------------------------------------------------------------
    # OAuth2 authentication
    # ------------------------------------------------------------------

    def _authenticate(self) -> bool:
        """Obtain an OAuth2 access token from FINRA FIP.

        Returns ``True`` if authentication succeeds.
        """
        if not self._client_id or not self._client_secret:
            logger.info(
                "FINRA credentials not configured — "
                "set FINRA_CLIENT_ID and FINRA_CLIENT_SECRET env vars"
            )
            return False

        credentials = f"{self._client_id}:{self._client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        try:
            resp = requests.post(
                _FINRA_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {encoded}",
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._access_token = data.get("access_token")
                if self._access_token:
                    self._session.headers["Authorization"] = (
                        f"Bearer {self._access_token}"
                    )
                    logger.info("FINRA OAuth2 authentication successful")
                    return True
            logger.warning(
                "FINRA OAuth2 failed: %d %s",
                resp.status_code,
                resp.text[:200],
            )
        except Exception as exc:
            logger.warning("FINRA OAuth2 error: %s", exc)

        return False
