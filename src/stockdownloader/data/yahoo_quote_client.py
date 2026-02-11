"""HTTP client for Yahoo Finance that handles authentication via the modern
cookie/crumb flow (fc.yahoo.com + /v1/test/getcrumb) and historical
CSV data downloads.

Replaces the legacy crumb extraction that parsed "CrumbStore" from
Yahoo Finance page HTML, which no longer works.
"""
from __future__ import annotations

import logging
from pathlib import Path

from stockdownloader.data.yahoo_auth_helper import YahooAuthHelper

logger = logging.getLogger(__name__)

_QUOTE_PAGE_URL = "https://finance.yahoo.com/quote/{symbol}/"
_DOWNLOAD_URL = (
    "https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
    "?period1={start}&period2={end}&interval=1d&events=history&crumb={crumb}"
)


class YahooQuoteClient:
    """Fetches Yahoo Finance quote pages and downloads historical CSV data."""

    def __init__(self, auth: YahooAuthHelper | None = None) -> None:
        self._auth = auth or YahooAuthHelper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_page(self, symbol: str) -> str | None:
        """Fetch the full HTML of the Yahoo Finance quote page for *symbol*.

        Returns ``None`` on failure.
        """
        url = _QUOTE_PAGE_URL.format(symbol=symbol)
        try:
            resp = self._auth.session.get(url, timeout=15)
            return resp.text
        except Exception:
            logger.warning("Failed to fetch page for symbol: %s", symbol)
            return None

    def get_crumb(self, symbol: str) -> str:
        """Obtain a crumb for Yahoo Finance API authentication using the
        modern fc.yahoo.com cookie flow.
        """
        if self._auth.authenticate():
            return self._auth.crumb or ""
        return ""

    def download_data(
        self,
        symbol: str,
        start_date: int,
        end_date: int,
        crumb: str,
    ) -> None:
        """Download historical CSV data for *symbol* and save to
        ``{symbol}.csv`` in the current directory.
        """
        filename = f"{symbol}.csv"
        url = _DOWNLOAD_URL.format(
            symbol=symbol,
            start=start_date,
            end=end_date,
            crumb=crumb,
        )
        try:
            resp = self._auth.session.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            Path(filename).write_bytes(resp.content)
        except Exception:
            logger.warning("Failed to download data for symbol: %s", symbol)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def auth(self) -> YahooAuthHelper:
        """Return the shared auth helper for use with other clients."""
        return self._auth
