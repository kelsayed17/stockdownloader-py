"""Fetches SEC EDGAR filing metadata for any company by CIK.

Uses the EDGAR submissions API (``data.sec.gov/submissions/CIK{cik}.json``)
which is free, requires no authentication, and returns structured filing
metadata.  Rate-limited to 10 requests per second (SEC fair-use policy).

Usage::

    client = SecEdgarClient()
    filings = client.fetch_gme_filings(form_types=["10-K", "10-Q", "8-K"])
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import requests

from stockdownloader.model.sec_models import SecFiling

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{filename}"
_ARCHIVE_BASE = (
    "https://www.sec.gov/Archives/edgar/data"
    "/{cik_raw}/{accession_nodashes}/{document}"
)
# SEC enforces 10 req/sec.  110 ms gap gives comfortable margin.
_RATE_LIMIT_DELAY = 0.11

# GameStop CIK
_GME_CIK = "0001326380"


class SecEdgarClient:
    """Downloads and parses SEC EDGAR filing metadata."""

    def __init__(
        self,
        user_agent: str = "StockDownloader admin@example.com",
    ) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_filings(
        self,
        cik: str,
        form_types: Sequence[str] | None = None,
    ) -> list[SecFiling]:
        """Fetch all filings for *cik*, optionally filtered by *form_types*.

        Returns filings sorted by ``filing_date`` ascending.
        """
        padded_cik = cik.lstrip("0").zfill(10)
        raw_cik = cik.lstrip("0")

        url = _SUBMISSIONS_URL.format(cik=padded_cik)
        data = self._fetch_json(url)
        if data is None:
            return []

        # Parse the "recent" filings block (up to ~1000 filings)
        filings = self._parse_filings_block(
            data.get("filings", {}).get("recent", {}),
            raw_cik,
        )

        # If more filings exist, they are in additional JSON files
        extra_files = data.get("filings", {}).get("files", [])
        for file_info in extra_files:
            filename = file_info.get("name", "")
            if not filename:
                continue
            page_url = _SUBMISSIONS_PAGE_URL.format(filename=filename)
            page_data = self._fetch_json(page_url)
            if page_data is not None:
                filings.extend(self._parse_filings_block(page_data, raw_cik))

        # Filter by form type if requested
        if form_types is not None:
            allowed = set(form_types)
            filings = [f for f in filings if f.form in allowed]

        # Sort by filing_date ascending
        filings.sort(key=lambda f: f.filing_date)
        return filings

    def fetch_gme_filings(
        self,
        form_types: Sequence[str] | None = None,
    ) -> list[SecFiling]:
        """Convenience method to fetch GameStop (CIK 0001326380) filings."""
        return self.fetch_filings(_GME_CIK, form_types=form_types)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if needed to maintain the SEC 10 req/sec rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()

    def _fetch_json(self, url: str) -> dict | None:
        """GET *url* with retry logic and rate limiting."""
        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limit()
                resp = self._session.get(url, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(
                    "SEC EDGAR returned %d for %s (attempt %d/%d)",
                    resp.status_code, url, attempt + 1, _MAX_RETRIES,
                )
            except (requests.RequestException, OSError) as exc:
                logger.warning(
                    "SEC EDGAR request failed: %s (attempt %d/%d)",
                    exc, attempt + 1, _MAX_RETRIES,
                )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(1.0)
        return None

    @staticmethod
    def _parse_filings_block(
        block: dict,
        raw_cik: str,
    ) -> list[SecFiling]:
        """Parse the parallel-array structure from EDGAR into SecFiling objects.

        The EDGAR JSON uses parallel arrays::

            {
                "accessionNumber": ["...", "..."],
                "filingDate": ["...", "..."],
                "form": ["...", "..."],
                ...
            }
        """
        accession_numbers: list[str] = block.get("accessionNumber", [])
        filing_dates: list[str] = block.get("filingDate", [])
        report_dates: list[str] = block.get("reportDate", [])
        forms: list[str] = block.get("form", [])
        primary_docs: list[str] = block.get("primaryDocument", [])
        descriptions: list[str] = block.get("primaryDocDescription", [])

        count = len(accession_numbers)
        filings: list[SecFiling] = []

        for i in range(count):
            accession = accession_numbers[i] if i < len(accession_numbers) else ""
            if not accession:
                continue

            # Build the full filing URL.
            # Accession format: "0001326380-24-000013"
            # URL path uses no dashes: "000132638024000013"
            accession_nodashes = accession.replace("-", "")
            doc = primary_docs[i] if i < len(primary_docs) else ""
            filing_url = _ARCHIVE_BASE.format(
                cik_raw=raw_cik,
                accession_nodashes=accession_nodashes,
                document=doc,
            )

            try:
                filings.append(SecFiling(
                    accession_number=accession,
                    filing_date=filing_dates[i] if i < len(filing_dates) else "",
                    report_date=report_dates[i] if i < len(report_dates) else "",
                    form=forms[i] if i < len(forms) else "",
                    primary_document=doc,
                    description=descriptions[i] if i < len(descriptions) else "",
                    filing_url=filing_url,
                ))
            except ValueError:
                # Skip filings that fail validation (empty required fields)
                continue

        return filings
