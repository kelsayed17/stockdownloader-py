"""Parses 13F institutional ownership from SEC EDGAR.

**Primary approach**: Downloads SEC Form 13F bulk data sets (TSV) from
``sec.gov/files/structureddata/data/form-13f-data-sets/``.  These contain
every 13F-HR filing for an entire quarter in flat TSV format, with one row
per holding.  Filtering by CUSIP gives complete institutional ownership.

**Fallback** (for the most recent quarter whose bulk file may not yet
exist): Uses the EDGAR full-text search API (``efts.sec.gov``) with
per-quarter date windowing to find individual 13F-HR filings, then
downloads and parses each filing's XML.

Rate-limited to 10 req/s (SEC fair-use policy).  Results are cached
locally as JSON.

Usage::

    client = SecOwnershipClient()
    snapshots = client.fetch_ownership_snapshots("GME", cusip="36467W109")
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path

import requests

from stockdownloader.model.regulatory_records import (
    InstitutionalHolding,
    OwnershipSnapshot,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
# SEC enforces 10 req/sec.  110 ms gap gives comfortable margin.
_RATE_LIMIT_DELAY = 0.11

# SEC bulk 13F data sets
_BULK_13F_BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets"

# EDGAR full-text search index (fallback for current quarter)
_EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# Direct archive access for filing documents (EFTS fallback)
_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

# 13F XML namespace variants
_NS_13F = "http://www.sec.gov/edgar/document/thirteenf/informationtable"
_NS_13F_ALT = "http://www.sec.gov/edgar/thirteenf"

# GameStop defaults
_GME_CUSIP = "36467W109"

# Quarter-end dates for each calendar quarter
_QUARTER_ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}

# Month names for bulk file URL construction
_MONTH_NAMES = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}


def _quarter_range(year: int, quarter: int) -> tuple[date, date]:
    """Return (start_date, end_date) for a calendar quarter."""
    start_month = (quarter - 1) * 3 + 1
    end_month = quarter * 3
    start = date(year, start_month, 1)
    # End of quarter
    if end_month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, end_month + 1, 1) - timedelta(days=1)
    return start, end


def _bulk_zip_urls(year: int, quarter: int) -> list[str]:
    """Build candidate SEC bulk 13F data set ZIP URLs for a quarter.

    Returns multiple URL candidates because SEC changed their naming
    convention in 2024:

    - **2024+** (date-range naming): ``01jun2024-31aug2024_form13f.zip``
    - **Pre-2024** (quarter naming): ``2023q4_form13f.zip``

    The date-range names correspond to the 13F filing window (roughly
    45–105 days after quarter end).
    """
    urls: list[str] = []

    # New format (2024+): date-range based on filing window
    #   Q1 (Mar 31) filings due May 15 -> window 01mar-31may
    #   Q2 (Jun 30) filings due Aug 14 -> window 01jun-31aug
    #   Q3 (Sep 30) filings due Nov 14 -> window 01sep-30nov
    #   Q4 (Dec 31) filings due Feb 14 -> window 01dec-28feb (next yr)
    #
    # NOTE: SEC uses the quarter-end filing window months, NOT the
    # quarter months themselves. Mapping from actual SEC URLs:
    #   01mar{Y}-31may{Y}     = Q4 of (Y-1)
    #   01jun{Y}-31aug{Y}     = Q1 of Y
    #   01sep{Y}-30nov{Y}     = Q2 of Y
    #   01dec{Y}-28feb{Y+1}   = Q3 of Y
    #   01jan{Y}-29feb{Y}     = special Q4 format (seen for 2023-Q4)
    if quarter == 1:
        urls.append(
            f"{_BULK_13F_BASE}/01jun{year}-31aug{year}_form13f.zip"
        )
    elif quarter == 2:
        urls.append(
            f"{_BULK_13F_BASE}/01sep{year}-30nov{year}_form13f.zip"
        )
    elif quarter == 3:
        # Feb end: 28 or 29 depending on leap year
        feb_end = 29 if _is_leap_year(year + 1) else 28
        urls.append(
            f"{_BULK_13F_BASE}/01dec{year}-{feb_end}feb{year + 1}"
            "_form13f.zip"
        )
    else:  # Q4
        urls.append(
            f"{_BULK_13F_BASE}/01mar{year + 1}-31may{year + 1}"
            "_form13f.zip"
        )
        # Alternate Q4 naming seen for 2023:
        # 01jan{Y+1}-29feb{Y+1}
        feb_end = 29 if _is_leap_year(year + 1) else 28
        urls.append(
            f"{_BULK_13F_BASE}/01jan{year + 1}-{feb_end}feb{year + 1}"
            "_form13f.zip"
        )

    # Old format (pre-2024): simple quarter naming
    urls.append(f"{_BULK_13F_BASE}/{year}q{quarter}_form13f.zip")

    return urls


def _is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


class SecOwnershipClient:
    """Parses 13F institutional ownership from SEC EDGAR.

    Uses SEC bulk 13F data sets (TSV) as the primary, authoritative source.
    Falls back to EFTS full-text search for the most recent quarter if the
    bulk file is not yet available.
    """

    def __init__(
        self,
        user_agent: str = "StockDownloader admin@example.com",
        cache_dir: str = "data/cache/ownership",
    ) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_time: float = 0.0
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Separate directory for downloaded bulk ZIP files
        self._bulk_dir = self._cache_dir / "bulk_13f"
        self._bulk_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ownership_snapshots(
        self,
        symbol: str,
        cusip: str = _GME_CUSIP,
        num_quarters: int = 24,
        *,
        force_refresh: bool = False,
    ) -> list[OwnershipSnapshot]:
        """Fetch institutional ownership snapshots for *symbol*.

        Downloads SEC bulk 13F data sets for each quarter, filters by
        CUSIP, and aggregates into quarterly :class:`OwnershipSnapshot`
        objects.

        Parameters
        ----------
        symbol:
            Ticker symbol (used for caching and labeling).
        cusip:
            CUSIP identifier to filter 13F holdings.
        num_quarters:
            Maximum number of quarterly snapshots to return.
        force_refresh:
            If ``True``, ignore the JSON cache and re-download.

        Returns
        -------
        List of :class:`OwnershipSnapshot` sorted by ``quarter_end``
        ascending.
        """
        symbol_upper = symbol.upper()

        # Try cache first (unless force refresh)
        if not force_refresh:
            cached = self._load_cache(symbol_upper)
            if cached is not None:
                logger.info(
                    "Using cached ownership data for %s (%d snapshots)",
                    symbol_upper, len(cached),
                )
                return cached[-num_quarters:]

        # Determine quarter range to fetch
        today = date.today()
        current_quarter = (today.month - 1) // 3 + 1
        current_year = today.year

        snapshots: list[OwnershipSnapshot] = []

        # Work backwards from the current quarter
        year, quarter = current_year, current_quarter
        quarters_fetched = 0

        while quarters_fetched < num_quarters:
            quarter_end_str = f"{year}-{_QUARTER_ENDS[quarter]}"
            quarter_end_date = date.fromisoformat(quarter_end_str)

            # Skip future quarters
            if quarter_end_date > today:
                year, quarter = _prev_quarter(year, quarter)
                continue

            logger.info(
                "Fetching 13F data for %s Q%d %d (%s)",
                symbol_upper, quarter, year, quarter_end_str,
            )

            # Try bulk data set first (primary approach)
            snap = self._fetch_from_bulk(
                symbol_upper, cusip, year, quarter, quarter_end_str,
            )

            # Fallback to EFTS search for recent quarters
            if snap is None:
                logger.info(
                    "Bulk data not available for Q%d %d, "
                    "trying EFTS fallback...", quarter, year,
                )
                snap = self._fetch_from_efts(
                    symbol_upper, cusip, year, quarter, quarter_end_str,
                )

            if snap is not None:
                snapshots.append(snap)

            quarters_fetched += 1
            year, quarter = _prev_quarter(year, quarter)

            # Stop if we go before 2013 Q3 (bulk data starts here)
            if year < 2013 or (year == 2013 and quarter < 3):
                break

        snapshots.sort(key=lambda s: s.quarter_end)

        if snapshots:
            self._save_cache(symbol_upper, snapshots)

        return snapshots[-num_quarters:]

    # ------------------------------------------------------------------
    # Primary: SEC Bulk 13F Data Sets
    # ------------------------------------------------------------------

    def _fetch_from_bulk(
        self,
        symbol: str,
        cusip: str,
        year: int,
        quarter: int,
        quarter_end: str,
    ) -> OwnershipSnapshot | None:
        """Download and parse a SEC bulk 13F data set for one quarter.

        Downloads the ZIP file, extracts INFOTABLE.tsv and SUBMISSION.tsv,
        filters by CUSIP, and aggregates into an OwnershipSnapshot.
        """
        zip_path = self._bulk_dir / f"{year}Q{quarter}_form13f.zip"

        # Download if not cached
        if not zip_path.exists():
            urls = _bulk_zip_urls(year, quarter)
            zip_data = None
            for url in urls:
                logger.info("Trying bulk 13F URL: %s", url)
                zip_data = self._download_with_retry(url)
                if zip_data is not None:
                    break
            if zip_data is None:
                return None
            try:
                zip_path.write_bytes(zip_data)
            except OSError as exc:
                logger.warning("Failed to cache ZIP: %s", exc)
                # Continue with in-memory data
                return self._parse_bulk_zip(
                    io.BytesIO(zip_data), symbol, cusip, quarter_end,
                )

        # Parse the cached ZIP
        try:
            return self._parse_bulk_zip(
                zip_path, symbol, cusip, quarter_end,
            )
        except (zipfile.BadZipFile, OSError) as exc:
            logger.warning(
                "Failed to parse bulk ZIP %s: %s", zip_path, exc,
            )
            # Delete corrupted file and retry next time
            zip_path.unlink(missing_ok=True)
            return None

    def _parse_bulk_zip(
        self,
        zip_source: Path | io.BytesIO,
        symbol: str,
        cusip: str,
        quarter_end: str,
    ) -> OwnershipSnapshot | None:
        """Parse a bulk 13F ZIP file and extract holdings for *cusip*.

        The ZIP contains (linked by ACCESSION_NUMBER):

        - INFOTABLE.tsv: one row per holding (CUSIP, shares, value, etc.)
        - SUBMISSION.tsv: one row per filing (CIK, filing date)
        - COVERPAGE.tsv: one row per filing (FILINGMANAGER_NAME, address)
        """
        cusip_upper = cusip.upper().replace(" ", "")

        try:
            zf = zipfile.ZipFile(zip_source)
        except zipfile.BadZipFile:
            return None

        with zf:
            # Find the TSV files (names may vary slightly)
            names = zf.namelist()
            infotable_name = None
            submission_name = None
            coverpage_name = None

            for name in names:
                lower = name.lower()
                if "infotable" in lower and lower.endswith(".tsv"):
                    infotable_name = name
                elif "submission" in lower and lower.endswith(".tsv"):
                    submission_name = name
                elif "coverpage" in lower and lower.endswith(".tsv"):
                    coverpage_name = name

            if not infotable_name:
                logger.warning(
                    "No INFOTABLE.tsv found in ZIP, files: %s", names,
                )
                return None

            # Build accession -> (manager_name, cik) from both
            # COVERPAGE.tsv (has FILINGMANAGER_NAME) and
            # SUBMISSION.tsv (has CIK).
            managers: dict[str, tuple[str, str]] = {}  # accession -> (name, cik)

            # Step 1: Read CIKs from SUBMISSION.tsv
            cik_map: dict[str, str] = {}  # accession -> cik
            if submission_name:
                try:
                    sub_data = zf.read(submission_name).decode(
                        "utf-8", errors="replace",
                    )
                    sub_reader = csv.DictReader(
                        io.StringIO(sub_data), delimiter="\t",
                    )
                    for row in sub_reader:
                        acc = row.get("ACCESSION_NUMBER", "").strip()
                        cik = row.get("CIK", "").strip()
                        if acc:
                            cik_map[acc] = cik
                except Exception as exc:
                    logger.warning(
                        "Failed to parse SUBMISSION.tsv: %s", exc,
                    )

            # Step 2: Read manager names from COVERPAGE.tsv
            if coverpage_name:
                try:
                    cp_data = zf.read(coverpage_name).decode(
                        "utf-8", errors="replace",
                    )
                    cp_reader = csv.DictReader(
                        io.StringIO(cp_data), delimiter="\t",
                    )
                    for row in cp_reader:
                        acc = row.get("ACCESSION_NUMBER", "").strip()
                        mgr_name = row.get(
                            "FILINGMANAGER_NAME", "",
                        ).strip()
                        if acc:
                            cik = cik_map.get(acc, "")
                            managers[acc] = (mgr_name or "Unknown", cik)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse COVERPAGE.tsv: %s", exc,
                    )

            # Fallback: if COVERPAGE was missing, use CIK as identifier
            if not managers and cik_map:
                for acc, cik in cik_map.items():
                    managers[acc] = (f"CIK-{cik}", cik)

            # Parse INFOTABLE.tsv and filter by CUSIP
            holdings: list[InstitutionalHolding] = []
            seen_accessions: set[str] = set()

            try:
                info_data = zf.read(infotable_name).decode(
                    "utf-8", errors="replace",
                )
                info_reader = csv.DictReader(
                    io.StringIO(info_data), delimiter="\t",
                )

                for row in info_reader:
                    row_cusip = (
                        row.get("CUSIP", "").strip().upper().replace(" ", "")
                    )
                    if row_cusip != cusip_upper:
                        continue

                    accession = row.get("ACCESSION_NUMBER", "").strip()
                    shares_text = row.get("SSHPRNAMT", "").strip()
                    value_text = row.get("VALUE", "").strip()
                    share_type = (
                        row.get("SSHPRNAMTTYPE", "").strip() or "SH"
                    )

                    # Only count shares, not principal amounts
                    if share_type.upper() not in ("SH", ""):
                        continue

                    try:
                        shares = int(float(shares_text)) if shares_text else 0
                    except (ValueError, OverflowError):
                        shares = 0

                    try:
                        value_usd = int(float(value_text)) if value_text else 0
                    except (ValueError, OverflowError):
                        value_usd = 0

                    if shares <= 0:
                        continue

                    # Look up manager from COVERPAGE + SUBMISSION
                    mgr_name, mgr_cik = managers.get(
                        accession, ("Unknown", ""),
                    )

                    # Deduplicate: some filers report the same CUSIP
                    # multiple times (e.g., different share classes).
                    # We sum them per accession at aggregation time.
                    try:
                        holdings.append(InstitutionalHolding(
                            filing_date=quarter_end,
                            manager_name=mgr_name or "Unknown",
                            manager_cik=mgr_cik,
                            shares=shares,
                            value_usd=value_usd,
                            share_class=share_type,
                        ))
                    except ValueError:
                        continue

                    seen_accessions.add(accession)

            except Exception as exc:
                logger.warning(
                    "Failed to parse INFOTABLE.tsv: %s", exc,
                )
                return None

        if not holdings:
            logger.info(
                "No holdings found for CUSIP %s in bulk data", cusip,
            )
            return None

        # Aggregate per-manager (sum shares within same accession)
        agg: dict[str, list[InstitutionalHolding]] = {}
        for h in holdings:
            key = f"{h.manager_name}|{h.manager_cik}"
            if key not in agg:
                agg[key] = []
            agg[key].append(h)

        # Merge holdings per manager
        merged: list[InstitutionalHolding] = []
        for key, manager_holdings in agg.items():
            total_shares = sum(h.shares for h in manager_holdings)
            total_value = sum(h.value_usd for h in manager_holdings)
            ref = manager_holdings[0]
            try:
                merged.append(InstitutionalHolding(
                    filing_date=ref.filing_date,
                    manager_name=ref.manager_name,
                    manager_cik=ref.manager_cik,
                    shares=total_shares,
                    value_usd=total_value,
                    share_class=ref.share_class,
                ))
            except ValueError:
                continue

        total_shares = sum(h.shares for h in merged)
        num_institutions = len(merged)
        sorted_holdings = sorted(
            merged, key=lambda h: h.shares, reverse=True,
        )
        top_10_shares = sum(h.shares for h in sorted_holdings[:10])
        top_10_conc = top_10_shares / total_shares if total_shares > 0 else 0.0

        logger.info(
            "Bulk 13F for %s Q%d: %d institutions, %s shares",
            symbol, (int(quarter_end[5:7]) - 1) // 3 + 1,
            num_institutions, f"{total_shares:,}",
        )

        try:
            return OwnershipSnapshot(
                quarter_end=quarter_end,
                symbol=symbol,
                total_institutional_shares=total_shares,
                num_institutions=num_institutions,
                top_10_concentration=top_10_conc,
                holdings=tuple(sorted_holdings),
            )
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Fallback: EDGAR Full-Text Search (EFTS) for current quarter
    # ------------------------------------------------------------------

    def _fetch_from_efts(
        self,
        symbol: str,
        cusip: str,
        year: int,
        quarter: int,
        quarter_end: str,
    ) -> OwnershipSnapshot | None:
        """Fetch 13F data via EFTS for a single quarter (fallback).

        Uses date-range windowing to stay under the 10,000-hit limit.
        """
        # 13F filings for a quarter are filed 0-45 days after quarter end.
        _, q_end = _quarter_range(year, quarter)
        search_start = q_end + timedelta(days=1)
        search_end = q_end + timedelta(days=60)

        start_str = search_start.strftime("%Y-%m-%d")
        end_str = search_end.strftime("%Y-%m-%d")

        base_url = (
            f"{_EFTS_SEARCH_URL}?q=%22{cusip}%22&forms=13F-HR"
            f"&startdt={start_str}&enddt={end_str}"
        )

        # Page through results
        all_hits: list[dict] = []
        page_size = 100
        offset = 0
        max_results = 5000  # Generous limit per quarter

        while offset < max_results:
            url = f"{base_url}&from={offset}&size={page_size}"
            page = self._fetch_efts_page(url)
            if page is None or not page:
                break
            all_hits.extend(page)
            if len(page) < page_size:
                break
            offset += page_size

        if not all_hits:
            return None

        logger.info(
            "EFTS returned %d hits for CUSIP %s Q%d %d",
            len(all_hits), cusip, quarter, year,
        )

        # Download and parse each filing's XML
        holdings: list[InstitutionalHolding] = []

        for hit in all_hits:
            source = hit.get("_source", {})
            ciks = source.get("ciks", [])
            cik = ciks[0].lstrip("0") if ciks else ""
            accession = source.get("adsh", "")
            display_names = source.get("display_names", [])
            manager_name = display_names[0] if display_names else ""

            hit_id = hit.get("_id", "")
            xml_filename = ""
            if ":" in hit_id:
                _, xml_filename = hit_id.split(":", 1)

            if not accession or not cik:
                continue

            acc_nodash = accession.replace("-", "")
            xml_url = ""
            if xml_filename:
                xml_url = (
                    f"{_ARCHIVE_BASE}/{cik}/{acc_nodash}/{xml_filename}"
                )

            if not xml_url:
                # Try index.json fallback
                index_url = f"{_ARCHIVE_BASE}/{cik}/{acc_nodash}/"
                xml_url = self._find_infotable_url(index_url) or ""

            if not xml_url:
                continue

            xml_content = self._fetch_url_text(xml_url)
            if xml_content is None:
                continue

            parsed = self._parse_13f_xml(xml_content, cusip)
            for h in parsed:
                # Override manager name with display name from EFTS
                try:
                    holdings.append(InstitutionalHolding(
                        filing_date=quarter_end,
                        manager_name=manager_name or h.manager_name,
                        manager_cik=cik,
                        shares=h.shares,
                        value_usd=h.value_usd,
                        share_class=h.share_class,
                    ))
                except ValueError:
                    continue

        if not holdings:
            return None

        # Aggregate
        total_shares = sum(h.shares for h in holdings)
        sorted_h = sorted(holdings, key=lambda h: h.shares, reverse=True)
        top10 = sum(h.shares for h in sorted_h[:10])
        top10_conc = top10 / total_shares if total_shares > 0 else 0.0

        try:
            return OwnershipSnapshot(
                quarter_end=quarter_end,
                symbol=symbol,
                total_institutional_shares=total_shares,
                num_institutions=len(holdings),
                top_10_concentration=top10_conc,
                holdings=tuple(sorted_h),
            )
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _download_with_retry(self, url: str) -> bytes | None:
        """Download binary content with retry logic."""
        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limit()
                resp = self._session.get(url, timeout=120)
                if resp.status_code == 200:
                    return resp.content
                logger.warning(
                    "Download returned %d: %s (attempt %d/%d)",
                    resp.status_code, url, attempt + 1, _MAX_RETRIES,
                )
            except (requests.RequestException, OSError) as exc:
                logger.warning(
                    "Download failed: %s (attempt %d/%d)",
                    exc, attempt + 1, _MAX_RETRIES,
                )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2.0)
        return None

    def _fetch_url_text(self, url: str) -> str | None:
        """Download text content with retry logic."""
        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limit()
                resp = self._session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.text
                logger.debug(
                    "Fetch returned %d: %s (attempt %d/%d)",
                    resp.status_code, url, attempt + 1, _MAX_RETRIES,
                )
            except (requests.RequestException, OSError) as exc:
                logger.debug(
                    "Fetch failed: %s (attempt %d/%d)",
                    exc, attempt + 1, _MAX_RETRIES,
                )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(1.0)
        return None

    def _fetch_efts_page(self, url: str) -> list[dict] | None:
        """Fetch a single page from the EFTS search-index API."""
        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limit()
                resp = self._session.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("hits", {}).get("hits", [])
                logger.warning(
                    "EFTS returned %d (attempt %d/%d): %s",
                    resp.status_code, attempt + 1, _MAX_RETRIES, url,
                )
            except (
                requests.RequestException, json.JSONDecodeError, OSError,
            ) as exc:
                logger.warning(
                    "EFTS failed: %s (attempt %d/%d)",
                    exc, attempt + 1, _MAX_RETRIES,
                )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(1.0)
        return None

    def _find_infotable_url(self, index_url: str) -> str | None:
        """Given a filing index URL, find the infotable XML document."""
        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limit()
                json_url = index_url.rstrip("/") + "/index.json"
                resp = self._session.get(json_url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("directory", {}).get("item", [])
                    for item in items:
                        name = item.get("name", "").lower()
                        if "infotable" in name and name.endswith(".xml"):
                            return index_url.rstrip("/") + "/" + item["name"]
                    for item in items:
                        name = item.get("name", "").lower()
                        if name.endswith(".xml") and "primary" not in name:
                            return index_url.rstrip("/") + "/" + item["name"]
                    break
            except (
                requests.RequestException, json.JSONDecodeError, OSError,
            ):
                pass
            if attempt < _MAX_RETRIES - 1:
                time.sleep(1.0)
        return None

    # ------------------------------------------------------------------
    # XML parsing (for EFTS fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_13f_xml(
        xml_content: str,
        cusip: str,
    ) -> list[InstitutionalHolding]:
        """Parse a 13F XML and extract holdings matching *cusip*."""
        holdings: list[InstitutionalHolding] = []
        cusip_upper = cusip.upper().replace(" ", "")

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return []

        namespaces = [
            {"ns": _NS_13F},
            {"ns": _NS_13F_ALT},
            {},
        ]

        for ns_dict in namespaces:
            prefix = f"{{{ns_dict['ns']}}}" if "ns" in ns_dict else ""
            entries = root.findall(f".//{prefix}infoTable")
            if not entries:
                entries = root.findall(f".//{prefix}InfoTable")
            if not entries:
                continue

            for entry in entries:
                entry_cusip = _get_xml_text(entry, f"{prefix}cusip")
                if not entry_cusip:
                    entry_cusip = _get_xml_text(entry, f"{prefix}CUSIP")
                if not entry_cusip:
                    continue
                if entry_cusip.upper().replace(" ", "") != cusip_upper:
                    continue

                name = (
                    _get_xml_text(entry, f"{prefix}nameOfIssuer") or "Unknown"
                )
                shares_text = _get_xml_text(
                    entry, f".//{prefix}sshPrnamt",
                )
                value_text = _get_xml_text(entry, f"{prefix}value")
                share_class = (
                    _get_xml_text(entry, f".//{prefix}sshPrnamtType")
                    or "SH"
                )

                try:
                    shares = int(shares_text) if shares_text else 0
                except ValueError:
                    shares = 0
                try:
                    value_usd = int(value_text) if value_text else 0
                except ValueError:
                    value_usd = 0

                if shares <= 0:
                    continue

                try:
                    holdings.append(InstitutionalHolding(
                        filing_date="",
                        manager_name=name,
                        manager_cik="",
                        shares=shares,
                        value_usd=value_usd,
                        share_class=share_class,
                    ))
                except ValueError:
                    continue

            if holdings:
                break

        return holdings

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cache(self, symbol: str) -> list[OwnershipSnapshot] | None:
        """Load cached ownership snapshots for *symbol*."""
        cache_file = self._cache_dir / f"{symbol}_13f.json"
        # Also check legacy filename
        legacy_file = self._cache_dir / f"{symbol}_ownership.json"

        target = cache_file if cache_file.exists() else (
            legacy_file if legacy_file.exists() else None
        )
        if target is None:
            return None

        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            snapshots: list[OwnershipSnapshot] = []
            for snap in data:
                holdings = tuple(
                    InstitutionalHolding(
                        filing_date=h["filing_date"],
                        manager_name=h["manager_name"],
                        manager_cik=h["manager_cik"],
                        shares=h["shares"],
                        value_usd=h["value_usd"],
                        share_class=h["share_class"],
                    )
                    for h in snap.get("holdings", [])
                )
                snapshots.append(OwnershipSnapshot(
                    quarter_end=snap["quarter_end"],
                    symbol=snap["symbol"],
                    total_institutional_shares=snap[
                        "total_institutional_shares"
                    ],
                    num_institutions=snap["num_institutions"],
                    top_10_concentration=snap["top_10_concentration"],
                    holdings=holdings,
                ))
            snapshots.sort(key=lambda s: s.quarter_end)
            return snapshots
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to load ownership cache for %s: %s", symbol, exc,
            )
            return None

    def _save_cache(
        self,
        symbol: str,
        snapshots: list[OwnershipSnapshot],
    ) -> None:
        """Persist ownership snapshots to JSON cache."""
        cache_file = self._cache_dir / f"{symbol}_13f.json"
        data = [
            {
                "quarter_end": s.quarter_end,
                "symbol": s.symbol,
                "total_institutional_shares": s.total_institutional_shares,
                "num_institutions": s.num_institutions,
                "top_10_concentration": s.top_10_concentration,
                "holdings": [
                    {
                        "filing_date": h.filing_date,
                        "manager_name": h.manager_name,
                        "manager_cik": h.manager_cik,
                        "shares": h.shares,
                        "value_usd": h.value_usd,
                        "share_class": h.share_class,
                    }
                    for h in s.holdings
                ],
            }
            for s in snapshots
        ]
        try:
            cache_file.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save ownership cache for %s: %s", symbol, exc,
            )

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if needed to maintain the SEC 10 req/sec rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _get_xml_text(element: ET.Element, tag: str) -> str | None:
    """Extract text from a child XML element."""
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None


def _filing_date_to_quarter_end(filing_date: str) -> str:
    """Map a 13F filing date to its reporting quarter-end date.

    13F filings report holdings as of a quarter-end date, but are filed
    during a window after that date:

    * Jan-Feb filings  → Q4 of prior year (Dec 31)
    * Mar-May filings  → Q1 (Mar 31)
    * Jun-Aug filings  → Q2 (Jun 30)
    * Sep-Nov filings  → Q3 (Sep 30)
    * Dec filings      → Q4 (Dec 31)
    """
    if not filing_date or len(filing_date) < 7:
        return filing_date
    try:
        month = int(filing_date[5:7])
        year = int(filing_date[:4])
    except (ValueError, IndexError):
        return filing_date

    if month <= 2:
        return f"{year - 1}-12-31"
    if month <= 5:
        return f"{year}-03-31"
    if month <= 8:
        return f"{year}-06-30"
    if month <= 11:
        return f"{year}-09-30"
    return f"{year}-12-31"


def _prev_quarter(year: int, quarter: int) -> tuple[int, int]:
    """Return (year, quarter) for the previous calendar quarter."""
    if quarter == 1:
        return year - 1, 4
    return year, quarter - 1
