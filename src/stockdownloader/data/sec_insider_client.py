"""Downloads and parses SEC insider ownership data.

**Form 3/4/5 insider transactions** (primary source):

Uses SEC bulk insider transaction data sets published quarterly as
ZIP files containing TSV tables.  Each ZIP includes:

* ``SUBMISSION.tsv`` — filing metadata (accession, issuer CIK, symbol)
* ``REPORTING_OWNER.tsv`` — owner info (name, CIK, title, relationship)
* ``NON_DERIVATIVE_TRANSACTION.tsv`` — common stock trades
* ``NON_DERIVATIVE_HOLDING.tsv`` — post-transaction holdings

Filter ``SUBMISSION.tsv`` by ``ISSUERTRADINGSYMBOL`` and join the
other tables by ``ACCESSION_NUMBER`` to reconstruct full insider
activity for a specific company.

**Schedule 13D/13G beneficial ownership** (secondary source):

Uses the EDGAR full-text search API (``efts.sec.gov``) to find
13D/13G filings containing the company's CUSIP, then downloads
and parses each filing's cover page for share counts.

Rate-limited to 10 req/s (SEC fair-use policy).  Downloaded zip files
are cached locally.

Usage::

    client = SecInsiderClient()
    txns = client.fetch_insider_transactions("GME")
    owners = client.fetch_beneficial_owners("GME")
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

import requests

from stockdownloader.data.base_client import BaseDataClient
from stockdownloader.data import sec_common
from stockdownloader.data import sec_insider_parsers as parsers
from stockdownloader.data.sec_common import SplitAdjustment, splits_for_symbol
from stockdownloader.model.regulatory_records import (
    BeneficialOwner,
    InsiderOwnershipSnapshot,
    InsiderTransaction,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
# SEC enforces 10 req/sec.  110 ms gap gives comfortable margin.
_RATE_LIMIT_DELAY = 0.11

# SEC bulk insider transaction data sets (quarterly ZIPs)
_BULK_INSIDER_BASE = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets"
)

# EDGAR full-text search index (for 13D/13G lookup)
_EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# Direct archive access for filing documents
_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

# GME default CUSIP (used only when no CUSIP is provided)
_GME_CUSIP = "36467W109"

class SecInsiderClient(BaseDataClient):
    """Downloads and parses SEC insider ownership data.

    Uses SEC bulk insider transaction data sets (TSV) as the primary
    source for Form 3/4/5 data, and EFTS full-text search for
    Schedule 13D/13G beneficial ownership filings.
    """

    def __init__(
        self,
        user_agent: str = "StockDownloader admin@example.com",
        data_dir: str = "data",
    ) -> None:
        super().__init__(
            rate_limit_delay=_RATE_LIMIT_DELAY,
            max_retries=_MAX_RETRIES,
            data_dir=data_dir,
            default_headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )
        # Bulk ZIP files are multi-ticker, keep under cache/
        self._bulk_dir = self._data_dir / "cache" / "bulk_insider"
        self._bulk_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_insider_transactions(
        self,
        symbol: str,
        cusip: str | None = None,
        num_quarters: int = 80,
        *,
        force_refresh: bool = False,
        extra_splits: list[SplitAdjustment] | None = None,
    ) -> list[InsiderTransaction]:
        """Fetch Form 3/4/5 insider transactions for *symbol*.

        Downloads SEC bulk insider transaction data sets for each
        quarter, filters by issuer trading symbol, and returns a
        list of :class:`InsiderTransaction` sorted by transaction
        date ascending.
        """
        symbol_upper = symbol.upper()

        # Try merged cache first
        if not force_refresh:
            cached = self._load_transaction_cache(symbol_upper)
            if cached is not None:
                logger.info(
                    "Using cached insider transactions for %s (%d records)",
                    symbol_upper, len(cached),
                )
                return cached

        # Determine quarter range
        today = date.today()
        current_quarter = (today.month - 1) // 3 + 1
        current_year = today.year

        # IPO-aware floor — skip quarters before the company existed.
        # SEC bulk data starts Q1 2006; for earlier filings we use
        # the EDGAR submissions API to fetch individual Forms 3/4/5.
        from stockdownloader.model.symbol_info import get_symbol_info
        _info = get_symbol_info(symbol_upper)

        # Default: earliest data is Q1 2003 (EDGAR individual filings)
        min_year, min_quarter = 2003, 1
        if _info is not None:
            ipo_y = _info.ipo_date.year
            ipo_q = (_info.ipo_date.month - 1) // 3 + 1
            if (ipo_y, ipo_q) > (min_year, min_quarter):
                min_year, min_quarter = ipo_y, ipo_q

        all_transactions: list[InsiderTransaction] = []

        # Work backwards from current quarter
        year, quarter = current_year, current_quarter
        quarters_fetched = 0

        while quarters_fetched < num_quarters:
            if year < min_year or (year == min_year and quarter < min_quarter):
                break

            # Skip future quarters
            if (year, quarter) > (current_year, current_quarter):
                year, quarter = sec_common.prev_quarter(year, quarter)
                continue

            # Check per-quarter cache
            cached_q = self._load_quarter_transactions(
                symbol_upper, year, quarter,
            )
            if cached_q is not None:
                all_transactions.extend(cached_q)
                quarters_fetched += 1
                year, quarter = sec_common.prev_quarter(year, quarter)
                continue

            logger.info(
                "Fetching insider transactions for %s Q%d %d",
                symbol_upper, quarter, year,
            )

            # SEC bulk data starts Q1 2006; earlier quarters use
            # individual EDGAR filing downloads instead.
            # Current quarter's bulk ZIP may not exist yet — fall back
            # to individual EDGAR filings in that case too.
            if (year, quarter) < (2006, 1):
                txns = self._fetch_individual_filings(
                    symbol_upper, year, quarter,
                )
            else:
                txns = self._fetch_from_bulk(symbol_upper, year, quarter)
                if txns is None and (year, quarter) >= (
                    current_year, current_quarter,
                ):
                    logger.info(
                        "Bulk ZIP unavailable for current quarter %dQ%d, "
                        "falling back to individual EDGAR filings",
                        year, quarter,
                    )
                    txns = self._fetch_individual_filings(
                        symbol_upper, year, quarter,
                    )

            if txns is not None:
                all_transactions.extend(txns)
                self._save_quarter_transactions(
                    symbol_upper, year, quarter, txns,
                )

            quarters_fetched += 1
            year, quarter = sec_common.prev_quarter(year, quarter)

        # Apply split adjustments
        symbol_splits = splits_for_symbol(symbol_upper, extra_splits)
        if symbol_splits:
            all_transactions = [
                parsers.apply_split_to_transaction(t, symbol_splits)
                for t in all_transactions
            ]

        all_transactions.sort(key=lambda t: t.transaction_date or t.filing_date)

        if all_transactions:
            self._save_transaction_cache(symbol_upper, all_transactions)

        return all_transactions

    def fetch_beneficial_owners(
        self,
        symbol: str,
        cusip: str | None = None,
        *,
        force_refresh: bool = False,
    ) -> list[BeneficialOwner]:
        """Fetch Schedule 13D/13G beneficial ownership filings for *symbol*.

        Uses EFTS to find filings containing the company's CUSIP,
        then downloads and parses each filing for share counts.
        """
        symbol_upper = symbol.upper()

        if not force_refresh:
            cached = self._load_beneficial_owners_cache(symbol_upper)
            if cached is not None:
                logger.info(
                    "Using cached beneficial owners for %s (%d records)",
                    symbol_upper, len(cached),
                )
                return cached

        # Auto-resolve CUSIP from registry
        if cusip is None:
            from stockdownloader.model.symbol_info import get_symbol_info
            _info = get_symbol_info(symbol_upper)
            if _info is not None and _info.cusip:
                cusip = _info.cusip
            elif symbol_upper == "GME":
                cusip = _GME_CUSIP
            else:
                logger.warning(
                    "No CUSIP for %s; cannot search for 13D/13G filings",
                    symbol_upper,
                )
                return []

        owners = self._fetch_13d_13g_from_efts(symbol_upper, cusip)

        if owners:
            self._save_beneficial_owners_cache(symbol_upper, owners)

        return owners

    def fetch_insider_snapshot(
        self,
        symbol: str,
        cusip: str | None = None,
    ) -> InsiderOwnershipSnapshot | None:
        """Build a combined insider ownership snapshot.

        Fetches both Form 3/4/5 transactions and 13D/13G beneficial
        ownership, then aggregates into a single snapshot showing
        current insider holdings.
        """
        symbol_upper = symbol.upper()
        transactions = self.fetch_insider_transactions(symbol_upper)
        owners = self.fetch_beneficial_owners(symbol_upper, cusip=cusip)

        if not transactions and not owners:
            return None

        # Compute latest insider shares from most recent holdings
        insider_shares: dict[str, int] = {}
        for t in transactions:
            if t.shares_owned_after > 0 and t.direct_or_indirect == "D":
                insider_shares[t.owner_name] = t.shares_owned_after

        # Latest beneficial owner shares (most recent filing per owner)
        owner_shares: dict[str, int] = {}
        for o in sorted(owners, key=lambda x: x.filing_date):
            owner_shares[o.owner_name] = o.shares_beneficially_owned

        today_str = date.today().isoformat()
        try:
            return InsiderOwnershipSnapshot(
                as_of_date=today_str,
                symbol=symbol_upper,
                total_insider_shares=sum(insider_shares.values()),
                total_beneficial_owner_shares=sum(owner_shares.values()),
                num_insiders=len(insider_shares),
                num_beneficial_owners=len(owner_shares),
                transactions=tuple(transactions),
                beneficial_owners=tuple(owners),
            )
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Bulk insider transactions (Form 3/4/5)
    # ------------------------------------------------------------------

    def _fetch_from_bulk(
        self,
        symbol: str,
        year: int,
        quarter: int,
    ) -> list[InsiderTransaction] | None:
        """Download and parse a SEC bulk insider transaction ZIP."""
        zip_path = self._bulk_dir / f"{year}Q{quarter}_form345.zip"

        if not zip_path.exists():
            url = f"{_BULK_INSIDER_BASE}/{year}q{quarter}_form345.zip"
            logger.info("Downloading bulk insider ZIP: %s", url)
            zip_data = sec_common.download_with_retry(
                self._session, url, rate_limit_fn=self._rate_limit,
            )
            if zip_data is None:
                return None
            try:
                zip_path.write_bytes(zip_data)
            except OSError as exc:
                logger.warning("Failed to cache ZIP: %s", exc)
                return parsers.parse_bulk_zip(
                    io.BytesIO(zip_data), symbol,
                )

        try:
            return parsers.parse_bulk_zip(zip_path, symbol)
        except (zipfile.BadZipFile, OSError) as exc:
            logger.warning(
                "Failed to parse bulk ZIP %s: %s", zip_path, exc,
            )
            zip_path.unlink(missing_ok=True)
            return None

    # ------------------------------------------------------------------
    # Individual EDGAR filing fetch (pre-2006 fallback)
    # ------------------------------------------------------------------

    def _fetch_individual_filings(
        self,
        symbol: str,
        year: int,
        quarter: int,
        issuer_cik: str | None = None,
    ) -> list[InsiderTransaction]:
        """Fetch Forms 3/4/5 from EDGAR submissions for a specific quarter.

        Used for pre-2006 data where SEC bulk data sets are not available.
        Queries the EDGAR submissions API for the issuer's CIK, finds
        Form 3/4/5 filings in the target quarter, then downloads and
        parses each filing's XML/HTML content.
        """
        if not issuer_cik:
            issuer_cik = self._resolve_cik(symbol)
            if not issuer_cik:
                logger.debug(
                    "No CIK for %s, cannot fetch individual filings", symbol,
                )
                return []

        cik_padded = issuer_cik.zfill(10)

        # Define the target quarter date range
        q_start_month = (quarter - 1) * 3 + 1
        q_end_month = quarter * 3
        q_start = f"{year}-{q_start_month:02d}-01"
        if q_end_month == 12:
            q_end = f"{year}-12-31"
        else:
            q_end = f"{year}-{q_end_month + 1:02d}-01"

        # Fetch the submissions JSON (includes recent + history files)
        all_form345: list[tuple[str, str, str]] = []  # (date, form, accession)

        try:
            self._rate_limit()
            url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
            resp = self._session.get(url, timeout=30)
            if resp.status_code != 200:
                return []
            data = resp.json()

            # Process recent filings
            recent = data.get("filings", {}).get("recent", {})
            parsers.collect_form345(
                recent, symbol, q_start, q_end, all_form345,
            )

            # Process history files
            for file_info in data.get("filings", {}).get("files", []):
                fname = file_info.get("name", "")
                if not fname:
                    continue
                self._rate_limit()
                hist_url = f"https://data.sec.gov/submissions/{fname}"
                hist_resp = self._session.get(hist_url, timeout=30)
                if hist_resp.status_code == 200:
                    parsers.collect_form345(
                        hist_resp.json(), symbol,
                        q_start, q_end, all_form345,
                    )

        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning(
                "Failed to fetch EDGAR submissions for %s: %s",
                symbol, exc,
            )
            return []

        if not all_form345:
            return []

        logger.info(
            "Found %d pre-bulk Form 3/4/5 filings for %s in %dQ%d",
            len(all_form345), symbol, year, quarter,
        )

        # Download and parse each filing
        transactions: list[InsiderTransaction] = []
        for fdate, form, accession in all_form345:
            txns = self._parse_individual_filing(
                symbol, cik_padded, accession, fdate, form,
            )
            transactions.extend(txns)

        return transactions

    def _parse_individual_filing(
        self,
        symbol: str,
        cik_padded: str,
        accession: str,
        filing_date: str,
        form_type: str,
    ) -> list[InsiderTransaction]:
        """Download and parse a single Form 3/4/5 filing.

        Tries the filing index first to find the primary XML document,
        then falls back to HTML parsing if XML is not available.
        """
        acc_nodash = accession.replace("-", "")
        index_url = (
            f"{_ARCHIVE_BASE}/{cik_padded.lstrip('0')}"
            f"/{acc_nodash}/index.json"
        )

        try:
            self._rate_limit()
            resp = self._session.get(index_url, timeout=30)
            if resp.status_code != 200:
                return []

            index_data = resp.json()
            items = index_data.get("directory", {}).get("item", [])

            # Find the primary XML document (form345 XML)
            xml_url = None
            html_url = None
            for item in items:
                name = item.get("name", "").lower()
                if name.endswith(".xml") and "primary_doc" not in name:
                    xml_url = (
                        f"{_ARCHIVE_BASE}/{cik_padded.lstrip('0')}"
                        f"/{acc_nodash}/{item['name']}"
                    )
                elif name.endswith(".htm") or name.endswith(".html"):
                    html_url = (
                        f"{_ARCHIVE_BASE}/{cik_padded.lstrip('0')}"
                        f"/{acc_nodash}/{item['name']}"
                    )

            # Try XML first (structured and reliable)
            if xml_url:
                content = sec_common.fetch_url_text(
                    self._session, xml_url,
                    rate_limit_fn=self._rate_limit,
                )
                if content:
                    txns = parsers.parse_form345_xml(
                        content, symbol, filing_date, form_type,
                    )
                    if txns:
                        return txns

            # Fallback to HTML
            if html_url:
                content = sec_common.fetch_url_text(
                    self._session, html_url,
                    rate_limit_fn=self._rate_limit,
                )
                if content:
                    return parsers.parse_form345_html(
                        content, symbol, filing_date, form_type,
                    )

        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.debug(
                "Failed to parse filing %s: %s", accession, exc,
            )

        return []

    # ------------------------------------------------------------------
    # Schedule 13D/13G via EFTS
    # ------------------------------------------------------------------

    def _fetch_13d_13g_from_efts(
        self,
        symbol: str,
        cusip: str,
    ) -> list[BeneficialOwner]:
        """Find and parse 13D/13G filings via EFTS full-text search."""
        forms = "SC 13D,SC 13D/A,SC 13G,SC 13G/A"
        base_url = (
            f"{_EFTS_SEARCH_URL}?q=%22{cusip}%22"
            f"&forms={forms.replace(' ', '%20').replace(',', '%2C')}"
        )

        all_hits: list[dict] = []
        page_size = 100
        offset = 0
        max_results = 5000

        while offset < max_results:
            url = f"{base_url}&from={offset}&size={page_size}"
            page = sec_common.fetch_efts_page(
                self._session, url, rate_limit_fn=self._rate_limit,
            )
            if page is None or not page:
                break
            all_hits.extend(page)
            if len(page) < page_size:
                break
            offset += page_size

        if not all_hits:
            logger.info("No 13D/13G filings found for CUSIP %s", cusip)
            return []

        logger.info(
            "EFTS returned %d hits for 13D/13G CUSIP %s",
            len(all_hits), cusip,
        )

        owners: list[BeneficialOwner] = []

        # The issuer CIK — exclude from owner identification
        from stockdownloader.model.symbol_info import get_symbol_info
        issuer_ciks: set[str] = set()
        _info = get_symbol_info(symbol)
        # GameStop's CIK is 0001326380
        _KNOWN_ISSUER_CIKS = {"0001326380": "GME"}
        for known_cik, known_sym in _KNOWN_ISSUER_CIKS.items():
            if known_sym == symbol:
                issuer_ciks.add(known_cik)
                issuer_ciks.add(known_cik.lstrip("0"))

        for hit in all_hits:
            source = hit.get("_source", {})
            ciks = source.get("ciks", [])
            accession = source.get("adsh", "")
            display_names = source.get("display_names", [])
            file_date = source.get("file_date", "")

            if not accession:
                continue

            # Determine filer (beneficial owner) vs issuer:
            # EFTS returns multiple CIKs and display_names — the issuer
            # is typically first, the filer second.  We want the filer.
            owner_name = ""
            filer_cik = ""
            for i, cik_val in enumerate(ciks):
                cleaned = cik_val.lstrip("0")
                if cik_val not in issuer_ciks and cleaned not in issuer_ciks:
                    filer_cik = cleaned
                    if i < len(display_names):
                        owner_name = display_names[i]
                    break

            # Fallback: if all CIKs are the issuer, use last entry
            if not owner_name and display_names:
                owner_name = display_names[-1] if len(display_names) > 1 else display_names[0]
            if not filer_cik and ciks:
                filer_cik = ciks[-1].lstrip("0") if len(ciks) > 1 else ciks[0].lstrip("0")

            # Clean owner name (EFTS often includes "(CIK ...)" suffix)
            if " (CIK " in owner_name:
                owner_name = owner_name.split(" (CIK ")[0].strip()
            # Strip trailing (GME) style tags
            if " (GME)" in owner_name:
                owner_name = owner_name.replace(" (GME)", "").strip()

            if not owner_name:
                continue

            # Determine form_type: EFTS sometimes returns None
            form_type = source.get("form_type") or ""
            if not form_type:
                # Infer from document filename in _id
                hit_id = hit.get("_id", "").lower()
                if "13d" in hit_id:
                    form_type = "SC 13D"
                    if "/a" in hit_id or "amendment" in hit_id:
                        form_type = "SC 13D/A"
                elif "13g" in hit_id:
                    form_type = "SC 13G"
                    if "/a" in hit_id or "amendment" in hit_id:
                        form_type = "SC 13G/A"
                else:
                    form_type = "SC 13D/G"  # generic fallback

            # Build filing URL.
            # EDGAR archives are accessible via either the issuer CIK
            # or the filer CIK (NOT the filing agent CIK embedded in
            # the accession number).  Use the filer CIK first.
            acc_nodash = accession.replace("-", "")
            acc_cik = filer_cik

            hit_id = hit.get("_id", "")
            doc_name = ""
            if ":" in hit_id:
                _, doc_name = hit_id.split(":", 1)

            if doc_name:
                filing_url = f"{_ARCHIVE_BASE}/{acc_cik}/{acc_nodash}/{doc_name}"
            else:
                filing_url = f"{_ARCHIVE_BASE}/{acc_cik}/{acc_nodash}/"

            # Download and parse the filing content for share data
            shares, pct, sole_vp, shared_vp, sole_dp, shared_dp = (
                self._parse_13d_13g_content(filing_url)
            )

            try:
                owners.append(BeneficialOwner(
                    filing_date=file_date,
                    owner_name=owner_name,
                    owner_cik=filer_cik,
                    form_type=form_type,
                    shares_beneficially_owned=shares,
                    percent_of_class=pct,
                    sole_voting_power=sole_vp,
                    shared_voting_power=shared_vp,
                    sole_dispositive_power=sole_dp,
                    shared_dispositive_power=shared_dp,
                    filing_url=filing_url,
                ))
            except ValueError as exc:
                logger.debug(
                    "Skipping 13D/13G hit %s: %s", accession, exc,
                )
                continue

        owners.sort(key=lambda o: o.filing_date)
        return owners

    def _parse_13d_13g_content(
        self,
        url: str,
    ) -> tuple[int, float, int, int, int, int]:
        """Download and parse a 13D/13G filing for share counts.

        Returns (shares_beneficially_owned, percent_of_class,
                 sole_voting_power, shared_voting_power,
                 sole_dispositive_power, shared_dispositive_power).

        Parsing is best-effort: 13D/13G filings are semi-structured
        HTML/text with no standard XML schema.  We use regex patterns
        to extract the cover page data fields that appear in most
        filings.
        """
        content = sec_common.fetch_url_text(
            self._session, url, rate_limit_fn=self._rate_limit,
        )
        if content is None:
            return 0, 0.0, 0, 0, 0, 0

        return parsers.extract_13d_13g_data(content)

    # ------------------------------------------------------------------
    # CIK resolution
    # ------------------------------------------------------------------

    def _resolve_cik(self, symbol: str) -> str | None:
        """Resolve an issuer CIK from a ticker symbol.

        Checks a hardcoded mapping first (fastest), then queries the
        SEC company tickers endpoint as a fallback.
        """
        _KNOWN_CIKS: dict[str, str] = {
            "GME": "1326380",
            "AAPL": "320193",
            "TSLA": "1318605",
            "MSFT": "789019",
            "AMZN": "1018724",
            "NVDA": "1045810",
            "GOOG": "1652044",
            "META": "1326801",
        }
        if symbol in _KNOWN_CIKS:
            return _KNOWN_CIKS[symbol]

        # Fallback: SEC company tickers JSON
        try:
            self._rate_limit()
            resp = self._session.get(
                "https://www.sec.gov/files/company_tickers.json",
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                for _key, entry in data.items():
                    if entry.get("ticker", "").upper() == symbol:
                        return str(entry["cik_str"])
        except (requests.RequestException, ValueError, KeyError):
            pass

        return None

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_transaction_cache(
        self, symbol: str,
    ) -> list[InsiderTransaction] | None:
        """Load cached insider transactions for *symbol*."""
        records = self._load_csv(symbol, "insider_transactions.csv", InsiderTransaction)
        if records is not None:
            return records
        # JSON fallback
        json_path = self._symbol_dir(symbol) / "insider_transactions.json"
        if not json_path.exists():
            return None
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            records = [InsiderTransaction(**r) for r in data]
            self._save_transaction_cache(symbol, records)
            logger.info("Migrated %s to CSV", json_path)
            return records
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to load insider cache for %s: %s", symbol, exc)
            return None

    def _save_transaction_cache(self, symbol, transactions):
        self._save_csv(symbol, "insider_transactions.csv", transactions)

    def _save_quarter_transactions(
        self,
        symbol: str,
        year: int,
        quarter: int,
        transactions: list[InsiderTransaction],
    ) -> None:
        """Save per-quarter insider transaction cache."""
        quarter_dir = self._progress_dir(symbol) / "insider"
        quarter_dir.mkdir(parents=True, exist_ok=True)
        cache_file = quarter_dir / f"{year}Q{quarter}.json"
        data = [asdict(t) for t in transactions]
        try:
            cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to save quarter insider cache %dQ%d: %s", year, quarter, exc)

    def _load_quarter_transactions(
        self,
        symbol: str,
        year: int,
        quarter: int,
    ) -> list[InsiderTransaction] | None:
        """Load per-quarter cached insider transactions."""
        cache_file = self._progress_dir(symbol) / "insider" / f"{year}Q{quarter}.json"
        # Legacy migration
        legacy = self._symbol_dir(symbol) / "insider" / f"{year}Q{quarter}.json"
        if not cache_file.exists() and legacy.exists():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            legacy.rename(cache_file)
            logger.info("Migrated %s → %s", legacy, cache_file)
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return [InsiderTransaction(**r) for r in data]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to load quarter insider cache %dQ%d: %s",
                year, quarter, exc,
            )
            return None

    def _load_beneficial_owners_cache(
        self, symbol: str,
    ) -> list[BeneficialOwner] | None:
        records = self._load_csv(symbol, "beneficial_owners.csv", BeneficialOwner)
        if records is not None:
            return records
        json_path = self._symbol_dir(symbol) / "beneficial_owners.json"
        if not json_path.exists():
            return None
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            records = [BeneficialOwner(**r) for r in data]
            self._save_beneficial_owners_cache(symbol, records)
            logger.info("Migrated %s to CSV", json_path)
            return records
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to load beneficial owners cache for %s: %s", symbol, exc)
            return None

    def _save_beneficial_owners_cache(self, symbol, owners):
        self._save_csv(symbol, "beneficial_owners.csv", owners)
