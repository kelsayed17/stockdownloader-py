"""Parsing helpers for SEC 13F institutional ownership data.

Contains all parsing functions and constants extracted from
``sec_ownership_client.py``.  These are pure-data transforms with no
HTTP or caching logic (except ``find_infotable_url`` which takes an
explicit session).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import requests

from stockdownloader.data.sec_ftd_client import SplitAdjustment, _KNOWN_SPLITS
from stockdownloader.model.regulatory_records import (
    InstitutionalHolding,
    OwnershipSnapshot,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

#: SEC bulk 13F data sets base URL.
BULK_13F_BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets"

#: 13F XML namespace variants.
NS_13F = "http://www.sec.gov/edgar/document/thirteenf/informationtable"
NS_13F_ALT = "http://www.sec.gov/edgar/thirteenf"

#: GameStop default CUSIP.
GME_CUSIP = "36467W109"

#: Quarter-end dates for each calendar quarter.
QUARTER_ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}

#: Month names for bulk file URL construction.
MONTH_NAMES = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}


# ------------------------------------------------------------------
# Quarter / date helpers
# ------------------------------------------------------------------


def quarter_range(year: int, quarter: int) -> tuple[date, date]:
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


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def bulk_zip_urls(year: int, quarter: int) -> list[str]:
    """Build candidate SEC bulk 13F data set ZIP URLs for a quarter.

    Returns multiple URL candidates because SEC changed their naming
    convention in 2024:

    - **2024+** (date-range naming): ``01jun2024-31aug2024_form13f.zip``
    - **Pre-2024** (quarter naming): ``2023q4_form13f.zip``

    The date-range names correspond to the 13F filing window (roughly
    45--105 days after quarter end).
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
            f"{BULK_13F_BASE}/01jun{year}-31aug{year}_form13f.zip"
        )
    elif quarter == 2:
        urls.append(
            f"{BULK_13F_BASE}/01sep{year}-30nov{year}_form13f.zip"
        )
    elif quarter == 3:
        # Feb end: 28 or 29 depending on leap year
        feb_end = 29 if is_leap_year(year + 1) else 28
        urls.append(
            f"{BULK_13F_BASE}/01dec{year}-{feb_end}feb{year + 1}"
            "_form13f.zip"
        )
    else:  # Q4
        urls.append(
            f"{BULK_13F_BASE}/01mar{year + 1}-31may{year + 1}"
            "_form13f.zip"
        )
        # Alternate Q4 naming seen for 2023:
        # 01jan{Y+1}-29feb{Y+1}
        feb_end = 29 if is_leap_year(year + 1) else 28
        urls.append(
            f"{BULK_13F_BASE}/01jan{year + 1}-{feb_end}feb{year + 1}"
            "_form13f.zip"
        )

    # Old format (pre-2024): simple quarter naming
    urls.append(f"{BULK_13F_BASE}/{year}q{quarter}_form13f.zip")

    return urls


# ------------------------------------------------------------------
# XML helpers
# ------------------------------------------------------------------


def get_xml_text(element: ET.Element, tag: str) -> str | None:
    """Extract text from a child XML element."""
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None


# ------------------------------------------------------------------
# Filing date mapping
# ------------------------------------------------------------------


def filing_date_to_quarter_end(filing_date: str) -> str:
    """Map a 13F filing date to its reporting quarter-end date.

    13F filings report holdings as of a quarter-end date, but are filed
    during a window after that date:

    * Jan-Feb filings  -> Q4 of prior year (Dec 31)
    * Mar-May filings  -> Q1 (Mar 31)
    * Jun-Aug filings  -> Q2 (Jun 30)
    * Sep-Nov filings  -> Q3 (Sep 30)
    * Dec filings      -> Q4 (Dec 31)
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


# ------------------------------------------------------------------
# Split adjustment
# ------------------------------------------------------------------


def apply_split_to_snapshot(
    snap: OwnershipSnapshot,
    split: SplitAdjustment,
) -> OwnershipSnapshot:
    """Apply a stock split adjustment to an ownership snapshot.

    If the snapshot's quarter-end date falls **before** the split date,
    share counts are multiplied and per-share values divided by the
    split ratio so that pre-split quarters are comparable to post-split.
    """
    split_date_str = split.split_date.isoformat()
    if snap.quarter_end >= split_date_str:
        return snap  # post-split -- no adjustment needed

    ratio = int(split.split_ratio)
    adjusted_holdings = tuple(
        InstitutionalHolding(
            filing_date=h.filing_date,
            manager_name=h.manager_name,
            manager_cik=h.manager_cik,
            shares=h.shares * ratio,
            value_usd=h.value_usd,  # dollar value unchanged
            share_class=h.share_class,
        )
        for h in snap.holdings
    )
    total_shares = sum(h.shares for h in adjusted_holdings)
    sorted_h = sorted(adjusted_holdings, key=lambda h: h.shares, reverse=True)
    top10 = sum(h.shares for h in sorted_h[:10])
    top10_conc = top10 / total_shares if total_shares > 0 else 0.0

    return OwnershipSnapshot(
        quarter_end=snap.quarter_end,
        symbol=snap.symbol,
        total_institutional_shares=total_shares,
        num_institutions=snap.num_institutions,
        top_10_concentration=top10_conc,
        holdings=tuple(sorted_h),
    )


# ------------------------------------------------------------------
# Bulk ZIP parsing
# ------------------------------------------------------------------


def parse_bulk_zip(
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
# 13F XML parsing (for EFTS fallback)
# ------------------------------------------------------------------


def parse_13f_xml(
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
        {"ns": NS_13F},
        {"ns": NS_13F_ALT},
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
            entry_cusip = get_xml_text(entry, f"{prefix}cusip")
            if not entry_cusip:
                entry_cusip = get_xml_text(entry, f"{prefix}CUSIP")
            if not entry_cusip:
                continue
            if entry_cusip.upper().replace(" ", "") != cusip_upper:
                continue

            name = (
                get_xml_text(entry, f"{prefix}nameOfIssuer") or "Unknown"
            )
            shares_text = get_xml_text(
                entry, f".//{prefix}sshPrnamt",
            )
            value_text = get_xml_text(entry, f"{prefix}value")
            share_class = (
                get_xml_text(entry, f".//{prefix}sshPrnamtType")
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
# Legacy text parsing (pre-2013 non-XML fallback)
# ------------------------------------------------------------------


def parse_13f_text(
    content: str,
    cusip: str,
) -> list[InstitutionalHolding]:
    """Parse a legacy (pre-2013) text infotable for holdings matching *cusip*.

    Pre-2013 13F filings use heterogeneous ASCII formats: tab-separated,
    comma-separated, or fixed-width.  This parser doesn't attempt full
    table parsing -- it scans for lines containing the CUSIP and extracts
    integer tokens as (value, shares).

    Parameters
    ----------
    content:
        Raw text content of the information table document.
    cusip:
        9-character CUSIP to search for (case-insensitive).

    Returns
    -------
    List of :class:`InstitutionalHolding` for rows matching *cusip*.
    Returns empty list if content looks like XML or CUSIP is absent.
    """
    # Skip XML content -- that's handled by parse_13f_xml.
    # Only reject actual XML declarations and HTML; SGML container
    # tags like <DOCUMENT> or <SEC-HEADER> are expected wrappers
    # around text tables in pre-2013 EDGAR filings.
    stripped = content.lstrip()
    lower_prefix = stripped[:10].lower()
    if lower_prefix.startswith("<?xml") or lower_prefix.startswith("<html"):
        return []

    cusip_upper = cusip.upper().replace(" ", "")
    holdings: list[InstitutionalHolding] = []

    for line in content.splitlines():
        # Case-insensitive CUSIP match
        if cusip_upper not in line.upper().replace(" ", ""):
            continue

        # Split line into fields.  Detect delimiter: tabs first,
        # then 2+ whitespace (fixed-width), then commas (CSV).
        if "\t" in line:
            fields = line.split("\t")
        elif re.search(r"\s{2,}", line):
            fields = re.split(r"\s{2,}", line)
        else:
            fields = line.split(",")

        # Skip any field containing the CUSIP so its digits
        # (e.g. "36467W109" -> "36467","109") don't pollute results.
        integers: list[int] = []
        for field in fields:
            if cusip_upper in field.upper().replace(" ", ""):
                continue
            # Extract numbers (strip commas from comma-formatted ints)
            for tok in re.findall(r"\d[\d,]*\d|\d+", field):
                cleaned = tok.replace(",", "")
                if cleaned.isdigit() and int(cleaned) > 0:
                    integers.append(int(cleaned))

        if len(integers) < 2:
            logger.debug(
                "Skipping line with < 2 numeric tokens: %s",
                line[:120],
            )
            continue

        # Convention: value is reported in $1000s (smaller number),
        # shares is the actual count (larger number).
        # Sort ascending and take the two largest: second-largest is
        # value ($1000s), largest is shares.
        #
        # Limitation: this heuristic inverts value/shares for expensive
        # securities with tiny share counts (e.g. 50 shares of BRK.A
        # worth $25,000k).  Acceptable for GME-focused scope where
        # shares always vastly outnumber value-in-$1000s.
        integers.sort()
        value_usd = integers[-2]  # second largest = value ($1000s)
        shares = integers[-1]     # largest = shares

        # Extract the issuer name from the first field.
        name = fields[0].strip() if fields else "Unknown"
        if not name:
            name = "Unknown"

        try:
            holdings.append(InstitutionalHolding(
                filing_date="",
                manager_name=name,
                manager_cik="",
                shares=shares,
                value_usd=value_usd,
                share_class="SH",
            ))
        except ValueError:
            continue

    return holdings


# ------------------------------------------------------------------
# Infotable URL finder (requires session)
# ------------------------------------------------------------------


def find_infotable_url(
    session: requests.Session,
    index_url: str,
    *,
    rate_limit_fn: Callable[[], None] | None = None,
    max_retries: int = 3,
) -> str | None:
    """Given a filing index URL, find the infotable document.

    Prefers XML (post-2013) but falls back to TXT (pre-2013).

    Parameters
    ----------
    session:
        An active ``requests.Session`` (with appropriate headers already set).
    index_url:
        The filing index directory URL on EDGAR.
    rate_limit_fn:
        Optional callable invoked before each HTTP request.
    max_retries:
        Number of attempts before giving up.
    """
    for attempt in range(max_retries):
        try:
            if rate_limit_fn is not None:
                rate_limit_fn()
            json_url = index_url.rstrip("/") + "/index.json"
            resp = session.get(json_url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("directory", {}).get("item", [])
                # Pass 1: prefer infotable XML
                for item in items:
                    name = item.get("name", "").lower()
                    if "infotable" in name and name.endswith(".xml"):
                        return index_url.rstrip("/") + "/" + item["name"]
                # Pass 2: any XML that isn't the primary doc
                for item in items:
                    name = item.get("name", "").lower()
                    if name.endswith(".xml") and "primary" not in name:
                        return index_url.rstrip("/") + "/" + item["name"]
                # Pass 3: infotable TXT (pre-2013 fallback)
                for item in items:
                    name = item.get("name", "").lower()
                    if "infotable" in name and name.endswith(".txt"):
                        return index_url.rstrip("/") + "/" + item["name"]
                break
        except (
            requests.RequestException, json.JSONDecodeError, OSError,
        ):
            pass
        if attempt < max_retries - 1:
            time.sleep(1.0)
    return None
