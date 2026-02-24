"""Parsing helpers for SEC 13F institutional ownership data.

Contains all parsing functions and constants extracted from
``sec_ownership_client.py``.  These are pure-data transforms with no
HTTP or caching logic (except ``find_infotable_url`` which takes an
explicit session).
"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Callable

import requests

from stockdownloader.data.sec_common import SplitAdjustment
from stockdownloader.data.sec_parser_utils import (
    GME_CUSIP,
    MONTH_NAMES,
    QUARTER_ENDS,
    filing_date_to_quarter_end,
    get_xml_text,
    is_leap_year,
    quarter_range,
)
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
# Bulk ZIP parsing (extracted to bulk_ownership_parser)
# ------------------------------------------------------------------

from stockdownloader.data.bulk_ownership_parser import parse_bulk_zip  # noqa: F401


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
