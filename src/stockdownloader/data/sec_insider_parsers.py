"""Parsing helpers for SEC insider ownership data.

Contains all parsing functions and constants extracted from
``sec_insider_client.py``.  These are pure-data transforms with no
HTTP or caching logic.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from pathlib import Path

from stockdownloader.data.sec_common import SplitAdjustment
from stockdownloader.data.sec_parser_utils import MONTH_ABBREVS, normalize_date, xml_text
from stockdownloader.model.regulatory_records import InsiderTransaction

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

#: Transaction codes that represent share acquisitions.
ACQUIRE_CODES = frozenset({"P", "A", "M", "J", "K", "I", "L", "G"})

#: Transaction codes that represent share dispositions.
DISPOSE_CODES = frozenset({"S", "D", "F", "W"})


# ------------------------------------------------------------------
# 13D/13G content extraction
# ------------------------------------------------------------------


def extract_13d_13g_data(
    content: str,
) -> tuple[int, float, int, int, int, int]:
    """Extract share counts from a 13D/13G filing's content.

    Returns (shares, percent, sole_vp, shared_vp, sole_dp, shared_dp).

    This is best-effort regex parsing -- 13D/13G filings lack a
    standard XML schema and vary widely in format.
    """
    shares = 0
    percent = 0.0
    sole_vp = 0
    shared_vp = 0
    sole_dp = 0
    shared_dp = 0

    # Strip HTML tags for text-based regex matching
    text = re.sub(r"<[^>]+>", " ", content)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Pattern: "aggregate amount beneficially owned" followed by a number
    agg_patterns = [
        r"aggregate\s+amount\s+beneficially\s+owned[^0-9]*?([0-9][0-9,]*)",
        r"amount\s+beneficially\s+owned[^0-9]*?([0-9][0-9,]*)",
        r"shares\s+beneficially\s+owned[^0-9]*?([0-9][0-9,]*)",
    ]
    for pattern in agg_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(",", "")
            try:
                shares = int(raw)
                break
            except ValueError:
                pass

    # Pattern: "percent of class" followed by a number (with optional % sign)
    pct_patterns = [
        r"percent\s+of\s+class[^%]*?(\d+\.\d+)\s*%",
        r"percent\s+of\s+class[^%]*?(\d+\.?\d*)\s*%",
        r"percentage\s+of\s+class[^%]*?(\d+\.\d+)\s*%",
    ]
    for pattern in pct_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                percent = float(match.group(1))
                break
            except ValueError:
                pass

    # Pattern: voting/dispositive power
    vp_patterns = [
        (r"sole\s+voting\s+power[^0-9]*?([0-9][0-9,]*)", "sole_vp"),
        (r"shared\s+voting\s+power[^0-9]*?([0-9][0-9,]*)", "shared_vp"),
        (r"sole\s+dispositive\s+power[^0-9]*?([0-9][0-9,]*)", "sole_dp"),
        (r"shared\s+dispositive\s+power[^0-9]*?([0-9][0-9,]*)", "shared_dp"),
    ]
    for pattern, field in vp_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(",", "")
            try:
                val = int(raw)
                if field == "sole_vp":
                    sole_vp = val
                elif field == "shared_vp":
                    shared_vp = val
                elif field == "sole_dp":
                    sole_dp = val
                elif field == "shared_dp":
                    shared_dp = val
            except ValueError:
                pass

    return shares, percent, sole_vp, shared_vp, sole_dp, shared_dp


# ------------------------------------------------------------------
# Split adjustment
# ------------------------------------------------------------------


def apply_split_to_transaction(
    txn: InsiderTransaction,
    splits: list[SplitAdjustment],
) -> InsiderTransaction:
    """Apply stock split adjustments to an insider transaction.

    Pre-split transactions have their share counts multiplied and
    prices divided by the split ratio.
    """
    shares = txn.shares
    price = txn.price_per_share
    shares_after = txn.shares_owned_after
    ref_date = txn.transaction_date or txn.filing_date

    for split in sorted(splits, key=lambda s: s.split_date, reverse=True):
        cutoff = split.split_date.isoformat()
        if ref_date < cutoff:
            ratio = int(split.split_ratio)
            shares = shares * ratio
            shares_after = shares_after * ratio
            if price > 0:
                price = price / ratio

    if (
        shares == txn.shares
        and price == txn.price_per_share
        and shares_after == txn.shares_owned_after
    ):
        return txn

    return InsiderTransaction(
        filing_date=txn.filing_date,
        transaction_date=txn.transaction_date,
        owner_name=txn.owner_name,
        owner_cik=txn.owner_cik,
        owner_title=txn.owner_title,
        is_director=txn.is_director,
        is_officer=txn.is_officer,
        is_ten_pct_owner=txn.is_ten_pct_owner,
        transaction_code=txn.transaction_code,
        shares=shares,
        price_per_share=price,
        shares_owned_after=shares_after,
        direct_or_indirect=txn.direct_or_indirect,
    )


# ------------------------------------------------------------------
# Bulk ZIP parsing
# ------------------------------------------------------------------


def parse_bulk_zip(
    zip_source: Path | io.BytesIO,
    symbol: str,
) -> list[InsiderTransaction]:
    """Parse a bulk insider transaction ZIP and extract data for *symbol*.

    The ZIP contains TSV files linked by ACCESSION_NUMBER:

    - SUBMISSION.tsv: filing metadata (issuer symbol, CIK, date)
    - REPORTING_OWNER.tsv: owner details (name, title, relationship)
    - NON_DERIVATIVE_TRANSACTION.tsv: stock trades
    - NON_DERIVATIVE_HOLDING.tsv: stock holdings at filing time
    """
    try:
        zf = zipfile.ZipFile(zip_source)
    except zipfile.BadZipFile:
        return []

    with zf:
        names = zf.namelist()
        submission_name = None
        owner_name_file = None
        ndt_name = None
        ndh_name = None

        for name in names:
            lower = name.lower()
            if not lower.endswith(".tsv"):
                continue
            if "submission" in lower:
                submission_name = name
            elif "reportingowner" in lower or "reporting_owner" in lower:
                owner_name_file = name
            elif (
                "nonderiv_trans" in lower
                or "non_derivative_transaction" in lower
            ):
                ndt_name = name
            elif (
                "nonderiv_holding" in lower
                or "non_derivative_holding" in lower
            ):
                ndh_name = name

        if not submission_name:
            logger.warning("No SUBMISSION.tsv in ZIP, files: %s", names)
            return []

        # Step 1: Find accession numbers for this symbol
        symbol_accessions: dict[str, str] = {}  # accession -> filing_date
        try:
            sub_data = zf.read(submission_name).decode(
                "utf-8", errors="replace",
            )
            sub_reader = csv.DictReader(
                io.StringIO(sub_data), delimiter="\t",
            )
            for row in sub_reader:
                issuer_sym = (
                    row.get("ISSUERTRADINGSYMBOL", "").strip().upper()
                )
                if issuer_sym != symbol:
                    continue
                acc = row.get("ACCESSION_NUMBER", "").strip()
                fdate = row.get("FILING_DATE", "").strip()
                if acc:
                    symbol_accessions[acc] = fdate
        except Exception as exc:
            logger.warning("Failed to parse SUBMISSION.tsv: %s", exc)
            return []

        if not symbol_accessions:
            return []

        # Step 2: Read owner details
        owners: dict[str, dict] = {}  # accession -> owner info
        if owner_name_file:
            try:
                own_data = zf.read(owner_name_file).decode(
                    "utf-8", errors="replace",
                )
                own_reader = csv.DictReader(
                    io.StringIO(own_data), delimiter="\t",
                )
                for row in own_reader:
                    acc = row.get("ACCESSION_NUMBER", "").strip()
                    if acc not in symbol_accessions:
                        continue
                    # Column names vary between SEC data set versions:
                    # Newer: RPTOWNER_RELATIONSHIP, RPTOWNER_TITLE
                    # Older: ISOFFICER, ISDIRECTOR, ISTENPERCENTOWNER, OFFICERTITLE
                    relationship = row.get(
                        "RPTOWNER_RELATIONSHIP", "",
                    ).strip().upper()
                    title = (
                        row.get("RPTOWNER_TITLE", "")
                        or row.get("OFFICERTITLE", "")
                    ).strip()
                    owners[acc] = {
                        "name": row.get("RPTOWNERNAME", "").strip(),
                        "cik": row.get("RPTOWNERCIK", "").strip(),
                        "title": title,
                        "is_director": (
                            "DIRECTOR" in relationship
                            or row.get("ISDIRECTOR", "").strip() == "1"
                        ),
                        "is_officer": (
                            "OFFICER" in relationship
                            or row.get("ISOFFICER", "").strip() == "1"
                        ),
                        "is_ten_pct": (
                            "10%" in relationship
                            or "TEN" in relationship
                            or row.get(
                                "ISTENPERCENTOWNER", "",
                            ).strip() == "1"
                        ),
                    }
            except Exception as exc:
                logger.warning(
                    "Failed to parse REPORTINGOWNER.tsv: %s", exc,
                )

        # Step 3: Read transactions
        transactions: list[InsiderTransaction] = []
        if ndt_name:
            try:
                ndt_data = zf.read(ndt_name).decode(
                    "utf-8", errors="replace",
                )
                ndt_reader = csv.DictReader(
                    io.StringIO(ndt_data), delimiter="\t",
                )
                for row in ndt_reader:
                    acc = row.get("ACCESSION_NUMBER", "").strip()
                    if acc not in symbol_accessions:
                        continue

                    code = row.get("TRANS_CODE", "").strip()
                    shares_text = row.get("TRANS_SHARES", "").strip()
                    price_text = row.get("TRANS_PRICEPERSHARE", "").strip()
                    acq_disp = (
                        row.get("TRANS_ACQUIRED_DISP_CD", "")
                        or row.get("TRANS_ACQUIRED_DISPOSED_CD", "")
                    ).strip()
                    shares_after_text = (
                        row.get("SHRS_OWND_FOLWNG_TRANS", "").strip()
                    )
                    direct_indirect = (
                        row.get("DIRECT_INDIRECT_OWNERSHIP", "").strip()
                    )
                    trans_date = (
                        row.get("TRANS_DATE", "").strip()
                    )

                    try:
                        shares = int(float(shares_text)) if shares_text else 0
                    except (ValueError, OverflowError):
                        shares = 0

                    try:
                        price = float(price_text) if price_text else 0.0
                    except (ValueError, OverflowError):
                        price = 0.0

                    try:
                        shares_after = (
                            int(float(shares_after_text))
                            if shares_after_text else 0
                        )
                    except (ValueError, OverflowError):
                        shares_after = 0

                    # Sign the shares: dispositions are negative
                    if acq_disp == "D" and shares > 0:
                        shares = -shares

                    owner = owners.get(acc, {})
                    raw_fdate = symbol_accessions.get(acc, "")
                    filing_date = normalize_date(raw_fdate)
                    trans_date = normalize_date(trans_date)

                    try:
                        transactions.append(InsiderTransaction(
                            filing_date=filing_date,
                            transaction_date=trans_date or filing_date,
                            owner_name=owner.get("name", "Unknown"),
                            owner_cik=owner.get("cik", ""),
                            owner_title=owner.get("title", ""),
                            is_director=owner.get("is_director", False),
                            is_officer=owner.get("is_officer", False),
                            is_ten_pct_owner=owner.get("is_ten_pct", False),
                            transaction_code=code,
                            shares=shares,
                            price_per_share=price,
                            shares_owned_after=shares_after,
                            direct_or_indirect=direct_indirect or "D",
                        ))
                    except ValueError:
                        continue
            except Exception as exc:
                logger.warning(
                    "Failed to parse NON_DERIVATIVE_TRANSACTION.tsv: %s",
                    exc,
                )

        # Step 4: If no transactions, check holdings
        if not transactions and ndh_name:
            try:
                ndh_data = zf.read(ndh_name).decode(
                    "utf-8", errors="replace",
                )
                ndh_reader = csv.DictReader(
                    io.StringIO(ndh_data), delimiter="\t",
                )
                for row in ndh_reader:
                    acc = row.get("ACCESSION_NUMBER", "").strip()
                    if acc not in symbol_accessions:
                        continue

                    shares_text = (
                        row.get("SHRS_OWND_FOLWNG_TRANS", "").strip()
                    )
                    direct_indirect = (
                        row.get("DIRECT_INDIRECT_OWNERSHIP", "").strip()
                    )

                    try:
                        shares_after = (
                            int(float(shares_text))
                            if shares_text else 0
                        )
                    except (ValueError, OverflowError):
                        shares_after = 0

                    if shares_after <= 0:
                        continue

                    owner = owners.get(acc, {})
                    raw_fdate = symbol_accessions.get(acc, "")
                    filing_date = normalize_date(raw_fdate)

                    try:
                        transactions.append(InsiderTransaction(
                            filing_date=filing_date,
                            transaction_date=filing_date,
                            owner_name=owner.get("name", "Unknown"),
                            owner_cik=owner.get("cik", ""),
                            owner_title=owner.get("title", ""),
                            is_director=owner.get("is_director", False),
                            is_officer=owner.get("is_officer", False),
                            is_ten_pct_owner=owner.get("is_ten_pct", False),
                            transaction_code="H",  # holding, not trade
                            shares=0,
                            price_per_share=0.0,
                            shares_owned_after=shares_after,
                            direct_or_indirect=direct_indirect or "D",
                        ))
                    except ValueError:
                        continue
            except Exception as exc:
                logger.warning(
                    "Failed to parse NON_DERIVATIVE_HOLDING.tsv: %s", exc,
                )

    logger.info(
        "Parsed %d insider transactions for %s",
        len(transactions), symbol,
    )
    return transactions


# ------------------------------------------------------------------
# Form 3/4/5 filing collection helper
# ------------------------------------------------------------------


def collect_form345(
    filing_data: dict,
    symbol: str,
    q_start: str,
    q_end: str,
    out: list[tuple[str, str, str]],
) -> None:
    """Collect Form 3/4/5 filings within a date range."""
    dates = filing_data.get("filingDate", [])
    forms = filing_data.get("form", [])
    accessions = filing_data.get("accessionNumber", [])

    for d, f, a in zip(dates, forms, accessions):
        if f not in ("3", "4", "5", "4/A", "5/A"):
            continue
        if d < q_start or d >= q_end:
            continue
        out.append((d, f, a))


# ------------------------------------------------------------------
# Form 3/4/5 XML and HTML parsing
# ------------------------------------------------------------------


def parse_form345_xml(
    content: str,
    symbol: str,
    filing_date: str,
    form_type: str,
) -> list[InsiderTransaction]:
    """Parse a Form 3/4/5 XML filing.

    *content* is the raw XML text (already fetched by the caller).
    """
    if not content:
        return []

    transactions: list[InsiderTransaction] = []

    # Extract owner info from XML
    owner_name = xml_text(content, "rptOwnerName") or "Unknown"
    owner_cik = xml_text(content, "rptOwnerCik") or ""
    title = xml_text(content, "officerTitle") or ""
    is_director = xml_text(content, "isDirector") == "1"
    is_officer = xml_text(content, "isOfficer") == "1"
    is_ten_pct = xml_text(content, "isTenPercentOwner") == "1"

    # Parse non-derivative transactions
    for txn_block in re.finditer(
        r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>",
        content, re.DOTALL | re.IGNORECASE,
    ):
        block = txn_block.group(1)
        trans_date = xml_text(block, "transactionDate") or ""
        # Handle nested value tags
        td_match = re.search(
            r"<transactionDate>.*?<value>(.*?)</value>",
            block, re.DOTALL | re.IGNORECASE,
        )
        if td_match:
            trans_date = td_match.group(1).strip()

        code = xml_text(block, "transactionCode") or ""
        shares_text = xml_text(block, "transactionShares")
        # Shares might be in a <value> sub-tag
        shares_match = re.search(
            r"<transactionShares>.*?<value>(.*?)</value>",
            block, re.DOTALL | re.IGNORECASE,
        )
        if shares_match:
            shares_text = shares_match.group(1).strip()

        price_text = ""
        price_match = re.search(
            r"<transactionPricePerShare>.*?<value>(.*?)</value>",
            block, re.DOTALL | re.IGNORECASE,
        )
        if price_match:
            price_text = price_match.group(1).strip()

        acq_disp = xml_text(block, "transactionAcquiredDisposedCode")
        acq_match = re.search(
            r"<transactionAcquiredDisposedCode>.*?<value>(.*?)</value>",
            block, re.DOTALL | re.IGNORECASE,
        )
        if acq_match:
            acq_disp = acq_match.group(1).strip()

        shares_after_text = ""
        after_match = re.search(
            r"<sharesOwnedFollowingTransaction>.*?<value>(.*?)</value>",
            block, re.DOTALL | re.IGNORECASE,
        )
        if after_match:
            shares_after_text = after_match.group(1).strip()

        direct_indirect = xml_text(
            block, "directOrIndirectOwnership",
        ) or ""
        di_match = re.search(
            r"<directOrIndirectOwnership>.*?<value>(.*?)</value>",
            block, re.DOTALL | re.IGNORECASE,
        )
        if di_match:
            direct_indirect = di_match.group(1).strip()

        try:
            shares = int(float(shares_text)) if shares_text else 0
        except (ValueError, TypeError):
            shares = 0
        try:
            price = float(price_text) if price_text else 0.0
        except (ValueError, TypeError):
            price = 0.0
        try:
            shares_after = int(float(shares_after_text)) if shares_after_text else 0
        except (ValueError, TypeError):
            shares_after = 0

        if acq_disp and acq_disp.upper() == "D" and shares > 0:
            shares = -shares

        try:
            transactions.append(InsiderTransaction(
                filing_date=filing_date,
                transaction_date=trans_date or filing_date,
                owner_name=owner_name,
                owner_cik=owner_cik,
                owner_title=title,
                is_director=is_director,
                is_officer=is_officer,
                is_ten_pct_owner=is_ten_pct,
                transaction_code=code or ("H" if form_type == "3" else ""),
                shares=shares,
                price_per_share=price,
                shares_owned_after=shares_after,
                direct_or_indirect=direct_indirect or "D",
            ))
        except ValueError:
            continue

    # Parse non-derivative holdings (Form 3 initial declarations)
    if not transactions:
        for hold_block in re.finditer(
            r"<nonDerivativeHolding>(.*?)</nonDerivativeHolding>",
            content, re.DOTALL | re.IGNORECASE,
        ):
            block = hold_block.group(1)
            shares_after_text = ""
            after_match = re.search(
                r"<sharesOwnedFollowingTransaction>.*?<value>(.*?)</value>",
                block, re.DOTALL | re.IGNORECASE,
            )
            if after_match:
                shares_after_text = after_match.group(1).strip()

            direct_indirect = ""
            di_match = re.search(
                r"<directOrIndirectOwnership>.*?<value>(.*?)</value>",
                block, re.DOTALL | re.IGNORECASE,
            )
            if di_match:
                direct_indirect = di_match.group(1).strip()

            try:
                shares_after = int(float(shares_after_text)) if shares_after_text else 0
            except (ValueError, TypeError):
                shares_after = 0

            if shares_after <= 0:
                continue

            try:
                transactions.append(InsiderTransaction(
                    filing_date=filing_date,
                    transaction_date=filing_date,
                    owner_name=owner_name,
                    owner_cik=owner_cik,
                    owner_title=title,
                    is_director=is_director,
                    is_officer=is_officer,
                    is_ten_pct_owner=is_ten_pct,
                    transaction_code="H",
                    shares=0,
                    price_per_share=0.0,
                    shares_owned_after=shares_after,
                    direct_or_indirect=direct_indirect or "D",
                ))
            except ValueError:
                continue

    return transactions


def parse_form345_html(
    content: str,
    symbol: str,
    filing_date: str,
    form_type: str,
) -> list[InsiderTransaction]:
    """Fallback HTML parsing for Form 3/4/5 filings.

    Extracts owner name and share holdings from HTML tables.
    Much less reliable than XML but covers very old filings.

    *content* is the raw HTML text (already fetched by the caller).
    """
    if not content:
        return []

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", content)
    text = re.sub(r"\s+", " ", text)

    # Try to find owner name
    owner_match = re.search(
        r"(?i)name\s+of\s+reporting\s+person[^A-Z]*([A-Z][A-Za-z\s,.'()-]+)",
        text,
    )
    owner_name = owner_match.group(1).strip() if owner_match else "Unknown"

    # Try to find shares held
    shares_match = re.search(
        r"(?i)amount\s+of\s+securities\s+beneficially\s+owned[^0-9]*([0-9][0-9,]*)",
        text,
    )
    if not shares_match:
        shares_match = re.search(
            r"(?i)shares\s+owned\s+following[^0-9]*([0-9][0-9,]*)",
            text,
        )

    if shares_match:
        try:
            shares_after = int(shares_match.group(1).replace(",", ""))
            return [InsiderTransaction(
                filing_date=filing_date,
                transaction_date=filing_date,
                owner_name=owner_name,
                owner_cik="",
                owner_title="",
                is_director=False,
                is_officer=False,
                is_ten_pct_owner=False,
                transaction_code="H",
                shares=0,
                price_per_share=0.0,
                shares_owned_after=shares_after,
                direct_or_indirect="D",
            )]
        except (ValueError, TypeError):
            pass

    return []
