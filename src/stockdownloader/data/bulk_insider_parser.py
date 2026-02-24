"""SEC bulk insider ownership ZIP parser.

Parses the quarterly bulk ZIP files from SEC EDGAR that contain
insider transaction data (Forms 3, 4, 5) in a multi-file TSV format:
SUBMISSION.tsv, REPORTING_OWNER.tsv, NON_DERIVATIVE_TRANSACTION.tsv,
and NON_DERIVATIVE_HOLDING.tsv.

Extracted from :mod:`sec_insider_parsers` for modularity.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from pathlib import Path

from stockdownloader.data.sec_parser_utils import normalize_date
from stockdownloader.core.models.regulatory import InsiderTransaction

logger = logging.getLogger(__name__)


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
