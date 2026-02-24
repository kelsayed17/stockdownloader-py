"""SEC bulk 13F institutional ownership ZIP parser.

Parses the quarterly bulk ZIP files from SEC EDGAR that contain
13F institutional ownership data in a multi-file TSV format:
INFOTABLE.tsv, SUBMISSION.tsv, and COVERPAGE.tsv.

Extracted from :mod:`sec_ownership_parsers` for modularity.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from pathlib import Path

from stockdownloader.model.regulatory_records import (
    InstitutionalHolding,
    OwnershipSnapshot,
)

logger = logging.getLogger(__name__)


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
