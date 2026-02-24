"""Unit tests for SEC insider-filing parsers — pure parsing logic, no client/cache."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from stockdownloader.data.sec_insider_client import SecInsiderClient
from stockdownloader.data.sec_insider_parsers import (
    extract_13d_13g_data,
    normalize_date,
    parse_bulk_zip,
    parse_form345_xml,
    xml_text,
)
from stockdownloader.data.sec_common import prev_quarter


# ------------------------------------------------------------------
# Sample TSV data for bulk ZIP
# ------------------------------------------------------------------

_SUBMISSION_TSV = (
    "ACCESSION_NUMBER\tFILING_DATE\tISSUERCIK\tISSUERTRADINGSYMBOL\n"
    "0001234-24-000001\t2024-03-15\t1326380\tGME\n"
    "0001234-24-000002\t2024-03-16\t1326380\tGME\n"
    "0001234-24-000003\t2024-03-17\t320193\tAAPL\n"
)

_REPORTING_OWNER_TSV = (
    "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\t"
    "ISOFFICER\tISDIRECTOR\tISTENPERCENTOWNER\tOFFICERTITLE\n"
    "0001234-24-000001\t9999999\tCohen Ryan\t0\t1\t1\t\n"
    "0001234-24-000002\t8888888\tSmith John\t1\t0\t0\tCFO\n"
    "0001234-24-000003\t7777777\tCook Tim\t1\t0\t0\tCEO\n"
)

_NON_DERIVATIVE_TRANSACTION_TSV = (
    "ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\t"
    "TRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISPOSED_CD\t"
    "SHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\n"
    "0001234-24-000001\t2024-03-15\tP\t100000\t15.50\tA\t9100000\tD\n"
    "0001234-24-000002\t2024-03-16\tS\t5000\t16.25\tD\t50000\tD\n"
    "0001234-24-000003\t2024-03-17\tP\t10000\t185.00\tA\t1000000\tD\n"
)

_NON_DERIVATIVE_HOLDING_TSV = (
    "ACCESSION_NUMBER\tSHRS_OWND_FOLWNG_TRANS\t"
    "DIRECT_INDIRECT_OWNERSHIP\n"
    "0001234-24-000001\t9100000\tD\n"
)


def _make_bulk_zip(
    submission: str = _SUBMISSION_TSV,
    owner: str = _REPORTING_OWNER_TSV,
    ndt: str = _NON_DERIVATIVE_TRANSACTION_TSV,
    ndh: str = _NON_DERIVATIVE_HOLDING_TSV,
) -> bytes:
    """Create an in-memory ZIP file mimicking SEC bulk data."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SUBMISSION.tsv", submission)
        zf.writestr("REPORTING_OWNER.tsv", owner)
        zf.writestr("NON_DERIVATIVE_TRANSACTION.tsv", ndt)
        zf.writestr("NON_DERIVATIVE_HOLDING.tsv", ndh)
    return buf.getvalue()


def _make_client(tmp_path: Path) -> SecInsiderClient:
    return SecInsiderClient(data_dir=str(tmp_path))


# ------------------------------------------------------------------
# Tests: Quarter helper
# ------------------------------------------------------------------


class TestPrevQuarter:
    def test_q4_wraps_to_q3(self) -> None:
        assert prev_quarter(2024, 4) == (2024, 3)

    def test_q1_wraps_to_prev_year_q4(self) -> None:
        assert prev_quarter(2024, 1) == (2023, 4)

    def test_q2_goes_to_q1(self) -> None:
        assert prev_quarter(2024, 2) == (2024, 1)


# ------------------------------------------------------------------
# Tests: Bulk ZIP parsing
# ------------------------------------------------------------------


class TestBulkZipParsing:
    def test_parses_gme_transactions(self) -> None:
        """Parse sample ZIP and extract GME transactions only."""
        zip_data = _make_bulk_zip()
        transactions = parse_bulk_zip(io.BytesIO(zip_data), "GME")

        assert len(transactions) == 2  # Two GME transactions, AAPL excluded
        # First transaction: Ryan Cohen purchase
        cohen = transactions[0]
        assert cohen.owner_name == "Cohen Ryan"
        assert cohen.transaction_code == "P"
        assert cohen.shares == 100000
        assert cohen.price_per_share == 15.50
        assert cohen.shares_owned_after == 9100000
        assert cohen.is_ten_pct_owner is True
        assert cohen.is_director is True
        assert cohen.is_officer is False

    def test_sale_has_negative_shares(self) -> None:
        """Dispositions should have negative share counts."""
        zip_data = _make_bulk_zip()
        transactions = parse_bulk_zip(io.BytesIO(zip_data), "GME")

        # Second transaction: John Smith sale
        sale = transactions[1]
        assert sale.transaction_code == "S"
        assert sale.shares == -5000  # negative for sale
        assert sale.owner_title == "CFO"

    def test_filters_by_symbol(self) -> None:
        """Only transactions for the requested symbol are returned."""
        zip_data = _make_bulk_zip()

        gme = parse_bulk_zip(io.BytesIO(zip_data), "GME")
        aapl = parse_bulk_zip(io.BytesIO(zip_data), "AAPL")

        assert len(gme) == 2
        assert len(aapl) == 1
        assert aapl[0].owner_name == "Cook Tim"

    def test_empty_zip_returns_empty(self) -> None:
        """ZIP with no data for symbol returns empty list."""
        zip_data = _make_bulk_zip()
        result = parse_bulk_zip(io.BytesIO(zip_data), "MSFT")
        assert result == []

    def test_bad_zip_returns_empty(self) -> None:
        """Corrupt ZIP data returns empty list."""
        result = parse_bulk_zip(io.BytesIO(b"not a zip"), "GME")
        assert result == []

    def test_missing_transaction_tsv_uses_holdings(self) -> None:
        """If no transaction TSV, fall back to holdings TSV."""
        # Build ZIP without NON_DERIVATIVE_TRANSACTION.tsv
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("SUBMISSION.tsv", _SUBMISSION_TSV)
            zf.writestr("REPORTING_OWNER.tsv", _REPORTING_OWNER_TSV)
            zf.writestr("NON_DERIVATIVE_HOLDING.tsv", _NON_DERIVATIVE_HOLDING_TSV)
        result = parse_bulk_zip(io.BytesIO(buf.getvalue()), "GME")
        assert len(result) >= 1
        assert result[0].transaction_code == "H"  # holding marker
        assert result[0].shares_owned_after == 9100000


# ------------------------------------------------------------------
# Tests: Abbreviated TSV file names (real SEC format)
# ------------------------------------------------------------------

# Real SEC bulk data uses abbreviated file names and column names
_SUBMISSION_TSV_REAL = (
    "ACCESSION_NUMBER\tFILING_DATE\tISSUERCIK\tISSUERTRADINGSYMBOL\n"
    "0001234-24-000001\t15-MAR-2024\t1326380\tGME\n"
)

_REPORTING_OWNER_TSV_REAL = (
    "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\t"
    "RPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
    "0001234-24-000001\t9999999\tCohen Ryan\tDIRECTOR,TEN PERCENT OWNER\tChairman\n"
)

_NONDERIV_TRANS_TSV_REAL = (
    "ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\t"
    "TRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\t"
    "SHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\n"
    "0001234-24-000001\t14-MAR-2024\tP\t100000\t15.50\tA\t9100000\tD\n"
)

_NONDERIV_HOLDING_TSV_REAL = (
    "ACCESSION_NUMBER\tSHRS_OWND_FOLWNG_TRANS\t"
    "DIRECT_INDIRECT_OWNERSHIP\n"
    "0001234-24-000001\t9100000\tD\n"
)


def _make_real_bulk_zip() -> bytes:
    """Create a ZIP mimicking real SEC abbreviated file names."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SUBMISSION.tsv", _SUBMISSION_TSV_REAL)
        zf.writestr("REPORTINGOWNER.tsv", _REPORTING_OWNER_TSV_REAL)
        zf.writestr("NONDERIV_TRANS.tsv", _NONDERIV_TRANS_TSV_REAL)
        zf.writestr("NONDERIV_HOLDING.tsv", _NONDERIV_HOLDING_TSV_REAL)
    return buf.getvalue()


class TestRealBulkZipFormat:
    """Tests using abbreviated file names and column names (real SEC format)."""

    def test_parses_abbreviated_file_names(self) -> None:
        """Parser finds TSV files with abbreviated names like NONDERIV_TRANS."""
        zip_data = _make_real_bulk_zip()
        transactions = parse_bulk_zip(io.BytesIO(zip_data), "GME")

        assert len(transactions) == 1
        assert transactions[0].owner_name == "Cohen Ryan"
        assert transactions[0].shares == 100000

    def test_rptowner_relationship_column(self) -> None:
        """RPTOWNER_RELATIONSHIP parses director/ten pct flags."""
        zip_data = _make_real_bulk_zip()
        transactions = parse_bulk_zip(io.BytesIO(zip_data), "GME")

        assert transactions[0].is_director is True
        assert transactions[0].is_ten_pct_owner is True
        assert transactions[0].owner_title == "Chairman"

    def test_trans_acquired_disp_cd_column(self) -> None:
        """TRANS_ACQUIRED_DISP_CD (abbreviated) correctly signs shares."""
        # Make a sale with abbreviated column name
        sale_tsv = (
            "ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\t"
            "TRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\t"
            "SHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\n"
            "0001234-24-000001\t14-MAR-2024\tS\t5000\t16.25\tD\t50000\tD\n"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("SUBMISSION.tsv", _SUBMISSION_TSV_REAL)
            zf.writestr("REPORTINGOWNER.tsv", _REPORTING_OWNER_TSV_REAL)
            zf.writestr("NONDERIV_TRANS.tsv", sale_tsv)
        transactions = parse_bulk_zip(io.BytesIO(buf.getvalue()), "GME")
        assert transactions[0].shares == -5000  # Disposition -> negative

    def test_date_normalization_dd_mon_yyyy(self) -> None:
        """DD-MON-YYYY dates are normalized to YYYY-MM-DD."""
        zip_data = _make_real_bulk_zip()
        transactions = parse_bulk_zip(io.BytesIO(zip_data), "GME")

        assert transactions[0].filing_date == "2024-03-15"
        assert transactions[0].transaction_date == "2024-03-14"


# ------------------------------------------------------------------
# Tests: Date normalization
# ------------------------------------------------------------------


class TestNormalizeDate:
    def test_iso_format_unchanged(self) -> None:
        assert normalize_date("2024-03-15") == "2024-03-15"

    def test_dd_mon_yyyy(self) -> None:
        assert normalize_date("15-MAR-2024") == "2024-03-15"
        assert normalize_date("02-JAN-2024") == "2024-01-02"

    def test_us_slash_format(self) -> None:
        assert normalize_date("03/15/2024") == "2024-03-15"

    def test_compact_yyyymmdd(self) -> None:
        assert normalize_date("20240315") == "2024-03-15"

    def test_empty_returns_empty(self) -> None:
        assert normalize_date("") == ""

    def test_whitespace_stripped(self) -> None:
        assert normalize_date("  2024-03-15  ") == "2024-03-15"


# ------------------------------------------------------------------
# Tests: 13D/13G content extraction
# ------------------------------------------------------------------


_SAMPLE_13D_HTML = """
<html>
<body>
<table>
<tr><td>AGGREGATE AMOUNT BENEFICIALLY OWNED BY EACH REPORTING PERSON</td>
<td>36,300,000</td></tr>
<tr><td>PERCENT OF CLASS REPRESENTED BY AMOUNT IN ROW (11)</td>
<td>11.9%</td></tr>
<tr><td>SOLE VOTING POWER</td><td>36,300,000</td></tr>
<tr><td>SHARED VOTING POWER</td><td>0</td></tr>
<tr><td>SOLE DISPOSITIVE POWER</td><td>36,300,000</td></tr>
<tr><td>SHARED DISPOSITIVE POWER</td><td>0</td></tr>
</table>
</body>
</html>
"""


class TestExtract13d13gData:
    def test_extracts_shares_and_percent(self) -> None:
        shares, pct, svp, shvp, sdp, shdp = extract_13d_13g_data(
            _SAMPLE_13D_HTML,
        )
        assert shares == 36300000
        assert pct == 11.9

    def test_extracts_voting_power(self) -> None:
        _, _, svp, shvp, _, _ = extract_13d_13g_data(_SAMPLE_13D_HTML)
        assert svp == 36300000
        assert shvp == 0

    def test_extracts_dispositive_power(self) -> None:
        _, _, _, _, sdp, shdp = extract_13d_13g_data(_SAMPLE_13D_HTML)
        assert sdp == 36300000
        assert shdp == 0

    def test_empty_content_returns_zeros(self) -> None:
        result = extract_13d_13g_data("")
        assert result == (0, 0.0, 0, 0, 0, 0)

    def test_plain_text_format(self) -> None:
        """13D filed as plain text (no HTML)."""
        text = """
        Item 5. Interest in Securities of the Issuer
        (a) Aggregate amount beneficially owned: 9,001,000 shares
        (b) Percent of class: 12.5%
        (c) Sole voting power: 9,001,000
        (d) Shared voting power: 0
        (e) Sole dispositive power: 9,001,000
        (f) Shared dispositive power: 0
        """
        shares, pct, svp, shvp, sdp, shdp = extract_13d_13g_data(text)
        assert shares == 9001000
        assert pct == 12.5
        assert svp == 9001000


# ------------------------------------------------------------------
# Tests: XML text extraction
# ------------------------------------------------------------------


class TestXmlText:
    def test_extracts_simple_tag(self) -> None:
        assert xml_text("<root><name>John</name></root>", "name") == "John"

    def test_returns_none_for_missing_tag(self) -> None:
        assert xml_text("<root><name>John</name></root>", "missing") is None

    def test_case_insensitive(self) -> None:
        assert xml_text("<Root><Name>John</Name></Root>", "name") == "John"


# ------------------------------------------------------------------
# Tests: Individual EDGAR filing parsing (pre-2006)
# ------------------------------------------------------------------

_SAMPLE_FORM3_XML = """<?xml version="1.0"?>
<ownershipDocument>
<issuer><issuerCik>1326380</issuerCik><issuerName>GameStop Corp</issuerName>
<issuerTradingSymbol>GME</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId>
<rptOwnerCik>9999999</rptOwnerCik><rptOwnerName>Test Owner</rptOwnerName>
</reportingOwnerId>
<reportingOwnerRelationship>
<isDirector>1</isDirector><isOfficer>0</isOfficer>
<isTenPercentOwner>1</isTenPercentOwner><officerTitle></officerTitle>
</reportingOwnerRelationship>
</reportingOwner>
<nonDerivativeTable>
<nonDerivativeHolding>
<securityTitle><value>Common Stock</value></securityTitle>
<ownershipNature>
<directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
</ownershipNature>
<postTransactionAmounts>
<sharesOwnedFollowingTransaction><value>500000</value></sharesOwnedFollowingTransaction>
</postTransactionAmounts>
</nonDerivativeHolding>
</nonDerivativeTable>
</ownershipDocument>"""

_SAMPLE_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
<issuer><issuerCik>1326380</issuerCik><issuerName>GameStop Corp</issuerName>
<issuerTradingSymbol>GME</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId>
<rptOwnerCik>8888888</rptOwnerCik><rptOwnerName>Buyer Jane</rptOwnerName>
</reportingOwnerId>
<reportingOwnerRelationship>
<isDirector>0</isDirector><isOfficer>1</isOfficer>
<isTenPercentOwner>0</isTenPercentOwner><officerTitle>CFO</officerTitle>
</reportingOwnerRelationship>
</reportingOwner>
<nonDerivativeTable>
<nonDerivativeTransaction>
<transactionDate><value>2005-11-15</value></transactionDate>
<transactionCoding><transactionCode>P</transactionCode></transactionCoding>
<transactionAmounts>
<transactionShares><value>10000</value></transactionShares>
<transactionPricePerShare><value>25.50</value></transactionPricePerShare>
<transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
</transactionAmounts>
<postTransactionAmounts>
<sharesOwnedFollowingTransaction><value>50000</value></sharesOwnedFollowingTransaction>
</postTransactionAmounts>
<ownershipNature>
<directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
</ownershipNature>
</nonDerivativeTransaction>
</nonDerivativeTable>
</ownershipDocument>"""


class TestForm345XmlParsing:
    """Tests for individual EDGAR Form 3/4/5 XML parsing."""

    def test_form3_holdings(self) -> None:
        """Form 3 XML with holdings (no transactions) parses correctly."""
        txns = parse_form345_xml(
            _SAMPLE_FORM3_XML, "GME", "2005-10-11", "3",
        )

        assert len(txns) == 1
        assert txns[0].owner_name == "Test Owner"
        assert txns[0].shares_owned_after == 500000
        assert txns[0].transaction_code == "H"
        assert txns[0].is_director is True
        assert txns[0].is_ten_pct_owner is True

    def test_form4_transaction(self) -> None:
        """Form 4 XML with a purchase transaction."""
        txns = parse_form345_xml(
            _SAMPLE_FORM4_XML, "GME", "2005-11-16", "4",
        )

        assert len(txns) == 1
        assert txns[0].owner_name == "Buyer Jane"
        assert txns[0].transaction_code == "P"
        assert txns[0].shares == 10000
        assert txns[0].price_per_share == 25.50
        assert txns[0].shares_owned_after == 50000
        assert txns[0].is_officer is True
        assert txns[0].owner_title == "CFO"

    def test_pre_2006_uses_individual_filings(self, tmp_path: Path) -> None:
        """Quarters before Q1 2006 use individual EDGAR filing fetch."""
        client = _make_client(tmp_path)

        with patch.object(
            client, "_fetch_individual_filings", return_value=[],
        ) as mock_individual, patch.object(
            client, "_fetch_from_bulk",
        ) as mock_bulk:
            client.fetch_insider_transactions(
                "GME", num_quarters=200, force_refresh=True,
            )

        # Should have called _fetch_individual_filings for pre-2006
        assert mock_individual.call_count > 0
        # Verify it was called for at least one pre-2006 quarter
        pre_2006_calls = [
            c for c in mock_individual.call_args_list
            if c[0][1] < 2006  # year argument
        ]
        assert len(pre_2006_calls) > 0
