"""Unit tests for SecInsiderClient — bulk ZIP parsing, EFTS 13D/13G, caching."""

from __future__ import annotations

import io
import json
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
from stockdownloader.model.regulatory_records import (
    BeneficialOwner,
    InsiderTransaction,
)


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
# Tests: Caching
# ------------------------------------------------------------------


class TestInsiderCaching:
    def test_save_and_load_transactions_csv(self, tmp_path: Path) -> None:
        """Transactions round-trip through CSV."""
        client = _make_client(tmp_path)
        txns = [
            InsiderTransaction(
                filing_date="2024-03-15",
                transaction_date="2024-03-15",
                owner_name="Cohen Ryan",
                owner_cik="9999999",
                owner_title="",
                is_director=True,
                is_officer=False,
                is_ten_pct_owner=True,
                transaction_code="P",
                shares=100000,
                price_per_share=15.50,
                shares_owned_after=9100000,
                direct_or_indirect="D",
            ),
        ]
        client._save_transaction_cache("GME", txns)

        # CSV file must exist (not JSON)
        assert (tmp_path / "GME" / "insider_transactions.csv").exists()
        assert not (tmp_path / "GME" / "insider_transactions.json").exists()

        loaded = client._load_transaction_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].owner_name == "Cohen Ryan"
        assert loaded[0].shares == 100000

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert client._load_transaction_cache("NOSYMBOL") is None

    def test_load_corrupt_csv_returns_none(self, tmp_path: Path) -> None:
        """Corrupt CSV falls through (no JSON fallback either)."""
        client = _make_client(tmp_path)
        sym_dir = tmp_path / "BAD"
        sym_dir.mkdir()
        # Write a CSV whose header doesn't match the dataclass fields
        # so _load_csv returns None (KeyError on missing column).
        (sym_dir / "insider_transactions.csv").write_text(
            "bad_col\nvalue\n", encoding="utf-8",
        )
        assert client._load_transaction_cache("BAD") is None

    def test_json_fallback_migrates_to_csv(self, tmp_path: Path) -> None:
        """Legacy JSON cache is loaded and migrated to CSV."""
        client = _make_client(tmp_path)
        sym_dir = tmp_path / "GME"
        sym_dir.mkdir()
        data = [
            {
                "filing_date": "2024-03-15",
                "transaction_date": "2024-03-15",
                "owner_name": "Legacy Owner",
                "owner_cik": "1111111",
                "owner_title": "",
                "is_director": False,
                "is_officer": False,
                "is_ten_pct_owner": False,
                "transaction_code": "P",
                "shares": 500,
                "price_per_share": 10.0,
                "shares_owned_after": 500,
                "direct_or_indirect": "D",
            },
        ]
        (sym_dir / "insider_transactions.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        loaded = client._load_transaction_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].owner_name == "Legacy Owner"
        # CSV should now exist after migration
        assert (sym_dir / "insider_transactions.csv").exists()

    def test_save_and_load_beneficial_owners_csv(self, tmp_path: Path) -> None:
        """Beneficial owners round-trip through CSV."""
        client = _make_client(tmp_path)
        owners = [
            BeneficialOwner(
                filing_date="2024-01-15",
                owner_name="RC Ventures LLC",
                owner_cik="1822844",
                form_type="SC 13D/A",
                shares_beneficially_owned=36300000,
                percent_of_class=11.9,
                sole_voting_power=36300000,
                shared_voting_power=0,
                sole_dispositive_power=36300000,
                shared_dispositive_power=0,
                filing_url="https://sec.gov/example",
            ),
        ]
        client._save_beneficial_owners_cache("GME", owners)

        # CSV file must exist (not JSON)
        assert (tmp_path / "GME" / "beneficial_owners.csv").exists()
        assert not (tmp_path / "GME" / "beneficial_owners.json").exists()

        loaded = client._load_beneficial_owners_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].owner_name == "RC Ventures LLC"
        assert loaded[0].shares_beneficially_owned == 36300000

    def test_beneficial_owners_json_fallback(self, tmp_path: Path) -> None:
        """Legacy JSON beneficial owners cache migrates to CSV."""
        client = _make_client(tmp_path)
        sym_dir = tmp_path / "GME"
        sym_dir.mkdir()
        data = [
            {
                "filing_date": "2024-01-15",
                "owner_name": "Legacy Fund",
                "owner_cik": "9999",
                "form_type": "SC 13D",
                "shares_beneficially_owned": 100000,
                "percent_of_class": 5.0,
                "sole_voting_power": 100000,
                "shared_voting_power": 0,
                "sole_dispositive_power": 100000,
                "shared_dispositive_power": 0,
                "filing_url": "https://sec.gov/legacy",
            },
        ]
        (sym_dir / "beneficial_owners.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        loaded = client._load_beneficial_owners_cache("GME")
        assert loaded is not None
        assert loaded[0].owner_name == "Legacy Fund"
        assert (sym_dir / "beneficial_owners.csv").exists()

    def test_quarter_cache_roundtrip(self, tmp_path: Path) -> None:
        """Quarter cache writes to .progress/insider/."""
        client = _make_client(tmp_path)
        txns = [
            InsiderTransaction(
                filing_date="2024-03-15",
                transaction_date="2024-03-15",
                owner_name="Test Owner",
                owner_cik="1111111",
                owner_title="CEO",
                is_director=False,
                is_officer=True,
                is_ten_pct_owner=False,
                transaction_code="P",
                shares=5000,
                price_per_share=20.0,
                shares_owned_after=10000,
                direct_or_indirect="D",
            ),
        ]
        client._save_quarter_transactions("GME", 2024, 1, txns)

        # File must be under .progress/insider/ (not insider/)
        progress_file = tmp_path / "GME" / ".progress" / "insider" / "2024Q1.json"
        legacy_file = tmp_path / "GME" / "insider" / "2024Q1.json"
        assert progress_file.exists()
        assert not legacy_file.exists()

        loaded = client._load_quarter_transactions("GME", 2024, 1)
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].owner_name == "Test Owner"

    def test_quarter_cache_legacy_migration(self, tmp_path: Path) -> None:
        """Legacy insider/{YYYY}Q{Q}.json migrates to .progress/insider/."""
        client = _make_client(tmp_path)
        legacy_dir = tmp_path / "GME" / "insider"
        legacy_dir.mkdir(parents=True)
        data = [
            {
                "filing_date": "2023-06-15",
                "transaction_date": "2023-06-15",
                "owner_name": "Legacy Quarter",
                "owner_cik": "2222222",
                "owner_title": "",
                "is_director": False,
                "is_officer": False,
                "is_ten_pct_owner": False,
                "transaction_code": "P",
                "shares": 1000,
                "price_per_share": 25.0,
                "shares_owned_after": 1000,
                "direct_or_indirect": "D",
            },
        ]
        (legacy_dir / "2023Q2.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        loaded = client._load_quarter_transactions("GME", 2023, 2)
        assert loaded is not None
        assert loaded[0].owner_name == "Legacy Quarter"
        # Legacy file should have been moved
        assert not (legacy_dir / "2023Q2.json").exists()
        assert (tmp_path / "GME" / ".progress" / "insider" / "2023Q2.json").exists()

    def test_quarter_cache_missing_returns_none(
        self, tmp_path: Path,
    ) -> None:
        client = _make_client(tmp_path)
        assert client._load_quarter_transactions("GME", 2020, 1) is None


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


# ------------------------------------------------------------------
# Tests: CUSIP auto-resolution
# ------------------------------------------------------------------


class TestCusipAutoResolution:
    def test_gme_uses_default_cusip(self, tmp_path: Path) -> None:
        """GME auto-resolves to known CUSIP from registry."""
        client = _make_client(tmp_path)
        with patch.object(
            client, "_fetch_13d_13g_from_efts", return_value=[],
        ) as mock_fetch:
            client.fetch_beneficial_owners("GME")

        # Should have been called with GME's CUSIP
        mock_fetch.assert_called_once()
        args = mock_fetch.call_args
        assert args[0][1] == "36467W109"

    def test_explicit_cusip_not_overridden(self, tmp_path: Path) -> None:
        """Explicit CUSIP takes precedence over registry."""
        client = _make_client(tmp_path)
        with patch.object(
            client, "_fetch_13d_13g_from_efts", return_value=[],
        ) as mock_fetch:
            client.fetch_beneficial_owners("GME", cusip="CUSTOM123")

        args = mock_fetch.call_args
        assert args[0][1] == "CUSTOM123"


# ------------------------------------------------------------------
# Tests: IPO-aware quarter floor
# ------------------------------------------------------------------


class TestIpoDateNarrowing:
    def test_tsla_skips_pre_ipo_quarters(self, tmp_path: Path) -> None:
        """TSLA (IPO 2010-06-29) should not fetch quarters before 2010."""
        client = _make_client(tmp_path)

        fetched_quarters: list[tuple[int, int]] = []

        def mock_fetch_from_bulk(symbol, year, quarter):
            fetched_quarters.append((year, quarter))
            return []

        with patch.object(
            client, "_fetch_from_bulk", side_effect=mock_fetch_from_bulk,
        ):
            client.fetch_insider_transactions(
                "TSLA", num_quarters=80, force_refresh=True,
            )

        # TSLA IPO is 2010-06-29 (Q2) — no quarter before 2010-Q2
        for y, q in fetched_quarters:
            assert (y, q) >= (2010, 2), (
                f"Fetched pre-IPO quarter {y}Q{q} for TSLA"
            )


# ------------------------------------------------------------------
# Tests: Model validation
# ------------------------------------------------------------------


class TestInsiderTransactionModel:
    def test_create_valid(self) -> None:
        txn = InsiderTransaction(
            filing_date="2024-01-15",
            transaction_date="2024-01-15",
            owner_name="Test",
            owner_cik="123",
            owner_title="CEO",
            is_director=False,
            is_officer=True,
            is_ten_pct_owner=False,
            transaction_code="P",
            shares=1000,
            price_per_share=10.0,
            shares_owned_after=5000,
            direct_or_indirect="D",
        )
        assert txn.shares == 1000
        assert txn.is_officer is True

    def test_empty_filing_date_raises(self) -> None:
        with pytest.raises(ValueError, match="filing_date"):
            InsiderTransaction(
                filing_date="",
                transaction_date="2024-01-15",
                owner_name="Test",
                owner_cik="123",
                owner_title="",
                is_director=False,
                is_officer=False,
                is_ten_pct_owner=False,
                transaction_code="P",
                shares=100,
                price_per_share=10.0,
                shares_owned_after=100,
                direct_or_indirect="D",
            )

    def test_empty_owner_name_raises(self) -> None:
        with pytest.raises(ValueError, match="owner_name"):
            InsiderTransaction(
                filing_date="2024-01-15",
                transaction_date="2024-01-15",
                owner_name="",
                owner_cik="123",
                owner_title="",
                is_director=False,
                is_officer=False,
                is_ten_pct_owner=False,
                transaction_code="P",
                shares=100,
                price_per_share=10.0,
                shares_owned_after=100,
                direct_or_indirect="D",
            )

    def test_frozen(self) -> None:
        txn = InsiderTransaction(
            filing_date="2024-01-15",
            transaction_date="2024-01-15",
            owner_name="Test",
            owner_cik="123",
            owner_title="",
            is_director=False,
            is_officer=False,
            is_ten_pct_owner=False,
            transaction_code="P",
            shares=100,
            price_per_share=10.0,
            shares_owned_after=100,
            direct_or_indirect="D",
        )
        with pytest.raises(AttributeError):
            txn.shares = 200  # type: ignore[misc]


class TestBeneficialOwnerModel:
    def test_create_valid(self) -> None:
        bo = BeneficialOwner(
            filing_date="2024-01-15",
            owner_name="RC Ventures LLC",
            owner_cik="1822844",
            form_type="SC 13D/A",
            shares_beneficially_owned=36300000,
            percent_of_class=11.9,
            sole_voting_power=36300000,
            shared_voting_power=0,
            sole_dispositive_power=36300000,
            shared_dispositive_power=0,
            filing_url="https://sec.gov/example",
        )
        assert bo.shares_beneficially_owned == 36300000

    def test_empty_owner_name_raises(self) -> None:
        with pytest.raises(ValueError, match="owner_name"):
            BeneficialOwner(
                filing_date="2024-01-15",
                owner_name="",
                owner_cik="123",
                form_type="SC 13D",
                shares_beneficially_owned=0,
                percent_of_class=0.0,
                sole_voting_power=0,
                shared_voting_power=0,
                sole_dispositive_power=0,
                shared_dispositive_power=0,
                filing_url="",
            )

    def test_empty_form_type_raises(self) -> None:
        with pytest.raises(ValueError, match="form_type"):
            BeneficialOwner(
                filing_date="2024-01-15",
                owner_name="Test",
                owner_cik="123",
                form_type="",
                shares_beneficially_owned=0,
                percent_of_class=0.0,
                sole_voting_power=0,
                shared_voting_power=0,
                sole_dispositive_power=0,
                shared_dispositive_power=0,
                filing_url="",
            )
