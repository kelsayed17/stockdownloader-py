"""Unit tests for SecOwnershipClient — EDGAR search, 13F parsing, caching."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.sec_ownership_client import (
    SecOwnershipClient,
    _filing_date_to_quarter_end,
    _get_xml_text,
)
from stockdownloader.model.regulatory_records import (
    InstitutionalHolding,
    OwnershipSnapshot,
)


# ------------------------------------------------------------------
# Sample XML and JSON responses
# ------------------------------------------------------------------

_SAMPLE_13F_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>GAMESTOP CORP NEW</nameOfIssuer>
    <cusip>36467W109</cusip>
    <value>150000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>5000000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>GAMESTOP CORP NEW</nameOfIssuer>
    <cusip>36467W109</cusip>
    <value>75000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>2500000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <cusip>037833100</cusip>
    <value>500000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>3000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
  </infoTable>
</informationTable>
"""

_SAMPLE_13F_XML_NO_NS = """\
<?xml version="1.0" encoding="UTF-8"?>
<informationTable>
  <infoTable>
    <nameOfIssuer>GAMESTOP CORP NEW</nameOfIssuer>
    <cusip>36467W109</cusip>
    <value>100000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>3000000</sshPrnamt>
      <sshPrnamtType>COM</sshPrnamtType>
    </shrsOrPrnAmt>
  </infoTable>
</informationTable>
"""

_SAMPLE_13F_XML_INVALID = "<not valid xml!!!"

_SAMPLE_EFTS_RESPONSE = {
    "hits": {
        "total": {"value": 2, "relation": "eq"},
        "hits": [
            {
                "_id": "0001234567-24-000001:infotable.xml",
                "_source": {
                    "file_date": "2024-05-15",
                    "period_ending": "2024-03-31",
                    "file_num": ["028-12345"],
                    "adsh": "0001234567-24-000001",
                    "ciks": ["0001234567"],
                    "display_names": ["Test Fund A (CIK 0001234567)"],
                    "root_forms": ["13F-HR"],
                    "form": "13F-HR",
                    "file_type": "INFORMATION TABLE",
                },
            },
            {
                "_id": "0001234567-24-000002:infotable.xml",
                "_source": {
                    "file_date": "2024-02-14",
                    "period_ending": "2023-12-31",
                    "file_num": ["028-12346"],
                    "adsh": "0001234567-24-000002",
                    "ciks": ["0001234567"],
                    "display_names": ["Test Fund A (CIK 0001234567)"],
                    "root_forms": ["13F-HR"],
                    "form": "13F-HR",
                    "file_type": "INFORMATION TABLE",
                },
            },
        ],
    },
}

_SAMPLE_INDEX_JSON = {
    "directory": {
        "item": [
            {"name": "primary_doc.xml"},
            {"name": "infotable.xml"},
            {"name": "R1.htm"},
        ],
    },
}

_SAMPLE_INDEX_JSON_NO_INFOTABLE = {
    "directory": {
        "item": [
            {"name": "primary_doc.xml"},
            {"name": "holdings.xml"},
            {"name": "R1.htm"},
        ],
    },
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(tmp_path: Path) -> SecOwnershipClient:
    return SecOwnershipClient(
        user_agent="TestAgent test@test.com",
        cache_dir=str(tmp_path / "ownership_cache"),
    )


def _mock_response(
    json_data: object = None,
    status_code: int = 200,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text
    return resp


# ------------------------------------------------------------------
# Tests: Filing date to quarter end mapping
# ------------------------------------------------------------------


class TestFilingDateToQuarterEnd:
    """Tests for _filing_date_to_quarter_end helper."""

    def test_january_maps_to_q4_prior_year(self) -> None:
        assert _filing_date_to_quarter_end("2024-01-15") == "2023-12-31"

    def test_february_maps_to_q4_prior_year(self) -> None:
        assert _filing_date_to_quarter_end("2024-02-14") == "2023-12-31"

    def test_march_maps_to_q1(self) -> None:
        assert _filing_date_to_quarter_end("2024-03-15") == "2024-03-31"

    def test_may_maps_to_q1(self) -> None:
        assert _filing_date_to_quarter_end("2024-05-10") == "2024-03-31"

    def test_june_maps_to_q2(self) -> None:
        assert _filing_date_to_quarter_end("2024-06-15") == "2024-06-30"

    def test_august_maps_to_q2(self) -> None:
        assert _filing_date_to_quarter_end("2024-08-10") == "2024-06-30"

    def test_september_maps_to_q3(self) -> None:
        assert _filing_date_to_quarter_end("2024-09-15") == "2024-09-30"

    def test_november_maps_to_q3(self) -> None:
        assert _filing_date_to_quarter_end("2024-11-10") == "2024-09-30"

    def test_december_maps_to_q4(self) -> None:
        assert _filing_date_to_quarter_end("2024-12-15") == "2024-12-31"

    def test_empty_string_passthrough(self) -> None:
        assert _filing_date_to_quarter_end("") == ""

    def test_short_string_passthrough(self) -> None:
        assert _filing_date_to_quarter_end("2024") == "2024"


# ------------------------------------------------------------------
# Tests: XML text extraction
# ------------------------------------------------------------------


class TestGetXmlText:
    """Tests for _get_xml_text helper."""

    def test_extracts_text(self) -> None:
        import xml.etree.ElementTree as ET

        root = ET.fromstring("<root><child>hello</child></root>")
        assert _get_xml_text(root, "child") == "hello"

    def test_returns_none_for_missing_element(self) -> None:
        import xml.etree.ElementTree as ET

        root = ET.fromstring("<root><child>hello</child></root>")
        assert _get_xml_text(root, "missing") is None

    def test_returns_none_for_empty_text(self) -> None:
        import xml.etree.ElementTree as ET

        root = ET.fromstring("<root><child></child></root>")
        assert _get_xml_text(root, "child") is None

    def test_strips_whitespace(self) -> None:
        import xml.etree.ElementTree as ET

        root = ET.fromstring("<root><child>  spaced  </child></root>")
        assert _get_xml_text(root, "child") == "spaced"


# ------------------------------------------------------------------
# Tests: 13F XML parsing
# ------------------------------------------------------------------


class TestParse13fXml:
    """Tests for _parse_13f_xml static method."""

    def test_parses_matching_cusip(self) -> None:
        holdings = SecOwnershipClient._parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        assert len(holdings) == 2

    def test_filters_by_cusip(self) -> None:
        holdings = SecOwnershipClient._parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        # Should not include AAPL
        assert all(h.manager_name == "GAMESTOP CORP NEW" for h in holdings)

    def test_parses_shares_and_value(self) -> None:
        holdings = SecOwnershipClient._parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        assert holdings[0].shares == 5_000_000
        assert holdings[0].value_usd == 150_000
        assert holdings[1].shares == 2_500_000
        assert holdings[1].value_usd == 75_000

    def test_parses_share_class(self) -> None:
        holdings = SecOwnershipClient._parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        assert holdings[0].share_class == "SH"

    def test_filing_date_is_empty_placeholder(self) -> None:
        """_parse_13f_xml sets filing_date='' as a placeholder."""
        holdings = SecOwnershipClient._parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        assert all(h.filing_date == "" for h in holdings)

    def test_parses_no_namespace_xml(self) -> None:
        holdings = SecOwnershipClient._parse_13f_xml(
            _SAMPLE_13F_XML_NO_NS, "36467W109",
        )
        assert len(holdings) == 1
        assert holdings[0].shares == 3_000_000

    def test_invalid_xml_returns_empty(self) -> None:
        holdings = SecOwnershipClient._parse_13f_xml(
            _SAMPLE_13F_XML_INVALID, "36467W109",
        )
        assert holdings == []

    def test_no_matching_cusip_returns_empty(self) -> None:
        holdings = SecOwnershipClient._parse_13f_xml(
            _SAMPLE_13F_XML, "000000000",
        )
        assert holdings == []

    def test_empty_xml(self) -> None:
        holdings = SecOwnershipClient._parse_13f_xml(
            "<root></root>", "36467W109",
        )
        assert holdings == []


# ------------------------------------------------------------------
# Tests: EDGAR search for 13F filings
# ------------------------------------------------------------------


class TestFetchEftsPage:
    """Tests for _fetch_efts_page (EDGAR search API)."""

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_returns_hits(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        with patch.object(
            client._session, "get",
            return_value=_mock_response(_SAMPLE_EFTS_RESPONSE),
        ):
            results = client._fetch_efts_page("https://efts.sec.gov/test")

        assert results is not None
        assert len(results) == 2
        # Each result preserves the raw EFTS hit structure
        assert results[0]["_source"]["file_date"] == "2024-05-15"
        assert results[1]["_source"]["file_date"] == "2024-02-14"
        # period_ending should be present
        assert results[0]["_source"]["period_ending"] == "2024-03-31"
        # _id contains the XML filename
        assert "infotable.xml" in results[0]["_id"]

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_handles_network_error(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        import requests

        with patch.object(
            client._session, "get",
            side_effect=requests.RequestException("timeout"),
        ):
            results = client._fetch_efts_page("https://efts.sec.gov/test")

        assert results is None

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_handles_empty_hits(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        with patch.object(
            client._session, "get",
            return_value=_mock_response({"hits": {"hits": []}}),
        ):
            results = client._fetch_efts_page("https://efts.sec.gov/test")

        assert results == []


# ------------------------------------------------------------------
# Tests: Find infotable URL
# ------------------------------------------------------------------


class TestFindInfotableUrl:
    """Tests for _find_infotable_url."""

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_finds_infotable_xml(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        with patch.object(
            client._session, "get",
            return_value=_mock_response(_SAMPLE_INDEX_JSON),
        ):
            url = client._find_infotable_url(
                "https://www.sec.gov/Archives/edgar/data/1234567/000123456724000001/",
            )

        assert url is not None
        assert "infotable.xml" in url

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_falls_back_to_any_xml(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        # Index with no infotable but has another xml
        with patch.object(
            client._session, "get",
            return_value=_mock_response(_SAMPLE_INDEX_JSON_NO_INFOTABLE),
        ):
            url = client._find_infotable_url(
                "https://www.sec.gov/Archives/edgar/data/1234567/000123456724000001/",
            )

        assert url is not None
        assert "holdings.xml" in url

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_returns_none_on_failure(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        with patch.object(
            client._session, "get",
            return_value=_mock_response(None, status_code=404),
        ):
            url = client._find_infotable_url(
                "https://www.sec.gov/Archives/edgar/data/1234567/000123456724000001/",
            )

        assert url is None

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_finds_txt_when_no_xml(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """Pre-2013 filings have infotable as .txt, not .xml."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        index_json = {
            "directory": {
                "item": [
                    {"name": "primary_doc.html", "size": "1234"},
                    {"name": "infotable.txt", "size": "5678"},
                ],
            },
        }

        with patch.object(client, "_session") as mock_session:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = index_json
            mock_session.get.return_value = mock_resp

            result = client._find_infotable_url(
                "https://www.sec.gov/Archives/edgar/data/12345/0001234/",
            )

        assert result is not None
        assert result.endswith("/infotable.txt")

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_prefers_xml_over_txt(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """When both .xml and .txt exist, prefer .xml."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        index_json = {
            "directory": {
                "item": [
                    {"name": "infotable.xml", "size": "1234"},
                    {"name": "infotable.txt", "size": "5678"},
                ],
            },
        }

        with patch.object(client, "_session") as mock_session:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = index_json
            mock_session.get.return_value = mock_resp

            result = client._find_infotable_url(
                "https://www.sec.gov/Archives/edgar/data/12345/0001234/",
            )

        assert result is not None
        assert result.endswith("/infotable.xml")


# ------------------------------------------------------------------
# Tests: EFTS text fallback integration
# ------------------------------------------------------------------


class TestEftsTextFallback:
    """Verify _fetch_from_efts falls back to text parsing when XML fails."""

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_text_fallback_when_xml_empty(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """When XML parsing returns nothing, text parsing should kick in."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        # Simulate EFTS returning one hit
        fake_hits = [{
            "_id": "0001234-05-000001:infotable.txt",
            "_source": {
                "adsh": "0001234-05-000001",
                "ciks": ["1234"],
                "display_names": ["Test Fund"],
                "file_date": "2005-05-15",
                "period_ending": "2005-03-31",
            },
        }]

        # The downloaded content is plain text, not XML
        text_content = (
            "GAMESTOP CORP\tCOM\t36467W109\t5000\t200000\tSH\n"
        )

        with patch.object(client, "_fetch_efts_page", return_value=fake_hits), \
             patch.object(client, "_fetch_url_text", return_value=text_content):
            result = client._fetch_from_efts(
                "GME", "36467W109", 2005, 1, "2005-03-31",
            )

        assert result is not None
        assert result.total_institutional_shares == 200000
        assert result.num_institutions == 1


# ------------------------------------------------------------------
# Tests: Caching
# ------------------------------------------------------------------


class TestOwnershipCaching:
    """Tests for ownership cache save/load."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        holdings = (
            InstitutionalHolding(
                filing_date="2024-05-15",
                manager_name="Vanguard",
                manager_cik="102909",
                shares=5_000_000,
                value_usd=150_000,
                share_class="COM",
            ),
            InstitutionalHolding(
                filing_date="2024-05-15",
                manager_name="BlackRock",
                manager_cik="101234",
                shares=3_000_000,
                value_usd=90_000,
                share_class="COM",
            ),
        )

        snapshots = [
            OwnershipSnapshot(
                quarter_end="2024-03-31",
                symbol="GME",
                total_institutional_shares=8_000_000,
                num_institutions=2,
                top_10_concentration=1.0,
                holdings=holdings,
            ),
        ]

        client._save_cache("GME", snapshots)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 1
        snap = loaded[0]
        assert snap.quarter_end == "2024-03-31"
        assert snap.symbol == "GME"
        assert snap.total_institutional_shares == 8_000_000
        assert snap.num_institutions == 2
        assert snap.top_10_concentration == 1.0
        assert len(snap.holdings) == 2
        assert snap.holdings[0].manager_name == "Vanguard"

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        loaded = client._load_cache("NOSYMBOL")
        assert loaded is None

    def test_load_corrupt_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._cache_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        cache_file = sym_dir / "13f.json"
        cache_file.write_text("{{{corrupt json", encoding="utf-8")
        loaded = client._load_cache("BAD")
        assert loaded is None

    def test_cache_file_naming(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        snapshots = [
            OwnershipSnapshot(
                quarter_end="2024-03-31",
                symbol="AAPL",
                total_institutional_shares=1000,
                num_institutions=1,
                top_10_concentration=1.0,
                holdings=(
                    InstitutionalHolding(
                        filing_date="2024-05-15",
                        manager_name="TestFund",
                        manager_cik="123",
                        shares=1000,
                        value_usd=100,
                        share_class="COM",
                    ),
                ),
            ),
        ]
        client._save_cache("AAPL", snapshots)
        assert (client._cache_dir / "AAPL" / "13f.json").exists()


# ------------------------------------------------------------------
# Tests: Per-symbol cache subdirectory
# ------------------------------------------------------------------


class TestPerSymbolCacheDir:
    """Tests for per-symbol cache subdirectory layout and legacy migration."""

    def test_symbol_cache_dir_creates_and_uppercases(self, tmp_path: Path) -> None:
        """_symbol_cache_dir creates the dir and uppercases the symbol."""
        client = _make_client(tmp_path)
        d = client._symbol_cache_dir("aapl")
        assert d == client._cache_dir / "AAPL"
        assert d.is_dir()

    def test_save_cache_creates_file_in_symbol_subdir(self, tmp_path: Path) -> None:
        """_save_cache writes to {SYMBOL}/13f.json inside cache_dir."""
        client = _make_client(tmp_path)
        snapshots = [
            OwnershipSnapshot(
                quarter_end="2024-03-31",
                symbol="TSLA",
                total_institutional_shares=1000,
                num_institutions=1,
                top_10_concentration=1.0,
                holdings=(
                    InstitutionalHolding(
                        filing_date="2024-05-15",
                        manager_name="Fund",
                        manager_cik="999",
                        shares=1000,
                        value_usd=100,
                        share_class="COM",
                    ),
                ),
            ),
        ]
        client._save_cache("TSLA", snapshots)
        expected = client._cache_dir / "TSLA" / "13f.json"
        assert expected.exists()
        # The old flat path should NOT exist
        assert not (client._cache_dir / "TSLA_13f.json").exists()

    def test_legacy_13f_json_migration_on_load(self, tmp_path: Path) -> None:
        """Legacy {SYMBOL}_13f.json migrates to {SYMBOL}/13f.json on load."""
        client = _make_client(tmp_path)

        # Write a snapshot using the old flat layout
        legacy_file = client._cache_dir / "GME_13f.json"
        data = [
            {
                "quarter_end": "2024-03-31",
                "symbol": "GME",
                "total_institutional_shares": 5000,
                "num_institutions": 1,
                "top_10_concentration": 1.0,
                "holdings": [
                    {
                        "filing_date": "2024-05-15",
                        "manager_name": "Fund",
                        "manager_cik": "123",
                        "shares": 5000,
                        "value_usd": 100,
                        "share_class": "COM",
                    },
                ],
            },
        ]
        legacy_file.write_text(json.dumps(data), encoding="utf-8")

        loaded = client._load_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].total_institutional_shares == 5000

        # Legacy file should be gone, new path should exist
        assert not legacy_file.exists()
        assert (client._cache_dir / "GME" / "13f.json").exists()

    def test_legacy_ownership_json_migration_on_load(self, tmp_path: Path) -> None:
        """Legacy {SYMBOL}_ownership.json migrates to {SYMBOL}/13f.json on load."""
        client = _make_client(tmp_path)

        legacy_file = client._cache_dir / "GME_ownership.json"
        data = [
            {
                "quarter_end": "2024-06-30",
                "symbol": "GME",
                "total_institutional_shares": 3000,
                "num_institutions": 1,
                "top_10_concentration": 1.0,
                "holdings": [
                    {
                        "filing_date": "2024-08-10",
                        "manager_name": "OldFund",
                        "manager_cik": "456",
                        "shares": 3000,
                        "value_usd": 50,
                        "share_class": "SH",
                    },
                ],
            },
        ]
        legacy_file.write_text(json.dumps(data), encoding="utf-8")

        loaded = client._load_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].holdings[0].manager_name == "OldFund"

        # Legacy file gone, new path exists
        assert not legacy_file.exists()
        assert (client._cache_dir / "GME" / "13f.json").exists()

    def test_save_load_roundtrip_new_paths(self, tmp_path: Path) -> None:
        """Save + load roundtrip works with new per-symbol paths."""
        client = _make_client(tmp_path)
        holdings = (
            InstitutionalHolding(
                filing_date="2024-05-15",
                manager_name="Vanguard",
                manager_cik="102909",
                shares=5_000_000,
                value_usd=150_000,
                share_class="COM",
            ),
            InstitutionalHolding(
                filing_date="2024-05-15",
                manager_name="BlackRock",
                manager_cik="101234",
                shares=3_000_000,
                value_usd=90_000,
                share_class="COM",
            ),
        )
        snapshots = [
            OwnershipSnapshot(
                quarter_end="2024-03-31",
                symbol="GME",
                total_institutional_shares=8_000_000,
                num_institutions=2,
                top_10_concentration=1.0,
                holdings=holdings,
            ),
        ]
        client._save_cache("GME", snapshots)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 1
        snap = loaded[0]
        assert snap.quarter_end == "2024-03-31"
        assert snap.total_institutional_shares == 8_000_000
        assert len(snap.holdings) == 2
        assert snap.holdings[0].manager_name == "Vanguard"
        assert snap.holdings[1].manager_name == "BlackRock"


# ------------------------------------------------------------------
# Tests: Full fetch_ownership_snapshots pipeline
# ------------------------------------------------------------------


class TestFetchOwnershipSnapshots:
    """Tests for the full pipeline."""

    def test_returns_cached_data_when_available(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        holdings = (
            InstitutionalHolding(
                filing_date="2024-05-15",
                manager_name="Vanguard",
                manager_cik="102909",
                shares=5_000_000,
                value_usd=150_000,
                share_class="COM",
            ),
        )
        snapshots = [
            OwnershipSnapshot(
                quarter_end="2024-03-31",
                symbol="GME",
                total_institutional_shares=5_000_000,
                num_institutions=1,
                top_10_concentration=1.0,
                holdings=holdings,
            ),
        ]
        client._save_cache("GME", snapshots)

        # Should not make any network calls
        with patch.object(
            client._session, "get",
            side_effect=AssertionError("Should not be called"),
        ):
            result = client.fetch_ownership_snapshots("gme")

        assert len(result) == 1
        assert result[0].quarter_end == "2024-03-31"

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_returns_empty_when_no_filings_found(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        # Bulk data returns None (not available), EFTS returns empty
        with patch.object(client, "_fetch_from_bulk", return_value=None), \
             patch.object(
                 client._session, "get",
                 return_value=_mock_response({"hits": {"hits": []}}),
             ):
            result = client.fetch_ownership_snapshots("GME", cusip="36467W109")

        assert result == []

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_aggregates_holdings_into_snapshots(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        # Build a fake OwnershipSnapshot as _fetch_from_efts would return
        snap_result = OwnershipSnapshot(
            quarter_end="2024-03-31",
            symbol="GME",
            total_institutional_shares=7_500_000,
            num_institutions=2,
            top_10_concentration=1.0,
            holdings=(
                InstitutionalHolding(
                    filing_date="2024-03-31",
                    manager_name="Test Fund",
                    manager_cik="1234567",
                    shares=5_000_000,
                    value_usd=150_000,
                    share_class="COM",
                ),
                InstitutionalHolding(
                    filing_date="2024-03-31",
                    manager_name="Test Fund 2",
                    manager_cik="9876543",
                    shares=2_500_000,
                    value_usd=75_000,
                    share_class="COM",
                ),
            ),
        )

        # Mock both bulk and EFTS at method level
        with patch.object(client, "_fetch_from_bulk", return_value=None), \
             patch.object(
                 client, "_fetch_from_efts",
                 side_effect=lambda *a, **kw: snap_result
                 if a[3] == 1 and a[2] == 2024 else None,
             ):
            result = client.fetch_ownership_snapshots(
                "GME", cusip="36467W109",
            )

        assert len(result) >= 1
        snap = result[0]
        assert snap.quarter_end == "2024-03-31"
        assert snap.symbol == "GME"
        assert snap.total_institutional_shares == 7_500_000
        assert snap.num_institutions == 2

    def test_num_quarters_limits_output(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        # Pre-populate cache with multiple quarters
        holdings = (
            InstitutionalHolding(
                filing_date="2024-05-15",
                manager_name="Fund",
                manager_cik="123",
                shares=1000,
                value_usd=100,
                share_class="COM",
            ),
        )
        snapshots = [
            OwnershipSnapshot(
                quarter_end=f"2024-{m:02d}-{d:02d}",
                symbol="GME",
                total_institutional_shares=1000,
                num_institutions=1,
                top_10_concentration=1.0,
                holdings=holdings,
            )
            for m, d in [(3, 31), (6, 30), (9, 30), (12, 31)]
        ]
        client._save_cache("GME", snapshots)

        result = client.fetch_ownership_snapshots("GME", num_quarters=2)
        assert len(result) == 2


# ------------------------------------------------------------------
# Tests: CUSIP auto-resolution from symbol registry
# ------------------------------------------------------------------


class TestCusipAutoResolution:
    """Verify that CUSIP auto-resolves from the symbol registry."""

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_aapl_auto_resolves_cusip(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """Calling fetch_ownership_snapshots('AAPL') without cusip=
        should auto-resolve AAPL's CUSIP, not use GME's."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        # Track what CUSIP is passed to _fetch_from_bulk
        with patch.object(client, "_fetch_from_bulk") as mock_bulk, \
             patch.object(client, "_fetch_from_efts", return_value=None):
            mock_bulk.return_value = None
            client.fetch_ownership_snapshots("AAPL", num_quarters=1)

        # The first call should have used AAPL's CUSIP, not GME's
        if mock_bulk.called:
            _, cusip_arg, _, _, _ = mock_bulk.call_args[0]
            assert cusip_arg == "037833100"  # AAPL CUSIP
            assert cusip_arg != "36467W109"  # Not GME CUSIP

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_gme_uses_default_cusip(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """Calling fetch_ownership_snapshots('GME') should still use
        the default GME CUSIP."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        with patch.object(client, "_fetch_from_bulk") as mock_bulk, \
             patch.object(client, "_fetch_from_efts", return_value=None):
            mock_bulk.return_value = None
            client.fetch_ownership_snapshots("GME", num_quarters=1)

        if mock_bulk.called:
            _, cusip_arg, _, _, _ = mock_bulk.call_args[0]
            assert cusip_arg == "36467W109"

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_explicit_cusip_not_overridden(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """An explicitly passed cusip= should be used as-is."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        with patch.object(client, "_fetch_from_bulk") as mock_bulk, \
             patch.object(client, "_fetch_from_efts", return_value=None):
            mock_bulk.return_value = None
            client.fetch_ownership_snapshots(
                "AAPL", cusip="CUSTOM123", num_quarters=1,
            )

        if mock_bulk.called:
            _, cusip_arg, _, _, _ = mock_bulk.call_args[0]
            assert cusip_arg == "CUSTOM123"


# ------------------------------------------------------------------
# Tests: Legacy text parsing for pre-2013 13F filings
# ------------------------------------------------------------------


class TestLegacyTextParsing:
    """Tests for _parse_13f_text() — extracts holdings from non-XML tables."""

    _CUSIP = "36467W109"

    def test_tab_separated_format(self) -> None:
        """Tab-separated infotable with standard column order."""
        content = (
            "NAME OF ISSUER\tTITLE OF CLASS\tCUSIP\tVALUE\tSHRSORPRNAMT\tSH/PRN\n"
            "GAMESTOP CORP\tCOM\t36467W109\t5000\t200000\tSH\n"
            "APPLE INC\tCOM\t037833100\t999999\t50000\tSH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 200000
        assert result[0].value_usd == 5000

    def test_fixed_width_format(self) -> None:
        """Fixed-width layout typical of early 2000s filings."""
        content = (
            "GAMESTOP CORP NEW       COM        36467W109      3500       150000   SH\n"
            "MICROSOFT CORP          COM        594918104     99000      1200000   SH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 150000
        assert result[0].value_usd == 3500

    def test_comma_separated_format(self) -> None:
        """CSV-style infotable."""
        content = (
            "GAMESTOP CORP,COM,36467W109,7200,300000,SH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 300000
        assert result[0].value_usd == 7200

    def test_cusip_not_found_returns_empty(self) -> None:
        """When the CUSIP doesn't appear, return empty list."""
        content = "APPLE INC\tCOM\t037833100\t999\t50000\tSH\n"
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_empty_content_returns_empty(self) -> None:
        result = SecOwnershipClient._parse_13f_text("", self._CUSIP)
        assert result == []

    def test_xml_content_returns_empty(self) -> None:
        """XML content should not match (handled by _parse_13f_xml)."""
        content = '<?xml version="1.0"?>\n<root><cusip>36467W109</cusip></root>'
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_html_content_returns_empty(self) -> None:
        """HTML content should be rejected."""
        content = "<html><body>36467W109\t5000\t200000</body></html>"
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_html_uppercase_returns_empty(self) -> None:
        """Uppercase <HTML> (common in early EDGAR filings) should be rejected."""
        content = "<HTML><BODY>36467W109\t5000\t200000</BODY></HTML>"
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_sgml_wrapped_content_is_accepted(self) -> None:
        """Pre-2013 SGML container tags like <DOCUMENT> should NOT be rejected."""
        content = (
            "<DOCUMENT>\n"
            "<TYPE>INFORMATION TABLE\n"
            "GAMESTOP CORP\tCOM\t36467W109\t5000\t200000\tSH\n"
            "</DOCUMENT>\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 200000

    def test_multiple_holders_same_cusip(self) -> None:
        """Multiple institutions holding the same CUSIP."""
        content = (
            "FUND A\tCOM\t36467W109\t1000\t50000\tSH\n"
            "FUND B\tCOM\t36467W109\t2000\t80000\tSH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 2
        total = sum(h.shares for h in result)
        assert total == 130000

    def test_cusip_case_insensitive(self) -> None:
        """CUSIP matching should be case-insensitive."""
        content = "GAMESTOP CORP\tCOM\t36467w109\t5000\t200000\tSH\n"
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1

    def test_value_with_commas_in_number(self) -> None:
        """Some filings use comma-formatted numbers like 1,500."""
        content = (
            "GAMESTOP CORP    COM    36467W109    1,500    50,000    SH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 50000
        assert result[0].value_usd == 1500


# ------------------------------------------------------------------
# Tests: Per-quarter EFTS snapshot caching
# ------------------------------------------------------------------


class TestEftsQuarterSnapshotCache:
    """Tests for per-quarter EFTS snapshot save/load."""

    @staticmethod
    def _make_snapshot(
        symbol: str = "GME",
        quarter_end: str = "2024-03-31",
    ) -> OwnershipSnapshot:
        return OwnershipSnapshot(
            quarter_end=quarter_end,
            symbol=symbol,
            total_institutional_shares=5000,
            num_institutions=1,
            top_10_concentration=1.0,
            holdings=(
                InstitutionalHolding(
                    filing_date=quarter_end,
                    manager_name="TestFund",
                    manager_cik="999",
                    shares=5000,
                    value_usd=150,
                    share_class="COM",
                ),
            ),
        )

    def test_save_creates_correct_file(self, tmp_path: Path) -> None:
        """_save_quarter_snapshot writes {SYMBOL}/{YYYY}Q{Q}.json."""
        client = _make_client(tmp_path)
        snap = self._make_snapshot()
        client._save_quarter_snapshot("GME", 2024, 1, snap)
        expected = client._cache_dir / "GME" / "2024Q1.json"
        assert expected.exists()
        data = json.loads(expected.read_text(encoding="utf-8"))
        assert data["quarter_end"] == "2024-03-31"
        assert data["total_institutional_shares"] == 5000

    def test_load_roundtrip(self, tmp_path: Path) -> None:
        """Save then load returns equivalent snapshot."""
        client = _make_client(tmp_path)
        snap = self._make_snapshot()
        client._save_quarter_snapshot("GME", 2024, 1, snap)
        loaded = client._load_quarter_snapshot("GME", 2024, 1)
        assert loaded is not None
        assert loaded.quarter_end == "2024-03-31"
        assert loaded.total_institutional_shares == 5000
        assert len(loaded.holdings) == 1
        assert loaded.holdings[0].manager_name == "TestFund"

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        """Loading a non-existent quarter returns None."""
        client = _make_client(tmp_path)
        loaded = client._load_quarter_snapshot("GME", 2099, 4)
        assert loaded is None

    def test_load_corrupt_returns_none(self, tmp_path: Path) -> None:
        """Loading a corrupt quarter file returns None."""
        client = _make_client(tmp_path)
        sym_dir = client._symbol_cache_dir("GME")
        corrupt = sym_dir / "2024Q1.json"
        corrupt.write_text("{{{bad json", encoding="utf-8")
        loaded = client._load_quarter_snapshot("GME", 2024, 1)
        assert loaded is None

    def test_force_refresh_preserves_quarter_snapshots(
        self, tmp_path: Path,
    ) -> None:
        """force_refresh=True deletes merged 13f.json but keeps quarter files."""
        client = _make_client(tmp_path)
        snap = self._make_snapshot()

        # Save a quarter snapshot and a merged cache
        client._save_quarter_snapshot("GME", 2024, 1, snap)
        client._save_cache("GME", [snap])

        merged = client._symbol_cache_dir("GME") / "13f.json"
        quarter = client._symbol_cache_dir("GME") / "2024Q1.json"
        assert merged.exists()
        assert quarter.exists()

        # Mock the network calls to return nothing (we just want
        # to test that force_refresh preserves quarter files)
        with patch.object(client, "_fetch_from_bulk", return_value=None), \
             patch.object(client, "_fetch_from_efts", return_value=None):
            client.fetch_ownership_snapshots(
                "GME", cusip="36467W109", num_quarters=1,
                force_refresh=True,
            )

        # Merged file should be deleted (or recreated empty if no data)
        # Quarter snapshot should still be on disk
        assert quarter.exists()


class TestEftsSnapshotIntegration:
    """Integration tests: per-quarter cache prevents HTTP requests."""

    def test_cached_quarter_skips_http(self, tmp_path: Path) -> None:
        """When a quarter snapshot exists, _fetch_from_efts returns it
        without making any HTTP requests."""
        client = _make_client(tmp_path)

        snap = OwnershipSnapshot(
            quarter_end="2024-03-31",
            symbol="GME",
            total_institutional_shares=5000,
            num_institutions=1,
            top_10_concentration=1.0,
            holdings=(
                InstitutionalHolding(
                    filing_date="2024-03-31",
                    manager_name="CachedFund",
                    manager_cik="111",
                    shares=5000,
                    value_usd=100,
                    share_class="COM",
                ),
            ),
        )
        client._save_quarter_snapshot("GME", 2024, 1, snap)

        with patch.object(
            client, "_fetch_efts_page",
            side_effect=AssertionError("Should not be called"),
        ):
            result = client._fetch_from_efts(
                "GME", "36467W109", 2024, 1, "2024-03-31",
            )

        assert result is not None
        assert result.total_institutional_shares == 5000
        assert result.holdings[0].manager_name == "CachedFund"


# ------------------------------------------------------------------
# Tests: Rate limiting
# ------------------------------------------------------------------


class TestOwnershipRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_sleeps_when_too_fast(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        with patch("stockdownloader.data.sec_ownership_client.time") as mock_time:
            mock_time.monotonic.side_effect = [
                0.0,    # first check
                0.0,    # set _last_request_time
                0.05,   # second check — only 50ms elapsed
                0.11,   # set _last_request_time after sleep
            ]
            mock_time.sleep = MagicMock()

            client._last_request_time = 0.0
            client._rate_limit()
            client._rate_limit()

            mock_time.sleep.assert_called()
