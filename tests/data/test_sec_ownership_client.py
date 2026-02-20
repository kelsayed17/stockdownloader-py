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
        cache_file = client._cache_dir / "BAD_ownership.json"
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
        assert (client._cache_dir / "AAPL_13f.json").exists()


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
