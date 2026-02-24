"""Unit tests for SEC ownership parsers — pure parsing logic, no network."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from stockdownloader.data.sec_ownership_parsers import (
    filing_date_to_quarter_end,
    get_xml_text,
    parse_13f_text,
    parse_13f_xml,
)


# ------------------------------------------------------------------
# Sample XML constants
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


# ------------------------------------------------------------------
# Tests: Filing date to quarter end mapping
# ------------------------------------------------------------------


class TestFilingDateToQuarterEnd:
    """Tests for filing_date_to_quarter_end helper."""

    def test_january_maps_to_q4_prior_year(self) -> None:
        assert filing_date_to_quarter_end("2024-01-15") == "2023-12-31"

    def test_february_maps_to_q4_prior_year(self) -> None:
        assert filing_date_to_quarter_end("2024-02-14") == "2023-12-31"

    def test_march_maps_to_q1(self) -> None:
        assert filing_date_to_quarter_end("2024-03-15") == "2024-03-31"

    def test_may_maps_to_q1(self) -> None:
        assert filing_date_to_quarter_end("2024-05-10") == "2024-03-31"

    def test_june_maps_to_q2(self) -> None:
        assert filing_date_to_quarter_end("2024-06-15") == "2024-06-30"

    def test_august_maps_to_q2(self) -> None:
        assert filing_date_to_quarter_end("2024-08-10") == "2024-06-30"

    def test_september_maps_to_q3(self) -> None:
        assert filing_date_to_quarter_end("2024-09-15") == "2024-09-30"

    def test_november_maps_to_q3(self) -> None:
        assert filing_date_to_quarter_end("2024-11-10") == "2024-09-30"

    def test_december_maps_to_q4(self) -> None:
        assert filing_date_to_quarter_end("2024-12-15") == "2024-12-31"

    def test_empty_string_passthrough(self) -> None:
        assert filing_date_to_quarter_end("") == ""

    def test_short_string_passthrough(self) -> None:
        assert filing_date_to_quarter_end("2024") == "2024"


# ------------------------------------------------------------------
# Tests: XML text extraction
# ------------------------------------------------------------------


class TestGetXmlText:
    """Tests for get_xml_text helper."""

    def test_extracts_text(self) -> None:
        root = ET.fromstring("<root><child>hello</child></root>")
        assert get_xml_text(root, "child") == "hello"

    def test_returns_none_for_missing_element(self) -> None:
        root = ET.fromstring("<root><child>hello</child></root>")
        assert get_xml_text(root, "missing") is None

    def test_returns_none_for_empty_text(self) -> None:
        root = ET.fromstring("<root><child></child></root>")
        assert get_xml_text(root, "child") is None

    def test_strips_whitespace(self) -> None:
        root = ET.fromstring("<root><child>  spaced  </child></root>")
        assert get_xml_text(root, "child") == "spaced"


# ------------------------------------------------------------------
# Tests: 13F XML parsing
# ------------------------------------------------------------------


class TestParse13fXml:
    """Tests for parse_13f_xml standalone function."""

    def test_parses_matching_cusip(self) -> None:
        holdings = parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        assert len(holdings) == 2

    def test_filters_by_cusip(self) -> None:
        holdings = parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        # Should not include AAPL
        assert all(h.manager_name == "GAMESTOP CORP NEW" for h in holdings)

    def test_parses_shares_and_value(self) -> None:
        holdings = parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        assert holdings[0].shares == 5_000_000
        assert holdings[0].value_usd == 150_000
        assert holdings[1].shares == 2_500_000
        assert holdings[1].value_usd == 75_000

    def test_parses_share_class(self) -> None:
        holdings = parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        assert holdings[0].share_class == "SH"

    def test_filing_date_is_empty_placeholder(self) -> None:
        """parse_13f_xml sets filing_date='' as a placeholder."""
        holdings = parse_13f_xml(
            _SAMPLE_13F_XML, "36467W109",
        )
        assert all(h.filing_date == "" for h in holdings)

    def test_parses_no_namespace_xml(self) -> None:
        holdings = parse_13f_xml(
            _SAMPLE_13F_XML_NO_NS, "36467W109",
        )
        assert len(holdings) == 1
        assert holdings[0].shares == 3_000_000

    def test_invalid_xml_returns_empty(self) -> None:
        holdings = parse_13f_xml(
            _SAMPLE_13F_XML_INVALID, "36467W109",
        )
        assert holdings == []

    def test_no_matching_cusip_returns_empty(self) -> None:
        holdings = parse_13f_xml(
            _SAMPLE_13F_XML, "000000000",
        )
        assert holdings == []

    def test_empty_xml(self) -> None:
        holdings = parse_13f_xml(
            "<root></root>", "36467W109",
        )
        assert holdings == []


# ------------------------------------------------------------------
# Tests: Legacy text parsing for pre-2013 13F filings
# ------------------------------------------------------------------


class TestLegacyTextParsing:
    """Tests for parse_13f_text() — extracts holdings from non-XML tables."""

    _CUSIP = "36467W109"

    def test_tab_separated_format(self) -> None:
        """Tab-separated infotable with standard column order."""
        content = (
            "NAME OF ISSUER\tTITLE OF CLASS\tCUSIP\tVALUE\tSHRSORPRNAMT\tSH/PRN\n"
            "GAMESTOP CORP\tCOM\t36467W109\t5000\t200000\tSH\n"
            "APPLE INC\tCOM\t037833100\t999999\t50000\tSH\n"
        )
        result = parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 200000
        assert result[0].value_usd == 5000

    def test_fixed_width_format(self) -> None:
        """Fixed-width layout typical of early 2000s filings."""
        content = (
            "GAMESTOP CORP NEW       COM        36467W109      3500       150000   SH\n"
            "MICROSOFT CORP          COM        594918104     99000      1200000   SH\n"
        )
        result = parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 150000
        assert result[0].value_usd == 3500

    def test_comma_separated_format(self) -> None:
        """CSV-style infotable."""
        content = (
            "GAMESTOP CORP,COM,36467W109,7200,300000,SH\n"
        )
        result = parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 300000
        assert result[0].value_usd == 7200

    def test_cusip_not_found_returns_empty(self) -> None:
        """When the CUSIP doesn't appear, return empty list."""
        content = "APPLE INC\tCOM\t037833100\t999\t50000\tSH\n"
        result = parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_empty_content_returns_empty(self) -> None:
        result = parse_13f_text("", self._CUSIP)
        assert result == []

    def test_xml_content_returns_empty(self) -> None:
        """XML content should not match (handled by parse_13f_xml)."""
        content = '<?xml version="1.0"?>\n<root><cusip>36467W109</cusip></root>'
        result = parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_html_content_returns_empty(self) -> None:
        """HTML content should be rejected."""
        content = "<html><body>36467W109\t5000\t200000</body></html>"
        result = parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_html_uppercase_returns_empty(self) -> None:
        """Uppercase <HTML> (common in early EDGAR filings) should be rejected."""
        content = "<HTML><BODY>36467W109\t5000\t200000</BODY></HTML>"
        result = parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_sgml_wrapped_content_is_accepted(self) -> None:
        """Pre-2013 SGML container tags like <DOCUMENT> should NOT be rejected."""
        content = (
            "<DOCUMENT>\n"
            "<TYPE>INFORMATION TABLE\n"
            "GAMESTOP CORP\tCOM\t36467W109\t5000\t200000\tSH\n"
            "</DOCUMENT>\n"
        )
        result = parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 200000

    def test_multiple_holders_same_cusip(self) -> None:
        """Multiple institutions holding the same CUSIP."""
        content = (
            "FUND A\tCOM\t36467W109\t1000\t50000\tSH\n"
            "FUND B\tCOM\t36467W109\t2000\t80000\tSH\n"
        )
        result = parse_13f_text(content, self._CUSIP)
        assert len(result) == 2
        total = sum(h.shares for h in result)
        assert total == 130000

    def test_cusip_case_insensitive(self) -> None:
        """CUSIP matching should be case-insensitive."""
        content = "GAMESTOP CORP\tCOM\t36467w109\t5000\t200000\tSH\n"
        result = parse_13f_text(content, self._CUSIP)
        assert len(result) == 1

    def test_value_with_commas_in_number(self) -> None:
        """Some filings use comma-formatted numbers like 1,500."""
        content = (
            "GAMESTOP CORP    COM    36467W109    1,500    50,000    SH\n"
        )
        result = parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 50000
        assert result[0].value_usd == 1500
