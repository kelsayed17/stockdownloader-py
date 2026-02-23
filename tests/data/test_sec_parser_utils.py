"""Unit tests for shared SEC parser utilities."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date

import pytest

from stockdownloader.data.sec_parser_utils import (
    MONTH_ABBREVS,
    MONTH_NAMES,
    GME_CUSIP,
    QUARTER_ENDS,
    filing_date_to_quarter_end,
    get_xml_text,
    is_leap_year,
    normalize_date,
    quarter_range,
    xml_text,
)


# ------------------------------------------------------------------
# normalize_date
# ------------------------------------------------------------------


class TestNormalizeDate:
    def test_normalize_date_iso(self) -> None:
        assert normalize_date("2024-01-15") == "2024-01-15"

    def test_normalize_date_slashes(self) -> None:
        assert normalize_date("01/15/2024") == "2024-01-15"

    def test_normalize_date_month_name(self) -> None:
        """DD-MON-YYYY SEC bulk TSV format."""
        assert normalize_date("15-Jan-2024") == "2024-01-15"

    def test_normalize_date_empty(self) -> None:
        assert normalize_date("") == ""

    def test_normalize_date_whitespace(self) -> None:
        assert normalize_date("  2024-01-15  ") == "2024-01-15"

    def test_normalize_date_compact(self) -> None:
        assert normalize_date("20240115") == "2024-01-15"


# ------------------------------------------------------------------
# xml_text (regex-based, works on raw strings)
# ------------------------------------------------------------------


class TestXmlText:
    def test_xml_text_found(self) -> None:
        content = "<root><name>John Doe</name><age>30</age></root>"
        assert xml_text(content, "name") == "John Doe"

    def test_xml_text_missing(self) -> None:
        content = "<root><name>John Doe</name></root>"
        assert xml_text(content, "missing") is None

    def test_xml_text_strips_whitespace(self) -> None:
        content = "<root><name>  John Doe  </name></root>"
        assert xml_text(content, "name") == "John Doe"

    def test_xml_text_case_insensitive(self) -> None:
        content = "<ROOT><Name>Test</Name></ROOT>"
        assert xml_text(content, "name") == "Test"


# ------------------------------------------------------------------
# get_xml_text (ElementTree-based, works on ET.Element)
# ------------------------------------------------------------------


class TestGetXmlText:
    def test_get_xml_text_found(self) -> None:
        root = ET.fromstring("<root><name>Jane</name></root>")
        assert get_xml_text(root, "name") == "Jane"

    def test_get_xml_text_missing(self) -> None:
        root = ET.fromstring("<root><name>Jane</name></root>")
        assert get_xml_text(root, "missing") is None

    def test_get_xml_text_empty_element(self) -> None:
        root = ET.fromstring("<root><name></name></root>")
        assert get_xml_text(root, "name") is None

    def test_get_xml_text_strips_whitespace(self) -> None:
        root = ET.fromstring("<root><name>  Jane  </name></root>")
        assert get_xml_text(root, "name") == "Jane"


# ------------------------------------------------------------------
# quarter_range
# ------------------------------------------------------------------


class TestQuarterRange:
    def test_quarter_range_q1(self) -> None:
        start, end = quarter_range(2024, 1)
        assert start == date(2024, 1, 1)
        assert end == date(2024, 3, 31)

    def test_quarter_range_q2(self) -> None:
        start, end = quarter_range(2024, 2)
        assert start == date(2024, 4, 1)
        assert end == date(2024, 6, 30)

    def test_quarter_range_q3(self) -> None:
        start, end = quarter_range(2024, 3)
        assert start == date(2024, 7, 1)
        assert end == date(2024, 9, 30)

    def test_quarter_range_q4(self) -> None:
        start, end = quarter_range(2024, 4)
        assert start == date(2024, 10, 1)
        assert end == date(2024, 12, 31)


# ------------------------------------------------------------------
# filing_date_to_quarter_end
# ------------------------------------------------------------------


class TestFilingDateToQuarterEnd:
    def test_jan_filing_maps_to_q4_prior_year(self) -> None:
        assert filing_date_to_quarter_end("2024-01-15") == "2023-12-31"

    def test_feb_filing_maps_to_q4_prior_year(self) -> None:
        assert filing_date_to_quarter_end("2024-02-14") == "2023-12-31"

    def test_mar_filing_maps_to_q1(self) -> None:
        assert filing_date_to_quarter_end("2024-03-31") == "2024-03-31"

    def test_may_filing_maps_to_q1(self) -> None:
        assert filing_date_to_quarter_end("2024-05-15") == "2024-03-31"

    def test_aug_filing_maps_to_q2(self) -> None:
        assert filing_date_to_quarter_end("2024-08-14") == "2024-06-30"

    def test_nov_filing_maps_to_q3(self) -> None:
        assert filing_date_to_quarter_end("2024-11-14") == "2024-09-30"

    def test_dec_filing_maps_to_q4(self) -> None:
        assert filing_date_to_quarter_end("2024-12-15") == "2024-12-31"

    def test_empty_string(self) -> None:
        assert filing_date_to_quarter_end("") == ""

    def test_short_string(self) -> None:
        assert filing_date_to_quarter_end("2024") == "2024"


# ------------------------------------------------------------------
# is_leap_year
# ------------------------------------------------------------------


class TestIsLeapYear:
    def test_common_leap_year(self) -> None:
        assert is_leap_year(2024) is True

    def test_non_leap_year(self) -> None:
        assert is_leap_year(2023) is False

    def test_century_not_leap(self) -> None:
        assert is_leap_year(1900) is False

    def test_400_year_leap(self) -> None:
        assert is_leap_year(2000) is True


# ------------------------------------------------------------------
# Constants sanity checks
# ------------------------------------------------------------------


class TestConstants:
    def test_month_abbrevs_length(self) -> None:
        assert len(MONTH_ABBREVS) == 12

    def test_month_names_length(self) -> None:
        assert len(MONTH_NAMES) == 12

    def test_gme_cusip(self) -> None:
        assert GME_CUSIP == "36467W109"

    def test_quarter_ends_length(self) -> None:
        assert len(QUARTER_ENDS) == 4
        assert QUARTER_ENDS[1] == "03-31"
        assert QUARTER_ENDS[4] == "12-31"
