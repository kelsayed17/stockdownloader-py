"""Unit tests for SecEdgarClient — filing download and parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.sec_edgar_client import SecEdgarClient
from stockdownloader.model.regulatory_records import SecFiling


# ------------------------------------------------------------------
# Sample EDGAR JSON responses
# ------------------------------------------------------------------

_SAMPLE_SUBMISSIONS = {
    "cik": "1326380",
    "entityType": "operating",
    "name": "GameStop Corp",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0001326380-24-000013",
                "0001326380-24-000010",
                "0001326380-23-000050",
            ],
            "filingDate": [
                "2024-03-26",
                "2024-02-01",
                "2023-12-05",
            ],
            "reportDate": [
                "2024-02-03",
                "2024-01-31",
                "2023-10-28",
            ],
            "form": [
                "10-K",
                "8-K",
                "10-Q",
            ],
            "primaryDocument": [
                "gme-20240203.htm",
                "gme-20240131.htm",
                "gme-20231028.htm",
            ],
            "primaryDocDescription": [
                "Annual Report",
                "Current Report",
                "Quarterly Report",
            ],
        },
        "files": [],
    },
}

_SAMPLE_PAGE = {
    "accessionNumber": ["0001326380-22-000005"],
    "filingDate": ["2022-06-01"],
    "reportDate": ["2022-04-30"],
    "form": ["10-K"],
    "primaryDocument": ["gme-20220430.htm"],
    "primaryDocDescription": ["Annual Report"],
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client() -> SecEdgarClient:
    return SecEdgarClient(user_agent="TestAgent test@test.com")


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


# ------------------------------------------------------------------
# Tests: Parsing
# ------------------------------------------------------------------


class TestParseFilingsBlock:
    """Tests for the static _parse_filings_block method."""

    def test_parses_basic_filings(self) -> None:
        block = _SAMPLE_SUBMISSIONS["filings"]["recent"]
        filings = SecEdgarClient._parse_filings_block(block, "1326380")
        assert len(filings) == 3

    def test_filing_fields_populated(self) -> None:
        block = _SAMPLE_SUBMISSIONS["filings"]["recent"]
        filings = SecEdgarClient._parse_filings_block(block, "1326380")
        f = filings[0]
        assert f.accession_number == "0001326380-24-000013"
        assert f.filing_date == "2024-03-26"
        assert f.report_date == "2024-02-03"
        assert f.form == "10-K"
        assert f.primary_document == "gme-20240203.htm"
        assert f.description == "Annual Report"

    def test_builds_correct_filing_url(self) -> None:
        block = _SAMPLE_SUBMISSIONS["filings"]["recent"]
        filings = SecEdgarClient._parse_filings_block(block, "1326380")
        f = filings[0]
        assert "1326380" in f.filing_url
        assert "000132638024000013" in f.filing_url  # dashes removed
        assert "gme-20240203.htm" in f.filing_url

    def test_handles_empty_block(self) -> None:
        filings = SecEdgarClient._parse_filings_block({}, "1326380")
        assert filings == []

    def test_skips_empty_accession_numbers(self) -> None:
        block = {
            "accessionNumber": ["", "0001326380-24-000010"],
            "filingDate": ["2024-01-01", "2024-02-01"],
            "reportDate": ["", "2024-01-31"],
            "form": ["10-K", "8-K"],
            "primaryDocument": ["doc1.htm", "doc2.htm"],
            "primaryDocDescription": ["Report 1", "Report 2"],
        }
        filings = SecEdgarClient._parse_filings_block(block, "1326380")
        assert len(filings) == 1
        assert filings[0].form == "8-K"


# ------------------------------------------------------------------
# Tests: Fetch
# ------------------------------------------------------------------


class TestFetchFilings:
    def test_fetch_returns_sorted_filings(self) -> None:
        client = _make_client()
        with patch.object(
            client._session, "get",
            return_value=_mock_response(_SAMPLE_SUBMISSIONS),
        ):
            filings = client.fetch_filings("1326380")

        assert len(filings) == 3
        # Should be sorted ascending by filing_date
        assert filings[0].filing_date == "2023-12-05"
        assert filings[1].filing_date == "2024-02-01"
        assert filings[2].filing_date == "2024-03-26"

    def test_fetch_filters_by_form_type(self) -> None:
        client = _make_client()
        with patch.object(
            client._session, "get",
            return_value=_mock_response(_SAMPLE_SUBMISSIONS),
        ):
            filings = client.fetch_filings("1326380", form_types=["10-K"])

        assert len(filings) == 1
        assert filings[0].form == "10-K"

    def test_fetch_handles_network_error(self) -> None:
        client = _make_client()
        with patch.object(
            client._session, "get",
            side_effect=ConnectionError("fail"),
        ):
            filings = client.fetch_filings("1326380")

        assert filings == []

    def test_fetch_handles_non_200_status(self) -> None:
        client = _make_client()
        with patch.object(
            client._session, "get",
            return_value=_mock_response({}, status_code=429),
        ):
            filings = client.fetch_filings("1326380")

        assert filings == []

    def test_fetch_handles_pagination(self) -> None:
        client = _make_client()
        submissions_with_page = {
            **_SAMPLE_SUBMISSIONS,
            "filings": {
                **_SAMPLE_SUBMISSIONS["filings"],
                "files": [{"name": "CIK0001326380-submissions-001.json"}],
            },
        }

        responses = [
            _mock_response(submissions_with_page),
            _mock_response(_SAMPLE_PAGE),
        ]
        with patch.object(
            client._session, "get",
            side_effect=responses,
        ):
            filings = client.fetch_filings("1326380")

        # 3 from recent + 1 from pagination
        assert len(filings) == 4
        # The oldest (from page) should appear first when sorted
        assert filings[0].filing_date == "2022-06-01"

    def test_pads_cik_to_ten_digits(self) -> None:
        client = _make_client()
        calls = []

        def capture_get(url, **kwargs):
            calls.append(url)
            return _mock_response(_SAMPLE_SUBMISSIONS)

        with patch.object(client._session, "get", side_effect=capture_get):
            client.fetch_filings("1326380")

        assert "CIK0001326380" in calls[0]


class TestFetchGmeFilings:
    def test_uses_gme_cik(self) -> None:
        client = _make_client()
        calls = []

        def capture_get(url, **kwargs):
            calls.append(url)
            return _mock_response(_SAMPLE_SUBMISSIONS)

        with patch.object(client._session, "get", side_effect=capture_get):
            client.fetch_gme_filings()

        assert "CIK0001326380" in calls[0]


# ------------------------------------------------------------------
# Tests: Rate Limiting
# ------------------------------------------------------------------


class TestRateLimiting:
    def test_rate_limit_sleeps_between_requests(self) -> None:
        client = _make_client()
        with patch("stockdownloader.data.sec_edgar_client.time") as mock_time:
            mock_time.monotonic.side_effect = [
                0.0,   # first _rate_limit check
                0.0,   # set _last_request_time
                0.05,  # second _rate_limit check — only 50ms elapsed
                0.11,  # set _last_request_time after sleep
            ]
            mock_time.sleep = MagicMock()

            client._last_request_time = 0.0
            client._rate_limit()  # first call — no sleep needed (0 elapsed)
            client._rate_limit()  # second call — should sleep ~60ms

            # Sleep should have been called at least once
            mock_time.sleep.assert_called()
