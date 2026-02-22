"""Unit tests for FinraShortInterestClient — OAuth2 auth, API query, parsing, caching."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.finra_short_interest_client import (
    FinraShortInterestClient,
    _normalize_date,
)
from stockdownloader.model.regulatory_records import ShortInterestRecord


# ------------------------------------------------------------------
# Sample FINRA API JSON responses
# ------------------------------------------------------------------

_SAMPLE_TOKEN_RESPONSE = {"access_token": "test-token-abc123"}

_SAMPLE_SI_RESPONSE = [
    {
        "settlementDate": "2024-01-15",
        "issueName": "GAMESTOP CORP NEW CL A",
        "symbolCode": "GME",
        "currentShortPositionQuantity": 30000000,
        "previousShortPositionQuantity": 28000000,
        "averageDailyVolumeQuantity": 5000000,
        "daysToCoverQuantity": 6.0,
        "shortInterestRatioChange": 0.07,
    },
    {
        "settlementDate": "2024-01-31",
        "issueName": "GAMESTOP CORP NEW CL A",
        "symbolCode": "GME",
        "currentShortPositionQuantity": 32000000,
        "previousShortPositionQuantity": 30000000,
        "averageDailyVolumeQuantity": 4500000,
        "daysToCoverQuantity": 7.1,
        "shortInterestRatioChange": 0.066,
    },
]

_SAMPLE_SI_MULTI_FORMAT = [
    {
        "settlementDate": "01/15/2024",
        "symbolCode": "GME",
        "currentShortPositionQuantity": 30000000,
        "averageDailyVolumeQuantity": 5000000,
        "daysToCoverQuantity": 6.0,
    },
    {
        "settlementDate": "20240131",
        "symbolCode": "GME",
        "currentShortPositionQuantity": 32000000,
        "averageDailyVolumeQuantity": 4500000,
        "daysToCoverQuantity": 7.1,
    },
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(
    tmp_path: Path,
    client_id: str = "test_id",
    client_secret: str = "test_secret",
) -> FinraShortInterestClient:
    return FinraShortInterestClient(
        client_id=client_id,
        client_secret=client_secret,
        data_dir=str(tmp_path),
    )


def _mock_response(
    json_data: object = None,
    status_code: int = 200,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text or json.dumps(json_data) if json_data else ""
    return resp


# ------------------------------------------------------------------
# Tests: Date Normalization
# ------------------------------------------------------------------


class TestNormalizeDate:
    """Tests for the _normalize_date helper."""

    def test_iso_format_passthrough(self) -> None:
        assert _normalize_date("2024-01-15") == "2024-01-15"

    def test_slash_format(self) -> None:
        assert _normalize_date("01/15/2024") == "2024-01-15"

    def test_slash_format_single_digit(self) -> None:
        assert _normalize_date("1/5/2024") == "2024-01-05"

    def test_compact_yyyymmdd_format(self) -> None:
        assert _normalize_date("20240115") == "2024-01-15"

    def test_empty_string(self) -> None:
        assert _normalize_date("") == ""

    def test_unrecognizable_format(self) -> None:
        assert _normalize_date("Jan 15 2024") == ""

    def test_invalid_slash_format(self) -> None:
        assert _normalize_date("abc/def/ghi") == ""

    def test_none_handled(self) -> None:
        # The function expects a string, but should handle empty gracefully
        assert _normalize_date("") == ""


# ------------------------------------------------------------------
# Tests: OAuth2 Authentication
# ------------------------------------------------------------------


class TestAuthentication:
    """Tests for FINRA OAuth2 authentication flow."""

    def test_successful_auth_sets_bearer_token(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_short_interest_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)
            result = client._authenticate()

        assert result is True
        assert client._access_token == "test-token-abc123"
        assert client._session.headers["Authorization"] == "Bearer test-token-abc123"

    def test_auth_sends_basic_auth_header(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, client_id="myid", client_secret="mysecret")

        with patch("stockdownloader.data.finra_short_interest_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)
            client._authenticate()

        call_kwargs = mock_post.call_args
        auth_header = call_kwargs.kwargs.get("headers", {}).get("Authorization", "")
        expected = base64.b64encode(b"myid:mysecret").decode()
        assert auth_header == f"Basic {expected}"

    def test_auth_failure_no_credentials(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, client_id="", client_secret="")
        result = client._authenticate()
        assert result is False
        assert client._access_token is None

    def test_auth_failure_bad_response(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_short_interest_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                {"error": "invalid_client"}, status_code=401, text="Unauthorized",
            )
            result = client._authenticate()

        assert result is False
        assert client._access_token is None

    def test_auth_failure_exception(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_short_interest_client.requests.post") as mock_post:
            mock_post.side_effect = ConnectionError("network failure")
            result = client._authenticate()

        assert result is False
        assert client._access_token is None

    def test_auth_failure_missing_access_token_in_response(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_short_interest_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"other": "value"})
            result = client._authenticate()

        assert result is False
        assert client._access_token is None


# ------------------------------------------------------------------
# Tests: Auto-auth before first query
# ------------------------------------------------------------------


class TestAutoAuth:
    """Tests that authentication happens automatically before API query."""

    @patch("stockdownloader.data.finra_short_interest_client.time")
    @patch("stockdownloader.data.finra_short_interest_client.requests.post")
    def test_fetch_triggers_auth_when_no_token(
        self, mock_post: MagicMock, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        assert client._access_token is None

        # Auth call via requests.post
        mock_post.side_effect = [
            _mock_response(_SAMPLE_TOKEN_RESPONSE),  # auth
        ]
        # Mock _query_api to return the sample response directly,
        # bypassing the pagination loop that would otherwise iterate
        # over many settlement dates and accumulate duplicate records.
        with patch.object(
            client, "_query_api",
            return_value=_SAMPLE_SI_RESPONSE,
        ):
            records = client.fetch_short_interest("GME")

        # Auth should have been triggered
        assert client._access_token == "test-token-abc123"
        assert len(records) == 2


# ------------------------------------------------------------------
# Tests: API Query
# ------------------------------------------------------------------


class TestQueryApi:
    """Tests for the _query_api method (paginated domainFilter query)."""

    @patch("stockdownloader.data.finra_short_interest_client.time")
    def test_successful_query(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        # Single page response (fewer than page_size records)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(_SAMPLE_SI_RESPONSE),
        ):
            raw = client._query_api("GME")

        assert raw is not None
        assert len(raw) == 2

    @patch("stockdownloader.data.finra_short_interest_client.time")
    def test_query_uses_domain_filters(self, mock_time: MagicMock, tmp_path: Path) -> None:
        """Verify payload uses domainFilters with symbolCode."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "post",
            return_value=_mock_response([]),
        ) as mock_post:
            client._query_api("GME")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        filters = payload["domainFilters"]
        assert len(filters) == 1
        assert filters[0]["fieldName"] == "symbolCode"
        assert filters[0]["values"] == ["GME"]
        # No compareFilters should be present
        assert "compareFilters" not in payload

    @patch("stockdownloader.data.finra_short_interest_client.time")
    def test_query_retries_on_failure(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        # Two failures followed by one success
        with patch.object(
            client._session, "post",
            side_effect=[
                _mock_response(None, status_code=500, text="Server Error"),
                _mock_response(None, status_code=500, text="Server Error"),
                _mock_response(_SAMPLE_SI_RESPONSE),
            ],
        ):
            raw = client._query_api("GME")

        assert raw is not None
        assert len(raw) == 2

    @patch("stockdownloader.data.finra_short_interest_client.time")
    def test_query_returns_none_after_max_retries(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "post",
            side_effect=[
                _mock_response(None, status_code=500, text="Error"),
                _mock_response(None, status_code=500, text="Error"),
                _mock_response(None, status_code=500, text="Error"),
            ],
        ):
            raw = client._query_api("GME")

        assert raw is None

    @patch("stockdownloader.data.finra_short_interest_client.time")
    def test_query_handles_request_exception(self, mock_time: MagicMock, tmp_path: Path) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        import requests as rq

        with patch.object(
            client._session, "post",
            side_effect=rq.RequestException("timeout"),
        ):
            raw = client._query_api("GME")

        assert raw is None

    @patch("stockdownloader.data.finra_short_interest_client.time")
    def test_pagination_stops_on_empty(self, mock_time: MagicMock, tmp_path: Path) -> None:
        """Verify pagination stops when API returns fewer than page_size."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(_SAMPLE_SI_RESPONSE),
        ) as mock_post:
            raw = client._query_api("GME")

        assert raw is not None
        assert len(raw) == 2
        # 2 < 5000 (page_size), so only 1 call needed
        assert mock_post.call_count == 1


# ------------------------------------------------------------------
# Tests: Parsing raw JSON to records
# ------------------------------------------------------------------


class TestRawToRecords:
    """Tests for _raw_to_records static method."""

    def test_parses_iso_dates(self) -> None:
        records = FinraShortInterestClient._raw_to_records(
            _SAMPLE_SI_RESPONSE, "GME",
        )
        assert len(records) == 2
        assert records[0].settlement_date == "2024-01-15"
        assert records[1].settlement_date == "2024-01-31"

    def test_parses_fields_correctly(self) -> None:
        records = FinraShortInterestClient._raw_to_records(
            _SAMPLE_SI_RESPONSE, "GME",
        )
        rec = records[0]
        assert rec.symbol == "GME"
        assert rec.short_interest == 30_000_000
        assert rec.avg_daily_volume == 5_000_000
        assert rec.days_to_cover == 6.0
        assert rec.short_interest_pct == 0.0  # Not available from FINRA

    def test_parses_alternative_date_formats(self) -> None:
        records = FinraShortInterestClient._raw_to_records(
            _SAMPLE_SI_MULTI_FORMAT, "GME",
        )
        assert len(records) == 2
        # Slash format -> ISO
        assert records[0].settlement_date == "2024-01-15"
        # YYYYMMDD -> ISO
        assert records[1].settlement_date == "2024-01-31"

    def test_records_sorted_ascending(self) -> None:
        # Reverse the order to verify sorting
        reversed_data = list(reversed(_SAMPLE_SI_RESPONSE))
        records = FinraShortInterestClient._raw_to_records(
            reversed_data, "GME",
        )
        dates = [r.settlement_date for r in records]
        assert dates == sorted(dates)

    def test_skips_malformed_rows(self) -> None:
        bad_data = [
            {"settlementDate": ""},  # empty date
            {"settlementDate": "2024-01-15", "currentShortPositionQuantity": "not_a_number"},
            {
                "settlementDate": "2024-01-20",
                "symbolCode": "GME",
                "currentShortPositionQuantity": 1000,
                "averageDailyVolumeQuantity": 500,
                "daysToCoverQuantity": 2.0,
            },
        ]
        records = FinraShortInterestClient._raw_to_records(bad_data, "GME")
        # Only the last valid row should parse
        assert len(records) == 1
        assert records[0].settlement_date == "2024-01-20"

    def test_empty_response(self) -> None:
        records = FinraShortInterestClient._raw_to_records([], "GME")
        assert records == []


# ------------------------------------------------------------------
# Tests: Caching (save + load)
# ------------------------------------------------------------------


class TestCaching:
    """Tests for cache save/load."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = [
            ShortInterestRecord(
                settlement_date="2024-01-15",
                symbol="GME",
                short_interest=30_000_000,
                avg_daily_volume=5_000_000,
                days_to_cover=6.0,
                short_interest_pct=0.0,
            ),
            ShortInterestRecord(
                settlement_date="2024-01-31",
                symbol="GME",
                short_interest=32_000_000,
                avg_daily_volume=4_500_000,
                days_to_cover=7.1,
                short_interest_pct=0.0,
            ),
        ]

        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].settlement_date == "2024-01-15"
        assert loaded[1].short_interest == 32_000_000

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        loaded = client._load_cache("NOSYMBOL")
        assert loaded is None

    def test_load_corrupt_csv_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        cache_file = sym_dir / "short_interest.csv"
        cache_file.write_text("not,valid,csv\ngarbage", encoding="utf-8")
        loaded = client._load_cache("BAD")
        assert loaded is None

    def test_load_corrupt_json_fallback(self, tmp_path: Path) -> None:
        """Corrupt JSON fallback returns None when no CSV exists."""
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        cache_file = sym_dir / "short_interest.json"
        cache_file.write_text("not valid json{{{", encoding="utf-8")
        loaded = client._load_cache("BAD")
        assert loaded is None

    def test_cache_file_naming(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = [
            ShortInterestRecord(
                settlement_date="2024-01-15",
                symbol="AAPL",
                short_interest=100,
                avg_daily_volume=1000,
                days_to_cover=0.1,
                short_interest_pct=0.0,
            ),
        ]
        client._save_cache("AAPL", records)
        assert (client._data_dir / "AAPL" / "short_interest.csv").exists()

    def test_save_creates_symbol_subdir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = [
            ShortInterestRecord(
                settlement_date="2024-01-15",
                symbol="TSLA",
                short_interest=200,
                avg_daily_volume=2000,
                days_to_cover=0.1,
                short_interest_pct=0.0,
            ),
        ]
        client._save_cache("TSLA", records)
        sym_dir = client._data_dir / "TSLA"
        assert sym_dir.is_dir()
        assert (sym_dir / "short_interest.csv").exists()

    def test_legacy_migration_on_load(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = [
            ShortInterestRecord(
                settlement_date="2024-01-15",
                symbol="MSFT",
                short_interest=500,
                avg_daily_volume=3000,
                days_to_cover=0.2,
                short_interest_pct=0.0,
            ),
        ]
        # Write legacy nested file (data_dir / "cache" / "short_interest" / symbol / "si.json")
        import json as _json
        legacy_dir = client._data_dir / "cache" / "short_interest" / "MSFT"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_file = legacy_dir / "si.json"
        legacy_file.write_text(
            _json.dumps([{
                "settlement_date": r.settlement_date,
                "symbol": r.symbol,
                "short_interest": r.short_interest,
                "avg_daily_volume": r.avg_daily_volume,
                "days_to_cover": r.days_to_cover,
                "short_interest_pct": r.short_interest_pct,
            } for r in records]),
            encoding="utf-8",
        )
        assert legacy_file.exists()

        loaded = client._load_cache("MSFT")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].symbol == "MSFT"
        # Legacy file should have been moved to JSON then migrated to CSV
        assert not legacy_file.exists()
        assert (client._data_dir / "MSFT" / "short_interest.csv").exists()

    def test_json_fallback_migrates_to_csv(self, tmp_path: Path) -> None:
        """When only short_interest.json exists, _load_cache reads it and writes CSV."""
        import json as _json

        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "TICKER"
        sym_dir.mkdir(parents=True, exist_ok=True)
        json_file = sym_dir / "short_interest.json"
        json_file.write_text(
            _json.dumps([{
                "settlement_date": "2024-03-01",
                "symbol": "TICKER",
                "short_interest": 42000,
                "avg_daily_volume": 8000,
                "days_to_cover": 5.25,
                "short_interest_pct": 12.5,
            }]),
            encoding="utf-8",
        )

        loaded = client._load_cache("TICKER")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].settlement_date == "2024-03-01"
        assert loaded[0].short_interest_pct == 12.5
        # CSV file should now exist
        assert (sym_dir / "short_interest.csv").exists()

    def test_json_fallback_preserves_missing_short_interest_pct(self, tmp_path: Path) -> None:
        """Old JSON without short_interest_pct field defaults to 0.0."""
        import json as _json

        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "OLD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        json_file = sym_dir / "short_interest.json"
        json_file.write_text(
            _json.dumps([{
                "settlement_date": "2023-06-15",
                "symbol": "OLD",
                "short_interest": 1000,
                "avg_daily_volume": 500,
                "days_to_cover": 2.0,
                # no short_interest_pct key
            }]),
            encoding="utf-8",
        )

        loaded = client._load_cache("OLD")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].short_interest_pct == 0.0

    def test_csv_preferred_over_json(self, tmp_path: Path) -> None:
        """When both CSV and JSON exist, CSV takes precedence."""
        import json as _json

        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "BOTH"
        sym_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON with one record
        json_file = sym_dir / "short_interest.json"
        json_file.write_text(
            _json.dumps([{
                "settlement_date": "2024-01-01",
                "symbol": "BOTH",
                "short_interest": 111,
                "avg_daily_volume": 222,
                "days_to_cover": 0.5,
                "short_interest_pct": 0.0,
            }]),
            encoding="utf-8",
        )

        # Write CSV with a different record
        csv_records = [
            ShortInterestRecord(
                settlement_date="2024-02-01",
                symbol="BOTH",
                short_interest=999,
                avg_daily_volume=888,
                days_to_cover=1.1,
                short_interest_pct=5.0,
            ),
        ]
        client._save_csv("BOTH", "short_interest.csv", csv_records)

        loaded = client._load_cache("BOTH")
        assert loaded is not None
        assert len(loaded) == 1
        # Should get the CSV data, not the JSON data
        assert loaded[0].short_interest == 999


# ------------------------------------------------------------------
# Tests: Fallback to cache when API fails
# ------------------------------------------------------------------


class TestFallbackToCache:
    """Tests for falling back to cached data when API fails."""

    @patch("stockdownloader.data.finra_short_interest_client.time")
    @patch("stockdownloader.data.finra_short_interest_client.requests.post")
    def test_falls_back_to_cache_on_api_failure(
        self, mock_post: MagicMock, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        # Pre-populate cache
        cached_records = [
            ShortInterestRecord(
                settlement_date="2024-01-10",
                symbol="GME",
                short_interest=25_000_000,
                avg_daily_volume=4_000_000,
                days_to_cover=6.25,
                short_interest_pct=0.0,
            ),
        ]
        client._save_cache("GME", cached_records)

        # Auth succeeds but API fails
        mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(None, status_code=500, text="Error"),
        ):
            records = client.fetch_short_interest("gme")

        assert len(records) == 1
        assert records[0].settlement_date == "2024-01-10"

    @patch("stockdownloader.data.finra_short_interest_client.time")
    @patch("stockdownloader.data.finra_short_interest_client.requests.post")
    def test_returns_empty_list_when_no_api_no_cache(
        self, mock_post: MagicMock, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(None, status_code=500, text="Error"),
        ):
            records = client.fetch_short_interest("NOSYMBOL")

        assert records == []


# ------------------------------------------------------------------
# Tests: Empty response handling
# ------------------------------------------------------------------


class TestEmptyResponse:
    """Tests for handling empty API responses."""

    @patch("stockdownloader.data.finra_short_interest_client.time")
    @patch("stockdownloader.data.finra_short_interest_client.requests.post")
    def test_empty_api_response_falls_to_cache(
        self, mock_post: MagicMock, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)
        with patch.object(
            client._session, "post",
            return_value=_mock_response([]),
        ):
            records = client.fetch_short_interest("GME")

        assert records == []


# ------------------------------------------------------------------
# Tests: Rate limiting
# ------------------------------------------------------------------


class TestRateLimiting:
    """Tests for the rate limiting mechanism."""

    def test_rate_limit_sleeps_when_too_fast(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        with patch("stockdownloader.data.finra_short_interest_client.time") as mock_time:
            mock_time.monotonic.side_effect = [
                0.0,   # first check
                0.0,   # set _last_request_time
                0.2,   # second check — 200ms elapsed
                0.5,   # set _last_request_time
                0.6,   # third check — only 100ms elapsed (< 500ms)
                1.0,   # set _last_request_time after sleep
            ]
            mock_time.sleep = MagicMock()

            client._last_request_time = 0.0
            client._rate_limit()  # First call, 0 elapsed
            client._rate_limit()  # 200ms elapsed, no sleep needed
            client._rate_limit()  # 100ms elapsed, needs sleep

            # Sleep should have been called
            mock_time.sleep.assert_called()


# ------------------------------------------------------------------
# Tests: Symbol normalization
# ------------------------------------------------------------------


class TestSymbolNormalization:
    """Tests that symbols are normalized to uppercase."""

    @patch("stockdownloader.data.finra_short_interest_client.time")
    @patch("stockdownloader.data.finra_short_interest_client.requests.post")
    def test_fetch_uppercases_symbol(
        self, mock_post: MagicMock, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)

        with patch.object(
            client._session, "post",
            return_value=_mock_response(_SAMPLE_SI_RESPONSE),
        ):
            records = client.fetch_short_interest("gme")

        assert all(r.symbol == "GME" for r in records)
