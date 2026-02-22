"""Unit tests for FinraDarkPoolClient — OAuth2 auth, API query, parsing, caching."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.finra_dark_pool_client import (
    FinraDarkPoolClient,
    _normalize_date,
)
from stockdownloader.model.regulatory_records import DarkPoolRecord


# ------------------------------------------------------------------
# Sample FINRA API JSON responses
# ------------------------------------------------------------------

_SAMPLE_TOKEN_RESPONSE = {"access_token": "dp-token-xyz789"}

# Combined OTC + ATS response from weeklySummary endpoint
# (both types come from the same dataset, distinguished by summaryTypeCode)
_SAMPLE_WEEKLY_RESPONSE = [
    {
        "weekStartDate": "2024-01-08",
        "issueSymbolIdentifier": "GME",
        "totalWeeklyShareQuantity": 15000000,
        "totalWeeklyTradeCount": 42000,
        "lastUpdateDate": "2024-01-14",
        "summaryTypeCode": "OTC_W_SMBL",
    },
    {
        "weekStartDate": "2024-01-08",
        "issueSymbolIdentifier": "GME",
        "totalWeeklyShareQuantity": 5000000,
        "totalWeeklyTradeCount": 12000,
        "lastUpdateDate": "2024-01-14",
        "summaryTypeCode": "ATS_W_SMBL",
    },
    {
        "weekStartDate": "2024-01-15",
        "issueSymbolIdentifier": "GME",
        "totalWeeklyShareQuantity": 18000000,
        "totalWeeklyTradeCount": 50000,
        "lastUpdateDate": "2024-01-21",
        "summaryTypeCode": "OTC_W_SMBL",
    },
    {
        "weekStartDate": "2024-01-15",
        "issueSymbolIdentifier": "GME",
        "totalWeeklyShareQuantity": 6000000,
        "totalWeeklyTradeCount": 14000,
        "lastUpdateDate": "2024-01-21",
        "summaryTypeCode": "ATS_W_SMBL",
    },
]

# OTC-only response (for simpler tests)
_SAMPLE_OTC_RESPONSE = [
    {
        "weekStartDate": "2024-01-08",
        "issueSymbolIdentifier": "GME",
        "totalWeeklyShareQuantity": 15000000,
        "totalWeeklyTradeCount": 42000,
        "summaryTypeCode": "OTC_W_SMBL",
    },
    {
        "weekStartDate": "2024-01-15",
        "issueSymbolIdentifier": "GME",
        "totalWeeklyShareQuantity": 18000000,
        "totalWeeklyTradeCount": 50000,
        "summaryTypeCode": "OTC_W_SMBL",
    },
]

# Pre-tagged rows for _raw_to_records testing
_SAMPLE_DP_RESPONSE = [
    {
        "weekStartDate": "2024-01-08",
        "issueSymbolIdentifier": "GME",
        "totalWeeklyShareQuantity": 15000000,
        "totalWeeklyTradeCount": 42000,
        "lastUpdateDate": "2024-01-14",
        "_source": "otc",
    },
    {
        "weekStartDate": "2024-01-15",
        "issueSymbolIdentifier": "GME",
        "totalWeeklyShareQuantity": 18000000,
        "totalWeeklyTradeCount": 50000,
        "lastUpdateDate": "2024-01-21",
        "_source": "otc",
    },
]

_SAMPLE_DP_MULTI_DATE_FORMAT = [
    {
        "weekEndDate": "01/12/2024",
        "totalWeeklyShareQuantity": 15000000,
        "_source": "otc",
    },
    {
        "weekEndDate": "20240119",
        "totalWeeklyShareQuantity": 18000000,
        "_source": "otc",
    },
]

_SAMPLE_DP_DUPLICATE_WEEKS = [
    {
        "weekEndDate": "2024-01-12",
        "totalWeeklyShareQuantity": 8000000,
        "_source": "otc",
    },
    {
        "weekEndDate": "2024-01-12",
        "totalWeeklyShareQuantity": 7000000,
        "_source": "otc",
    },
]

# Mixed OTC + ATS rows for aggregation testing
_SAMPLE_MIXED_OTC_ATS = [
    {
        "weekStartDate": "2024-01-08",
        "totalWeeklyShareQuantity": 15000000,
        "_source": "otc",
    },
    {
        "weekStartDate": "2024-01-08",
        "totalWeeklyShareQuantity": 5000000,
        "_source": "ats",
    },
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(
    tmp_path: Path,
    client_id: str = "test_id",
    client_secret: str = "test_secret",
) -> FinraDarkPoolClient:
    return FinraDarkPoolClient(
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
    resp.text = text or (json.dumps(json_data) if json_data else "")
    return resp


# ------------------------------------------------------------------
# Tests: Date Normalization
# ------------------------------------------------------------------


class TestNormalizeDateDP:
    """Tests for the dark pool _normalize_date helper."""

    def test_iso_format_passthrough(self) -> None:
        assert _normalize_date("2024-01-12") == "2024-01-12"

    def test_slash_format(self) -> None:
        assert _normalize_date("01/12/2024") == "2024-01-12"

    def test_compact_format(self) -> None:
        assert _normalize_date("20240112") == "2024-01-12"

    def test_empty_string(self) -> None:
        assert _normalize_date("") == ""

    def test_unrecognizable_returns_empty(self) -> None:
        assert _normalize_date("Week of Jan 12") == ""


# ------------------------------------------------------------------
# Tests: OAuth2 Authentication
# ------------------------------------------------------------------


class TestDPAuthentication:
    """Tests for FINRA OAuth2 authentication in dark pool client."""

    def test_successful_auth_sets_bearer_token(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_dark_pool_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)
            result = client._authenticate()

        assert result is True
        assert client._access_token == "dp-token-xyz789"
        assert client._session.headers["Authorization"] == "Bearer dp-token-xyz789"

    def test_auth_sends_correct_basic_header(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, client_id="cid", client_secret="csec")

        with patch("stockdownloader.data.finra_dark_pool_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)
            client._authenticate()

        call_kwargs = mock_post.call_args
        auth_header = call_kwargs.kwargs.get("headers", {}).get("Authorization", "")
        expected = base64.b64encode(b"cid:csec").decode()
        assert auth_header == f"Basic {expected}"

    def test_auth_failure_no_credentials(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, client_id="", client_secret="")
        result = client._authenticate()
        assert result is False

    def test_auth_failure_bad_status(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_dark_pool_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                {"error": "forbidden"}, status_code=403, text="Forbidden",
            )
            result = client._authenticate()

        assert result is False

    def test_auth_failure_network_exception(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_dark_pool_client.requests.post") as mock_post:
            mock_post.side_effect = ConnectionError("network down")
            result = client._authenticate()

        assert result is False

    def test_auth_failure_missing_token_in_response(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with patch("stockdownloader.data.finra_dark_pool_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"status": "ok"})
            result = client._authenticate()

        assert result is False


# ------------------------------------------------------------------
# Tests: Auto-auth before first query
# ------------------------------------------------------------------


class TestDPAutoAuth:
    """Tests that authentication happens automatically."""

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    @patch("stockdownloader.data.finra_dark_pool_client.requests.post")
    def test_fetch_triggers_auth_when_no_token(
        self, mock_post: MagicMock, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        assert client._access_token is None

        mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(_SAMPLE_WEEKLY_RESPONSE),
        ):
            records = client.fetch_dark_pool_volume("GME")

        assert client._access_token == "dp-token-xyz789"
        # 4 rows aggregate into 2 weeks
        assert len(records) == 2


# ------------------------------------------------------------------
# Tests: API Query
# ------------------------------------------------------------------


class TestDPQueryApi:
    """Tests for _query_api."""

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    def test_successful_query(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(_SAMPLE_WEEKLY_RESPONSE),
        ):
            raw = client._query_api("GME")

        assert raw is not None
        assert len(raw) == 4  # 2 OTC + 2 ATS rows

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    def test_query_tags_rows_by_summary_type(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(_SAMPLE_WEEKLY_RESPONSE),
        ):
            raw = client._query_api("GME")

        assert raw is not None
        ats_rows = [r for r in raw if r["_source"] == "ats"]
        otc_rows = [r for r in raw if r["_source"] == "otc"]
        assert len(ats_rows) == 2
        assert len(otc_rows) == 2

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    def test_query_retries_on_failure(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "post",
            side_effect=[
                _mock_response(None, status_code=503, text="Unavailable"),
                _mock_response(None, status_code=503, text="Unavailable"),
                _mock_response(_SAMPLE_WEEKLY_RESPONSE),
            ],
        ):
            raw = client._query_api("GME")

        assert raw is not None
        assert len(raw) == 4

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    def test_query_returns_none_after_max_retries(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(None, status_code=500, text="Error"),
        ):
            raw = client._query_api("GME")

        assert raw is None

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    def test_query_uses_domain_filters(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """Verify payload uses domainFilters with issueSymbolIdentifier."""
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
        field_names = [f["fieldName"] for f in filters]
        assert "issueSymbolIdentifier" in field_names
        assert "summaryTypeCode" in field_names

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    def test_query_requests_ats_and_otc_types(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """Verify payload requests both ATS_W_SMBL and OTC_W_SMBL."""
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
        type_filter = [
            f for f in payload["domainFilters"]
            if f["fieldName"] == "summaryTypeCode"
        ]
        assert len(type_filter) == 1
        assert set(type_filter[0]["values"]) == {"ATS_W_SMBL", "OTC_W_SMBL"}

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    def test_pagination_stops_on_empty(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """Verify pagination stops when API returns fewer than page_size."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(_SAMPLE_WEEKLY_RESPONSE),
        ) as mock_post:
            raw = client._query_api("GME")

        assert raw is not None
        assert len(raw) == 4
        # 4 < 5000 (page_size), so only 1 call needed
        assert mock_post.call_count == 1


# ------------------------------------------------------------------
# Tests: Parsing raw JSON to records
# ------------------------------------------------------------------


class TestDPRawToRecords:
    """Tests for _raw_to_records static method."""

    def test_parses_iso_dates(self) -> None:
        records = FinraDarkPoolClient._raw_to_records(
            _SAMPLE_DP_RESPONSE, "GME",
        )
        assert len(records) == 2
        assert records[0].week_ending == "2024-01-08"
        assert records[1].week_ending == "2024-01-15"

    def test_parses_volume_correctly(self) -> None:
        records = FinraDarkPoolClient._raw_to_records(
            _SAMPLE_DP_RESPONSE, "GME",
        )
        rec = records[0]
        assert rec.symbol == "GME"
        assert rec.total_weekly_volume == 15_000_000
        assert rec.otc_volume == 15_000_000
        assert rec.ats_volume == 0
        assert rec.ats_pct == 0.0

    def test_parses_alternative_date_formats(self) -> None:
        records = FinraDarkPoolClient._raw_to_records(
            _SAMPLE_DP_MULTI_DATE_FORMAT, "GME",
        )
        assert len(records) == 2
        assert records[0].week_ending == "2024-01-12"
        assert records[1].week_ending == "2024-01-19"

    def test_aggregates_duplicate_weeks(self) -> None:
        records = FinraDarkPoolClient._raw_to_records(
            _SAMPLE_DP_DUPLICATE_WEEKS, "GME",
        )
        assert len(records) == 1
        # 8M + 7M = 15M
        assert records[0].total_weekly_volume == 15_000_000

    def test_records_sorted_ascending(self) -> None:
        reversed_data = list(reversed(_SAMPLE_DP_RESPONSE))
        records = FinraDarkPoolClient._raw_to_records(
            reversed_data, "GME",
        )
        dates = [r.week_ending for r in records]
        assert dates == sorted(dates)

    def test_skips_rows_with_bad_dates(self) -> None:
        bad_data = [
            {"weekEndDate": "", "totalWeeklyShareQuantity": 1000, "_source": "otc"},
            {"weekEndDate": "bad_date", "totalWeeklyShareQuantity": 2000, "_source": "otc"},
            {"weekEndDate": "2024-01-12", "totalWeeklyShareQuantity": 3000, "_source": "otc"},
        ]
        records = FinraDarkPoolClient._raw_to_records(bad_data, "GME")
        assert len(records) == 1
        assert records[0].total_weekly_volume == 3000

    def test_empty_response(self) -> None:
        records = FinraDarkPoolClient._raw_to_records([], "GME")
        assert records == []

    def test_aggregates_otc_and_ats_for_same_week(self) -> None:
        """OTC + ATS rows for the same week are properly separated."""
        records = FinraDarkPoolClient._raw_to_records(
            _SAMPLE_MIXED_OTC_ATS, "GME",
        )
        assert len(records) == 1
        rec = records[0]
        assert rec.total_weekly_volume == 20_000_000  # 15M + 5M
        assert rec.otc_volume == 15_000_000
        assert rec.ats_volume == 5_000_000
        assert rec.ats_pct == pytest.approx(0.25)  # 5M / 20M

    def test_rows_without_source_default_to_otc(self) -> None:
        """Rows missing _source tag are treated as OTC."""
        data = [
            {
                "weekStartDate": "2024-01-08",
                "totalWeeklyShareQuantity": 10000000,
                # no _source key
            },
        ]
        records = FinraDarkPoolClient._raw_to_records(data, "GME")
        assert len(records) == 1
        assert records[0].otc_volume == 10_000_000
        assert records[0].ats_volume == 0


# ------------------------------------------------------------------
# Tests: Caching (save + load)
# ------------------------------------------------------------------


class TestDPCaching:
    """Tests for cache save/load."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = [
            DarkPoolRecord(
                week_ending="2024-01-12",
                symbol="GME",
                total_weekly_volume=15_000_000,
                ats_volume=5_000_000,
                otc_volume=10_000_000,
                ats_pct=0.333,
            ),
            DarkPoolRecord(
                week_ending="2024-01-19",
                symbol="GME",
                total_weekly_volume=18_000_000,
                ats_volume=6_000_000,
                otc_volume=12_000_000,
                ats_pct=0.333,
            ),
        ]

        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].week_ending == "2024-01-12"
        assert loaded[1].total_weekly_volume == 18_000_000
        assert loaded[0].ats_volume == 5_000_000

    def test_csv_roundtrip_preserves_types(self, tmp_path: Path) -> None:
        """CSV save/load preserves int and float field types."""
        client = _make_client(tmp_path)
        records = [
            DarkPoolRecord(
                week_ending="2024-01-12",
                symbol="GME",
                total_weekly_volume=15_000_000,
                ats_volume=5_000_000,
                otc_volume=10_000_000,
                ats_pct=0.333,
            ),
        ]
        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        rec = loaded[0]
        assert isinstance(rec.total_weekly_volume, int)
        assert isinstance(rec.ats_volume, int)
        assert isinstance(rec.otc_volume, int)
        assert isinstance(rec.ats_pct, float)
        assert rec.ats_pct == pytest.approx(0.333)

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        loaded = client._load_cache("UNKNOWN")
        assert loaded is None

    def test_load_corrupt_csv_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "CORRUPT"
        sym_dir.mkdir(parents=True, exist_ok=True)
        cache_file = sym_dir / "dark_pool.csv"
        cache_file.write_text("not,valid,csv,header\n\x00\x01\x02", encoding="utf-8")
        loaded = client._load_cache("CORRUPT")
        assert loaded is None

    def test_save_cache_creates_csv(self, tmp_path: Path) -> None:
        """_save_cache writes dark_pool.csv (not .json)."""
        client = _make_client(tmp_path)
        records = [
            DarkPoolRecord(
                week_ending="2024-01-12",
                symbol="AAPL",
                total_weekly_volume=100,
                ats_volume=50,
                otc_volume=50,
                ats_pct=0.5,
            ),
        ]
        client._save_cache("AAPL", records)
        assert (client._data_dir / "AAPL" / "dark_pool.csv").exists()
        assert not (client._data_dir / "AAPL" / "dark_pool.json").exists()

    def test_save_creates_symbol_subdir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = [
            DarkPoolRecord(
                week_ending="2024-01-12",
                symbol="TSLA",
                total_weekly_volume=200,
                ats_volume=100,
                otc_volume=100,
                ats_pct=0.5,
            ),
        ]
        client._save_cache("TSLA", records)
        sym_dir = client._data_dir / "TSLA"
        assert sym_dir.is_dir()
        assert (sym_dir / "dark_pool.csv").exists()

    def test_json_fallback_migration(self, tmp_path: Path) -> None:
        """_load_cache migrates a dark_pool.json file to CSV and returns records."""
        client = _make_client(tmp_path)
        # Write a JSON cache file (the old format)
        import json as _json
        sym_dir = client._data_dir / "NVDA"
        sym_dir.mkdir(parents=True, exist_ok=True)
        json_path = sym_dir / "dark_pool.json"
        json_path.write_text(
            _json.dumps([{
                "week_ending": "2024-01-12",
                "symbol": "NVDA",
                "total_weekly_volume": 500,
                "ats_volume": 200,
                "otc_volume": 300,
                "ats_pct": 0.4,
            }]),
            encoding="utf-8",
        )

        loaded = client._load_cache("NVDA")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].symbol == "NVDA"
        assert loaded[0].ats_pct == pytest.approx(0.4)
        # CSV file should now exist (migration result)
        assert (sym_dir / "dark_pool.csv").exists()

    def test_json_fallback_migration_preserves_default_ats_pct(
        self, tmp_path: Path,
    ) -> None:
        """JSON fallback uses .get('ats_pct', 0.0) for records missing ats_pct."""
        client = _make_client(tmp_path)
        import json as _json
        sym_dir = client._data_dir / "META"
        sym_dir.mkdir(parents=True, exist_ok=True)
        json_path = sym_dir / "dark_pool.json"
        # Intentionally omit ats_pct from JSON data
        json_path.write_text(
            _json.dumps([{
                "week_ending": "2024-02-01",
                "symbol": "META",
                "total_weekly_volume": 1000,
                "ats_volume": 400,
                "otc_volume": 600,
            }]),
            encoding="utf-8",
        )

        loaded = client._load_cache("META")
        assert loaded is not None
        assert loaded[0].ats_pct == 0.0

    def test_legacy_migration_on_load(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = [
            DarkPoolRecord(
                week_ending="2024-01-12",
                symbol="MSFT",
                total_weekly_volume=300,
                ats_volume=100,
                otc_volume=200,
                ats_pct=0.333,
            ),
        ]
        # Write legacy flat file
        import json as _json
        legacy_file = client._data_dir / "cache" / "dark_pool" / "MSFT_dp.json"
        legacy_file.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.write_text(
            _json.dumps([{
                "week_ending": r.week_ending,
                "symbol": r.symbol,
                "total_weekly_volume": r.total_weekly_volume,
                "ats_volume": r.ats_volume,
                "otc_volume": r.otc_volume,
                "ats_pct": r.ats_pct,
            } for r in records]),
            encoding="utf-8",
        )
        assert legacy_file.exists()

        loaded = client._load_cache("MSFT")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].symbol == "MSFT"
        # Legacy file should have been moved to JSON path
        assert not legacy_file.exists()
        assert (client._data_dir / "MSFT" / "dark_pool.json").exists()
        # And CSV should now exist after migration
        assert (client._data_dir / "MSFT" / "dark_pool.csv").exists()


# ------------------------------------------------------------------
# Tests: Fallback to cache when API fails
# ------------------------------------------------------------------


class TestDPFallbackToCache:
    """Tests for falling back to cached data when API fails."""

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    @patch("stockdownloader.data.finra_dark_pool_client.requests.post")
    def test_falls_back_to_cache_on_api_failure(
        self, mock_post: MagicMock, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        # Pre-populate cache
        cached_records = [
            DarkPoolRecord(
                week_ending="2024-01-05",
                symbol="GME",
                total_weekly_volume=12_000_000,
                ats_volume=4_000_000,
                otc_volume=8_000_000,
                ats_pct=0.333,
            ),
        ]
        client._save_cache("GME", cached_records)

        mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)
        with patch.object(
            client._session, "post",
            return_value=_mock_response(None, status_code=500, text="Error"),
        ):
            records = client.fetch_dark_pool_volume("gme")

        assert len(records) == 1
        assert records[0].week_ending == "2024-01-05"

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    @patch("stockdownloader.data.finra_dark_pool_client.requests.post")
    def test_returns_empty_when_no_api_no_cache(
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
            records = client.fetch_dark_pool_volume("NOSYMBOL")

        assert records == []


# ------------------------------------------------------------------
# Tests: Rate limiting
# ------------------------------------------------------------------


class TestDPRateLimiting:
    """Tests for rate limiting mechanism."""

    def test_rate_limit_sleeps_when_too_fast(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        with patch("stockdownloader.data.finra_dark_pool_client.time") as mock_time:
            mock_time.monotonic.side_effect = [
                0.0,   # first check
                0.0,   # set _last_request_time
                0.1,   # second check — 100ms elapsed (< 500ms)
                0.5,   # set _last_request_time after sleep
            ]
            mock_time.sleep = MagicMock()

            client._last_request_time = 0.0
            client._rate_limit()
            client._rate_limit()

            mock_time.sleep.assert_called()


# ------------------------------------------------------------------
# Tests: Symbol normalization
# ------------------------------------------------------------------


class TestDPSymbolNormalization:
    """Tests that symbols are normalized to uppercase."""

    @patch("stockdownloader.data.finra_dark_pool_client.time")
    @patch("stockdownloader.data.finra_dark_pool_client.requests.post")
    def test_fetch_uppercases_symbol(
        self, mock_post: MagicMock, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)
        mock_post.return_value = _mock_response(_SAMPLE_TOKEN_RESPONSE)

        with patch.object(
            client._session, "post",
            return_value=_mock_response(_SAMPLE_OTC_RESPONSE),
        ):
            records = client.fetch_dark_pool_volume("gme")

        assert all(r.symbol == "GME" for r in records)
