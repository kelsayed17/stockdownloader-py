"""Unit tests for RegShoThresholdClient — caching, OCC, NYSE, Nasdaq, CBOE."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.regsho_threshold_client import (
    RegShoThresholdClient,
    ThresholdRecord,
)
from stockdownloader.data.regsho_sources import (
    CBOE_URL_TEMPLATE,
    NASDAQ_URL_TEMPLATE,
    NYSE_URL_TEMPLATE,
    OCC_URL_TEMPLATE,
    cffi_get,
    curl_get,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(tmp_path: Path) -> RegShoThresholdClient:
    return RegShoThresholdClient(
        client_id="test_id",
        client_secret="test_secret",
        data_dir=str(tmp_path),
    )


def _sample_records(symbol: str = "GME") -> list[ThresholdRecord]:
    return [
        ThresholdRecord(
            date="2024-06-10",
            symbol=symbol,
            market="NYSE",
            threshold_shares=15000,
            consecutive_days=3,
        ),
        ThresholdRecord(
            date="2024-06-11",
            symbol=symbol,
            market="NYSE",
            threshold_shares=16000,
            consecutive_days=4,
        ),
    ]


# ------------------------------------------------------------------
# Tests: Cache file naming (per-symbol subdirectory, CSV)
# ------------------------------------------------------------------


class TestThresholdCacheFileNaming:
    """Tests that cache uses per-symbol subdirectories with CSV format."""

    def test_cache_file_naming(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_cache("AAPL", _sample_records("AAPL"))
        assert (client._data_dir / "AAPL" / "regsho_threshold.csv").exists()

    def test_save_creates_symbol_subdir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_cache("TSLA", _sample_records("TSLA"))
        sym_dir = client._data_dir / "TSLA"
        assert sym_dir.is_dir()
        assert (sym_dir / "regsho_threshold.csv").exists()


# ------------------------------------------------------------------
# Tests: Save and load roundtrip (CSV)
# ------------------------------------------------------------------


class TestThresholdCacheRoundtrip:
    """Tests for save/load roundtrip with CSV format."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = _sample_records()
        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].date == "2024-06-10"
        assert loaded[1].threshold_shares == 16000

    def test_csv_roundtrip_preserves_all_fields(self, tmp_path: Path) -> None:
        """CSV save/load roundtrip preserves all ThresholdRecord fields."""
        client = _make_client(tmp_path)
        records = _sample_records()
        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        for orig, back in zip(records, loaded):
            assert orig.date == back.date
            assert orig.symbol == back.symbol
            assert orig.market == back.market
            assert orig.threshold_shares == back.threshold_shares
            assert orig.consecutive_days == back.consecutive_days

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        loaded = client._load_cache("NOSYMBOL")
        assert loaded is None

    def test_load_corrupt_csv_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        (sym_dir / "regsho_threshold.csv").write_text(
            "bad,header\nno,data", encoding="utf-8",
        )
        loaded = client._load_cache("BAD")
        assert loaded is None


# ------------------------------------------------------------------
# Tests: JSON fallback migration
# ------------------------------------------------------------------


class TestThresholdJsonFallbackMigration:
    """Tests that existing JSON caches are migrated to CSV on load."""

    def test_json_fallback_migrates_to_csv(self, tmp_path: Path) -> None:
        """Loading from JSON auto-migrates to CSV."""
        client = _make_client(tmp_path)
        records = _sample_records("GME")

        # Write JSON at the standard location
        sym_dir = client._data_dir / "GME"
        sym_dir.mkdir(parents=True, exist_ok=True)
        json_path = sym_dir / "regsho_threshold.json"
        json_path.write_text(
            json.dumps([asdict(r) for r in records]),
            encoding="utf-8",
        )

        loaded = client._load_cache("GME")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].symbol == "GME"
        # CSV should now exist
        assert (sym_dir / "regsho_threshold.csv").exists()

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        """Corrupt JSON file returns None."""
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        (sym_dir / "regsho_threshold.json").write_text(
            "{{invalid", encoding="utf-8",
        )
        loaded = client._load_cache("BAD")
        assert loaded is None


# ------------------------------------------------------------------
# Tests: Legacy migration (old cache/regsho/ paths)
# ------------------------------------------------------------------


class TestThresholdLegacyMigration:
    """Tests that legacy flat cache files are migrated on load."""

    def test_legacy_migration_on_load(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = _sample_records("MSFT")

        # Write legacy flat file at the old location
        legacy_file = client._data_dir / "cache" / "regsho" / "MSFT_threshold.json"
        legacy_file.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.write_text(
            json.dumps([asdict(r) for r in records]),
            encoding="utf-8",
        )
        assert legacy_file.exists()

        loaded = client._load_cache("MSFT")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].symbol == "MSFT"
        # Legacy file should have been moved to standard JSON location
        # then migrated to CSV
        assert not legacy_file.exists()
        assert (client._data_dir / "MSFT" / "regsho_threshold.csv").exists()


# ------------------------------------------------------------------
# Tests: NYSE progress in .progress/ directory
# ------------------------------------------------------------------


class TestNyseProgressDir:
    """Tests that NYSE progress files live under .progress/."""

    def test_save_and_load_progress(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_nyse_progress("GME", {"2021-01-25", "2021-01-26"})
        loaded = client._load_nyse_progress("GME")
        assert loaded == {"2021-01-25", "2021-01-26"}

    def test_progress_file_in_progress_dir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_nyse_progress("GME", {"2021-01-25"})
        progress_file = client._data_dir / "GME" / ".progress" / "regsho_nyse.json"
        assert progress_file.exists()

    def test_legacy_progress_migrated(self, tmp_path: Path) -> None:
        """Legacy regsho_nyse_progress.json is migrated to .progress/."""
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "GME"
        sym_dir.mkdir(parents=True, exist_ok=True)
        legacy = sym_dir / "regsho_nyse_progress.json"
        legacy.write_text(
            json.dumps(["2021-01-25", "2021-01-26"]),
            encoding="utf-8",
        )

        loaded = client._load_nyse_progress("GME")
        assert loaded == {"2021-01-25", "2021-01-26"}
        # Legacy file should be gone
        assert not legacy.exists()
        # New location should exist
        assert (sym_dir / ".progress" / "regsho_nyse.json").exists()

    def test_load_nonexistent_progress(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        loaded = client._load_nyse_progress("NOSYMBOL")
        assert loaded == set()


# ------------------------------------------------------------------
# Tests: URL templates
# ------------------------------------------------------------------


class TestUrlTemplates:
    """Tests that URL templates produce correct URLs."""

    def test_nyse_url_template(self) -> None:
        url = NYSE_URL_TEMPLATE.format(date="2021-01-25")
        assert url == (
            "https://www.nyse.com/api/regulatory/threshold-securities/"
            "download?selectedDate=2021-01-25"
        )

    def test_nasdaq_url_template(self) -> None:
        url = NASDAQ_URL_TEMPLATE.format(date="20210125")
        assert url == (
            "https://www.nasdaqtrader.com/dynamic/symdir/regsho/"
            "nasdaqth20210125.txt"
        )

    def test_occ_url_template(self) -> None:
        url = OCC_URL_TEMPLATE.format(date="20260220")
        assert url == (
            "https://marketdata.theocc.com/threshold-securities"
            "?reportDate=20260220"
        )

    def test_cboe_url_template(self) -> None:
        url = CBOE_URL_TEMPLATE.format(date="2021-01-25")
        assert url == (
            "https://www.cboe.com/us/equities/market_statistics/"
            "reg_sho_threshold/2021-01-25/csv/"
        )


# ------------------------------------------------------------------
# Tests: NYSE query (uses curl_cffi / curl subprocess)
# ------------------------------------------------------------------

_NYSE_SAMPLE_RESPONSE = (
    "Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Filler|Filler\n"
    "ATAC|Altimar Acquisition Corporation|NYSE|Y||\n"
    "GME|GameStop Corp. Class A|NYSE|Y||\n"
    "FUBO|fuboTV Inc.|NYSE|Y||\n"
    "DFAE|Dimensional Emerging Core|NYSE Arca|Y||\n"
    "20210125210501\n"
)

_NYSE_HEADER_ONLY = (
    "Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Filler|Filler\n"
    "20210104210500\n"
)


class TestNyseQuery:
    """Tests for NYSE threshold list query (uses curl_cffi)."""

    def test_nyse_parses_symbol_with_market(self, tmp_path: Path) -> None:
        """NYSE response correctly parses symbol and market category."""
        client = _make_client(tmp_path)

        with patch(
            "stockdownloader.data.regsho_sources.cffi_get",
            return_value=(_NYSE_SAMPLE_RESPONSE, 200),
        ):
            records = client._query_nyse("GME", lookback_days=7)

        assert len(records) >= 1
        assert records[0].symbol == "GME"
        assert records[0].market == "NYSE"

    def test_nyse_arca_market_category(self, tmp_path: Path) -> None:
        """NYSE Arca market category is preserved."""
        client = _make_client(tmp_path)

        with patch(
            "stockdownloader.data.regsho_sources.cffi_get",
            return_value=(_NYSE_SAMPLE_RESPONSE, 200),
        ):
            records = client._query_nyse("DFAE", lookback_days=7)

        assert len(records) >= 1
        assert records[0].market == "NYSE Arca"

    def test_nyse_not_found_returns_empty(self, tmp_path: Path) -> None:
        """Symbol not in NYSE response returns empty list."""
        client = _make_client(tmp_path)

        with patch(
            "stockdownloader.data.regsho_sources.cffi_get",
            return_value=(_NYSE_SAMPLE_RESPONSE, 200),
        ):
            records = client._query_nyse("TSLA", lookback_days=7)

        assert len(records) == 0

    def test_nyse_header_only_skipped(self, tmp_path: Path) -> None:
        """Header-only NYSE response (no threshold securities) is handled."""
        client = _make_client(tmp_path)

        with patch(
            "stockdownloader.data.regsho_sources.cffi_get",
            return_value=(_NYSE_HEADER_ONLY, 200),
        ):
            records = client._query_nyse("GME", lookback_days=7)

        assert len(records) == 0

    def test_nyse_cloudflare_backoff(self, tmp_path: Path) -> None:
        """NYSE 403 triggers backoff, not immediate stop."""
        client = _make_client(tmp_path)

        # Some requests fail with 403, others succeed — should keep going
        side_effects = [("", 403), (_NYSE_SAMPLE_RESPONSE, 200)] * 5
        with patch(
            "stockdownloader.data.regsho_sources.cffi_get",
            side_effect=side_effects,
        ):
            records = client._query_nyse("GME", lookback_days=14)

        # Should still get records from successful requests
        assert len(records) >= 1

    def test_nyse_skips_cached_dates(self, tmp_path: Path) -> None:
        """NYSE query skips dates already in cache."""
        client = _make_client(tmp_path)

        call_count = 0
        original_return = (_NYSE_SAMPLE_RESPONSE, 200)

        def counting_cffi_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_return

        # Provide cached_dates that cover all weekdays in the lookback
        from datetime import date as dt_date, timedelta as td
        today = dt_date.today()
        cached = set()
        for i in range(7):
            d = today - td(days=i)
            if d.weekday() < 5:
                cached.add(d.strftime("%Y-%m-%d"))

        with patch(
            "stockdownloader.data.regsho_sources.cffi_get",
            side_effect=counting_cffi_get,
        ):
            records = client._query_nyse(
                "GME", lookback_days=7, cached_dates=cached,
            )

        # No HTTP requests should have been made — all dates cached
        assert call_count == 0
        assert len(records) == 0

    def test_cffi_get_uses_curl_session(self, tmp_path: Path) -> None:
        """_cffi_get uses curl_cffi session when available."""
        client = _make_client(tmp_path)

        mock_resp = MagicMock()
        mock_resp.text = "test body"
        mock_resp.status_code = 200

        if client._curl_session is not None:
            with patch.object(
                client._curl_session, "get", return_value=mock_resp,
            ):
                text, status = client._cffi_get("https://example.com")
                assert status == 200
                assert text == "test body"

    def test_cffi_get_fallback_to_curl(self, tmp_path: Path) -> None:
        """_cffi_get falls back to subprocess curl when curl_cffi unavailable."""
        client = _make_client(tmp_path)
        client._curl_session = None  # Simulate curl_cffi not available

        with patch(
            "stockdownloader.data.regsho_sources.curl_get",
            return_value=("fallback body", 200),
        ):
            text, status = client._cffi_get("https://example.com")
            assert status == 200
            assert text == "fallback body"

    def test_curl_get_static_method(self) -> None:
        """curl_get parses curl output correctly."""
        # This tests the actual curl integration (requires curl binary)
        body, status = curl_get(
            "https://httpbin.org/status/200", timeout=5,
        )
        assert status == 200


# ------------------------------------------------------------------
# Tests: OCC combined endpoint
# ------------------------------------------------------------------

_OCC_SAMPLE_RESPONSE = (
    "Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Flag2|\n"
    "AAPU|DIREXION SHS ETF TR DAILY AAPL|G|Y|N|\n"
    "AIM|AIM ImmunoTech Inc.|NYSE American|Y||\n"
    "GME|GameStop Corp New|NYSE|Y||\n"
    "SPXS|Direxion Daily S&P 500 Bear 3x|NYSE Arca|Y||\n"
)


class TestOccQuery:
    """Tests for the OCC combined threshold list query."""

    def test_occ_parses_symbol(self, tmp_path: Path) -> None:
        """OCC response correctly identifies symbol on threshold list."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _OCC_SAMPLE_RESPONSE

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_occ("GME", lookback_days=7)

        assert len(records) >= 1
        assert records[0].symbol == "GME"
        assert records[0].market == "NYSE"

    def test_occ_not_found_returns_empty(self, tmp_path: Path) -> None:
        """Symbol not in OCC response returns empty list."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _OCC_SAMPLE_RESPONSE

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_occ("TSLA", lookback_days=7)

        assert len(records) == 0

    def test_occ_file_not_exist_counted_as_failure(self, tmp_path: Path) -> None:
        """OCC 'File does not exist' message increments failure counter."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "File requested does not exist."

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_occ("GME", lookback_days=5)

        assert records == []

    def test_occ_404_not_counted_as_failure(self, tmp_path: Path) -> None:
        """OCC 404 responses should not increment failure counter."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_occ("GME", lookback_days=5)

        assert records == []

    def test_occ_market_category_preserved(self, tmp_path: Path) -> None:
        """OCC market category is used as the record market field."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _OCC_SAMPLE_RESPONSE

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_occ("SPXS", lookback_days=7)

        assert len(records) >= 1
        assert records[0].market == "NYSE Arca"

    def test_occ_lookback_capped(self, tmp_path: Path) -> None:
        """OCC lookback is capped at 60 days even if larger requested."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        call_count = 0

        def counting_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp

        with patch.object(client._session, "get", side_effect=counting_get):
            client._query_occ("GME", lookback_days=5000)

        # Should be capped at 60 days worth of weekday checks
        # 60 calendar days = ~42 weekdays, but we can't predict exact count
        # due to weekend placement.  Just verify it's much less than 5000.
        assert call_count <= 45


# ------------------------------------------------------------------
# Tests: CBOE query
# ------------------------------------------------------------------

_CBOE_SAMPLE_RESPONSE = (
    "Symbol|CompanyName\n"
    "MSTU|T-Rex 2X Long MSTR Daily Target ETF\n"
    "GME|GameStop Corp Class A\n"
    "20210104030216\n"
)


class TestCboeQuery:
    """Tests for the CBOE BZX threshold list query."""

    def test_cboe_parses_symbol(self, tmp_path: Path) -> None:
        """CBOE response correctly identifies symbol."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _CBOE_SAMPLE_RESPONSE

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_cboe("GME", lookback_days=7)

        assert len(records) >= 1
        assert records[0].symbol == "GME"
        assert records[0].market == "CBOE BZX"

    def test_cboe_not_found_returns_empty(self, tmp_path: Path) -> None:
        """Symbol not in CBOE response returns empty."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _CBOE_SAMPLE_RESPONSE

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_cboe("TSLA", lookback_days=7)

        assert len(records) == 0


# ------------------------------------------------------------------
# Tests: Nasdaq query
# ------------------------------------------------------------------

_NASDAQ_SAMPLE_RESPONSE = (
    "Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Rule 3210|Filler\n"
    "ACMR|ACM RESH INC CL A|G|Y|N|\n"
    "GME|GAMESTOP CORP CL A|S|Y|N|\n"
)


class TestNasdaqQuery:
    """Tests for Nasdaq threshold list query."""

    def test_nasdaq_parses_symbol(self, tmp_path: Path) -> None:
        """Nasdaq response correctly identifies symbol."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _NASDAQ_SAMPLE_RESPONSE
        mock_resp.headers = {"Content-Type": "text/plain"}

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_nasdaq("GME", lookback_days=7)

        assert len(records) >= 1
        assert records[0].symbol == "GME"
        assert records[0].market == "NASDAQ (S)"

    def test_nasdaq_302_not_failure(self, tmp_path: Path) -> None:
        """Nasdaq 302 redirect (file doesn't exist) is not a failure."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 302

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_nasdaq("GME", lookback_days=5)

        assert records == []


# ------------------------------------------------------------------
# Tests: fetch_threshold_status (integration of all sources)
# ------------------------------------------------------------------


class TestFetchThresholdStatus:
    """Tests for the top-level fetch_threshold_status method."""

    def test_all_sources_queried(self, tmp_path: Path) -> None:
        """All four sources are queried and results merged."""
        client = _make_client(tmp_path)

        with (
            patch.object(
                client, "_query_nyse",
                return_value=[ThresholdRecord(
                    date="2021-01-25", symbol="GME",
                    market="NYSE", threshold_shares=0, consecutive_days=0,
                )],
            ) as mock_nyse,
            patch.object(
                client, "_query_nasdaq",
                return_value=[],
            ) as mock_nasdaq,
            patch.object(
                client, "_query_cboe",
                return_value=[],
            ) as mock_cboe,
            patch.object(
                client, "_query_occ",
                return_value=[ThresholdRecord(
                    date="2021-01-25", symbol="GME",
                    market="NYSE", threshold_shares=0, consecutive_days=0,
                )],
            ) as mock_occ,
        ):
            records = client.fetch_threshold_status("GME", lookback_days=5)

        # All sources should have been called
        mock_nyse.assert_called_once()
        mock_nasdaq.assert_called_once()
        mock_cboe.assert_called_once()
        mock_occ.assert_called_once()
        # Deduplicated: same date+market from NYSE and OCC = 1 record
        assert len(records) == 1

    def test_results_merged_with_cache(self, tmp_path: Path) -> None:
        """Fresh results are merged with cached historical data."""
        client = _make_client(tmp_path)

        # Pre-populate cache
        historical = [ThresholdRecord(
            date="2020-01-02", symbol="GME",
            market="NYSE", threshold_shares=0, consecutive_days=0,
        )]
        client._save_cache("GME", historical)

        with (
            patch.object(
                client, "_query_nyse",
                return_value=[ThresholdRecord(
                    date="2021-01-25", symbol="GME",
                    market="NYSE", threshold_shares=0, consecutive_days=0,
                )],
            ),
            patch.object(client, "_query_nasdaq", return_value=[]),
            patch.object(client, "_query_cboe", return_value=[]),
            patch.object(client, "_query_occ", return_value=[]),
        ):
            records = client.fetch_threshold_status("GME", lookback_days=5)

        # Should have both cached and fresh records
        assert len(records) == 2
        assert records[0].date == "2020-01-02"
        assert records[1].date == "2021-01-25"

    def test_multi_market_records_preserved(self, tmp_path: Path) -> None:
        """Records from different markets for same date are kept separate."""
        client = _make_client(tmp_path)

        with (
            patch.object(
                client, "_query_nyse",
                return_value=[ThresholdRecord(
                    date="2021-01-25", symbol="GME",
                    market="NYSE", threshold_shares=0, consecutive_days=0,
                )],
            ),
            patch.object(client, "_query_nasdaq", return_value=[]),
            patch.object(client, "_query_cboe", return_value=[]),
            patch.object(
                client, "_query_occ",
                return_value=[ThresholdRecord(
                    date="2021-01-25", symbol="GME",
                    market="NYSE Arca", threshold_shares=0, consecutive_days=0,
                )],
            ),
        ):
            records = client.fetch_threshold_status("GME", lookback_days=5)

        # Different markets = different records
        assert len(records) == 2
        markets = {r.market for r in records}
        assert markets == {"NYSE", "NYSE Arca"}

    def test_empty_all_sources_returns_cache(self, tmp_path: Path) -> None:
        """When all sources return empty, cached data is returned."""
        client = _make_client(tmp_path)

        # Pre-populate cache
        historical = [ThresholdRecord(
            date="2020-01-02", symbol="GME",
            market="NYSE", threshold_shares=0, consecutive_days=0,
        )]
        client._save_cache("GME", historical)

        with (
            patch.object(client, "_query_nyse", return_value=[]),
            patch.object(client, "_query_nasdaq", return_value=[]),
            patch.object(client, "_query_cboe", return_value=[]),
            patch.object(client, "_query_occ", return_value=[]),
        ):
            records = client.fetch_threshold_status("GME", lookback_days=5)

        assert len(records) == 1
        assert records[0].date == "2020-01-02"

    def test_fetch_threshold_targeted_queries_only_remaining(
        self, tmp_path: Path,
    ) -> None:
        """Targeted fetch only queries dates not already in progress file."""
        client = _make_client(tmp_path)

        # Pre-populate progress in .progress/ directory
        progress_dir = client._data_dir / "GME" / ".progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        (progress_dir / "regsho_nyse.json").write_text(
            json.dumps(["2021-01-25", "2021-01-26"]),
            encoding="utf-8",
        )

        call_count = 0

        def counting_cffi_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return (_NYSE_SAMPLE_RESPONSE, 200)

        with patch.object(client, "_cffi_get", side_effect=counting_cffi_get):
            records = client.fetch_threshold_targeted(
                "GME",
                target_dates=["2021-01-25", "2021-01-26", "2021-01-27"],
            )

        # Only 2021-01-27 should be queried (the other two are in progress)
        assert call_count == 1

    def test_fetch_threshold_targeted_saves_incrementally(
        self, tmp_path: Path,
    ) -> None:
        """Targeted fetch saves progress even for dates where symbol not found."""
        client = _make_client(tmp_path)

        with patch.object(
            client, "_cffi_get",
            return_value=(_NYSE_HEADER_ONLY, 200),
        ):
            client.fetch_threshold_targeted(
                "GME",
                target_dates=["2021-01-04", "2021-01-05"],
            )

        # Progress should be saved in .progress/ directory
        progress_file = client._data_dir / "GME" / ".progress" / "regsho_nyse.json"
        assert progress_file.exists()
        progress = set(json.loads(progress_file.read_text(encoding="utf-8")))
        assert "2021-01-04" in progress
        assert "2021-01-05" in progress

    def test_cached_dates_passed_to_nyse(self, tmp_path: Path) -> None:
        """Cached dates are passed to NYSE query for skip optimization."""
        client = _make_client(tmp_path)

        # Pre-populate cache
        historical = [ThresholdRecord(
            date="2020-01-02", symbol="GME",
            market="NYSE", threshold_shares=0, consecutive_days=0,
        )]
        client._save_cache("GME", historical)

        with (
            patch.object(
                client, "_query_nyse", return_value=[],
            ) as mock_nyse,
            patch.object(client, "_query_nasdaq", return_value=[]),
            patch.object(client, "_query_cboe", return_value=[]),
            patch.object(client, "_query_occ", return_value=[]),
        ):
            client.fetch_threshold_status("GME", lookback_days=5)

        # NYSE should receive cached_dates kwarg
        call_kwargs = mock_nyse.call_args
        assert "cached_dates" in call_kwargs.kwargs
        assert "2020-01-02" in call_kwargs.kwargs["cached_dates"]
