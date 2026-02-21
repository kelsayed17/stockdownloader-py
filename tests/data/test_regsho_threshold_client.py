"""Unit tests for RegShoThresholdClient — caching, OCC, per-symbol subdirectories."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.regsho_threshold_client import (
    RegShoThresholdClient,
    ThresholdRecord,
    _OCC_URL_TEMPLATE,
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
            market="FINRA",
            threshold_shares=15000,
            consecutive_days=3,
        ),
        ThresholdRecord(
            date="2024-06-11",
            symbol=symbol,
            market="FINRA",
            threshold_shares=16000,
            consecutive_days=4,
        ),
    ]


# ------------------------------------------------------------------
# Tests: Cache file naming (per-symbol subdirectory)
# ------------------------------------------------------------------


class TestThresholdCacheFileNaming:
    """Tests that cache uses per-symbol subdirectories."""

    def test_cache_file_naming(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_cache("AAPL", _sample_records("AAPL"))
        assert (client._data_dir / "AAPL" / "regsho_threshold.json").exists()

    def test_save_creates_symbol_subdir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_cache("TSLA", _sample_records("TSLA"))
        sym_dir = client._data_dir / "TSLA"
        assert sym_dir.is_dir()
        assert (sym_dir / "regsho_threshold.json").exists()


# ------------------------------------------------------------------
# Tests: Save and load roundtrip
# ------------------------------------------------------------------


class TestThresholdCacheRoundtrip:
    """Tests for save/load roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = _sample_records()
        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].date == "2024-06-10"
        assert loaded[1].threshold_shares == 16000

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        loaded = client._load_cache("NOSYMBOL")
        assert loaded is None

    def test_load_corrupt_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        (sym_dir / "regsho_threshold.json").write_text("{{invalid", encoding="utf-8")
        loaded = client._load_cache("BAD")
        assert loaded is None


# ------------------------------------------------------------------
# Tests: Legacy migration
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
        # Legacy file should have been moved
        assert not legacy_file.exists()
        assert (client._data_dir / "MSFT" / "regsho_threshold.json").exists()


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

    def test_occ_url_template(self) -> None:
        """Verify the OCC URL template produces correct URLs."""
        url = _OCC_URL_TEMPLATE.format(date="20260220")
        assert url == (
            "https://marketdata.theocc.com/threshold-securities"
            "?reportDate=20260220"
        )

    def test_occ_parses_symbol(self, tmp_path: Path) -> None:
        """OCC response correctly identifies symbol on threshold list."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _OCC_SAMPLE_RESPONSE

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_occ("GME", lookback_days=1)

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
            records = client._query_occ("TSLA", lookback_days=1)

        assert len(records) == 0

    def test_occ_404_not_counted_as_failure(self, tmp_path: Path) -> None:
        """OCC 404 responses should not increment failure counter."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_occ("GME", lookback_days=5)

        # Should not stop early due to failures
        assert records == []

    def test_occ_market_category_preserved(self, tmp_path: Path) -> None:
        """OCC market category is used as the record market field."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _OCC_SAMPLE_RESPONSE

        with patch.object(client._session, "get", return_value=mock_resp):
            records = client._query_occ("SPXS", lookback_days=1)

        assert len(records) >= 1
        assert records[0].market == "NYSE Arca"


class TestOccFallback:
    """Tests that OCC is tried first, NYSE/Nasdaq as fallback."""

    def test_occ_results_skip_nyse_nasdaq(self, tmp_path: Path) -> None:
        """When OCC returns results, NYSE/Nasdaq are not queried."""
        client = _make_client(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _OCC_SAMPLE_RESPONSE

        with (
            patch.object(
                client, "_query_occ",
                return_value=[ThresholdRecord(
                    date="2024-06-10", symbol="GME",
                    market="NYSE", threshold_shares=0, consecutive_days=0,
                )],
            ) as mock_occ,
            patch.object(client, "_query_nyse") as mock_nyse,
            patch.object(client, "_query_nasdaq") as mock_nasdaq,
        ):
            records = client.fetch_threshold_status("GME", lookback_days=5)

        mock_occ.assert_called_once()
        mock_nyse.assert_not_called()
        mock_nasdaq.assert_not_called()
        assert len(records) >= 1

    def test_empty_occ_triggers_nyse_nasdaq(self, tmp_path: Path) -> None:
        """When OCC returns nothing, NYSE and Nasdaq are queried."""
        client = _make_client(tmp_path)

        with (
            patch.object(client, "_query_occ", return_value=[]) as mock_occ,
            patch.object(
                client, "_query_nyse", return_value=[],
            ) as mock_nyse,
            patch.object(
                client, "_query_nasdaq", return_value=[],
            ) as mock_nasdaq,
        ):
            client.fetch_threshold_status("GME", lookback_days=5)

        mock_occ.assert_called_once()
        mock_nyse.assert_called_once()
        mock_nasdaq.assert_called_once()
