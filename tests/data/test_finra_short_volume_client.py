"""Unit tests for FinraShortVolumeClient — caching with per-symbol subdirectories."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from stockdownloader.data.finra_short_volume_client import (
    FinraShortVolumeClient,
    ShortVolumeRecord,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(tmp_path: Path) -> FinraShortVolumeClient:
    return FinraShortVolumeClient(
        client_id="test_id",
        client_secret="test_secret",
        data_dir=str(tmp_path),
    )


def _sample_records(symbol: str = "GME") -> list[ShortVolumeRecord]:
    return [
        ShortVolumeRecord(
            date="2024-06-10",
            symbol=symbol,
            short_volume=5_000_000,
            total_volume=10_000_000,
            short_exempt_volume=50_000,
            short_volume_ratio=0.5,
        ),
        ShortVolumeRecord(
            date="2024-06-11",
            symbol=symbol,
            short_volume=6_000_000,
            total_volume=11_000_000,
            short_exempt_volume=60_000,
            short_volume_ratio=0.545455,
        ),
    ]


# ------------------------------------------------------------------
# Tests: Cache file naming (per-symbol subdirectory)
# ------------------------------------------------------------------


class TestSVCacheFileNaming:
    """Tests that cache uses per-symbol subdirectories with CSV format."""

    def test_cache_file_is_csv(self, tmp_path: Path) -> None:
        """_save_cache creates a .csv file, not .json."""
        client = _make_client(tmp_path)
        client._save_cache("AAPL", _sample_records("AAPL"))
        assert (client._data_dir / "AAPL" / "short_volume.csv").exists()
        assert not (client._data_dir / "AAPL" / "short_volume.json").exists()

    def test_save_creates_symbol_subdir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_cache("TSLA", _sample_records("TSLA"))
        sym_dir = client._data_dir / "TSLA"
        assert sym_dir.is_dir()
        assert (sym_dir / "short_volume.csv").exists()


# ------------------------------------------------------------------
# Tests: Save and load roundtrip
# ------------------------------------------------------------------


class TestSVCacheRoundtrip:
    """Tests for save/load roundtrip using CSV format."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = _sample_records()
        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].date == "2024-06-10"
        assert loaded[1].short_volume == 6_000_000

    def test_csv_roundtrip_preserves_all_fields(self, tmp_path: Path) -> None:
        """Save then load via CSV preserves every field with correct types."""
        client = _make_client(tmp_path)
        records = _sample_records("NVDA")
        client._save_cache("NVDA", records)

        csv_path = client._data_dir / "NVDA" / "short_volume.csv"
        assert csv_path.exists()

        loaded = client._load_cache("NVDA")
        assert loaded is not None
        assert len(loaded) == len(records)
        for orig, got in zip(records, loaded):
            assert orig.date == got.date
            assert orig.symbol == got.symbol
            assert orig.short_volume == got.short_volume
            assert orig.total_volume == got.total_volume
            assert orig.short_exempt_volume == got.short_exempt_volume
            assert orig.short_volume_ratio == got.short_volume_ratio

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        loaded = client._load_cache("NOSYMBOL")
        assert loaded is None

    def test_load_corrupt_csv_cache(self, tmp_path: Path) -> None:
        """Corrupt CSV (bad data row) returns None."""
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        # Write a CSV with the correct header but non-coercible data values
        (sym_dir / "short_volume.csv").write_text(
            "date,symbol,short_volume,total_volume,short_exempt_volume,short_volume_ratio\n"
            "not-a-date,X,bad,bad,bad,bad\n",
            encoding="utf-8",
        )
        loaded = client._load_cache("BAD")
        assert loaded is None


# ------------------------------------------------------------------
# Tests: Legacy migration
# ------------------------------------------------------------------


class TestSVLegacyMigration:
    """Tests that legacy flat cache files are migrated on load."""

    def test_legacy_migration_on_load(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = _sample_records("MSFT")

        # Write legacy file in cache/short_volume subdirectory structure
        legacy_dir = client._data_dir / "cache" / "short_volume" / "MSFT"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_file = legacy_dir / "sv.json"
        legacy_file.write_text(
            json.dumps([asdict(r) for r in records]),
            encoding="utf-8",
        )
        assert legacy_file.exists()

        loaded = client._load_cache("MSFT")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].symbol == "MSFT"
        # Legacy file should have been moved to JSON, then migrated to CSV
        assert not legacy_file.exists()
        assert (client._data_dir / "MSFT" / "short_volume.csv").exists()

    def test_json_fallback_migrates_to_csv(self, tmp_path: Path) -> None:
        """If only .json exists (no .csv), _load_cache reads JSON and migrates to CSV."""
        client = _make_client(tmp_path)
        records = _sample_records("AMZN")

        # Write a JSON cache file directly (simulating pre-migration state)
        sym_dir = client._data_dir / "AMZN"
        sym_dir.mkdir(parents=True, exist_ok=True)
        json_path = sym_dir / "short_volume.json"
        json_path.write_text(
            json.dumps([asdict(r) for r in records]),
            encoding="utf-8",
        )
        assert json_path.exists()
        assert not (sym_dir / "short_volume.csv").exists()

        loaded = client._load_cache("AMZN")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].symbol == "AMZN"
        assert loaded[0].date == "2024-06-10"
        # After loading, CSV should have been created for future loads
        assert (sym_dir / "short_volume.csv").exists()
