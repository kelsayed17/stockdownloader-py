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
        cache_dir=str(tmp_path / "sv_cache"),
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
    """Tests that cache uses per-symbol subdirectories."""

    def test_cache_file_naming(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_cache("AAPL", _sample_records("AAPL"))
        assert (client._cache_dir / "AAPL" / "sv.json").exists()

    def test_save_creates_symbol_subdir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_cache("TSLA", _sample_records("TSLA"))
        sym_dir = client._cache_dir / "TSLA"
        assert sym_dir.is_dir()
        assert (sym_dir / "sv.json").exists()


# ------------------------------------------------------------------
# Tests: Save and load roundtrip
# ------------------------------------------------------------------


class TestSVCacheRoundtrip:
    """Tests for save/load roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = _sample_records()
        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].date == "2024-06-10"
        assert loaded[1].short_volume == 6_000_000

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        loaded = client._load_cache("NOSYMBOL")
        assert loaded is None

    def test_load_corrupt_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._cache_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        (sym_dir / "sv.json").write_text("{{invalid", encoding="utf-8")
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

        # Write legacy flat file
        legacy_file = client._cache_dir / "MSFT_sv.json"
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
        assert (client._cache_dir / "MSFT" / "sv.json").exists()
