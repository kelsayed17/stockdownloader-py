"""Unit tests for OccOptionsClient — OCC bulk download parsing and caching."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from stockdownloader.data.market.occ_client import OccOptionsClient
from stockdownloader.core.models.regulatory import OccOpenInterestRecord


# ------------------------------------------------------------------
# Sample data — matches OCC cont-volume-download fixed-width format
# ------------------------------------------------------------------

# Header line (always first line)
_OCC_HEADER = "        H02202026      02202026"

# GME records — 52 chars each: symbol(6)+underlying(6)+exch(1)+vol(9)+exerc(9)+oi(9)+kind(4)+exp(8)
_GME_LINE_1 = "GME   GME   A000005227000000000000005121OSTK20260220"
_GME_LINE_2 = "GME   GME   A000004015000000008000003975OSTK20260227"
_GME_LINE_3 = "GME   GME   C000012345000000000000067890OSTK20260321"

# Non-GME record
_AAPL_LINE  = "AAPL  AAPL  A000100000000000000000200000OSTK20260220"

# Full sample response (as OCC would return)
_SAMPLE_BULK_TEXT = "\r\n".join([
    _OCC_HEADER,
    _AAPL_LINE,
    _GME_LINE_1,
    _GME_LINE_2,
    _GME_LINE_3,
    "",  # trailing blank
])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(tmp_path: Path) -> OccOptionsClient:
    return OccOptionsClient(data_dir=str(tmp_path))


# ------------------------------------------------------------------
# Tests: Fixed-width line parsing
# ------------------------------------------------------------------


class TestOccBulkParsing:
    """Tests for parsing OCC fixed-width bulk download format."""

    def test_parse_single_line(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        record = client._parse_bulk_line(_GME_LINE_1, "2026-02-20")

        assert record is not None
        assert record.symbol == "GME"
        assert record.exchange == "A"
        assert record.volume == 5227
        assert record.exercised == 0
        assert record.open_interest == 5121
        assert record.product_kind == "OSTK"
        assert record.expiration == "2026-02-20"
        assert record.date == "2026-02-20"

    def test_parse_line_with_exercised(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        record = client._parse_bulk_line(_GME_LINE_2, "2026-02-20")

        assert record is not None
        assert record.volume == 4015
        assert record.exercised == 8
        assert record.open_interest == 3975
        assert record.expiration == "2026-02-27"

    def test_parse_short_line_returns_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        result = client._parse_bulk_line("too short", "2026-02-20")
        assert result is None

    def test_parse_header_line_returns_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        result = client._parse_bulk_line(_OCC_HEADER, "2026-02-20")
        assert result is None

    def test_filter_bulk_text_for_symbol(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = client._parse_bulk_text(_SAMPLE_BULK_TEXT, "GME", "2026-02-20")

        assert len(records) == 3
        assert all(r.symbol == "GME" for r in records)
        assert records[0].volume == 5227
        assert records[1].volume == 4015
        assert records[2].volume == 12345

    def test_filter_bulk_text_no_matches(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = client._parse_bulk_text(_SAMPLE_BULK_TEXT, "TSLA", "2026-02-20")
        assert len(records) == 0


# ------------------------------------------------------------------
# Tests: Cache save/load roundtrip
# ------------------------------------------------------------------


def _sample_records() -> list[OccOpenInterestRecord]:
    return [
        OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=5227, exercised=0, open_interest=5121,
            product_kind="OSTK", expiration="2021-02-19",
        ),
        OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="C",
            volume=12345, exercised=0, open_interest=67890,
            product_kind="OSTK", expiration="2021-02-19",
        ),
    ]


class TestOccCacheRoundtrip:
    """Tests for save/load cache roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = _sample_records()
        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].date == "2021-01-27"
        assert loaded[0].exchange == "A"
        assert loaded[1].exchange == "C"

    def test_csv_roundtrip_preserves_types(self, tmp_path: Path) -> None:
        """CSV save/load preserves int fields (not strings)."""
        client = _make_client(tmp_path)
        records = _sample_records()
        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        r = loaded[0]
        assert isinstance(r.volume, int)
        assert isinstance(r.exercised, int)
        assert isinstance(r.open_interest, int)
        assert r.volume == 5227
        assert r.exercised == 0
        assert r.open_interest == 5121

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert client._load_cache("NOSYMBOL") is None

    def test_load_corrupt_csv_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        (sym_dir / "occ_open_interest.csv").write_text(
            "{{invalid", encoding="utf-8",
        )
        # Corrupt CSV with no valid rows returns empty list (not None)
        result = client._load_cache("BAD")
        assert result is not None
        assert len(result) == 0

    def test_cache_file_path_is_csv(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_cache("GME", _sample_records())
        assert (client._data_dir / "GME" / "occ_open_interest.csv").exists()

    def test_json_fallback_migrates_to_csv(self, tmp_path: Path) -> None:
        """Legacy JSON cache is read and migrated to CSV on first load."""
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "GME"
        sym_dir.mkdir(parents=True, exist_ok=True)

        records = _sample_records()
        json_path = sym_dir / "occ_open_interest.json"
        json_path.write_text(
            json.dumps([asdict(r) for r in records], indent=2),
            encoding="utf-8",
        )

        # No CSV file yet
        assert not (sym_dir / "occ_open_interest.csv").exists()

        loaded = client._load_cache("GME")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].exchange == "A"
        assert loaded[1].exchange == "C"

        # Migration should have created CSV
        assert (sym_dir / "occ_open_interest.csv").exists()

    def test_json_fallback_corrupt_returns_none(self, tmp_path: Path) -> None:
        """Corrupt JSON cache returns None without crashing."""
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "GME"
        sym_dir.mkdir(parents=True, exist_ok=True)

        json_path = sym_dir / "occ_open_interest.json"
        json_path.write_text("{{invalid json", encoding="utf-8")

        loaded = client._load_cache("GME")
        assert loaded is None


class TestOccProgress:
    """Tests for progress tracking (incremental backfill)."""

    def test_save_and_load_progress(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        progress = {"2021-01-27", "2021-01-28"}
        client._save_progress("GME", progress)
        loaded = client._load_progress("GME")

        assert loaded == progress

    def test_empty_progress_on_first_load(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert client._load_progress("GME") == set()

    def test_progress_file_path(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_progress("GME", {"2021-01-27"})
        assert (client._data_dir / "GME" / ".progress" / "occ_options.json").exists()

    def test_legacy_progress_migration(self, tmp_path: Path) -> None:
        """Legacy occ_options_progress.json is moved to .progress/."""
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "GME"
        sym_dir.mkdir(parents=True, exist_ok=True)

        legacy_path = sym_dir / "occ_options_progress.json"
        legacy_path.write_text(
            json.dumps(["2021-01-27", "2021-01-28"]), encoding="utf-8",
        )

        loaded = client._load_progress("GME")
        assert loaded == {"2021-01-27", "2021-01-28"}

        # Legacy file should have been moved
        assert not legacy_path.exists()
        assert (sym_dir / ".progress" / "occ_options.json").exists()


class TestOccMergeRecords:
    """Tests for record merging with deduplication."""

    def test_merge_no_overlap(self, tmp_path: Path) -> None:
        existing = [OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=100, exercised=0, open_interest=50,
            product_kind="OSTK", expiration="2021-02-19",
        )]
        new = [OccOpenInterestRecord(
            date="2021-01-28", symbol="GME", exchange="A",
            volume=200, exercised=0, open_interest=75,
            product_kind="OSTK", expiration="2021-02-19",
        )]
        merged = OccOptionsClient._merge_records(existing, new)
        assert len(merged) == 2
        assert merged[0].date == "2021-01-27"
        assert merged[1].date == "2021-01-28"

    def test_merge_deduplicates_by_key(self, tmp_path: Path) -> None:
        old = OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=100, exercised=0, open_interest=50,
            product_kind="OSTK", expiration="2021-02-19",
        )
        updated = OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=999, exercised=0, open_interest=888,
            product_kind="OSTK", expiration="2021-02-19",
        )
        merged = OccOptionsClient._merge_records([old], [updated])
        assert len(merged) == 1
        assert merged[0].volume == 999  # New wins


# ------------------------------------------------------------------
# Tests: fetch_open_interest integration (mocked HTTP)
# ------------------------------------------------------------------


class TestFetchOpenInterest:
    """Tests for the full fetch_open_interest flow with mocked HTTP."""

    def test_fetch_downloads_and_caches(self, tmp_path: Path) -> None:
        """Fetch downloads bulk files, filters for symbol, saves cache."""
        client = _make_client(tmp_path)

        # Use recent weekdays within the rolling window
        from datetime import date as dt, timedelta
        d = dt.today() - timedelta(days=3)
        # Ensure it's a weekday
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        recent = d.isoformat()

        call_dates: list[str] = []

        def mock_download(date_str, symbol):
            call_dates.append(date_str)
            if date_str == recent:
                return [OccOpenInterestRecord(
                    date=recent, symbol="GME", exchange="A",
                    volume=100, exercised=0, open_interest=50,
                    product_kind="OSTK", expiration="2026-03-21",
                )]
            return []

        with patch.object(client, "_download_day", side_effect=mock_download):
            records = client.fetch_open_interest("GME")

        # Should have called download for weekdays
        assert len(call_dates) > 0

        # Should have cached records
        cached = client._load_cache("GME")
        assert cached is not None
        assert len(cached) >= 1

        # Progress should be saved
        progress = client._load_progress("GME")
        assert len(progress) > 0

    def test_fetch_skips_already_downloaded(self, tmp_path: Path) -> None:
        """Dates in progress file are not re-downloaded."""
        client = _make_client(tmp_path)

        # Use dates within the rolling window
        from datetime import date as dt, timedelta
        d1 = (dt.today() - timedelta(days=5)).isoformat()
        d2 = (dt.today() - timedelta(days=4)).isoformat()

        client._save_progress("GME", {d1, d2})

        call_dates: list[str] = []

        def mock_download(date_str, symbol):
            call_dates.append(date_str)
            return []

        with patch.object(client, "_download_day", side_effect=mock_download):
            client.fetch_open_interest("GME")

        # Should NOT have downloaded pre-populated dates
        assert d1 not in call_dates
        assert d2 not in call_dates

    def test_fetch_returns_cached_when_all_done(self, tmp_path: Path) -> None:
        """When all dates are in progress, returns cache without downloading."""
        client = _make_client(tmp_path)

        # Pre-populate cache and mark all dates as done
        records = _sample_records()
        client._save_cache("GME", records)

        # Mark the entire rolling window as done
        from datetime import date as dt, timedelta
        all_dates = set()
        current = dt.today() - timedelta(days=60)
        while current <= dt.today():
            if current.weekday() < 5:
                all_dates.add(current.isoformat())
            current += timedelta(days=1)
        client._save_progress("GME", all_dates)

        call_count = 0

        def mock_download(date_str, symbol):
            nonlocal call_count
            call_count += 1
            return []

        with patch.object(client, "_download_day", side_effect=mock_download):
            result = client.fetch_open_interest("GME")

        assert call_count == 0  # No downloads
        assert len(result) == 2  # Returns cached
