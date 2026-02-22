"""Unit tests for OccOptionsClient — OCC bulk download parsing and caching."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from stockdownloader.data.occ_options_client import OccOptionsClient
from stockdownloader.model.regulatory_records import OccOpenInterestRecord


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
