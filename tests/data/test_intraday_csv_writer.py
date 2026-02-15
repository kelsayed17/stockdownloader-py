"""Tests for intraday_csv_writer."""

from decimal import Decimal
from pathlib import Path

import pytest

from stockdownloader.data.intraday_csv_writer import write_to_file
from stockdownloader.data.intraday_csv_loader import IntradayCsvLoader
from stockdownloader.model.intraday_price_data import IntradayPriceData


def _make_bar(date: str, close: str = "600.00", volume: int = 1000) -> IntradayPriceData:
    return IntradayPriceData(
        date=date,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        adj_close=Decimal(close),
        volume=volume,
    )


class TestIntradayCsvWriter:

    def test_write_creates_file(self, tmp_path):
        path = tmp_path / "test.csv"
        bars = [_make_bar("2025-01-15 09:30:00-05:00")]
        count = write_to_file(bars, path)
        assert count == 1
        assert path.exists()

    def test_write_returns_row_count(self, tmp_path):
        path = tmp_path / "test.csv"
        bars = [
            _make_bar("2025-01-15 09:30:00-05:00"),
            _make_bar("2025-01-15 09:35:00-05:00"),
            _make_bar("2025-01-15 09:40:00-05:00"),
        ]
        assert write_to_file(bars, path) == 3

    def test_write_empty_list(self, tmp_path):
        path = tmp_path / "empty.csv"
        count = write_to_file([], path)
        assert count == 0
        assert path.exists()
        # File should contain just the header
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert "Datetime" in lines[0]

    def test_roundtrip_with_loader(self, tmp_path):
        """Write bars then read them back with IntradayCsvLoader."""
        path = tmp_path / "roundtrip.csv"
        original = [
            _make_bar("2025-01-15 09:30:00-05:00", "601.50", 5000),
            _make_bar("2025-01-15 09:35:00-05:00", "602.25", 3000),
        ]
        write_to_file(original, path)

        loaded = IntradayCsvLoader.load_from_file(path)
        assert len(loaded) == 2
        assert loaded[0].date == original[0].date
        assert loaded[0].close == original[0].close
        assert loaded[0].volume == original[0].volume
        assert loaded[1].date == original[1].date

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dirs" / "test.csv"
        write_to_file([_make_bar("2025-01-15 09:30:00-05:00")], path)
        assert path.exists()

    def test_header_format(self, tmp_path):
        path = tmp_path / "header.csv"
        write_to_file([_make_bar("2025-01-15 09:30:00-05:00")], path)
        header = path.read_text().splitlines()[0]
        assert header == "Datetime,Open,High,Low,Close,Volume"

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "overwrite.csv"
        write_to_file(
            [_make_bar("2025-01-15 09:30:00-05:00"), _make_bar("2025-01-15 09:35:00-05:00")],
            path,
        )
        # Overwrite with fewer bars
        write_to_file([_make_bar("2025-01-15 09:30:00-05:00")], path)
        loaded = IntradayCsvLoader.load_from_file(path)
        assert len(loaded) == 1
