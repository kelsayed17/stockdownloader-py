"""Tests for CsvPriceDataLoader."""

import io
from decimal import Decimal

import pytest

from stockdownloader.data.data_parsers import CsvPriceDataLoader


def test_load_from_stream():
    csv = (
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2024-01-02,100.00,105.00,99.00,103.00,103.00,1000000\n"
        "2024-01-03,103.00,107.00,102.00,106.00,106.00,1200000\n"
    )

    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)

    assert len(data) == 2
    assert data[0].date == "2024-01-02"
    assert data[0].open == Decimal("100.00")
    assert data[0].high == Decimal("105.00")
    assert data[0].low == Decimal("99.00")
    assert data[0].close == Decimal("103.00")
    assert data[0].volume == 1000000


def test_load_from_stream_skips_invalid_lines():
    csv = (
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2024-01-02,100.00,105.00,99.00,103.00,103.00,1000000\n"
        "2024-01-03,null,null,null,null,null,0\n"
        "2024-01-04,110.00,115.00,109.00,113.00,113.00,900000\n"
    )

    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)

    assert len(data) == 2
    assert data[0].date == "2024-01-02"
    assert data[1].date == "2024-01-04"


def test_load_from_stream_with_missing_adj_close_uses_close():
    csv = (
        "Date,Open,High,Low,Close\n"
        "2024-01-02,100.00,105.00,99.00,103.00\n"
    )

    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)

    assert len(data) == 1
    assert data[0].adj_close == Decimal("103.00")
    assert data[0].volume == 0


def test_load_from_file(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2024-01-02,50.00,55.00,49.00,53.00,53.00,500000\n"
    )

    data = CsvPriceDataLoader.load_from_file(str(csv_file))

    assert len(data) == 1
    assert data[0].date == "2024-01-02"
    assert data[0].close == Decimal("53.00")


def test_load_from_nonexistent_file_returns_empty():
    data = CsvPriceDataLoader.load_from_file("/nonexistent/file.csv")
    assert len(data) == 0


def test_load_empty_stream():
    csv = "Date,Open,High,Low,Close,Adj Close,Volume\n"
    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)
    assert len(data) == 0
