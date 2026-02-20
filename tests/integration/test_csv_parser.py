"""Integration test for the CSV parsing pipeline:
raw CSV data -> CsvParser -> CsvPriceDataLoader -> PriceData.
Tests the interaction between the parser and the data loader.
"""
from __future__ import annotations

import io
from decimal import Decimal

import pytest

from stockdownloader.data.data_parsers import CsvPriceDataLoader
from stockdownloader.model.price_data import PriceData
from stockdownloader.util.parsers import CsvParser


def test_parse_csv_to_price_data_end_to_end():
    csv = (
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2023-01-03,384.37,386.43,377.83,380.82,380.82,74850700\n"
        "2023-01-04,383.18,385.88,380.00,383.76,383.76,68860700\n"
        "2023-01-05,381.72,381.84,378.76,379.38,379.38,57510600\n"
    )

    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)

    assert len(data) == 3

    first = data[0]
    assert first.date == "2023-01-03"
    assert first.open == Decimal("384.37")
    assert first.high == Decimal("386.43")
    assert first.low == Decimal("377.83")
    assert first.close == Decimal("380.82")
    assert first.adj_close == Decimal("380.82")
    assert first.volume == 74850700


def test_csv_parser_handles_quoted_fields():
    csv = (
        'Name,Value\n'
        '"Hello, World",42\n'
        '"She said ""hi""",99\n'
    )

    with CsvParser(io.StringIO(csv)) as parser:
        header = parser.read_next()
        assert header == ["Name", "Value"]

        row1 = parser.read_next()
        assert row1[0] == "Hello, World"
        assert row1[1] == "42"

        row2 = parser.read_next()
        assert row2[0] == 'She said "hi"'
        assert row2[1] == "99"


def test_csv_parser_with_custom_delimiter():
    tsv = "Name\tValue\nAlice\t100\nBob\t200\n"

    with CsvParser(io.StringIO(tsv), separator='\t') as parser:
        header = parser.read_next()
        assert header[0] == "Name"
        assert header[1] == "Value"

        row1 = parser.read_next()
        assert row1[0] == "Alice"
        assert row1[1] == "100"

        row2 = parser.read_next()
        assert row2[0] == "Bob"
        assert row2[1] == "200"


def test_csv_parser_read_all_integration():
    csv = (
        "A,B,C\n"
        "1,2,3\n"
        "4,5,6\n"
        "7,8,9\n"
    )

    with CsvParser(io.StringIO(csv)) as parser:
        all_rows = parser.read_all()
        assert len(all_rows) == 4  # header + 3 rows
        assert all_rows[0] == ["A", "B", "C"]
        assert all_rows[1] == ["1", "2", "3"]
        assert all_rows[3] == ["7", "8", "9"]


def test_price_data_loader_skips_invalid_rows():
    csv = (
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2023-01-03,384.37,386.43,377.83,380.82,380.82,74850700\n"
        "2023-01-04,null,null,null,null,null,0\n"
        "2023-01-05,381.72,381.84,378.76,379.38,379.38,57510600\n"
    )

    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)

    assert len(data) == 2, "Invalid rows should be skipped"
    assert data[0].date == "2023-01-03"
    assert data[1].date == "2023-01-05"


def test_price_data_loader_handles_minimal_csv():
    csv = (
        "Date,Open,High,Low,Close\n"
        "2023-01-03,100,110,90,105\n"
    )

    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)

    assert len(data) == 1
    row = data[0]
    assert row.close == Decimal("105")
    assert row.adj_close == Decimal("105"), "Missing adjClose should default to close"
    assert row.volume == 0, "Missing volume should default to 0"


def test_price_data_loader_from_resource_file(test_data_path):
    with open(test_data_path, "rb") as f:
        data = CsvPriceDataLoader.load_from_stream(f)

    assert len(data) > 0
    assert len(data) > 100, "Resource file should have substantial data"

    # Verify data integrity across all rows
    for bar in data:
        assert bar.date is not None
        assert bar.high >= bar.low, \
            f"High should be >= Low for {bar.date}"
        assert bar.high >= bar.open, \
            f"High should be >= Open for {bar.date}"
        assert bar.high >= bar.close, \
            f"High should be >= Close for {bar.date}"
        assert bar.low <= bar.open, \
            f"Low should be <= Open for {bar.date}"
        assert bar.low <= bar.close, \
            f"Low should be <= Close for {bar.date}"


def test_empty_stream_returns_empty_list():
    csv = "Date,Open,High,Low,Close,Adj Close,Volume\n"
    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)
    assert len(data) == 0, "Empty CSV (header only) should return empty list"


def test_csv_parser_skip_lines_integration():
    csv = "Comment line 1\nComment line 2\nA,B\n1,2\n"

    with CsvParser(io.StringIO(csv)) as parser:
        parser.skip_lines(2)
        header = parser.read_next()
        assert header == ["A", "B"]

        data = parser.read_next()
        assert data == ["1", "2"]
