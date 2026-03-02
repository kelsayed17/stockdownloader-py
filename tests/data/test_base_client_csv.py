"""Tests for BaseDataClient CSV and progress-dir helpers."""

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from stockdownloader.data.base_client import BaseDataClient


@dataclass(frozen=True, slots=True)
class _SampleRecord:
    date: str
    symbol: str
    value: int
    ratio: float
    flag: bool


@dataclass(frozen=True, slots=True)
class _DecimalRecord:
    date: str
    price: Decimal


class TestSaveCsv:
    def test_save_creates_csv_with_header(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        records = [
            _SampleRecord("2024-01-01", "GME", 100, 0.5, True),
            _SampleRecord("2024-01-02", "GME", 200, 0.75, False),
        ]
        client._save_csv("GME", "test_data.csv", records)
        path = tmp_path / "GME" / "test_data.csv"
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert lines[0] == "date,symbol,value,ratio,flag"
        assert lines[1] == "2024-01-01,GME,100,0.5,True"
        assert lines[2] == "2024-01-02,GME,200,0.75,False"

    def test_save_empty_list_creates_header_only(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        client._save_csv("GME", "empty.csv", [], _SampleRecord)
        path = tmp_path / "GME" / "empty.csv"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert lines[0] == "date,symbol,value,ratio,flag"

    def test_save_decimal_field(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        records = [_DecimalRecord("2024-01-01", Decimal("12.50"))]
        client._save_csv("GME", "dec.csv", records)
        path = tmp_path / "GME" / "dec.csv"
        lines = path.read_text().strip().splitlines()
        assert lines[1] == "2024-01-01,12.50"


class TestLoadCsv:
    def test_load_roundtrip(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        original = [
            _SampleRecord("2024-01-01", "GME", 100, 0.5, True),
            _SampleRecord("2024-01-02", "GME", 200, 0.75, False),
        ]
        client._save_csv("GME", "rt.csv", original)
        loaded = client._load_csv("GME", "rt.csv", _SampleRecord)
        assert loaded == original

    def test_load_nonexistent_returns_none(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        (tmp_path / "GME").mkdir()
        assert client._load_csv("GME", "nope.csv", _SampleRecord) is None

    def test_load_corrupt_returns_none(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        d = tmp_path / "GME"
        d.mkdir()
        (d / "bad.csv").write_text("not,a,csv\n\x00\x01\x02")
        assert client._load_csv("GME", "bad.csv", _SampleRecord) is None

    def test_load_decimal_field(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        original = [_DecimalRecord("2024-01-01", Decimal("12.50"))]
        client._save_csv("GME", "dec.csv", original)
        loaded = client._load_csv("GME", "dec.csv", _DecimalRecord)
        assert loaded is not None
        assert loaded[0].price == Decimal("12.50")

    def test_load_bool_coercion(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        original = [_SampleRecord("2024-01-01", "GME", 1, 0.0, False)]
        client._save_csv("GME", "bool.csv", original)
        loaded = client._load_csv("GME", "bool.csv", _SampleRecord)
        assert loaded is not None
        assert loaded[0].flag is False


class TestProgressDir:
    def test_progress_dir_created(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        pdir = client._progress_dir("GME")
        assert pdir == tmp_path / "GME" / ".progress"
        assert pdir.is_dir()

    def test_progress_dir_idempotent(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        p1 = client._progress_dir("GME")
        p2 = client._progress_dir("GME")
        assert p1 == p2
