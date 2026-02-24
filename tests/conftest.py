"""Shared test fixtures for stockdownloader tests."""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from stockdownloader.core.models.price import PriceData


TEST_DATA_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    """Path to the test price data CSV file."""
    return TEST_DATA_DIR / "test-price-data.csv"


@pytest.fixture(scope="session")
def sample_price_data(test_data_path: Path) -> list[PriceData]:
    """Load sample price data from CSV for testing."""
    data: list[PriceData] = []
    with open(test_data_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                PriceData(
                    date=row["Date"],
                    open=Decimal(row["Open"]),
                    high=Decimal(row["High"]),
                    low=Decimal(row["Low"]),
                    close=Decimal(row["Close"]),
                    adj_close=Decimal(row["Adj Close"]),
                    volume=int(row["Volume"]),
                )
            )
    return data


@pytest.fixture
def small_price_data() -> list[PriceData]:
    """A small set of price data for simple unit tests."""
    return [
        PriceData("2024-01-01", Decimal("100"), Decimal("105"), Decimal("99"), Decimal("103"), Decimal("103"), 1000000),
        PriceData("2024-01-02", Decimal("103"), Decimal("107"), Decimal("102"), Decimal("106"), Decimal("106"), 1200000),
        PriceData("2024-01-03", Decimal("106"), Decimal("108"), Decimal("104"), Decimal("105"), Decimal("105"), 900000),
        PriceData("2024-01-04", Decimal("105"), Decimal("110"), Decimal("104"), Decimal("109"), Decimal("109"), 1100000),
        PriceData("2024-01-05", Decimal("109"), Decimal("112"), Decimal("108"), Decimal("111"), Decimal("111"), 1300000),
    ]
