"""Tests for PriceData model."""

from decimal import Decimal

import pytest

from stockdownloader.core.models.price import PriceData


def test_create_valid_price_data():
    pd = PriceData(
        date="2024-01-01",
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("95"),
        close=Decimal("105"),
        adj_close=Decimal("105"),
        volume=1000,
    )

    assert pd.date == "2024-01-01"
    assert pd.open == Decimal("100")
    assert pd.high == Decimal("110")
    assert pd.low == Decimal("95")
    assert pd.close == Decimal("105")
    assert pd.adj_close == Decimal("105")
    assert pd.volume == 1000


def test_null_date_throws():
    with pytest.raises((ValueError, TypeError)):
        PriceData(
            date=None,
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("1"),
            close=Decimal("1"),
            adj_close=Decimal("1"),
            volume=0,
        )


def test_null_close_throws():
    with pytest.raises((ValueError, TypeError)):
        PriceData(
            date="2024-01-01",
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("1"),
            close=None,
            adj_close=Decimal("1"),
            volume=0,
        )


def test_negative_volume_throws():
    with pytest.raises((ValueError, TypeError)):
        PriceData(
            date="2024-01-01",
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("1"),
            close=Decimal("1"),
            adj_close=Decimal("1"),
            volume=-1,
        )


def test_zero_volume_is_valid():
    pd = PriceData(
        date="2024-01-01",
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
        adj_close=Decimal("1"),
        volume=0,
    )
    assert pd.volume == 0


def test_to_string_contains_all_fields():
    pd = PriceData(
        date="2024-01-01",
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("95"),
        close=Decimal("105"),
        adj_close=Decimal("105"),
        volume=5000,
    )
    s = str(pd)
    assert "2024-01-01" in s
    assert "100" in s
    assert "5000" in s


def test_record_equality():
    pd1 = PriceData(
        date="2024-01-01",
        open=Decimal("10"),
        high=Decimal("10"),
        low=Decimal("10"),
        close=Decimal("10"),
        adj_close=Decimal("10"),
        volume=100,
    )
    pd2 = PriceData(
        date="2024-01-01",
        open=Decimal("10"),
        high=Decimal("10"),
        low=Decimal("10"),
        close=Decimal("10"),
        adj_close=Decimal("10"),
        volume=100,
    )
    assert pd1 == pd2
    assert hash(pd1) == hash(pd2)
