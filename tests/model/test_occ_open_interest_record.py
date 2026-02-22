"""Unit tests for OccOpenInterestRecord dataclass."""

from __future__ import annotations

import pytest

from stockdownloader.model.regulatory_records import OccOpenInterestRecord


class TestOccOpenInterestRecord:
    """Tests for OccOpenInterestRecord creation and validation."""

    def test_create_valid_record(self) -> None:
        record = OccOpenInterestRecord(
            date="2021-01-27",
            symbol="GME",
            exchange="A",
            volume=5227,
            exercised=0,
            open_interest=5121,
            product_kind="OSTK",
            expiration="2021-02-19",
        )
        assert record.date == "2021-01-27"
        assert record.symbol == "GME"
        assert record.exchange == "A"
        assert record.volume == 5227
        assert record.exercised == 0
        assert record.open_interest == 5121
        assert record.product_kind == "OSTK"
        assert record.expiration == "2021-02-19"

    def test_record_is_frozen(self) -> None:
        record = OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=100, exercised=0, open_interest=50,
            product_kind="OSTK", expiration="2021-02-19",
        )
        with pytest.raises(AttributeError):
            record.volume = 999  # type: ignore[misc]

    def test_empty_date_raises(self) -> None:
        with pytest.raises(ValueError, match="date must not be empty"):
            OccOpenInterestRecord(
                date="", symbol="GME", exchange="A",
                volume=100, exercised=0, open_interest=50,
                product_kind="OSTK", expiration="2021-02-19",
            )

    def test_empty_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="symbol must not be empty"):
            OccOpenInterestRecord(
                date="2021-01-27", symbol="", exchange="A",
                volume=100, exercised=0, open_interest=50,
                product_kind="OSTK", expiration="2021-02-19",
            )

    def test_negative_open_interest_raises(self) -> None:
        with pytest.raises(ValueError, match="open_interest must be non-negative"):
            OccOpenInterestRecord(
                date="2021-01-27", symbol="GME", exchange="A",
                volume=100, exercised=0, open_interest=-1,
                product_kind="OSTK", expiration="2021-02-19",
            )
