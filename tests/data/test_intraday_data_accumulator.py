"""Tests for IntradayDataAccumulator.

Uses mocked Yahoo data so tests run offline without hitting real APIs.
"""

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.intraday_data_accumulator import (
    IntradayDataAccumulator,
    _merge_bars,
    default_csv_path,
)
from stockdownloader.data.intraday_csv_writer import write_to_file
from stockdownloader.model.intraday_price_data import IntradayPriceData


def _bar(date: str, close: str = "600.00") -> IntradayPriceData:
    return IntradayPriceData(
        date=date,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        adj_close=Decimal(close),
        volume=1000,
    )


class TestDefaultCsvPath:

    def test_spy(self):
        assert default_csv_path("SPY") == Path("data/spy_5m_bars.csv")

    def test_aapl(self):
        assert default_csv_path("AAPL") == Path("data/aapl_5m_bars.csv")

    def test_lowercase_input(self):
        assert default_csv_path("aapl") == Path("data/aapl_5m_bars.csv")


class TestMergeBars:

    def test_merge_empty_lists(self):
        assert _merge_bars([], []) == []

    def test_merge_fresh_into_empty(self):
        fresh = [_bar("2025-01-15 09:30:00-05:00")]
        merged = _merge_bars([], fresh)
        assert len(merged) == 1
        assert merged[0].date == "2025-01-15 09:30:00-05:00"

    def test_merge_empty_fresh(self):
        existing = [_bar("2025-01-15 09:30:00-05:00")]
        merged = _merge_bars(existing, [])
        assert len(merged) == 1

    def test_merge_deduplicates(self):
        existing = [
            _bar("2025-01-15 09:30:00-05:00", "600.00"),
            _bar("2025-01-15 09:35:00-05:00", "601.00"),
        ]
        fresh = [
            _bar("2025-01-15 09:35:00-05:00", "601.50"),  # overlap
            _bar("2025-01-15 09:40:00-05:00", "602.00"),
        ]
        merged = _merge_bars(existing, fresh)
        assert len(merged) == 3

    def test_fresh_overwrites_existing_on_overlap(self):
        existing = [_bar("2025-01-15 09:30:00-05:00", "600.00")]
        fresh = [_bar("2025-01-15 09:30:00-05:00", "605.00")]
        merged = _merge_bars(existing, fresh)
        assert len(merged) == 1
        assert merged[0].close == Decimal("605.00")

    def test_merge_sorts_by_date(self):
        existing = [_bar("2025-01-16 09:30:00-05:00")]
        fresh = [_bar("2025-01-15 09:30:00-05:00")]
        merged = _merge_bars(existing, fresh)
        assert merged[0].date < merged[1].date

    def test_merge_large_overlap(self):
        """Simulate two ~60-day windows with 30 days overlap."""
        existing = [_bar(f"2025-01-{d:02d} 09:30:00-05:00") for d in range(1, 31)]
        fresh = [_bar(f"2025-01-{d:02d} 09:30:00-05:00") for d in range(15, 31)]
        fresh += [_bar(f"2025-02-{d:02d} 09:30:00-05:00") for d in range(1, 16)]
        merged = _merge_bars(existing, fresh)
        # 30 Jan days + 15 Feb days = 45 unique
        assert len(merged) == 45


class TestAccumulator:

    def test_accumulate_creates_new_file(self, tmp_path):
        """When no CSV exists, creates a new one from Yahoo data."""
        csv_path = tmp_path / "spy_5m_bars.csv"
        mock_client = MagicMock()
        mock_client.fetch_intraday_history.return_value = [
            _bar("2025-01-15 09:30:00-05:00"),
            _bar("2025-01-15 09:35:00-05:00"),
        ]

        acc = IntradayDataAccumulator(client=mock_client)
        result = acc.accumulate("SPY", csv_path)

        assert len(result) == 2
        assert csv_path.exists()
        mock_client.fetch_intraday_history.assert_called_once()

    def test_accumulate_merges_with_existing(self, tmp_path):
        """When CSV exists, merges fresh data with it."""
        csv_path = tmp_path / "spy_5m_bars.csv"

        # Pre-populate CSV with existing data
        existing = [
            _bar("2025-01-10 09:30:00-05:00"),
            _bar("2025-01-10 09:35:00-05:00"),
        ]
        write_to_file(existing, csv_path)

        # Mock Yahoo returning new bars
        mock_client = MagicMock()
        mock_client.fetch_intraday_history.return_value = [
            _bar("2025-01-15 09:30:00-05:00"),
            _bar("2025-01-15 09:35:00-05:00"),
        ]

        acc = IntradayDataAccumulator(client=mock_client)
        result = acc.accumulate("SPY", csv_path)

        assert len(result) == 4  # 2 existing + 2 new

    def test_accumulate_deduplicates_overlap(self, tmp_path):
        """Overlapping bars are deduplicated during merge."""
        csv_path = tmp_path / "spy_5m_bars.csv"

        existing = [
            _bar("2025-01-15 09:30:00-05:00"),
            _bar("2025-01-15 09:35:00-05:00"),
        ]
        write_to_file(existing, csv_path)

        mock_client = MagicMock()
        mock_client.fetch_intraday_history.return_value = [
            _bar("2025-01-15 09:35:00-05:00"),  # overlap
            _bar("2025-01-15 09:40:00-05:00"),  # new
        ]

        acc = IntradayDataAccumulator(client=mock_client)
        result = acc.accumulate("SPY", csv_path)

        assert len(result) == 3  # not 4

    def test_accumulate_uses_default_path(self, tmp_path, monkeypatch):
        """When no path is given, uses data/<symbol>_5m_bars.csv."""
        monkeypatch.chdir(tmp_path)
        mock_client = MagicMock()
        mock_client.fetch_intraday_history.return_value = [
            _bar("2025-01-15 09:30:00-05:00"),
        ]

        acc = IntradayDataAccumulator(client=mock_client)
        result = acc.accumulate("AAPL")

        expected_path = tmp_path / "data" / "aapl_5m_bars.csv"
        assert expected_path.exists()
        assert len(result) == 1

    def test_accumulate_with_empty_fetch(self, tmp_path):
        """If Yahoo returns nothing, existing data is preserved."""
        csv_path = tmp_path / "spy_5m_bars.csv"
        existing = [_bar("2025-01-15 09:30:00-05:00")]
        write_to_file(existing, csv_path)

        mock_client = MagicMock()
        mock_client.fetch_intraday_history.return_value = []

        acc = IntradayDataAccumulator(client=mock_client)
        result = acc.accumulate("SPY", csv_path)

        assert len(result) == 1

    def test_accumulate_custom_fetch_days(self, tmp_path):
        """Fetch days parameter is passed through to client."""
        csv_path = tmp_path / "spy_5m_bars.csv"
        mock_client = MagicMock()
        mock_client.fetch_intraday_history.return_value = []

        acc = IntradayDataAccumulator(client=mock_client, fetch_days=30)
        acc.accumulate("SPY", csv_path)

        mock_client.fetch_intraday_history.assert_called_once_with(
            "SPY", total_days=30, interval="5m"
        )

    def test_accumulate_result_is_sorted(self, tmp_path):
        """Merged result is always sorted chronologically."""
        csv_path = tmp_path / "spy_5m_bars.csv"

        existing = [_bar("2025-01-20 09:30:00-05:00")]
        write_to_file(existing, csv_path)

        mock_client = MagicMock()
        mock_client.fetch_intraday_history.return_value = [
            _bar("2025-01-10 09:30:00-05:00"),
        ]

        acc = IntradayDataAccumulator(client=mock_client)
        result = acc.accumulate("SPY", csv_path)

        assert result[0].date < result[1].date

    def test_multiple_accumulations(self, tmp_path):
        """Simulate accumulating data over 3 runs."""
        csv_path = tmp_path / "spy_5m_bars.csv"

        mock_client = MagicMock()

        # Run 1: Jan data
        mock_client.fetch_intraday_history.return_value = [
            _bar(f"2025-01-{d:02d} 09:30:00-05:00") for d in range(1, 11)
        ]
        acc = IntradayDataAccumulator(client=mock_client)
        result = acc.accumulate("SPY", csv_path)
        assert len(result) == 10

        # Run 2: Feb data (no overlap)
        mock_client.fetch_intraday_history.return_value = [
            _bar(f"2025-02-{d:02d} 09:30:00-05:00") for d in range(1, 11)
        ]
        result = acc.accumulate("SPY", csv_path)
        assert len(result) == 20

        # Run 3: Mar data with overlap into Feb
        mock_client.fetch_intraday_history.return_value = [
            _bar(f"2025-02-{d:02d} 09:30:00-05:00") for d in range(5, 11)
        ] + [
            _bar(f"2025-03-{d:02d} 09:30:00-05:00") for d in range(1, 11)
        ]
        result = acc.accumulate("SPY", csv_path)
        assert len(result) == 30  # 10 Jan + 10 Feb + 10 Mar
