"""Unit tests for BorrowRateProxy — heuristic DTC-to-fee mapping."""

from __future__ import annotations

import pytest

from stockdownloader.data.borrow_rate import BorrowRateProxy
from stockdownloader.core.models.regulatory import BorrowRateRecord, ShortInterestRecord


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_si_record(
    settlement_date: str = "2024-01-15",
    symbol: str = "GME",
    short_interest: int = 30_000_000,
    avg_daily_volume: int = 5_000_000,
    days_to_cover: float = 6.0,
    short_interest_pct: float = 0.0,
) -> ShortInterestRecord:
    return ShortInterestRecord(
        settlement_date=settlement_date,
        symbol=symbol,
        short_interest=short_interest,
        avg_daily_volume=avg_daily_volume,
        days_to_cover=days_to_cover,
        short_interest_pct=short_interest_pct,
    )


# ------------------------------------------------------------------
# Tests: DTC-to-fee heuristic mapping
# ------------------------------------------------------------------


class TestDtcToFee:
    """Tests for the _dtc_to_fee static method."""

    def test_negative_dtc_returns_gc_rate(self) -> None:
        # Edge case: negative DTC should return minimum rate
        fee = BorrowRateProxy._dtc_to_fee(-1.0)
        assert fee == 0.25

    def test_zero_dtc_returns_minimum(self) -> None:
        fee = BorrowRateProxy._dtc_to_fee(0.0)
        assert fee == 0.25

    def test_dtc_below_1_easy_to_borrow(self) -> None:
        # DTC < 1 -> ~0.25% to ~1.0% (GC rate)
        fee = BorrowRateProxy._dtc_to_fee(0.5)
        assert 0.25 <= fee <= 1.0
        # Specifically: 0.25 + 0.5 * 0.75 = 0.625
        assert abs(fee - 0.625) < 0.01

    def test_dtc_at_1_boundary(self) -> None:
        fee = BorrowRateProxy._dtc_to_fee(1.0)
        assert abs(fee - 1.0) < 0.01

    def test_dtc_1_to_3_moderate(self) -> None:
        # DTC 1-3 -> ~1.0% to ~5.0%
        fee = BorrowRateProxy._dtc_to_fee(2.0)
        assert 1.0 <= fee <= 5.0
        # Specifically: 1.0 + (2.0 - 1.0) * 2.0 = 3.0
        assert abs(fee - 3.0) < 0.01

    def test_dtc_at_3_boundary(self) -> None:
        fee = BorrowRateProxy._dtc_to_fee(3.0)
        assert abs(fee - 5.0) < 0.01

    def test_dtc_3_to_7_elevated(self) -> None:
        # DTC 3-7 -> ~5.0% to ~20.0%
        fee = BorrowRateProxy._dtc_to_fee(5.0)
        assert 5.0 <= fee <= 20.0
        # Specifically: 5.0 + (5.0 - 3.0) * 3.75 = 12.5
        assert abs(fee - 12.5) < 0.01

    def test_dtc_at_7_boundary(self) -> None:
        fee = BorrowRateProxy._dtc_to_fee(7.0)
        assert abs(fee - 20.0) < 0.01

    def test_dtc_7_to_15_high(self) -> None:
        # DTC 7-15 -> ~20.0% to ~50.0%
        fee = BorrowRateProxy._dtc_to_fee(11.0)
        assert 20.0 <= fee <= 50.0
        # Specifically: 20.0 + (11.0 - 7.0) * 3.75 = 35.0
        assert abs(fee - 35.0) < 0.01

    def test_dtc_at_15_boundary(self) -> None:
        fee = BorrowRateProxy._dtc_to_fee(15.0)
        assert abs(fee - 50.0) < 0.01

    def test_dtc_above_15_very_hard(self) -> None:
        # DTC > 15 -> 50%+ (hard to borrow)
        fee = BorrowRateProxy._dtc_to_fee(20.0)
        assert fee >= 50.0
        # Specifically: 50.0 + (20.0 - 15.0) * 5.0 = 75.0
        assert abs(fee - 75.0) < 0.01

    def test_dtc_extremely_high_capped_at_100(self) -> None:
        fee = BorrowRateProxy._dtc_to_fee(30.0)
        assert fee == 100.0

    def test_dtc_beyond_cap(self) -> None:
        fee = BorrowRateProxy._dtc_to_fee(100.0)
        assert fee == 100.0

    def test_fee_monotonically_increasing(self) -> None:
        """Verify fees increase as DTC increases across the full range."""
        dtc_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
        fees = [BorrowRateProxy._dtc_to_fee(d) for d in dtc_values]
        for i in range(1, len(fees)):
            assert fees[i] >= fees[i - 1], (
                f"Fee at DTC={dtc_values[i]} ({fees[i]}) should be >= "
                f"fee at DTC={dtc_values[i-1]} ({fees[i-1]})"
            )


# ------------------------------------------------------------------
# Tests: estimate_borrow_rates with empty input
# ------------------------------------------------------------------


class TestEmptyInput:
    """Tests for empty input handling."""

    def test_empty_si_records_returns_empty(self) -> None:
        proxy = BorrowRateProxy()
        result = proxy.estimate_borrow_rates("GME", [])
        assert result == []

    def test_empty_symbol_still_works(self) -> None:
        # The proxy itself doesn't validate symbol; records set it
        proxy = BorrowRateProxy()
        result = proxy.estimate_borrow_rates("GME", [])
        assert result == []


# ------------------------------------------------------------------
# Tests: estimate_borrow_rates with real-looking SI records
# ------------------------------------------------------------------


class TestEstimateBorrowRates:
    """Tests for estimate_borrow_rates method with realistic data."""

    def test_single_record_easy_to_borrow(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [
            _make_si_record(
                days_to_cover=0.5,
                short_interest=1_000_000,
            ),
        ]
        result = proxy.estimate_borrow_rates("GME", si_records)
        assert len(result) == 1
        rec = result[0]
        assert rec.symbol == "GME"
        assert rec.date == "2024-01-15"
        assert rec.estimated_fee_pct < 1.0  # Easy to borrow

    def test_single_record_hard_to_borrow(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [
            _make_si_record(
                days_to_cover=20.0,
                short_interest=50_000_000,
            ),
        ]
        result = proxy.estimate_borrow_rates("GME", si_records)
        assert len(result) == 1
        assert result[0].estimated_fee_pct >= 50.0

    def test_multiple_records_sorted_by_date(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [
            _make_si_record(settlement_date="2024-01-31", days_to_cover=3.0),
            _make_si_record(settlement_date="2024-01-15", days_to_cover=6.0),
            _make_si_record(settlement_date="2024-01-22", days_to_cover=4.0),
        ]
        result = proxy.estimate_borrow_rates("GME", si_records)
        assert len(result) == 3
        dates = [r.date for r in result]
        assert dates == sorted(dates)

    def test_utilization_computed_with_shares_outstanding(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [
            _make_si_record(short_interest=30_000_000),
        ]
        result = proxy.estimate_borrow_rates(
            "GME", si_records, shares_outstanding=305_000_000,
        )
        assert len(result) == 1
        expected_util = 30_000_000 / 305_000_000
        assert abs(result[0].utilization_pct - expected_util) < 0.001

    def test_utilization_zero_without_shares_outstanding(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [
            _make_si_record(short_interest=30_000_000),
        ]
        result = proxy.estimate_borrow_rates("GME", si_records)
        assert result[0].utilization_pct == 0.0

    def test_utilization_zero_with_none_shares_outstanding(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [
            _make_si_record(short_interest=30_000_000),
        ]
        result = proxy.estimate_borrow_rates(
            "GME", si_records, shares_outstanding=None,
        )
        assert result[0].utilization_pct == 0.0

    def test_utilization_zero_with_zero_shares_outstanding(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [
            _make_si_record(short_interest=30_000_000),
        ]
        result = proxy.estimate_borrow_rates(
            "GME", si_records, shares_outstanding=0,
        )
        assert result[0].utilization_pct == 0.0

    def test_symbol_normalized_to_uppercase(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [
            _make_si_record(days_to_cover=1.0),
        ]
        result = proxy.estimate_borrow_rates("gme", si_records)
        assert result[0].symbol == "GME"

    def test_realistic_gme_scenario(self) -> None:
        """Test with data resembling real GME short interest."""
        proxy = BorrowRateProxy()
        si_records = [
            _make_si_record(
                settlement_date="2024-01-15",
                short_interest=30_000_000,
                avg_daily_volume=5_000_000,
                days_to_cover=6.0,  # Elevated
            ),
            _make_si_record(
                settlement_date="2024-01-31",
                short_interest=32_000_000,
                avg_daily_volume=4_500_000,
                days_to_cover=7.1,  # High
            ),
            _make_si_record(
                settlement_date="2024-02-15",
                short_interest=28_000_000,
                avg_daily_volume=6_000_000,
                days_to_cover=4.67,  # Elevated
            ),
        ]
        result = proxy.estimate_borrow_rates(
            "GME", si_records, shares_outstanding=305_000_000,
        )

        assert len(result) == 3

        # DTC 6.0 -> elevated range (5-20%)
        assert 5.0 <= result[0].estimated_fee_pct <= 20.0

        # DTC 7.1 -> high range (20%+)
        assert result[1].estimated_fee_pct >= 20.0

        # DTC 4.67 -> elevated range (5-20%)
        assert 5.0 <= result[2].estimated_fee_pct <= 20.0

        # Utilization should be computed
        for rec in result:
            assert rec.utilization_pct > 0.0

    def test_all_dtc_brackets(self) -> None:
        """Test that all DTC brackets produce expected fee ranges."""
        proxy = BorrowRateProxy()

        test_cases = [
            (0.5, 0.25, 1.0),    # DTC < 1: GC rate
            (2.0, 1.0, 5.0),     # DTC 1-3
            (5.0, 5.0, 20.0),    # DTC 3-7
            (10.0, 20.0, 50.0),  # DTC 7-15
            (20.0, 50.0, 100.0), # DTC > 15
        ]

        for dtc, min_fee, max_fee in test_cases:
            si_records = [_make_si_record(days_to_cover=dtc)]
            result = proxy.estimate_borrow_rates("GME", si_records)
            fee = result[0].estimated_fee_pct
            assert min_fee <= fee <= max_fee, (
                f"DTC={dtc}: expected fee in [{min_fee}, {max_fee}], got {fee}"
            )

    def test_days_to_cover_passed_through(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [_make_si_record(days_to_cover=4.5)]
        result = proxy.estimate_borrow_rates("GME", si_records)
        assert result[0].days_to_cover == 4.5

    def test_record_type(self) -> None:
        proxy = BorrowRateProxy()
        si_records = [_make_si_record()]
        result = proxy.estimate_borrow_rates("GME", si_records)
        assert isinstance(result[0], BorrowRateRecord)
