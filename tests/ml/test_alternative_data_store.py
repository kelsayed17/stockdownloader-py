"""Tests for AlternativeDataStore — date-aligned forward-fill store."""

from __future__ import annotations

from decimal import Decimal

from stockdownloader.ml.alternative_data_store import (
    AlternativeDataSnapshot,
    AlternativeDataStore,
)
from stockdownloader.model.regulatory_records import (
    BorrowRateRecord,
    DarkPoolRecord,
    FtdRecord,
    ShortInterestRecord,
)
from stockdownloader.model.institutional_holding import (
    InstitutionalHolding,
    OwnershipSnapshot,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

TRADING_DATES = [
    "2024-01-02",
    "2024-01-03",
    "2024-01-04",
    "2024-01-05",
    "2024-01-08",
    "2024-01-09",
    "2024-01-10",
    "2024-01-11",
    "2024-01-12",
    "2024-01-15",
    "2024-01-16",
    "2024-01-17",
    "2024-01-18",
    "2024-01-19",
]


def _ftd(date: str, qty: int, price: float = 20.0) -> FtdRecord:
    return FtdRecord(
        settlement_date=date,
        symbol="GME",
        cusip="36467W109",
        quantity=qty,
        description="GAMESTOP CORP",
        price=Decimal(str(price)),
    )


def _si(
    date: str,
    si: int = 10_000_000,
    si_pct: float = 15.0,
    dtc: float = 2.5,
) -> ShortInterestRecord:
    return ShortInterestRecord(
        settlement_date=date,
        symbol="GME",
        short_interest=si,
        avg_daily_volume=4_000_000,
        days_to_cover=dtc,
        short_interest_pct=si_pct,
    )


def _dp(
    week_ending: str,
    ats_vol: int = 500_000,
    ats_pct: float = 0.35,
) -> DarkPoolRecord:
    return DarkPoolRecord(
        week_ending=week_ending,
        symbol="GME",
        total_weekly_volume=1_500_000,
        ats_volume=ats_vol,
        otc_volume=1_000_000,
        ats_pct=ats_pct,
    )


def _own(
    quarter_end: str,
    concentration: float = 0.45,
    num_inst: int = 120,
) -> OwnershipSnapshot:
    return OwnershipSnapshot(
        quarter_end=quarter_end,
        symbol="GME",
        total_institutional_shares=30_000_000,
        num_institutions=num_inst,
        top_10_concentration=concentration,
        holdings=(),
    )


def _br(date: str, fee: float = 5.0) -> BorrowRateRecord:
    return BorrowRateRecord(
        date=date,
        symbol="GME",
        estimated_fee_pct=fee,
        days_to_cover=2.5,
        utilization_pct=0.80,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestEmptyStore:
    def test_not_loaded_initially(self) -> None:
        store = AlternativeDataStore()
        assert store.is_loaded is False

    def test_get_returns_none(self) -> None:
        store = AlternativeDataStore()
        assert store.get("2024-01-02") is None

    def test_get_or_default_returns_zeros(self) -> None:
        store = AlternativeDataStore()
        snap = store.get_or_default("2024-01-02")
        assert snap.date == "2024-01-02"
        assert snap.ftd_quantity == 0
        assert snap.short_interest == 0
        assert snap.dark_pool_volume == 0
        assert snap.borrow_rate_proxy == 0.0

    def test_dates_empty(self) -> None:
        store = AlternativeDataStore()
        assert store.dates == []
        assert store.record_count == 0

    def test_load_empty_sets_loaded(self) -> None:
        store = AlternativeDataStore()
        store.load(symbol="GME", price_dates=[])
        assert store.is_loaded is True
        assert store.record_count == 0

    def test_load_none_lists(self) -> None:
        store = AlternativeDataStore()
        store.load(symbol="GME", price_dates=TRADING_DATES)
        assert store.is_loaded is True
        # All snapshots exist but with zero values
        assert store.record_count == len(TRADING_DATES)
        for d in TRADING_DATES:
            snap = store.get(d)
            assert snap is not None
            assert snap.ftd_quantity == 0


class TestFtdOnly:
    def test_ftd_values_aligned(self) -> None:
        store = AlternativeDataStore()
        ftd_records = [
            _ftd("2024-01-03", 50_000, 20.0),
            _ftd("2024-01-10", 120_000, 22.0),
        ]
        store.load(
            symbol="GME",
            ftd_records=ftd_records,
            price_dates=TRADING_DATES,
        )
        assert store.is_loaded is True

        # Before any FTD data: zeros
        snap_02 = store.get("2024-01-02")
        assert snap_02 is not None
        assert snap_02.ftd_quantity == 0
        assert snap_02.ftd_value == 0.0

        # On FTD date
        snap_03 = store.get("2024-01-03")
        assert snap_03 is not None
        assert snap_03.ftd_quantity == 50_000
        assert snap_03.ftd_value == 50_000 * 20.0

        # Forward-filled between FTD dates
        snap_05 = store.get("2024-01-05")
        assert snap_05 is not None
        assert snap_05.ftd_quantity == 50_000  # carried forward

        # New FTD date
        snap_10 = store.get("2024-01-10")
        assert snap_10 is not None
        assert snap_10.ftd_quantity == 120_000
        assert snap_10.ftd_value == 120_000 * 22.0

        # Forward-filled after second FTD
        snap_15 = store.get("2024-01-15")
        assert snap_15 is not None
        assert snap_15.ftd_quantity == 120_000

    def test_ftd_prev_tracked(self) -> None:
        store = AlternativeDataStore()
        ftd_records = [
            _ftd("2024-01-03", 50_000),
            _ftd("2024-01-10", 120_000),
        ]
        store.load(
            symbol="GME",
            ftd_records=ftd_records,
            price_dates=TRADING_DATES,
        )
        # Before first FTD: prev is 0
        snap_03 = store.get("2024-01-03")
        assert snap_03 is not None
        assert snap_03.ftd_quantity_prev == 0

        # After second FTD: prev tracks the first
        snap_10 = store.get("2024-01-10")
        assert snap_10 is not None
        assert snap_10.ftd_quantity_prev == 50_000

    def test_ftd_same_date_sums(self) -> None:
        """Multiple FTD records on the same date should sum."""
        store = AlternativeDataStore()
        ftd_records = [
            _ftd("2024-01-03", 30_000, 20.0),
            _ftd("2024-01-03", 20_000, 25.0),
        ]
        store.load(
            symbol="GME",
            ftd_records=ftd_records,
            price_dates=TRADING_DATES,
        )
        snap = store.get("2024-01-03")
        assert snap is not None
        assert snap.ftd_quantity == 50_000
        assert snap.ftd_value == 30_000 * 20.0 + 20_000 * 25.0

    def test_other_fields_zero_when_ftd_only(self) -> None:
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            ftd_records=[_ftd("2024-01-05", 100_000)],
            price_dates=TRADING_DATES,
        )
        snap = store.get("2024-01-05")
        assert snap is not None
        assert snap.short_interest == 0
        assert snap.dark_pool_volume == 0
        assert snap.institutional_ownership_pct == 0.0
        assert snap.borrow_rate_proxy == 0.0


class TestAllSources:
    def test_all_sources_merged(self) -> None:
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            ftd_records=[_ftd("2024-01-03", 50_000)],
            si_records=[_si("2024-01-05", si=12_000_000, si_pct=18.0, dtc=3.0)],
            dp_records=[_dp("2024-01-05", ats_vol=600_000, ats_pct=0.40)],
            ownership=[_own("2024-01-02", concentration=0.50, num_inst=130)],
            borrow_rates=[_br("2024-01-04", fee=7.5)],
            price_dates=TRADING_DATES,
        )

        # Pick a date after all sources have reported
        snap = store.get("2024-01-08")
        assert snap is not None

        # FTD: forward-filled from Jan 3
        assert snap.ftd_quantity == 50_000

        # SI: forward-filled from Jan 5
        assert snap.short_interest == 12_000_000
        assert snap.short_interest_pct == 18.0
        assert snap.days_to_cover == 3.0

        # DP: forward-filled from Jan 5
        assert snap.dark_pool_volume == 600_000
        assert snap.dark_pool_pct == 0.40

        # Ownership: forward-filled from Jan 2
        assert snap.ownership_concentration_top10 == 0.50
        assert snap.num_institutions == 130

        # Borrow: forward-filled from Jan 4
        assert snap.borrow_rate_proxy == 7.5

    def test_record_count_matches_dates(self) -> None:
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            ftd_records=[_ftd("2024-01-03", 50_000)],
            si_records=[_si("2024-01-15")],
            price_dates=TRADING_DATES,
        )
        assert store.record_count == len(TRADING_DATES)


class TestForwardFill:
    def test_sparse_si_fills_forward(self) -> None:
        """Short interest reported bi-monthly fills all dates in between."""
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            si_records=[
                _si("2024-01-05", si=10_000_000, si_pct=15.0, dtc=2.5),
                _si("2024-01-15", si=11_000_000, si_pct=16.5, dtc=2.8),
            ],
            price_dates=TRADING_DATES,
        )

        # Before first SI report: zeros
        snap_02 = store.get("2024-01-02")
        assert snap_02 is not None
        assert snap_02.short_interest == 0

        # First report
        snap_05 = store.get("2024-01-05")
        assert snap_05 is not None
        assert snap_05.short_interest == 10_000_000

        # Between reports: forward-filled from first
        snap_10 = store.get("2024-01-10")
        assert snap_10 is not None
        assert snap_10.short_interest == 10_000_000
        assert snap_10.short_interest_pct == 15.0

        # Second report
        snap_15 = store.get("2024-01-15")
        assert snap_15 is not None
        assert snap_15.short_interest == 11_000_000
        assert snap_15.short_interest_pct == 16.5

    def test_sparse_dp_fills_forward(self) -> None:
        """Weekly dark pool data fills forward to daily."""
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            dp_records=[
                _dp("2024-01-05", ats_vol=500_000, ats_pct=0.35),
                _dp("2024-01-12", ats_vol=700_000, ats_pct=0.40),
            ],
            price_dates=TRADING_DATES,
        )
        # Between weeks: forward-filled
        snap_09 = store.get("2024-01-09")
        assert snap_09 is not None
        assert snap_09.dark_pool_volume == 500_000

        # After second week
        snap_15 = store.get("2024-01-15")
        assert snap_15 is not None
        assert snap_15.dark_pool_volume == 700_000

    def test_sparse_ownership_fills_forward(self) -> None:
        """Quarterly ownership fills across all subsequent dates."""
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            ownership=[_own("2024-01-02", concentration=0.45, num_inst=120)],
            price_dates=TRADING_DATES,
        )
        # Every date after the quarter_end should carry forward
        for d in TRADING_DATES:
            snap = store.get(d)
            assert snap is not None
            assert snap.ownership_concentration_top10 == 0.45
            assert snap.num_institutions == 120


class TestDatesBeforeData:
    def test_zeros_before_first_observation(self) -> None:
        """Dates before any source has data should get zeros."""
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            ftd_records=[_ftd("2024-01-10", 80_000)],
            si_records=[_si("2024-01-15")],
            borrow_rates=[_br("2024-01-12", fee=6.0)],
            price_dates=TRADING_DATES,
        )
        snap = store.get("2024-01-02")
        assert snap is not None
        assert snap.ftd_quantity == 0
        assert snap.ftd_value == 0.0
        assert snap.short_interest == 0
        assert snap.borrow_rate_proxy == 0.0
        assert snap.dark_pool_volume == 0

    def test_partial_coverage(self) -> None:
        """Some sources have data, others don't yet at a given date."""
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            ftd_records=[_ftd("2024-01-03", 50_000)],
            si_records=[_si("2024-01-10")],
            price_dates=TRADING_DATES,
        )
        # Jan 5: FTD filled, SI not yet
        snap_05 = store.get("2024-01-05")
        assert snap_05 is not None
        assert snap_05.ftd_quantity == 50_000
        assert snap_05.short_interest == 0

        # Jan 12: both filled
        snap_12 = store.get("2024-01-12")
        assert snap_12 is not None
        assert snap_12.ftd_quantity == 50_000
        assert snap_12.short_interest == 10_000_000


class TestPreviousValues:
    def test_si_prev_tracked(self) -> None:
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            si_records=[
                _si("2024-01-05", si=10_000_000, dtc=2.5),
                _si("2024-01-15", si=11_000_000, dtc=2.8),
            ],
            price_dates=TRADING_DATES,
        )
        # First report: prev is zero
        snap_05 = store.get("2024-01-05")
        assert snap_05 is not None
        assert snap_05.short_interest_prev == 0
        assert snap_05.days_to_cover_prev == 0.0

        # Between reports: prev stays zero (only updates on new observations)
        snap_10 = store.get("2024-01-10")
        assert snap_10 is not None
        assert snap_10.short_interest_prev == 0

        # Second report: prev is the first report's value
        snap_15 = store.get("2024-01-15")
        assert snap_15 is not None
        assert snap_15.short_interest_prev == 10_000_000
        assert snap_15.days_to_cover_prev == 2.5

    def test_dp_prev_tracked(self) -> None:
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            dp_records=[
                _dp("2024-01-05", ats_vol=500_000),
                _dp("2024-01-12", ats_vol=700_000),
            ],
            price_dates=TRADING_DATES,
        )
        snap_05 = store.get("2024-01-05")
        assert snap_05 is not None
        assert snap_05.dark_pool_volume_prev == 0

        snap_12 = store.get("2024-01-12")
        assert snap_12 is not None
        assert snap_12.dark_pool_volume_prev == 500_000

    def test_borrow_prev_tracked(self) -> None:
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            borrow_rates=[
                _br("2024-01-04", fee=5.0),
                _br("2024-01-11", fee=8.0),
            ],
            price_dates=TRADING_DATES,
        )
        snap_04 = store.get("2024-01-04")
        assert snap_04 is not None
        assert snap_04.borrow_rate_proxy == 5.0
        assert snap_04.borrow_rate_prev == 0.0

        snap_11 = store.get("2024-01-11")
        assert snap_11 is not None
        assert snap_11.borrow_rate_proxy == 8.0
        assert snap_11.borrow_rate_prev == 5.0

    def test_ftd_prev_on_consecutive_days(self) -> None:
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            ftd_records=[
                _ftd("2024-01-03", 50_000),
                _ftd("2024-01-04", 60_000),
                _ftd("2024-01-05", 70_000),
            ],
            price_dates=TRADING_DATES,
        )
        snap_03 = store.get("2024-01-03")
        assert snap_03 is not None
        assert snap_03.ftd_quantity_prev == 0

        snap_04 = store.get("2024-01-04")
        assert snap_04 is not None
        assert snap_04.ftd_quantity_prev == 50_000

        snap_05 = store.get("2024-01-05")
        assert snap_05 is not None
        assert snap_05.ftd_quantity_prev == 60_000


class TestPriceDatesFiltering:
    def test_only_trading_dates_get_snapshots(self) -> None:
        """Non-trading dates from records should NOT appear in output."""
        store = AlternativeDataStore()
        # FTD on a Saturday (not in trading dates)
        store.load(
            symbol="GME",
            ftd_records=[
                _ftd("2024-01-03", 50_000),
                _ftd("2024-01-06", 90_000),  # Saturday, not in TRADING_DATES
            ],
            price_dates=TRADING_DATES,
        )
        assert store.record_count == len(TRADING_DATES)
        # Saturday should not be in the store
        assert store.get("2024-01-06") is None
        # But forward-fill should pick up Saturday's record
        # Actually, since Jan 6 isn't a price_date, we only scan
        # price_dates, so the Jan 6 FTD is never reached in forward-fill.
        # The store correctly only iterates over price_dates.
        # Jan 8 should still carry the Jan 3 value.
        snap_08 = store.get("2024-01-08")
        assert snap_08 is not None
        assert snap_08.ftd_quantity == 50_000

    def test_no_price_dates_collects_from_records(self) -> None:
        """When price_dates is None, dates are collected from records."""
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            ftd_records=[
                _ftd("2024-01-03", 50_000),
                _ftd("2024-01-10", 80_000),
            ],
            si_records=[_si("2024-01-05")],
        )
        # Should have exactly the unique dates from all records
        assert store.record_count == 3
        assert store.dates == ["2024-01-03", "2024-01-05", "2024-01-10"]

    def test_dates_property_sorted(self) -> None:
        store = AlternativeDataStore()
        store.load(
            symbol="GME",
            ftd_records=[_ftd("2024-01-10", 100), _ftd("2024-01-03", 200)],
            price_dates=["2024-01-10", "2024-01-03", "2024-01-05"],
        )
        assert store.dates == ["2024-01-03", "2024-01-05", "2024-01-10"]


class TestAlternativeDataSnapshot:
    def test_default_snapshot_all_zeros(self) -> None:
        snap = AlternativeDataSnapshot()
        assert snap.date == ""
        assert snap.ftd_quantity == 0
        assert snap.short_interest == 0
        assert snap.dark_pool_volume == 0
        assert snap.institutional_ownership_pct == 0.0
        assert snap.borrow_rate_proxy == 0.0

    def test_snapshot_is_frozen(self) -> None:
        snap = AlternativeDataSnapshot(date="2024-01-02", ftd_quantity=100)
        try:
            snap.ftd_quantity = 200  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass
