"""Date-aligned store for alternative market data.

Consolidates data from multiple sources (SEC FTD, FINRA short interest,
dark pool volume, 13F ownership, borrow rate proxies) into a unified
date-indexed lookup with forward-fill for sparse data.

Each data source reports at a different frequency:
  - FTD: daily settlement dates
  - Short interest: bi-monthly (15th and end of month)
  - Dark pool: weekly
  - 13F ownership: quarterly
  - Borrow rate: derived from short interest (same frequency)

The store forward-fills missing dates so every trading day has a
complete :class:`AlternativeDataSnapshot`.

Usage::

    store = AlternativeDataStore()
    store.load(
        symbol="GME",
        ftd_records=ftd_list,
        si_records=si_list,
        price_dates=["2024-01-02", "2024-01-03", ...],
    )
    snap = store.get("2024-01-15")
    print(snap.ftd_quantity, snap.short_interest_pct)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stockdownloader.model.regulatory_records import (
        BorrowRateRecord,
        DarkPoolRecord,
        FtdRecord,
        ShortInterestRecord,
    )
    from stockdownloader.model.institutional_holding import OwnershipSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlternativeDataSnapshot:
    """All alternative data values for a single trading date."""

    date: str = ""

    # FTD features
    ftd_quantity: int = 0
    ftd_value: float = 0.0  # quantity * price
    ftd_quantity_prev: int = 0  # previous day (for change calc)

    # Short interest features
    short_interest: int = 0
    short_interest_pct: float = 0.0
    days_to_cover: float = 0.0
    short_interest_prev: int = 0  # previous report (for change calc)
    days_to_cover_prev: float = 0.0

    # Dark pool features
    dark_pool_volume: int = 0
    dark_pool_pct: float = 0.0  # ATS % of total
    dark_pool_volume_prev: int = 0  # previous week

    # Ownership features
    institutional_ownership_pct: float = 0.0
    ownership_concentration_top10: float = 0.0
    num_institutions: int = 0

    # Borrow rate proxy
    borrow_rate_proxy: float = 0.0
    borrow_rate_prev: float = 0.0


# ------------------------------------------------------------------
# Internal typed-dict aliases for per-source raw values
# ------------------------------------------------------------------

_FtdVal = tuple[int, float]  # (quantity, value)
_SiVal = tuple[int, float, float]  # (short_interest, si_pct, dtc)
_DpVal = tuple[int, float]  # (volume, ats_pct)
_OwnVal = tuple[float, float, int]  # (inst_pct, concentration, num_inst)
_BrVal = float  # fee_pct


class AlternativeDataStore:
    """Date-indexed store with forward-fill for alternative data."""

    def __init__(self) -> None:
        self._snapshots: dict[str, AlternativeDataSnapshot] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(
        self,
        symbol: str,
        ftd_records: list[FtdRecord] | None = None,
        si_records: list[ShortInterestRecord] | None = None,
        dp_records: list[DarkPoolRecord] | None = None,
        ownership: list[OwnershipSnapshot] | None = None,
        borrow_rates: list[BorrowRateRecord] | None = None,
        price_dates: list[str] | None = None,
    ) -> None:
        """Load all alternative data, align to trading dates, forward-fill.

        Parameters
        ----------
        symbol:
            Ticker symbol (used only for logging).
        ftd_records:
            SEC Failure-to-Deliver records.
        si_records:
            FINRA short interest reports.
        dp_records:
            FINRA OTC/ATS (dark pool) weekly records.
        ownership:
            Aggregated 13F ownership snapshots per quarter.
        borrow_rates:
            Estimated borrow rate records.
        price_dates:
            List of trading dates (``"YYYY-MM-DD"`` strings) that define
            the output calendar.  If *None*, dates are collected from the
            records themselves.
        """
        # 1. Build per-source date-indexed dicts
        ftd_map = self._build_ftd_map(ftd_records)
        si_map = self._build_si_map(si_records)
        dp_map = self._build_dp_map(dp_records)
        own_map = self._build_ownership_map(ownership)
        br_map = self._build_borrow_map(borrow_rates)

        # Log record counts
        logger.info(
            "%s alternative data: FTD=%d SI=%d DP=%d OWN=%d BR=%d",
            symbol,
            len(ftd_map),
            len(si_map),
            len(dp_map),
            len(own_map),
            len(br_map),
        )

        # 2. Determine trading-date calendar
        if price_dates is not None:
            all_dates = sorted(set(price_dates))
        else:
            all_dates = self._collect_dates(
                ftd_map, si_map, dp_map, own_map, br_map
            )

        if not all_dates:
            self._loaded = True
            return

        # 3. Forward-fill each source across trading dates and build snapshots
        self._snapshots = self._forward_fill_and_merge(
            all_dates, ftd_map, si_map, dp_map, own_map, br_map
        )

        logger.info(
            "%s alternative store built: %d dates",
            symbol,
            len(self._snapshots),
        )
        self._loaded = True

    def get(self, date: str) -> AlternativeDataSnapshot | None:
        """Look up snapshot for a specific date."""
        return self._snapshots.get(date)

    def get_or_default(self, date: str) -> AlternativeDataSnapshot:
        """Return snapshot or zero-filled default."""
        return self._snapshots.get(date, AlternativeDataSnapshot(date=date))

    @property
    def dates(self) -> list[str]:
        """All dates with data, sorted ascending."""
        return sorted(self._snapshots.keys())

    @property
    def record_count(self) -> int:
        return len(self._snapshots)

    # ------------------------------------------------------------------
    # Per-source map builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ftd_map(
        records: list[FtdRecord] | None,
    ) -> dict[str, _FtdVal]:
        if not records:
            return {}
        out: dict[str, _FtdVal] = {}
        for r in records:
            qty = r.quantity
            value = float(qty) * float(r.price)
            date = r.settlement_date
            # If multiple FTD records share the same date, sum them
            if date in out:
                prev_qty, prev_val = out[date]
                out[date] = (prev_qty + qty, prev_val + value)
            else:
                out[date] = (qty, value)
        return out

    @staticmethod
    def _build_si_map(
        records: list[ShortInterestRecord] | None,
    ) -> dict[str, _SiVal]:
        if not records:
            return {}
        out: dict[str, _SiVal] = {}
        for r in records:
            out[r.settlement_date] = (
                r.short_interest,
                r.short_interest_pct,
                r.days_to_cover,
            )
        return out

    @staticmethod
    def _build_dp_map(
        records: list[DarkPoolRecord] | None,
    ) -> dict[str, _DpVal]:
        if not records:
            return {}
        out: dict[str, _DpVal] = {}
        for r in records:
            out[r.week_ending] = (r.ats_volume, r.ats_pct)
        return out

    @staticmethod
    def _build_ownership_map(
        records: list[OwnershipSnapshot] | None,
    ) -> dict[str, _OwnVal]:
        if not records:
            return {}
        out: dict[str, _OwnVal] = {}
        for r in records:
            # OwnershipSnapshot doesn't carry institutional_ownership_pct
            # directly; we store the concentration and count.
            # The caller may pre-compute pct; we use top_10_concentration
            # as a proxy if institutional_ownership_pct isn't available.
            out[r.quarter_end] = (
                0.0,  # institutional_ownership_pct (to be set by caller)
                r.top_10_concentration,
                r.num_institutions,
            )
        return out

    @staticmethod
    def _build_borrow_map(
        records: list[BorrowRateRecord] | None,
    ) -> dict[str, _BrVal]:
        if not records:
            return {}
        return {r.date: r.estimated_fee_pct for r in records}

    # ------------------------------------------------------------------
    # Date collection (when price_dates not provided)
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_dates(*maps: dict[str, Any]) -> list[str]:
        """Gather all unique dates from all source maps, sorted."""
        dates: set[str] = set()
        for m in maps:
            dates.update(m.keys())
        return sorted(dates)

    # ------------------------------------------------------------------
    # Forward-fill and merge
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_fill_and_merge(
        all_dates: list[str],
        ftd_map: dict[str, _FtdVal],
        si_map: dict[str, _SiVal],
        dp_map: dict[str, _DpVal],
        own_map: dict[str, _OwnVal],
        br_map: dict[str, _BrVal],
    ) -> dict[str, AlternativeDataSnapshot]:
        """Forward-fill each source and build per-date snapshots.

        For each trading date the most recent available observation from
        each source is carried forward.  "Previous" fields track the
        observation *before* the current one so downstream code can
        compute deltas.
        """
        snapshots: dict[str, AlternativeDataSnapshot] = {}

        # Running state for forward-fill: current value + previous value
        cur_ftd: _FtdVal = (0, 0.0)
        prev_ftd_qty: int = 0

        cur_si: _SiVal = (0, 0.0, 0.0)
        prev_si: int = 0
        prev_dtc: float = 0.0

        cur_dp: _DpVal = (0, 0.0)
        prev_dp_vol: int = 0

        cur_own: _OwnVal = (0.0, 0.0, 0)

        cur_br: float = 0.0
        prev_br: float = 0.0

        for date in all_dates:
            # --- FTD ---
            if date in ftd_map:
                prev_ftd_qty = cur_ftd[0]
                cur_ftd = ftd_map[date]

            # --- Short interest ---
            if date in si_map:
                prev_si = cur_si[0]
                prev_dtc = cur_si[2]
                cur_si = si_map[date]

            # --- Dark pool ---
            if date in dp_map:
                prev_dp_vol = cur_dp[0]
                cur_dp = dp_map[date]

            # --- Ownership ---
            if date in own_map:
                cur_own = own_map[date]

            # --- Borrow rate ---
            if date in br_map:
                prev_br = cur_br
                cur_br = br_map[date]

            snapshots[date] = AlternativeDataSnapshot(
                date=date,
                # FTD
                ftd_quantity=cur_ftd[0],
                ftd_value=cur_ftd[1],
                ftd_quantity_prev=prev_ftd_qty,
                # SI
                short_interest=cur_si[0],
                short_interest_pct=cur_si[1],
                days_to_cover=cur_si[2],
                short_interest_prev=prev_si,
                days_to_cover_prev=prev_dtc,
                # DP
                dark_pool_volume=cur_dp[0],
                dark_pool_pct=cur_dp[1],
                dark_pool_volume_prev=prev_dp_vol,
                # Ownership
                institutional_ownership_pct=cur_own[0],
                ownership_concentration_top10=cur_own[1],
                num_institutions=cur_own[2],
                # Borrow
                borrow_rate_proxy=cur_br,
                borrow_rate_prev=prev_br,
            )

        return snapshots
