"""Estimates stock borrow fees from short interest data.

Pure computation module -- no network calls.  Uses a heuristic mapping
from days-to-cover (DTC) to estimated annual borrow fee percentage,
approximating the relationship between short demand and lending supply.

Usage::

    from stockdownloader.data.borrow_rate_proxy import BorrowRateProxy
    from stockdownloader.data.finra_short_interest_client import FinraShortInterestClient

    si_client = FinraShortInterestClient()
    records = si_client.fetch_short_interest("GME")

    proxy = BorrowRateProxy()
    borrow_rates = proxy.estimate_borrow_rates("GME", records,
                                                shares_outstanding=305_000_000)
"""

from __future__ import annotations

import logging

from stockdownloader.model.regulatory_records import BorrowRateRecord, ShortInterestRecord

logger = logging.getLogger(__name__)


class BorrowRateProxy:
    """Estimates stock borrow fees from short interest data.

    Uses a heuristic mapping based on days-to-cover::

        DTC < 1:   ~0.25% (easy to borrow, GC rate)
        DTC 1-3:   ~1.0%
        DTC 3-7:   ~5.0%
        DTC 7-15:  ~20.0%
        DTC > 15:  ~50.0%+ (hard to borrow)

    These are rough estimates.  Actual borrow fees depend on
    broker inventory, demand, and market conditions.
    """

    def estimate_borrow_rates(
        self,
        symbol: str,
        si_records: list[ShortInterestRecord],
        shares_outstanding: int | None = None,
    ) -> list[BorrowRateRecord]:
        """Estimate borrow rates from short interest data.

        Parameters
        ----------
        symbol:
            Ticker symbol for labeling.
        si_records:
            Short interest records (must have ``days_to_cover`` and
            ``short_interest`` fields populated).
        shares_outstanding:
            Total shares outstanding.  Used to compute utilization
            percentage.  If ``None``, utilization is set to 0.0.

        Returns
        -------
        List of :class:`BorrowRateRecord` sorted by date ascending.
        """
        symbol_upper = symbol.upper()
        records: list[BorrowRateRecord] = []

        for si in si_records:
            dtc = si.days_to_cover
            fee = self._dtc_to_fee(dtc)

            # Compute utilization if shares outstanding is known
            utilization = 0.0
            if shares_outstanding and shares_outstanding > 0:
                utilization = si.short_interest / shares_outstanding

            try:
                records.append(BorrowRateRecord(
                    date=si.settlement_date,
                    symbol=symbol_upper,
                    estimated_fee_pct=fee,
                    days_to_cover=dtc,
                    utilization_pct=utilization,
                ))
            except ValueError as exc:
                logger.debug(
                    "Skipping borrow rate record: %s", exc,
                )
                continue

        records.sort(key=lambda r: r.date)
        return records

    @staticmethod
    def _dtc_to_fee(dtc: float) -> float:
        """Convert days-to-cover to estimated annual borrow fee %.

        Uses a piecewise linear interpolation across the DTC
        spectrum.  Returns the estimated fee as a percentage
        (e.g., 5.0 means 5.0% annual).

        Parameters
        ----------
        dtc:
            Days-to-cover ratio (short interest / avg daily volume).

        Returns
        -------
        Estimated annual borrow fee percentage.
        """
        if dtc < 0:
            return 0.25

        if dtc < 1.0:
            # Easy to borrow: GC rate range (0.25% - 1.0%)
            return 0.25 + dtc * 0.75

        if dtc < 3.0:
            # Moderate demand: 1.0% - 5.0%
            return 1.0 + (dtc - 1.0) * 2.0

        if dtc < 7.0:
            # Elevated demand: 5.0% - 20.0%
            return 5.0 + (dtc - 3.0) * 3.75

        if dtc < 15.0:
            # High demand: 20.0% - 50.0%
            return 20.0 + (dtc - 7.0) * 3.75

        # Very hard to borrow: 50%+ (capped at 100%)
        fee = 50.0 + (dtc - 15.0) * 5.0
        return min(fee, 100.0)
