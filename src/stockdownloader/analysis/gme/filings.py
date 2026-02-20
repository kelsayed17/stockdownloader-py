"""Filing-price correlation and nearest-date lookup helpers."""

from __future__ import annotations

import bisect

from stockdownloader.analysis.gme.models import FilingImpact
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.sec_filing import SecFiling

__all__ = [
    "correlate_filings_with_price",
]


def correlate_filings_with_price(
    filings: list[SecFiling],
    daily_data: list[PriceData],
) -> list[FilingImpact]:
    """Correlate each filing with surrounding price action.

    For every filing, look up the filing_date in *daily_data* (falling
    back to the nearest subsequent trading day), then compute price
    changes and volume ratios.

    Returns a list of :class:`FilingImpact` sorted by filing_date.
    """
    if not filings or not daily_data:
        return []

    # Build date -> index mapping
    date_to_idx: dict[str, int] = {d.date: i for i, d in enumerate(daily_data)}

    # Sorted date list for nearest-day lookup
    sorted_dates = sorted(date_to_idx)

    results: list[FilingImpact] = []

    for filing in filings:
        idx = _find_nearest_index(filing.filing_date, date_to_idx, sorted_dates)
        if idx is None:
            continue

        bar = daily_data[idx]

        # Price before (1 trading day)
        price_1d_before = daily_data[idx - 1].close if idx >= 1 else bar.close

        # Price after (1 trading day)
        price_1d_after = daily_data[idx + 1].close if idx + 1 < len(daily_data) else bar.close

        # Price 5 trading days after
        idx_5d = min(idx + 5, len(daily_data) - 1)
        price_5d_after = daily_data[idx_5d].close

        # 20-day average volume before the filing
        vol_start = max(0, idx - 20)
        vol_window = [daily_data[j].volume for j in range(vol_start, idx)]
        avg_vol_20d = sum(vol_window) / len(vol_window) if vol_window else float(bar.volume)

        volume_ratio = float(bar.volume) / avg_vol_20d if avg_vol_20d > 0 else 0.0

        # Percentage changes vs prior close
        if price_1d_before > 0:
            change_1d = (float(bar.close) - float(price_1d_before)) / float(price_1d_before) * 100.0
            change_5d = (float(price_5d_after) - float(price_1d_before)) / float(price_1d_before) * 100.0
        else:
            change_1d = 0.0
            change_5d = 0.0

        results.append(FilingImpact(
            filing=filing,
            price_on_date=bar.close,
            price_1d_before=price_1d_before,
            price_1d_after=price_1d_after,
            price_5d_after=price_5d_after,
            volume_on_date=bar.volume,
            avg_volume_20d=avg_vol_20d,
            volume_ratio=round(volume_ratio, 2),
            price_change_1d_pct=round(change_1d, 2),
            price_change_5d_pct=round(change_5d, 2),
        ))

    results.sort(key=lambda r: r.filing.filing_date)
    return results


def _find_nearest_index(
    target_date: str,
    date_to_idx: dict[str, int],
    sorted_dates: list[str],
) -> int | None:
    """Find the index of *target_date* or the nearest subsequent trading day."""
    if target_date in date_to_idx:
        return date_to_idx[target_date]

    # Binary search for the next trading day after target_date
    pos = bisect.bisect_left(sorted_dates, target_date)
    if pos < len(sorted_dates):
        return date_to_idx[sorted_dates[pos]]
    # If target_date is after all data, use the last bar
    if sorted_dates:
        return date_to_idx[sorted_dates[-1]]
    return None
