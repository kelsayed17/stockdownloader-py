"""Filing-price correlation and statistical event study for SEC filings.

Includes filing correlation helpers (formerly filings.py) and
Campbell-Lo-MacKinlay event study analysis.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict

from scipy.stats import t as t_dist

from stockdownloader.gme.analysis.models import EventStudyResult, FilingImpact
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.models.regulatory import SecFiling

__all__ = [
    "correlate_filings_with_price",
    "run_event_study",
]


# =========================================================================
# Filing-price correlation (formerly filings.py)
# =========================================================================


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


# =========================================================================
# Event study (Campbell-Lo-MacKinlay)
# =========================================================================


def _compute_log_returns(daily_data: list[PriceData]) -> list[float]:
    """Compute daily log returns from close prices."""
    returns: list[float] = []
    for i in range(1, len(daily_data)):
        prev = float(daily_data[i - 1].close)
        curr = float(daily_data[i].close)
        if prev > 0 and curr > 0:
            returns.append(math.log(curr / prev))
    return returns


def _autocorrelation(series: list[float], max_lag: int = 5) -> tuple[float, ...]:
    """Compute autocorrelation at lags 1 through *max_lag*."""
    n = len(series)
    if n < max_lag + 2:
        return tuple(0.0 for _ in range(max_lag))
    mean = sum(series) / n
    var = sum((x - mean) ** 2 for x in series) / n
    if var == 0:
        return tuple(0.0 for _ in range(max_lag))
    result: list[float] = []
    for lag in range(1, max_lag + 1):
        cov = sum(
            (series[i] - mean) * (series[i + lag] - mean)
            for i in range(n - lag)
        ) / (n - lag)
        result.append(round(cov / var, 4))
    return tuple(result)


def _ols_market_model(
    stock_returns: list[float],
    bench_returns: list[float],
) -> tuple[float, float]:
    """Estimate alpha, beta via OLS: R_stock = alpha + beta * R_bench."""
    n = len(stock_returns)
    if n < 2:
        return 0.0, 1.0
    mean_s = sum(stock_returns) / n
    mean_b = sum(bench_returns) / n
    cov_sb = sum(
        (s - mean_s) * (b - mean_b)
        for s, b in zip(stock_returns, bench_returns)
    ) / (n - 1)
    var_b = sum((b - mean_b) ** 2 for b in bench_returns) / (n - 1)
    beta = cov_sb / var_b if var_b > 0 else 0.0
    alpha = mean_s - beta * mean_b
    return alpha, beta


def run_event_study(
    filings: list[SecFiling],
    stock_data: list[PriceData],
    benchmark_data: list[PriceData],
    estimation_window: int = 120,
    pre_event: int = 5,
    post_event: int = 20,
) -> list[EventStudyResult]:
    """Run a Campbell-Lo-MacKinlay event study on SEC filings.

    For each filing, estimates a market model from the *estimation_window*
    before the event, then computes cumulative abnormal returns (CAR)
    over [0, +post_event] trading days.  Results are grouped by form type.
    """
    if not filings or not stock_data or not benchmark_data:
        return []

    # Build date-aligned return series
    stock_date_idx = {d.date: i for i, d in enumerate(stock_data)}
    bench_date_idx = {d.date: i for i, d in enumerate(benchmark_data)}
    stock_sorted_dates = sorted(stock_date_idx)

    stock_returns = _compute_log_returns(stock_data)
    bench_returns = _compute_log_returns(benchmark_data)

    # Returns are indexed offset by 1: stock_returns[i] = log(close[i+1]/close[i])
    # So return at stock index j corresponds to stock_returns[j-1]

    # Group individual CARs by form type
    form_cars: dict[str, list[tuple[str, float, float, float]]] = defaultdict(list)
    # Each entry: (filing_date, car_20d, car_1d, car_5d)

    for filing in filings:
        idx = _find_nearest_index(filing.filing_date, stock_date_idx, stock_sorted_dates)
        if idx is None:
            continue

        # We need: estimation window of `estimation_window` days ending at idx - pre_event - 1
        # And event window of [idx, idx + post_event]
        est_end = idx - pre_event - 1  # last day of estimation window (return index)
        est_start = est_end - estimation_window + 1

        if est_start < 1 or idx + post_event >= len(stock_data):
            continue  # Not enough data

        # Collect aligned returns for estimation window
        est_stock_rets: list[float] = []
        est_bench_rets: list[float] = []
        for j in range(est_start, est_end + 1):
            s_date = stock_data[j].date
            if s_date in bench_date_idx:
                b_idx = bench_date_idx[s_date]
                if 1 <= j < len(stock_data) and 1 <= b_idx < len(benchmark_data):
                    sr = stock_returns[j - 1] if j - 1 < len(stock_returns) else 0.0
                    br = bench_returns[b_idx - 1] if b_idx - 1 < len(bench_returns) else 0.0
                    est_stock_rets.append(sr)
                    est_bench_rets.append(br)

        if len(est_stock_rets) < 30:
            continue  # Need minimum estimation data

        alpha, beta = _ols_market_model(est_stock_rets, est_bench_rets)

        # Compute abnormal returns in event window [idx, idx + post_event]
        car = 0.0
        car_1d = 0.0
        car_5d = 0.0
        for t in range(idx, min(idx + post_event + 1, len(stock_data))):
            if t < 1:
                continue
            s_date = stock_data[t].date
            sr = stock_returns[t - 1] if t - 1 < len(stock_returns) else 0.0
            # Find matching benchmark return
            br = 0.0
            if s_date in bench_date_idx:
                b_idx = bench_date_idx[s_date]
                if 1 <= b_idx and b_idx - 1 < len(bench_returns):
                    br = bench_returns[b_idx - 1]
            expected = alpha + beta * br
            ar = sr - expected
            car += ar
            days_from_event = t - idx
            if days_from_event <= 1:
                car_1d += ar
            if days_from_event <= 5:
                car_5d += ar

        form_cars[filing.form].append((filing.filing_date, car, car_1d, car_5d))

    # Build results per form type
    results: list[EventStudyResult] = []
    for form_type, events in form_cars.items():
        n = len(events)
        if n < 2:
            continue
        cars = [e[1] for e in events]
        cars_1d = [e[2] for e in events]
        cars_5d = [e[3] for e in events]

        mean_c = sum(cars) / n
        std_c = math.sqrt(sum((c - mean_c) ** 2 for c in cars) / (n - 1))
        median_c = sorted(cars)[n // 2]

        if std_c > 0:
            t_stat_val = mean_c / (std_c / math.sqrt(n))
            p_val = float(2 * t_dist.sf(abs(t_stat_val), df=n - 1))
        else:
            t_stat_val = 0.0
            p_val = 1.0

        results.append(EventStudyResult(
            form_type=form_type,
            event_count=n,
            mean_car=round(mean_c, 6),
            std_car=round(std_c, 6),
            t_stat=round(t_stat_val, 4),
            p_value=round(p_val, 4),
            median_car=round(median_c, 6),
            mean_car_1d=round(sum(cars_1d) / n, 6),
            mean_car_5d=round(sum(cars_5d) / n, 6),
            individual_cars=tuple((e[0], round(e[1], 6)) for e in events),
        ))

    results.sort(key=lambda r: abs(r.t_stat), reverse=True)
    return results
