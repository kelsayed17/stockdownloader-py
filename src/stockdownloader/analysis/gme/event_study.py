"""Statistical event study (Campbell-Lo-MacKinlay) for SEC filings."""

from __future__ import annotations

import math
from collections import defaultdict

from scipy.stats import t as t_dist

from stockdownloader.analysis.gme.filings import _find_nearest_index
from stockdownloader.analysis.gme.models import EventStudyResult
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.sec_filing import SecFiling

__all__ = [
    "run_event_study",
]


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
