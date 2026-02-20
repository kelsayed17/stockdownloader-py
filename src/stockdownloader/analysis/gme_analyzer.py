"""Analysis logic for the comprehensive GME analysis application.

Contains pure functions for:
- Statistical event study (Campbell-Lo-MacKinlay) for SEC filings
- Volatility regime detection (rolling realized vol)
- Return distribution analysis (skew, kurtosis, VaR, autocorrelation)
- Volume profile analysis (POC, value area, anomalies)
- Structural break detection (scipy.signal.find_peaks)
- Correlating SEC filings with surrounding price action (legacy)
- Detecting key periods (legacy)
- Analysing options chains (max pain, P/C ratios, unusual activity)
- Computing price statistics over a given history
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

from scipy.signal import find_peaks
from scipy.stats import (
    jarque_bera,
    kurtosis as scipy_kurtosis,
    skew as scipy_skew,
    t as t_dist,
)

from stockdownloader.model.options import OptionContract, OptionsChain
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.sec_filing import SecFiling

_ZERO = Decimal("0")


# ======================================================================
# Data classes
# ======================================================================


@dataclass(frozen=True, slots=True)
class FilingImpact:
    """SEC filing correlated with surrounding price action."""

    filing: SecFiling
    price_on_date: Decimal          # Close on filing date (or nearest trading day)
    price_1d_before: Decimal        # Close 1 trading day before
    price_1d_after: Decimal         # Close 1 trading day after
    price_5d_after: Decimal         # Close 5 trading days after
    volume_on_date: int             # Volume on filing date
    avg_volume_20d: float           # Average daily volume for 20 days before
    volume_ratio: float             # volume_on_date / avg_volume_20d
    price_change_1d_pct: float      # % change filing day (vs prior close)
    price_change_5d_pct: float      # % change 5d after (vs prior close)


@dataclass(frozen=True, slots=True)
class KeyPeriod:
    """A notable period in the price history where a large move occurred."""

    name: str
    start_date: str
    end_date: str
    start_price: Decimal
    end_price: Decimal
    price_change_pct: float
    peak_price: Decimal
    peak_date: str


@dataclass(frozen=True, slots=True)
class OptionsAnalysis:
    """Summary analysis of a current options chain snapshot."""

    underlying_price: Decimal
    total_call_volume: int
    total_put_volume: int
    total_call_oi: int
    total_put_oi: int
    put_call_volume_ratio: Decimal
    put_call_oi_ratio: Decimal
    max_pain_strike: Decimal
    nearest_expiry: str
    highest_oi_call_strike: Decimal
    highest_oi_put_strike: Decimal
    unusual_volume_contracts: list[tuple[str, Decimal, int, int]] = field(
        default_factory=list
    )  # (symbol, strike, volume, oi)
    iv_skew: float = 0.0  # ATM put IV minus ATM call IV


@dataclass(frozen=True, slots=True)
class PriceStatistics:
    """Summary statistics for a price history."""

    period_start: str
    period_end: str
    trading_days: int
    start_price: Decimal
    end_price: Decimal
    all_time_high: Decimal
    all_time_high_date: str
    all_time_low: Decimal
    all_time_low_date: str
    total_return_pct: float
    avg_daily_volume: int
    max_daily_volume: int
    max_volume_date: str
    volatility_annual: float  # annualised std dev of daily log returns


@dataclass(frozen=True, slots=True)
class EventStudyResult:
    """Cumulative Abnormal Return (CAR) event study for one form type."""

    form_type: str
    event_count: int
    mean_car: float          # Mean CAR over [0, +20] window
    std_car: float           # Std dev of CARs across events
    t_stat: float            # t-statistic for H0: mean_car = 0
    p_value: float           # Two-tailed p-value
    median_car: float        # Median CAR (robust to outliers)
    mean_car_1d: float       # Mean CAR over [0, +1]
    mean_car_5d: float       # Mean CAR over [0, +5]
    individual_cars: tuple[tuple[str, float], ...] = ()  # (filing_date, car_20d)


@dataclass(frozen=True, slots=True)
class VolatilityRegime:
    """Volatility regime classification and per-regime statistics."""

    regime_thresholds: tuple[float, float, float]  # (low_upper, high_lower, extreme_lower)
    regime_counts: dict[str, int] = field(default_factory=dict)
    regime_durations: dict[str, list[int]] = field(default_factory=dict)
    regime_stats: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )  # regime -> (mean_return, std_return, skew, kurtosis)
    transitions: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (date, from_regime, to_regime)
    vol_breakouts: list[tuple[str, float]] = field(
        default_factory=list
    )  # (date, vol_value)
    current_regime: str = "Normal"
    current_vol: float = 0.0


@dataclass(frozen=True, slots=True)
class ReturnDistribution:
    """Statistical properties of the daily log return distribution."""

    mean: float
    std: float
    skewness: float
    kurtosis: float           # Excess kurtosis (Fisher definition)
    jarque_bera_stat: float
    jarque_bera_p: float      # p < 0.05 => reject normality
    is_normal: bool           # Whether JB test fails to reject normality
    t_fit_df: float           # Degrees of freedom from Student-t fit
    t_fit_loc: float
    t_fit_scale: float
    var_95: float             # Value at Risk (5th percentile)
    var_99: float             # Value at Risk (1st percentile)
    cvar_95: float            # Expected Shortfall below VaR_95
    cvar_99: float            # Expected Shortfall below VaR_99
    autocorr_returns: tuple[float, ...] = ()     # lag 1..5
    autocorr_abs_returns: tuple[float, ...] = ()  # lag 1..5 (vol clustering)


@dataclass(frozen=True, slots=True)
class VolumeProfile:
    """Volume distribution across price levels and anomaly detection."""

    price_levels: tuple[float, ...] = ()
    volume_at_level: tuple[int, ...] = ()
    poc_price: float = 0.0                  # Point of Control
    value_area_low: float = 0.0
    value_area_high: float = 0.0
    anomaly_days: tuple[tuple[str, int, float], ...] = ()  # (date, vol, z_score)
    obv_trend_direction: int = 0            # +1 rising, -1 falling, 0 flat
    obv_price_divergence: bool = False


@dataclass(frozen=True, slots=True)
class StructuralBreak:
    """A detected structural break (event cluster) in the price series."""

    start_date: str
    end_date: str
    peak_date: str
    peak_abs_return: float
    num_spike_days: int
    pre_regime_vol: float    # Annualised vol in 20 days before
    post_regime_vol: float   # Annualised vol in 20 days after
    pre_mean_return: float
    post_mean_return: float
    cumulative_return: float


# ======================================================================
# Shared Helpers
# ======================================================================


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


# ======================================================================
# 1. Statistical Event Study (Campbell-Lo-MacKinlay)
# ======================================================================


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


# ======================================================================
# 2. Volatility Regime Detection
# ======================================================================


def detect_volatility_regimes(
    daily_data: list[PriceData],
    vol_window: int = 20,
) -> VolatilityRegime:
    """Detect volatility regimes using rolling realized volatility."""
    log_rets = _compute_log_returns(daily_data)
    if len(log_rets) < vol_window + 1:
        return VolatilityRegime(regime_thresholds=(0.0, 0.0, 0.0))

    # Compute rolling realized vol (annualised)
    rolling_vols: list[tuple[str, float]] = []
    for i in range(vol_window, len(log_rets) + 1):
        window = log_rets[i - vol_window:i]
        mean_w = sum(window) / len(window)
        var_w = sum((r - mean_w) ** 2 for r in window) / (len(window) - 1)
        ann_vol = math.sqrt(var_w) * math.sqrt(252)
        # i corresponds to daily_data index i (since log_rets is offset by 1)
        date = daily_data[i].date if i < len(daily_data) else daily_data[-1].date
        rolling_vols.append((date, ann_vol))

    vol_values = [v[1] for v in rolling_vols]
    sorted_vols = sorted(vol_values)
    median_vol = sorted_vols[len(sorted_vols) // 2]
    vol_mean = sum(vol_values) / len(vol_values)
    vol_std = math.sqrt(
        sum((v - vol_mean) ** 2 for v in vol_values) / (len(vol_values) - 1)
    ) if len(vol_values) > 1 else 0.0

    # Thresholds
    low_upper = median_vol - 0.5 * vol_std
    high_lower = median_vol + 0.5 * vol_std
    extreme_lower = median_vol + 2.0 * vol_std

    def _classify(vol: float) -> str:
        if vol < low_upper:
            return "Low"
        if vol <= high_lower:
            return "Normal"
        if vol <= extreme_lower:
            return "High"
        return "Extreme"

    # Classify each day + collect per-regime returns
    regime_labels: list[str] = []
    regime_returns: dict[str, list[float]] = defaultdict(list)
    for i, (date, vol) in enumerate(rolling_vols):
        regime = _classify(vol)
        regime_labels.append(regime)
        # The return on this day is log_rets[vol_window + i - 1] roughly
        ret_idx = vol_window + i - 1
        if 0 <= ret_idx < len(log_rets):
            regime_returns[regime].append(log_rets[ret_idx])

    # Counts
    counts: dict[str, int] = defaultdict(int)
    for r in regime_labels:
        counts[r] += 1

    # Transitions + durations
    transitions: list[tuple[str, str, str]] = []
    vol_breakouts: list[tuple[str, float]] = []
    durations: dict[str, list[int]] = defaultdict(list)
    run_length = 1
    for i in range(1, len(regime_labels)):
        if regime_labels[i] != regime_labels[i - 1]:
            durations[regime_labels[i - 1]].append(run_length)
            transitions.append((
                rolling_vols[i][0],
                regime_labels[i - 1],
                regime_labels[i],
            ))
            if regime_labels[i - 1] == "Normal" and regime_labels[i] in ("High", "Extreme"):
                vol_breakouts.append((rolling_vols[i][0], rolling_vols[i][1]))
            run_length = 1
        else:
            run_length += 1
    # Final run
    if regime_labels:
        durations[regime_labels[-1]].append(run_length)

    # Per-regime stats
    stats: dict[str, tuple[float, float, float, float]] = {}
    for regime, rets in regime_returns.items():
        if len(rets) >= 4:
            m = sum(rets) / len(rets)
            s = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))
            sk = float(scipy_skew(rets))
            ku = float(scipy_kurtosis(rets))
            stats[regime] = (round(m, 6), round(s, 6), round(sk, 4), round(ku, 4))
        elif len(rets) >= 2:
            m = sum(rets) / len(rets)
            s = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))
            stats[regime] = (round(m, 6), round(s, 6), 0.0, 0.0)

    return VolatilityRegime(
        regime_thresholds=(round(low_upper, 4), round(high_lower, 4), round(extreme_lower, 4)),
        regime_counts=dict(counts),
        regime_durations=dict(durations),
        regime_stats=stats,
        transitions=transitions,
        vol_breakouts=vol_breakouts,
        current_regime=regime_labels[-1] if regime_labels else "Normal",
        current_vol=round(rolling_vols[-1][1], 4) if rolling_vols else 0.0,
    )


# ======================================================================
# 3. Return Distribution Analysis
# ======================================================================


def analyze_return_distribution(
    daily_data: list[PriceData],
) -> ReturnDistribution:
    """Analyse the statistical properties of the daily log return distribution."""
    log_rets = _compute_log_returns(daily_data)
    n = len(log_rets)

    if n < 10:
        return ReturnDistribution(
            mean=0.0, std=0.0, skewness=0.0, kurtosis=0.0,
            jarque_bera_stat=0.0, jarque_bera_p=1.0, is_normal=True,
            t_fit_df=float("inf"), t_fit_loc=0.0, t_fit_scale=0.0,
            var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
        )

    # Moments
    mean_r = sum(log_rets) / n
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in log_rets) / (n - 1))
    sk = float(scipy_skew(log_rets))
    ku = float(scipy_kurtosis(log_rets))  # excess kurtosis

    # Normality test
    jb_stat, jb_p = jarque_bera(log_rets)
    is_normal = float(jb_p) > 0.05

    # Student-t fit
    try:
        df, loc, scale = t_dist.fit(log_rets)
    except Exception:
        df, loc, scale = float("inf"), mean_r, std_r

    # VaR / CVaR (empirical)
    sorted_rets = sorted(log_rets)
    idx_5 = max(0, int(0.05 * n) - 1)
    idx_1 = max(0, int(0.01 * n) - 1)
    var_95 = sorted_rets[idx_5]
    var_99 = sorted_rets[idx_1]
    tail_95 = [r for r in sorted_rets if r <= var_95]
    tail_99 = [r for r in sorted_rets if r <= var_99]
    cvar_95 = sum(tail_95) / len(tail_95) if tail_95 else var_95
    cvar_99 = sum(tail_99) / len(tail_99) if tail_99 else var_99

    # Autocorrelation
    acf_rets = _autocorrelation(log_rets)
    acf_abs = _autocorrelation([abs(r) for r in log_rets])

    return ReturnDistribution(
        mean=round(mean_r, 6),
        std=round(std_r, 6),
        skewness=round(sk, 4),
        kurtosis=round(ku, 4),
        jarque_bera_stat=round(float(jb_stat), 4),
        jarque_bera_p=round(float(jb_p), 6),
        is_normal=is_normal,
        t_fit_df=round(float(df), 4),
        t_fit_loc=round(float(loc), 6),
        t_fit_scale=round(float(scale), 6),
        var_95=round(var_95, 6),
        var_99=round(var_99, 6),
        cvar_95=round(cvar_95, 6),
        cvar_99=round(cvar_99, 6),
        autocorr_returns=acf_rets,
        autocorr_abs_returns=acf_abs,
    )


# ======================================================================
# 4. Volume Profile Analysis
# ======================================================================


def compute_volume_profile(
    daily_data: list[PriceData],
    num_buckets: int = 50,
    anomaly_window: int = 60,
    anomaly_z_threshold: float = 3.0,
) -> VolumeProfile:
    """Build a volume profile and detect volume anomalies."""
    if not daily_data:
        return VolumeProfile()

    closes = [float(d.close) for d in daily_data]
    volumes = [d.volume for d in daily_data]
    price_min = min(closes)
    price_max = max(closes)

    if price_max <= price_min:
        return VolumeProfile(poc_price=price_min)

    # Volume histogram
    bucket_width = (price_max - price_min) / num_buckets
    level_volumes: list[int] = [0] * num_buckets
    level_midpoints: list[float] = [
        price_min + (i + 0.5) * bucket_width for i in range(num_buckets)
    ]

    for close, vol in zip(closes, volumes):
        idx = int((close - price_min) / bucket_width)
        idx = min(idx, num_buckets - 1)
        level_volumes[idx] += vol

    # POC
    poc_idx = max(range(num_buckets), key=lambda i: level_volumes[i])
    poc_price = level_midpoints[poc_idx]

    # Value Area (70% of volume)
    total_vol = sum(level_volumes)
    target_vol = total_vol * 0.70
    va_low_idx = poc_idx
    va_high_idx = poc_idx
    accumulated = level_volumes[poc_idx]

    while accumulated < target_vol and (va_low_idx > 0 or va_high_idx < num_buckets - 1):
        expand_low = level_volumes[va_low_idx - 1] if va_low_idx > 0 else -1
        expand_high = level_volumes[va_high_idx + 1] if va_high_idx < num_buckets - 1 else -1
        if expand_low >= expand_high and va_low_idx > 0:
            va_low_idx -= 1
            accumulated += level_volumes[va_low_idx]
        elif va_high_idx < num_buckets - 1:
            va_high_idx += 1
            accumulated += level_volumes[va_high_idx]
        else:
            break

    va_low = price_min + va_low_idx * bucket_width
    va_high = price_min + (va_high_idx + 1) * bucket_width

    # Volume anomaly detection
    anomalies: list[tuple[str, int, float]] = []
    for i in range(anomaly_window, len(daily_data)):
        window_vols = volumes[i - anomaly_window:i]
        mean_v = sum(window_vols) / len(window_vols)
        std_v = math.sqrt(
            sum((v - mean_v) ** 2 for v in window_vols) / (len(window_vols) - 1)
        ) if len(window_vols) > 1 else 1.0
        if std_v > 0:
            z = (volumes[i] - mean_v) / std_v
            if z > anomaly_z_threshold:
                anomalies.append((daily_data[i].date, volumes[i], round(z, 2)))

    # OBV trend analysis
    obv = 0
    obv_series: list[int] = [0]
    for i in range(1, len(daily_data)):
        if daily_data[i].close > daily_data[i - 1].close:
            obv += daily_data[i].volume
        elif daily_data[i].close < daily_data[i - 1].close:
            obv -= daily_data[i].volume
        obv_series.append(obv)

    lookback = min(20, len(obv_series) - 1)
    if lookback > 0:
        obv_trend = 1 if obv_series[-1] > obv_series[-1 - lookback] else -1
        price_trend = 1 if closes[-1] > closes[-1 - lookback] else -1
        divergence = obv_trend != price_trend
    else:
        obv_trend = 0
        price_trend = 0
        divergence = False

    return VolumeProfile(
        price_levels=tuple(round(p, 2) for p in level_midpoints),
        volume_at_level=tuple(level_volumes),
        poc_price=round(poc_price, 2),
        value_area_low=round(va_low, 2),
        value_area_high=round(va_high, 2),
        anomaly_days=tuple(anomalies),
        obv_trend_direction=obv_trend,
        obv_price_divergence=divergence,
    )


# ======================================================================
# 5. Structural Break Detection
# ======================================================================


def detect_structural_breaks(
    daily_data: list[PriceData],
    prominence: float = 0.03,
    cluster_gap: int = 5,
    context_window: int = 20,
    max_breaks: int = 10,
) -> list[StructuralBreak]:
    """Detect structural breaks using scipy peak detection on absolute returns.

    Spikes within *cluster_gap* trading days are merged into clusters.
    """
    log_rets = _compute_log_returns(daily_data)
    if len(log_rets) < 3:
        return []

    abs_rets = [abs(r) for r in log_rets]

    # Find peaks (spike days)
    peaks, properties = find_peaks(abs_rets, prominence=prominence)

    if len(peaks) == 0:
        return []

    # Cluster nearby peaks
    clusters: list[list[int]] = []
    current_cluster: list[int] = [int(peaks[0])]
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] <= cluster_gap:
            current_cluster.append(int(peaks[i]))
        else:
            clusters.append(current_cluster)
            current_cluster = [int(peaks[i])]
    clusters.append(current_cluster)

    # Analyse each cluster
    breaks: list[StructuralBreak] = []
    for cluster in clusters:
        start_idx = cluster[0]
        end_idx = cluster[-1]

        # Peak within cluster (highest absolute return)
        peak_idx = max(cluster, key=lambda j: abs_rets[j])
        peak_abs_return = abs_rets[peak_idx]

        # Map return indices back to daily_data indices (offset by 1)
        # log_rets[i] = log(close[i+1]/close[i]), so return index i
        # corresponds to the move ON day i+1
        d_start = start_idx + 1
        d_end = end_idx + 1
        d_peak = peak_idx + 1

        if d_start >= len(daily_data) or d_end >= len(daily_data):
            continue

        # Cumulative return over cluster
        cum_ret = sum(log_rets[start_idx:end_idx + 1])

        # Pre-cluster stats
        pre_start = max(0, start_idx - context_window)
        pre_rets = log_rets[pre_start:start_idx]
        pre_vol = _annualised_vol(pre_rets)
        pre_mean = sum(pre_rets) / len(pre_rets) if pre_rets else 0.0

        # Post-cluster stats
        post_end = min(len(log_rets), end_idx + 1 + context_window)
        post_rets = log_rets[end_idx + 1:post_end]
        post_vol = _annualised_vol(post_rets)
        post_mean = sum(post_rets) / len(post_rets) if post_rets else 0.0

        breaks.append(StructuralBreak(
            start_date=daily_data[d_start].date,
            end_date=daily_data[d_end].date,
            peak_date=daily_data[d_peak].date,
            peak_abs_return=round(peak_abs_return, 6),
            num_spike_days=len(cluster),
            pre_regime_vol=round(pre_vol, 4),
            post_regime_vol=round(post_vol, 4),
            pre_mean_return=round(pre_mean, 6),
            post_mean_return=round(post_mean, 6),
            cumulative_return=round(cum_ret, 6),
        ))

    breaks.sort(key=lambda b: b.peak_abs_return, reverse=True)
    return breaks[:max_breaks]


def _annualised_vol(returns: list[float]) -> float:
    """Annualised volatility from a list of log returns."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var) * math.sqrt(252)


# ======================================================================
# Filing ↔ Price Correlation (Legacy — kept for backward compatibility)
# ======================================================================


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

    # Build date → index mapping
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
    import bisect
    pos = bisect.bisect_left(sorted_dates, target_date)
    if pos < len(sorted_dates):
        return date_to_idx[sorted_dates[pos]]
    # If target_date is after all data, use the last bar
    if sorted_dates:
        return date_to_idx[sorted_dates[-1]]
    return None


# ======================================================================
# Key Period Detection
# ======================================================================


def detect_key_periods(
    daily_data: list[PriceData],
    window: int = 20,
    threshold_pct: float = 50.0,
    max_periods: int = 10,
) -> list[KeyPeriod]:
    """Detect periods where price moved more than *threshold_pct* % within
    any *window*-day sliding window.

    Returns up to *max_periods* most significant moves, sorted by
    absolute price change descending.
    """
    if len(daily_data) < 2:
        return []

    raw_periods: list[KeyPeriod] = []

    i = 0
    while i < len(daily_data) - 1:
        end = min(i + window, len(daily_data) - 1)
        start_price = daily_data[i].close
        if start_price <= 0:
            i += 1
            continue

        # Find peak and trough within the window
        peak_price = start_price
        peak_date = daily_data[i].date
        trough_price = start_price
        end_price = daily_data[end].close

        for j in range(i, end + 1):
            if daily_data[j].close > peak_price:
                peak_price = daily_data[j].close
                peak_date = daily_data[j].date
            if daily_data[j].close < trough_price:
                trough_price = daily_data[j].close

        # Check both up-moves and down-moves
        up_pct = (float(peak_price) - float(start_price)) / float(start_price) * 100.0
        down_pct = (float(start_price) - float(trough_price)) / float(start_price) * 100.0
        change_pct = (float(end_price) - float(start_price)) / float(start_price) * 100.0

        if abs(up_pct) >= threshold_pct or abs(down_pct) >= threshold_pct:
            raw_periods.append(KeyPeriod(
                name=f"{'Rally' if change_pct > 0 else 'Decline'} "
                     f"({daily_data[i].date[:7]})",
                start_date=daily_data[i].date,
                end_date=daily_data[end].date,
                start_price=start_price,
                end_price=end_price,
                price_change_pct=round(change_pct, 1),
                peak_price=peak_price,
                peak_date=peak_date,
            ))
            # Skip forward past this period to avoid overlap
            i = end + 1
        else:
            i += 1

    # Sort by absolute price change and take top N
    raw_periods.sort(key=lambda p: abs(p.price_change_pct), reverse=True)
    return raw_periods[:max_periods]


# ======================================================================
# Options Chain Analysis
# ======================================================================


def analyze_options_chain(
    chain: OptionsChain,
    underlying_price: Decimal | None = None,
) -> OptionsAnalysis:
    """Analyse an options chain snapshot: P/C ratios, max pain, unusual activity."""
    price = underlying_price or chain.underlying_price or _ZERO

    all_calls = chain.all_calls
    all_puts = chain.all_puts

    total_call_vol = chain.total_call_volume
    total_put_vol = chain.total_put_volume
    total_call_oi = chain.total_call_open_interest
    total_put_oi = chain.total_put_open_interest

    # Put/call ratios
    pc_vol_ratio = chain.put_call_ratio
    pc_oi_ratio = (
        Decimal(total_put_oi) / Decimal(total_call_oi)
        if total_call_oi > 0 else _ZERO
    ).quantize(Decimal("0.0001"))

    # Nearest expiry
    expirations = chain.expiration_dates
    nearest_expiry = expirations[0] if expirations else ""

    # Max pain for nearest expiry
    max_pain = _compute_max_pain(chain, nearest_expiry) if nearest_expiry else _ZERO

    # Highest OI strikes
    highest_oi_call = _highest_oi_strike(all_calls)
    highest_oi_put = _highest_oi_strike(all_puts)

    # Unusual volume: contracts where volume > 3× open interest
    unusual = _find_unusual_volume(all_calls + all_puts)

    # IV skew: ATM put IV minus ATM call IV (nearest expiry)
    iv_skew = _compute_iv_skew(chain, nearest_expiry, price)

    return OptionsAnalysis(
        underlying_price=price,
        total_call_volume=total_call_vol,
        total_put_volume=total_put_vol,
        total_call_oi=total_call_oi,
        total_put_oi=total_put_oi,
        put_call_volume_ratio=pc_vol_ratio,
        put_call_oi_ratio=pc_oi_ratio,
        max_pain_strike=max_pain,
        nearest_expiry=nearest_expiry,
        highest_oi_call_strike=highest_oi_call,
        highest_oi_put_strike=highest_oi_put,
        unusual_volume_contracts=unusual,
        iv_skew=round(iv_skew, 4),
    )


def _compute_max_pain(chain: OptionsChain, expiration: str) -> Decimal:
    """Compute the max-pain strike for a given expiration.

    Max pain is the strike where the total value of expiring options
    (calls + puts) is minimised for holders.
    """
    calls = chain.get_calls(expiration)
    puts = chain.get_puts(expiration)
    if not calls and not puts:
        return _ZERO

    # Collect all unique strikes
    strikes: set[Decimal] = set()
    for c in calls:
        strikes.add(c.strike)
    for p in puts:
        strikes.add(p.strike)

    best_strike = _ZERO
    min_pain = float("inf")

    for strike in sorted(strikes):
        total_pain = 0.0
        # Call holders lose money when stock is below their strike
        # Put holders lose money when stock is above their strike
        for c in calls:
            if strike < c.strike:
                # Calls expire worthless → holders lose premium (irrelevant to pin)
                pass
            else:
                # Calls are ITM → holders gain (strike - call_strike) * OI
                total_pain += float(strike - c.strike) * c.open_interest
        for p in puts:
            if strike > p.strike:
                # Puts expire worthless → holders lose premium
                pass
            else:
                # Puts are ITM → holders gain (put_strike - strike) * OI
                total_pain += float(p.strike - strike) * p.open_interest

        if total_pain < min_pain:
            min_pain = total_pain
            best_strike = strike

    return best_strike


def _highest_oi_strike(contracts: list[OptionContract]) -> Decimal:
    """Return the strike with the highest open interest."""
    if not contracts:
        return _ZERO
    best = max(contracts, key=lambda c: c.open_interest)
    return best.strike


def _find_unusual_volume(
    contracts: list[OptionContract],
    ratio_threshold: float = 3.0,
) -> list[tuple[str, Decimal, int, int]]:
    """Find contracts where volume exceeds *ratio_threshold* × open interest."""
    unusual: list[tuple[str, Decimal, int, int]] = []
    for c in contracts:
        if c.open_interest > 0 and c.volume > ratio_threshold * c.open_interest:
            unusual.append((c.contract_symbol, c.strike, c.volume, c.open_interest))
    # Sort by volume descending
    unusual.sort(key=lambda x: x[2], reverse=True)
    return unusual[:20]  # cap at 20 most unusual


def _compute_iv_skew(
    chain: OptionsChain,
    expiration: str,
    underlying_price: Decimal,
) -> float:
    """Compute IV skew: average ATM put IV minus average ATM call IV.

    ATM is defined as contracts within 5% of the underlying price.
    """
    if underlying_price <= 0:
        return 0.0

    atm_range = float(underlying_price) * 0.05
    price_f = float(underlying_price)

    call_ivs: list[float] = []
    for c in chain.get_calls(expiration):
        if abs(float(c.strike) - price_f) <= atm_range:
            call_ivs.append(float(c.implied_volatility))

    put_ivs: list[float] = []
    for p in chain.get_puts(expiration):
        if abs(float(p.strike) - price_f) <= atm_range:
            put_ivs.append(float(p.implied_volatility))

    avg_call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else 0.0
    avg_put_iv = sum(put_ivs) / len(put_ivs) if put_ivs else 0.0

    return avg_put_iv - avg_call_iv


# ======================================================================
# Price Statistics
# ======================================================================


def compute_price_statistics(daily_data: list[PriceData]) -> PriceStatistics:
    """Compute summary statistics for a price history."""
    if not daily_data:
        return PriceStatistics(
            period_start="", period_end="", trading_days=0,
            start_price=_ZERO, end_price=_ZERO,
            all_time_high=_ZERO, all_time_high_date="",
            all_time_low=_ZERO, all_time_low_date="",
            total_return_pct=0.0,
            avg_daily_volume=0, max_daily_volume=0, max_volume_date="",
            volatility_annual=0.0,
        )

    # Basic
    first = daily_data[0]
    last = daily_data[-1]

    # ATH / ATL
    ath = first
    atl = first
    max_vol_bar = first
    for bar in daily_data:
        if bar.close > ath.close:
            ath = bar
        if bar.close < atl.close:
            atl = bar
        if bar.volume > max_vol_bar.volume:
            max_vol_bar = bar

    # Total return
    total_return = 0.0
    if first.close > 0:
        total_return = (float(last.close) - float(first.close)) / float(first.close) * 100.0

    # Average volume
    total_volume = sum(b.volume for b in daily_data)
    avg_volume = total_volume // len(daily_data) if daily_data else 0

    # Annualised volatility (std dev of daily log returns × √252)
    volatility = _compute_annual_volatility(daily_data)

    return PriceStatistics(
        period_start=first.date,
        period_end=last.date,
        trading_days=len(daily_data),
        start_price=first.close,
        end_price=last.close,
        all_time_high=ath.close,
        all_time_high_date=ath.date,
        all_time_low=atl.close,
        all_time_low_date=atl.date,
        total_return_pct=round(total_return, 2),
        avg_daily_volume=avg_volume,
        max_daily_volume=max_vol_bar.volume,
        max_volume_date=max_vol_bar.date,
        volatility_annual=round(volatility, 4),
    )


def _compute_annual_volatility(daily_data: list[PriceData]) -> float:
    """Compute annualised volatility from daily log returns."""
    if len(daily_data) < 2:
        return 0.0

    log_returns: list[float] = []
    for i in range(1, len(daily_data)):
        prev_close = float(daily_data[i - 1].close)
        curr_close = float(daily_data[i].close)
        if prev_close > 0 and curr_close > 0:
            log_returns.append(math.log(curr_close / prev_close))

    if len(log_returns) < 2:
        return 0.0

    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(252)
