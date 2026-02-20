"""Return distribution analysis, volume profile, and price statistics."""

from __future__ import annotations

import math
from decimal import Decimal

from scipy.stats import (
    jarque_bera,
    kurtosis as scipy_kurtosis,
    skew as scipy_skew,
    t as t_dist,
)

from stockdownloader.analysis.gme.models import (
    PriceStatistics,
    ReturnDistribution,
    VolumeProfile,
)
from stockdownloader.model.price_data import PriceData

_ZERO = Decimal("0")

__all__ = [
    "analyze_return_distribution",
    "compute_volume_profile",
    "compute_price_statistics",
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
    except (ValueError, RuntimeError):
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

    # Annualised volatility (std dev of daily log returns x sqrt(252))
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
