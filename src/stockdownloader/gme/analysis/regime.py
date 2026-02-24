"""Volatility regime detection, structural break detection, and key period detection."""

from __future__ import annotations

import math
from collections import defaultdict

from scipy.signal import find_peaks
from scipy.stats import (
    kurtosis as scipy_kurtosis,
    skew as scipy_skew,
)

from stockdownloader.gme.analysis.models import (
    KeyPeriod,
    StructuralBreak,
    VolatilityRegime,
)
from stockdownloader.core.models.price import PriceData

__all__ = [
    "detect_volatility_regimes",
    "detect_structural_breaks",
    "detect_key_periods",
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


def _annualised_vol(returns: list[float]) -> float:
    """Annualised volatility from a list of log returns."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var) * math.sqrt(252)


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
