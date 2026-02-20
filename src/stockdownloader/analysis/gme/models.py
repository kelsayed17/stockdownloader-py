"""GME analysis data classes shared across analysis modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from stockdownloader.model.regulatory_records import SecFiling

__all__ = [
    "FilingImpact",
    "KeyPeriod",
    "OptionsAnalysis",
    "PriceStatistics",
    "EventStudyResult",
    "VolatilityRegime",
    "ReturnDistribution",
    "VolumeProfile",
    "StructuralBreak",
]


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
