"""Extracts normalized feature vectors from price data for ML prediction.

Combines four feature groups — normalized technical indicators, atomic
signal generator scores, regime classification metadata, and enhanced
features (lags, volatility, momentum derivatives, interactions,
calendar) — into a single float vector suitable for sklearn models.

Usage::

    from stockdownloader.ml.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    fv = extractor.extract(daily_data, index=250)
    print(fv.values)   # tuple of 63 floats
    print(fv.names)    # tuple of 63 feature names
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, TYPE_CHECKING

from stockdownloader.model.indicator_values import IndicatorValues
from stockdownloader.strategy.regime.regime_detector import MarketRegimeDetector
from stockdownloader.strategy.signals.signal_generator import AtomicSignalGenerator
from stockdownloader.strategy.base_registry import SignalGeneratorRegistry
import stockdownloader.strategy.signals.generators  # noqa: F401 — trigger registry population
from stockdownloader.util.indicators.hub import IndicatorHub

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.ml.alternative_data_store import AlternativeDataStore
    from stockdownloader.ml.hmm_regime_detector import RegimeSnapshot
    from stockdownloader.model.price_data import PriceData


# ------------------------------------------------------------------
# Output type
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Immutable feature vector for ML prediction."""

    values: tuple[float, ...]
    names: tuple[str, ...]
    bar_date: str


# ------------------------------------------------------------------
# Registry-ordered generator names (deterministic)
# ------------------------------------------------------------------

_GENERATOR_NAMES: tuple[str, ...] = (
    "rsi", "macd", "stochastic", "cci", "williams_r",
    "sma_cross", "adx", "ema_trend", "ichimoku", "sar",
    "vwap", "sma_position",
    "bb_touch", "bb_squeeze",
    "obv", "mfi", "volume_surge",
)


# ------------------------------------------------------------------
# Feature names
# ------------------------------------------------------------------

_INDICATOR_NAMES: tuple[str, ...] = (
    "rsi14_norm", "stoch_k_norm", "stoch_d_norm",
    "adx14_norm", "plus_di_norm", "minus_di_norm",
    "mfi14_norm", "williams_r_norm", "cci20_norm",
    "bb_percent_b", "bb_width_norm", "roc12_clamped",
    "macd_hist_atr", "macd_signal_spread_atr",
    "price_vs_sma20_atr", "price_vs_sma50_atr",
    "price_vs_sma200_atr", "price_vs_vwap_atr",
    "ema_cross_atr",
    "volume_ratio", "atr_pct",
    "fib_position", "bb_position",
    "obv_rising_flag", "sar_bullish_flag", "price_above_cloud_flag",
)

_SIGNAL_NAMES: tuple[str, ...] = tuple(f"{n}_score" for n in _GENERATOR_NAMES)

_REGIME_NAMES: tuple[str, ...] = (
    "regime_confidence", "regime_adx_norm",
    "regime_bb_width_pctl", "regime_trend_slope", "regime_sma200_dist_norm",
)

_ENHANCED_NAMES: tuple[str, ...] = (
    # Lag features (4)
    "return_1d", "return_3d", "return_5d", "return_10d",
    # Volatility features (3)
    "realized_vol_5d", "vol_ratio_5d_20d", "intraday_range_pct",
    # Momentum derivative features (3)
    "rsi_roc_3", "macd_hist_slope_3", "adx_slope_3",
    # Cross-feature interactions (3)
    "rsi_x_adx", "bb_width_x_vol_ratio", "regime_conf_x_slope",
    # Calendar features (2)
    "day_of_week_norm", "month_of_year_norm",
)

# Group E: Alternative data features (17)
_ALT_DATA_NAMES: tuple[str, ...] = (
    # FTD features (4)
    "ftd_count_norm", "ftd_count_change_pct", "ftd_value_norm",
    "ftd_rolling_avg_20d",
    # Short interest features (4)
    "short_interest_pct", "short_interest_change",
    "si_days_to_cover", "si_days_to_cover_change",
    # Dark pool features (3)
    "dark_pool_volume_pct", "dark_pool_volume_change", "dark_pool_vs_avg",
    # Institutional ownership features (3)
    "institutional_ownership_pct", "ownership_concentration",
    "institution_count_norm",
    # Borrow rate features (2)
    "borrow_rate_proxy", "borrow_rate_change",
    # Cross-interaction (1)
    "ftd_x_si",
)

# Group F: HMM regime features (5)
_HMM_FEATURE_NAMES: tuple[str, ...] = (
    "hmm_regime_id",
    "hmm_prob_bull",
    "hmm_prob_bear",
    "hmm_prob_sideways",
    "hmm_regime_duration_norm",
)

_BASE_FEATURE_NAMES: tuple[str, ...] = (
    _INDICATOR_NAMES + _SIGNAL_NAMES + _REGIME_NAMES + _ENHANCED_NAMES
)

_ALL_FEATURE_NAMES: tuple[str, ...] = _BASE_FEATURE_NAMES

_ALL_FEATURE_NAMES_WITH_ALT: tuple[str, ...] = (
    _BASE_FEATURE_NAMES + _ALT_DATA_NAMES
)

_ALL_FEATURE_NAMES_WITH_HMM: tuple[str, ...] = (
    _BASE_FEATURE_NAMES + _HMM_FEATURE_NAMES
)

_ALL_FEATURE_NAMES_WITH_ALT_AND_HMM: tuple[str, ...] = (
    _BASE_FEATURE_NAMES + _ALT_DATA_NAMES + _HMM_FEATURE_NAMES
)

_MIN_INDEX = 201  # Minimum bar index for meaningful features


# ------------------------------------------------------------------
# FeatureExtractor
# ------------------------------------------------------------------


class FeatureExtractor:
    """Extracts a feature vector at any bar index.

    Combines normalized indicators, signal generator scores, regime
    features, and enhanced features (lags, volatility, momentum
    derivatives, interactions, calendar) into a single vector suitable
    for sklearn models.

    When an :class:`AlternativeDataStore` is provided, 17 additional
    alternative data features are appended.  When HMM regime snapshots
    are provided, 5 HMM features are appended.  Both can be active
    simultaneously.  Without extra data, the output is 63 features —
    fully backward compatible.

    Parameters
    ----------
    hub:
        Optional shared :class:`IndicatorHub` for caching.
    alt_data_store:
        Optional :class:`AlternativeDataStore` for alternative data
        features (FTD, short interest, dark pool, ownership, borrow rate).
    hmm_snapshots:
        Optional list of :class:`RegimeSnapshot` from HMM detector,
        one per bar aligned with the price data.
    """

    FEATURE_NAMES: ClassVar[tuple[str, ...]] = _ALL_FEATURE_NAMES

    def __init__(
        self,
        hub: IndicatorHub | None = None,
        alt_data_store: AlternativeDataStore | None = None,
        hmm_snapshots: list[RegimeSnapshot] | None = None,
    ) -> None:
        self._hub = hub or IndicatorHub()
        self._generators = self._build_generators()
        self._detector = MarketRegimeDetector(self._hub)
        self._alt_store = alt_data_store
        self._hmm_snapshots = hmm_snapshots

        # Dynamically set feature names based on data availability
        has_alt = self._alt_store is not None and self._alt_store.is_loaded
        has_hmm = self._hmm_snapshots is not None and len(self._hmm_snapshots) > 0

        if has_alt and has_hmm:
            self.FEATURE_NAMES = _ALL_FEATURE_NAMES_WITH_ALT_AND_HMM
        elif has_alt:
            self.FEATURE_NAMES = _ALL_FEATURE_NAMES_WITH_ALT
        elif has_hmm:
            self.FEATURE_NAMES = _ALL_FEATURE_NAMES_WITH_HMM
        else:
            self.FEATURE_NAMES = _BASE_FEATURE_NAMES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> FeatureVector:
        """Extract the feature vector at bar *index*.

        Raises :class:`ValueError` if *index* < ``201``.
        """
        if index < _MIN_INDEX:
            raise ValueError(
                f"Need index >= {_MIN_INDEX} for meaningful features, got {index}"
            )

        iv = IndicatorValues.compute(list(data), index, self._hub)

        indicator_feats = self._indicator_features(iv)
        signal_feats = self._signal_features(data, index)
        regime_feats = self._regime_features(data, index)
        enhanced_feats = self._enhanced_features(data, index, iv, regime_feats)

        all_feats = indicator_feats + signal_feats + regime_feats + enhanced_feats

        # Append alt data features when store is available
        if self._alt_store is not None and self._alt_store.is_loaded:
            bar_date = data[index].date[:10]
            alt_feats = self._alt_data_features(data, index, bar_date)
            all_feats = all_feats + alt_feats

        # Append HMM regime features when snapshots are available
        if self._hmm_snapshots is not None and len(self._hmm_snapshots) > 0:
            hmm_feats = self._hmm_features(index)
            all_feats = all_feats + hmm_feats

        values = tuple(all_feats)
        return FeatureVector(
            values=values,
            names=self.FEATURE_NAMES,
            bar_date=iv.date,
        )

    def extract_batch(
        self,
        data: list[PriceData],
        start: int,
        end: int,
    ) -> list[FeatureVector]:
        """Extract feature vectors for bars in ``[start, end)``."""
        return [self.extract(data, i) for i in range(start, end)]

    @property
    def feature_count(self) -> int:
        """Number of features in the output vector."""
        return len(self.FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Group A: Normalized indicator features (26)
    # ------------------------------------------------------------------

    def _indicator_features(self, iv: IndicatorValues) -> list[float]:
        """Convert :class:`IndicatorValues` to 26 normalized floats."""
        close = _f(iv.close)
        atr = _f(iv.atr14)

        return [
            # Oscillators scaled to [0, 1]
            _f(iv.rsi14) / 100.0,
            _f(iv.stoch_k) / 100.0,
            _f(iv.stoch_d) / 100.0,
            _f(iv.adx14) / 100.0,
            _f(iv.plus_di) / 100.0,
            _f(iv.minus_di) / 100.0,
            _f(iv.mfi14) / 100.0,
            _f(iv.williams_r) / 100.0,
            _clamp(_f(iv.cci20) / 200.0, -2.0, 2.0),
            # Bollinger
            _f(iv.bb_percent_b),
            _safe_div(_f(iv.bb_width), close),
            _clamp(_f(iv.roc12), -10.0, 10.0),
            # MACD relative to ATR
            _safe_div(_f(iv.macd_histogram), atr),
            _safe_div(_f(iv.macd_line) - _f(iv.macd_signal), atr),
            # Price distances relative to ATR
            _safe_div(close - _f(iv.sma20), atr),
            _safe_div(close - _f(iv.sma50), atr),
            _safe_div(close - _f(iv.sma200), atr),
            _safe_div(close - _f(iv.vwap), atr),
            _safe_div(_f(iv.ema12) - _f(iv.ema26), atr),
            # Volume & volatility ratios
            _safe_div(float(iv.volume), _f(iv.avg_volume20)),
            _safe_div(atr, close),
            # Position within ranges
            _safe_div(close - _f(iv.fib_low), _f(iv.fib_high) - _f(iv.fib_low)),
            _safe_div(close - _f(iv.bb_lower), _f(iv.bb_upper) - _f(iv.bb_lower)),
            # Binary flags
            1.0 if iv.obv_rising else 0.0,
            1.0 if iv.sar_bullish else 0.0,
            1.0 if iv.price_above_cloud else 0.0,
        ]

    # ------------------------------------------------------------------
    # Group B: Signal generator scores (17)
    # ------------------------------------------------------------------

    def _signal_features(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> list[float]:
        """Evaluate all 17 generators and return their scores."""
        scores: list[float] = []
        for gen in self._generators:
            result = gen.evaluate(data, index, self._hub)
            scores.append(result.score)
        return scores

    # ------------------------------------------------------------------
    # Group C: Regime features (5)
    # ------------------------------------------------------------------

    def _regime_features(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> list[float]:
        """Classify regime and return 5 numeric features."""
        rc = self._detector.classify(data, index)
        return [
            rc.confidence,
            rc.adx_value / 100.0,
            rc.bb_width_percentile,
            rc.trend_slope,
            rc.sma200_distance / 100.0,
        ]

    # ------------------------------------------------------------------
    # Group D: Enhanced features (15)
    # ------------------------------------------------------------------

    def _enhanced_features(
        self,
        data: Sequence[PriceData],
        index: int,
        iv: IndicatorValues,
        regime_feats: list[float],
    ) -> list[float]:
        """Compute 15 enhanced features: lags, vol, momentum, interactions, calendar."""
        close = float(data[index].close)
        high = float(data[index].high)
        low = float(data[index].low)
        atr = _f(iv.atr14)

        # --- Lag return features (4) ---
        def _ret(offset: int) -> float:
            prev_close = float(data[index - offset].close)
            return _safe_div(close - prev_close, prev_close)

        return_1d = _ret(1)
        return_3d = _ret(3)
        return_5d = _ret(5)
        return_10d = _ret(10)

        # --- Volatility features (3) ---
        # Realized vol = std dev of 5 daily returns
        daily_returns: list[float] = []
        for k in range(1, 6):
            c_now = float(data[index - k + 1].close)
            c_prev = float(data[index - k].close)
            daily_returns.append(_safe_div(c_now - c_prev, c_prev))
        mean_r = sum(daily_returns) / 5.0
        realized_vol_5d = math.sqrt(
            sum((r - mean_r) ** 2 for r in daily_returns) / 5.0
        )

        # 20-day realized vol for ratio
        daily_returns_20: list[float] = []
        for k in range(1, 21):
            c_now = float(data[index - k + 1].close)
            c_prev = float(data[index - k].close)
            daily_returns_20.append(_safe_div(c_now - c_prev, c_prev))
        mean_r20 = sum(daily_returns_20) / 20.0
        realized_vol_20d = math.sqrt(
            sum((r - mean_r20) ** 2 for r in daily_returns_20) / 20.0
        )
        vol_ratio_5d_20d = _safe_div(realized_vol_5d, realized_vol_20d)

        intraday_range_pct = _safe_div(high - low, close)

        # --- Momentum derivative features (3) ---
        rsi_now = float(self._hub.rsi(data, index, 14))
        rsi_3ago = float(self._hub.rsi(data, index - 3, 14))
        rsi_roc_3 = (rsi_now - rsi_3ago) / 100.0

        macd_hist_now = float(self._hub.macd_histogram(data, index))
        macd_hist_3ago = float(self._hub.macd_histogram(data, index - 3))
        macd_hist_slope_3 = _safe_div(macd_hist_now - macd_hist_3ago, atr)

        adx_now = float(self._hub.adx(data, index, 14).adx)
        adx_3ago = float(self._hub.adx(data, index - 3, 14).adx)
        adx_slope_3 = (adx_now - adx_3ago) / 100.0

        # --- Cross-feature interactions (3) ---
        rsi_x_adx = (rsi_now / 100.0) * (adx_now / 100.0)
        bb_width_norm = _safe_div(_f(iv.bb_width), close)
        volume_ratio = _safe_div(float(iv.volume), _f(iv.avg_volume20))
        bb_width_x_vol_ratio = bb_width_norm * volume_ratio

        # regime_feats = [confidence, adx_norm, bb_pctl, trend_slope, sma200_dist]
        regime_conf_x_slope = regime_feats[0] * regime_feats[3]

        # --- Calendar features (2) ---
        date_str = data[index].date
        try:
            from datetime import date as _date
            dt = _date.fromisoformat(date_str[:10])
            day_of_week_norm = dt.weekday() / 4.0  # Mon=0 → 0.0, Fri=4 → 1.0
            month_of_year_norm = (dt.month - 1) / 11.0
        except (ValueError, IndexError):
            day_of_week_norm = 0.5
            month_of_year_norm = 0.5

        return [
            return_1d, return_3d, return_5d, return_10d,
            realized_vol_5d, vol_ratio_5d_20d, intraday_range_pct,
            rsi_roc_3, macd_hist_slope_3, adx_slope_3,
            rsi_x_adx, bb_width_x_vol_ratio, regime_conf_x_slope,
            day_of_week_norm, month_of_year_norm,
        ]

    # ------------------------------------------------------------------
    # Group E: Alternative data features (17)
    # ------------------------------------------------------------------

    def _alt_data_features(
        self,
        data: Sequence[PriceData],
        index: int,
        bar_date: str,
    ) -> list[float]:
        """Compute 17 alternative data features from the alt data store.

        Features are normalised / clamped to keep magnitudes compatible
        with the existing 63-feature vector.  When a source has no data
        the forward-filled defaults from the store yield zeros.
        """
        assert self._alt_store is not None  # caller guarantees this

        snap = self._alt_store.get_or_default(bar_date)
        close = float(data[index].close)
        volume = float(data[index].volume) if data[index].volume else 1.0

        # --- FTD features (4) ---
        ftd_count_norm = _safe_div(float(snap.ftd_quantity), volume)
        ftd_count_change_pct = _safe_div(
            float(snap.ftd_quantity - snap.ftd_quantity_prev),
            max(float(snap.ftd_quantity_prev), 1.0),
        )
        ftd_value_norm = _safe_div(snap.ftd_value, close * volume)

        # Rolling average approximation: use current value as proxy
        # (the store already forward-fills, so this represents recent FTD)
        ftd_rolling_avg_20d = ftd_count_norm  # same normalisation

        # --- Short interest features (4) ---
        short_interest_pct = _clamp(snap.short_interest_pct, 0.0, 2.0)
        short_interest_change = _safe_div(
            float(snap.short_interest - snap.short_interest_prev),
            max(float(snap.short_interest_prev), 1.0),
        )
        si_days_to_cover = _clamp(snap.days_to_cover / 30.0, 0.0, 2.0)
        si_days_to_cover_change = _safe_div(
            snap.days_to_cover - snap.days_to_cover_prev,
            max(snap.days_to_cover_prev, 0.01),
        )

        # --- Dark pool features (3) ---
        dark_pool_volume_pct = _clamp(snap.dark_pool_pct, 0.0, 1.0)
        dark_pool_volume_change = _safe_div(
            float(snap.dark_pool_volume - snap.dark_pool_volume_prev),
            max(float(snap.dark_pool_volume_prev), 1.0),
        )
        dark_pool_vs_avg = _safe_div(
            float(snap.dark_pool_volume),
            max(float(snap.dark_pool_volume_prev), 1.0),
        )

        # --- Institutional ownership features (3) ---
        institutional_ownership_pct = _clamp(
            snap.institutional_ownership_pct, 0.0, 2.0,
        )
        ownership_concentration = _clamp(
            snap.ownership_concentration_top10, 0.0, 1.0,
        )
        institution_count_norm = _clamp(
            float(snap.num_institutions) / 500.0, 0.0, 2.0,
        )

        # --- Borrow rate features (2) ---
        borrow_rate_proxy = _clamp(snap.borrow_rate_proxy / 100.0, 0.0, 1.0)
        borrow_rate_change = _safe_div(
            snap.borrow_rate_proxy - snap.borrow_rate_prev,
            max(snap.borrow_rate_prev, 0.01),
        )

        # --- Cross-interaction (1) ---
        ftd_x_si = ftd_count_norm * short_interest_pct

        return [
            ftd_count_norm, ftd_count_change_pct, ftd_value_norm,
            ftd_rolling_avg_20d,
            short_interest_pct, short_interest_change,
            si_days_to_cover, si_days_to_cover_change,
            dark_pool_volume_pct, dark_pool_volume_change, dark_pool_vs_avg,
            institutional_ownership_pct, ownership_concentration,
            institution_count_norm,
            borrow_rate_proxy, borrow_rate_change,
            ftd_x_si,
        ]

    # ------------------------------------------------------------------
    # Group F: HMM regime features (5)
    # ------------------------------------------------------------------

    def _hmm_features(self, index: int) -> list[float]:
        """Extract 5 HMM regime features for bar at *index*.

        Features:
        - hmm_regime_id: regime ID normalised to [0, 1] (0/n_regimes)
        - hmm_prob_bull: posterior probability of bull regime
        - hmm_prob_bear: posterior probability of bear regime
        - hmm_prob_sideways: posterior probability of sideways regime
        - hmm_regime_duration_norm: duration in current regime / 100
        """
        assert self._hmm_snapshots is not None

        if index < len(self._hmm_snapshots):
            snap = self._hmm_snapshots[index]
            # Normalise regime_id to [0, 1] range (3 regimes → 0, 0.5, 1.0)
            n_regimes = 3
            regime_norm = float(snap.regime_id) / max(n_regimes - 1, 1)
            return [
                regime_norm,
                snap.prob_regime_0,  # bull (labelled by mean return)
                snap.prob_regime_1,  # bear
                snap.prob_regime_2,  # sideways
                min(snap.regime_duration / 100.0, 2.0),  # clamp to [0, 2]
            ]

        # Fallback: index beyond snapshot length → neutral
        return [0.5, 0.33, 0.33, 0.34, 0.0]

    # ------------------------------------------------------------------
    # Generator construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_generators() -> list[AtomicSignalGenerator]:
        """Create default instances of all 17 registered generators."""
        generators: list[AtomicSignalGenerator] = []
        for name in _GENERATOR_NAMES:
            entry = SignalGeneratorRegistry.get(name)
            gen = entry.factory(**entry.default_kwargs)
            generators.append(gen)
        return generators


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _f(val: Any) -> float:
    """Convert Decimal or numeric to float."""
    return float(val)


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 when denominator is zero or result is non-finite."""
    if denominator == 0.0:
        return 0.0
    result = numerator / denominator
    if not math.isfinite(result):
        return 0.0
    return result


def _clamp(val: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, val))
