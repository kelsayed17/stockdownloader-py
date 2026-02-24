"""Market regime classification using technical indicators.

Classifies each bar into one of five regimes using ADX, Bollinger Band
width, linear regression slope, and SMA(200) position.  Regime detection
feeds into the ensemble strategy (Phase 5) and the grand tournament
(Phase 6) for regime-aware strategy selection.

Usage::

    from stockdownloader.strategy.regime import (
        MarketRegimeDetector, MarketRegime,
    )

    detector = MarketRegimeDetector(hub)
    regime = detector.classify(data, current_index)
    if regime.regime == MarketRegime.STRONG_TREND_UP:
        # use trend-following strategy
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from stockdownloader.util.indicators.hub import IndicatorHub
from stockdownloader.core.math import ZERO

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.core.models.price import PriceData

class MarketRegime(Enum):
    """Five distinct market regime classifications."""

    STRONG_TREND_UP = "strong_trend_up"
    STRONG_TREND_DOWN = "strong_trend_down"
    WEAK_TREND = "weak_trend"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"

@dataclass(frozen=True, slots=True)
class RegimeClassification:
    """Result of classifying a single bar's market regime.

    Attributes
    ----------
    regime:
        The detected regime.
    confidence:
        Confidence level [0, 1] based on how clearly indicators agree.
    adx_value:
        ADX reading used for classification.
    bb_width_percentile:
        Bollinger Band width as a percentile of its lookback history.
    trend_slope:
        Linear regression slope normalised by ATR.
    sma200_distance:
        Price distance from SMA(200) as a percentage.
    """

    regime: MarketRegime
    confidence: float
    adx_value: float
    bb_width_percentile: float
    trend_slope: float
    sma200_distance: float


@dataclass(frozen=True, slots=True)
class RegimeDetectorConfig:
    """Tunable parameters and thresholds for :class:`MarketRegimeDetector`.

    Groups indicator lookback periods and classification thresholds into
    a single configuration object.
    """

    # Indicator lookback periods
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    slope_period: int = 15
    sma_period: int = 200

    # Classification thresholds
    adx_strong: float = 30.0
    adx_weak_low: float = 20.0
    bb_width_high_vol: float = 0.80     # 80th percentile
    bb_width_mean_rev: float = 0.40     # 40th percentile
    slope_threshold: float = 0.5        # ATR-normalised slope threshold


class MarketRegimeDetector:
    """Classifies market regime from technical indicators.

    Parameters
    ----------
    hub:
        Shared :class:`IndicatorHub` for indicator computation.
    config:
        Optional :class:`RegimeDetectorConfig` with periods and thresholds.
        Falls back to defaults when ``None``.
    """

    def __init__(
        self,
        hub: IndicatorHub,
        config: RegimeDetectorConfig | None = None,
        *,
        # Legacy keyword args for backward compatibility
        adx_period: int | None = None,
        bb_period: int | None = None,
        bb_std: float | None = None,
        slope_period: int | None = None,
        sma_period: int | None = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        else:
            # Build config from explicit kwargs or defaults
            self._cfg = RegimeDetectorConfig(
                adx_period=adx_period if adx_period is not None else 14,
                bb_period=bb_period if bb_period is not None else 20,
                bb_std=bb_std if bb_std is not None else 2.0,
                slope_period=slope_period if slope_period is not None else 15,
                sma_period=sma_period if sma_period is not None else 200,
            )
        self._hub = hub

    @property
    def warmup_period(self) -> int:
        """Minimum bars required before classification is meaningful."""
        return max(self._cfg.sma_period, 120) + 1

    def classify(
        self,
        data: Sequence[PriceData],
        index: int,
        lookback: int = 120,
    ) -> RegimeClassification:
        """Classify the market regime at *index*.

        Parameters
        ----------
        data:
            Price data sequence.
        index:
            Current bar index to classify.
        lookback:
            Number of historical bars for BB width percentile.

        Returns
        -------
        RegimeClassification with regime, confidence, and indicator values.
        """
        hub = self._hub
        cfg = self._cfg

        # 1. ADX — trend strength
        adx_result = hub.adx(data, index, cfg.adx_period)
        adx_val = float(adx_result.adx)
        plus_di = float(adx_result.plus_di)
        minus_di = float(adx_result.minus_di)

        # 2. BB width percentile — volatility regime
        bb_width_pctl = self._bb_width_percentile(data, index, lookback)

        # 3. Trend slope — LRS normalised by ATR
        slope = self._normalised_slope(data, index)

        # 4. SMA(200) distance — long-term trend position
        sma200_dist = self._sma_distance(data, index)

        # 5. Decision tree
        regime, confidence = self._classify_decision_tree(
            adx_val, plus_di, minus_di, bb_width_pctl, slope, cfg,
        )

        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            adx_value=adx_val,
            bb_width_percentile=bb_width_pctl,
            trend_slope=slope,
            sma200_distance=sma200_dist,
        )

    def classify_range(
        self,
        data: Sequence[PriceData],
        start: int,
        end: int,
        lookback: int = 120,
    ) -> list[RegimeClassification]:
        """Classify a range of bars [start, end).

        Returns one :class:`RegimeClassification` per bar.
        """
        return [
            self.classify(data, i, lookback) for i in range(start, end)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bb_width_percentile(
        self,
        data: Sequence[PriceData],
        index: int,
        lookback: int,
    ) -> float:
        """Compute current BB width as a percentile of recent history."""
        hub = self._hub
        cfg = self._cfg
        current_bb = hub.bollinger_bands(
            data, index, cfg.bb_period, cfg.bb_std,
        )
        current_width = float(current_bb.upper - current_bb.lower)

        # Collect historical BB widths
        start = max(cfg.bb_period, index - lookback)
        widths: list[float] = []
        for i in range(start, index + 1):
            bb = hub.bollinger_bands(data, i, cfg.bb_period, cfg.bb_std)
            widths.append(float(bb.upper - bb.lower))

        if not widths:
            return 0.5

        # Percentile: fraction of historical widths <= current
        count_below = sum(1 for w in widths if w <= current_width)
        return count_below / len(widths)

    def _normalised_slope(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> float:
        """Linear regression slope of close prices, normalised by ATR."""
        cfg = self._cfg
        if index < cfg.slope_period:
            return 0.0

        # Simple slope: (last - first) / period, normalised by ATR
        close_now = float(data[index].close)
        close_ago = float(data[index - cfg.slope_period].close)
        raw_slope = (close_now - close_ago) / cfg.slope_period

        atr_val = float(self._hub.atr(data, index, cfg.adx_period))
        if atr_val == 0:
            return 0.0

        return raw_slope / atr_val

    def _sma_distance(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> float:
        """Distance from SMA(200) as a percentage of price."""
        if index < self._cfg.sma_period:
            return 0.0

        price = float(data[index].close)
        sma = float(self._hub.sma(data, index, self._cfg.sma_period))
        if sma == 0:
            return 0.0

        return ((price - sma) / sma) * 100.0

    @staticmethod
    def _classify_decision_tree(
        adx: float,
        plus_di: float,
        minus_di: float,
        bb_pctl: float,
        slope: float,
        cfg: RegimeDetectorConfig,
    ) -> tuple[MarketRegime, float]:
        """Apply decision tree to classify regime and compute confidence.

        Returns ``(regime, confidence)`` where confidence is [0, 1].
        """
        # Priority 1: High volatility overrides everything
        if bb_pctl > cfg.bb_width_high_vol:
            conf = min(1.0, (bb_pctl - cfg.bb_width_high_vol) / 0.20 * 0.5 + 0.5)
            return MarketRegime.HIGH_VOLATILITY, conf

        # Priority 2: Strong trend (ADX > threshold and clear directional slope)
        if adx > cfg.adx_strong and abs(slope) > cfg.slope_threshold:
            conf = min(1.0, (adx - cfg.adx_strong) / 20.0 * 0.4 + 0.6)
            if slope > 0 and plus_di > minus_di:
                return MarketRegime.STRONG_TREND_UP, conf
            elif slope < 0 and minus_di > plus_di:
                return MarketRegime.STRONG_TREND_DOWN, conf
            # DI disagrees with slope — lower confidence
            if slope > 0:
                return MarketRegime.STRONG_TREND_UP, conf * 0.7
            return MarketRegime.STRONG_TREND_DOWN, conf * 0.7

        # Priority 3: Mean-reverting (low ADX + tight BB)
        if adx < cfg.adx_weak_low and bb_pctl < cfg.bb_width_mean_rev:
            conf = min(1.0, (cfg.adx_weak_low - adx) / 10.0 * 0.3 + 0.5)
            return MarketRegime.MEAN_REVERTING, conf

        # Priority 4: Weak trend (ADX 20-30 or ADX > 30 but no directional clarity)
        if adx >= cfg.adx_weak_low:
            conf = 0.4 + min(0.3, (adx - cfg.adx_weak_low) / 20.0 * 0.3)
            return MarketRegime.WEAK_TREND, conf

        # Default: low ADX but not tight enough for mean-reversion
        return MarketRegime.WEAK_TREND, 0.3
