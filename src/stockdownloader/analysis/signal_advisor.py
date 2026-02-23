"""Regime-aware signal advisory engine.

Combines :func:`generate_alert`, :class:`MarketRegimeDetector`, walk-forward
validation scores, and ATR-based risk levels into a single
:class:`~stockdownloader.model.signal_advisory.SignalAdvisory`.

Usage::

    from stockdownloader.analysis.signal_advisor import SignalAdvisor

    advisor = SignalAdvisor()
    advisory = advisor.evaluate("SPY", daily_price_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.analysis.signal_generator import generate_alert
from stockdownloader.model.alert_result import (
    AlertDirection,
    AlertResult,
    Action,
    OptionsRecommendation,
)
from stockdownloader.model.signal_advisory import (
    AdvisoryAction,
    AdvisoryReasoning,
    OptionsAdvisory,
    SignalAdvisory,
)
from stockdownloader.strategy.regime.regime_detector import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeClassification,
)
from stockdownloader.strategy.regime.regime_strategy_map import RegimeStrategyMapper
from stockdownloader.util.indicators.hub import IndicatorHub

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdvisorConfig:
    """Tunable parameters for :class:`SignalAdvisor`.

    Attributes
    ----------
    atr_stop_multiplier:
        ATR multiplier for stop-loss distance.
    atr_target_multiplier:
        ATR multiplier for take-profit distance.
    min_confidence:
        Below this threshold the action is forced to HOLD.
    max_position_pct:
        Maximum position size as a percentage of capital.
    regime_penalty:
        Confidence penalty when the signal is misaligned with regime.
    ml_model_path:
        Path to a trained ML model (``.joblib``).  ``None`` disables ML.
    ml_boost_max:
        Maximum confidence multiplier boost when ML agrees with signal.
    ml_penalty_max:
        Maximum confidence penalty when ML disagrees with signal.
    ml_min_probability:
        ML probability threshold before boost/penalty is applied.
    """

    atr_stop_multiplier: float = 2.0
    atr_target_multiplier: float = 3.0
    min_confidence: float = 0.4
    max_position_pct: float = 2.0
    regime_penalty: float = 0.3
    ml_model_path: str | None = None
    ml_boost_max: float = 0.3
    ml_penalty_max: float = 0.3
    ml_min_probability: float = 0.6


# ------------------------------------------------------------------
# Direction → Action mapping matrix
# ------------------------------------------------------------------

# (AlertDirection, MarketRegime) → AdvisoryAction
# Unspecified pairs fall through to the generic mapping.
_REGIME_OVERRIDE: dict[tuple[AlertDirection, MarketRegime], AdvisoryAction] = {
    # Strong buys
    (AlertDirection.STRONG_BUY, MarketRegime.STRONG_TREND_UP): AdvisoryAction.STRONG_BUY,
    (AlertDirection.STRONG_BUY, MarketRegime.WEAK_TREND): AdvisoryAction.BUY,
    (AlertDirection.STRONG_BUY, MarketRegime.STRONG_TREND_DOWN): AdvisoryAction.HOLD,
    (AlertDirection.STRONG_BUY, MarketRegime.MEAN_REVERTING): AdvisoryAction.BUY,
    (AlertDirection.STRONG_BUY, MarketRegime.HIGH_VOLATILITY): AdvisoryAction.BUY,
    # Buys
    (AlertDirection.BUY, MarketRegime.STRONG_TREND_UP): AdvisoryAction.BUY,
    (AlertDirection.BUY, MarketRegime.STRONG_TREND_DOWN): AdvisoryAction.HOLD,
    (AlertDirection.BUY, MarketRegime.WEAK_TREND): AdvisoryAction.BUY,
    (AlertDirection.BUY, MarketRegime.MEAN_REVERTING): AdvisoryAction.BUY,
    (AlertDirection.BUY, MarketRegime.HIGH_VOLATILITY): AdvisoryAction.HOLD,
    # Sells
    (AlertDirection.SELL, MarketRegime.STRONG_TREND_DOWN): AdvisoryAction.SELL,
    (AlertDirection.SELL, MarketRegime.STRONG_TREND_UP): AdvisoryAction.HOLD,
    (AlertDirection.SELL, MarketRegime.WEAK_TREND): AdvisoryAction.SELL,
    (AlertDirection.SELL, MarketRegime.MEAN_REVERTING): AdvisoryAction.SELL,
    (AlertDirection.SELL, MarketRegime.HIGH_VOLATILITY): AdvisoryAction.HOLD,
    # Strong sells
    (AlertDirection.STRONG_SELL, MarketRegime.STRONG_TREND_DOWN): AdvisoryAction.STRONG_SELL,
    (AlertDirection.STRONG_SELL, MarketRegime.WEAK_TREND): AdvisoryAction.SELL,
    (AlertDirection.STRONG_SELL, MarketRegime.STRONG_TREND_UP): AdvisoryAction.HOLD,
    (AlertDirection.STRONG_SELL, MarketRegime.MEAN_REVERTING): AdvisoryAction.SELL,
    (AlertDirection.STRONG_SELL, MarketRegime.HIGH_VOLATILITY): AdvisoryAction.SELL,
}


# ------------------------------------------------------------------
# SignalAdvisor
# ------------------------------------------------------------------


class SignalAdvisor:
    """Combines multi-indicator alerts with regime detection and walk-forward
    confidence to produce a :class:`SignalAdvisory`.

    Parameters
    ----------
    config:
        Tunable thresholds.  Defaults to :class:`AdvisorConfig` defaults.
    regime_mapper:
        Optional :class:`RegimeStrategyMapper` for regime alignment messages.
    walk_forward_scores:
        Optional dict mapping strategy names to ``(oos_score, is_score)``
        tuples.  The degradation ratio ``oos / is`` is used to scale
        confidence.
    hub:
        Optional shared :class:`IndicatorHub`.
    """

    def __init__(
        self,
        config: AdvisorConfig | None = None,
        regime_mapper: RegimeStrategyMapper | None = None,
        walk_forward_scores: dict[str, tuple[float, float]] | None = None,
        hub: IndicatorHub | None = None,
        ml_predictor: object | None = None,
    ) -> None:
        self._cfg = config or AdvisorConfig()
        self._mapper = regime_mapper
        self._wf_scores = walk_forward_scores or {}
        self._hub = hub or IndicatorHub()
        self._detector = MarketRegimeDetector(self._hub)
        self._ml_predictor = ml_predictor or self._load_ml_predictor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        data: list[PriceData] | Sequence[PriceData],
        index: int | None = None,
    ) -> SignalAdvisory:
        """Produce a regime-aware advisory for *symbol* at *index*.

        Parameters
        ----------
        symbol:
            Ticker (e.g. ``"SPY"``).
        data:
            Daily price data with at least 201 bars for a meaningful signal.
        index:
            Bar index to evaluate.  Defaults to the last bar.

        Returns
        -------
        SignalAdvisory
        """
        if index is None:
            index = len(data) - 1

        cfg = self._cfg

        # Insufficient data → neutral HOLD
        if index < 201:
            return self._hold_advisory(
                symbol, data, index, reason="Insufficient data (need 201+ bars)"
            )

        # 1. Multi-indicator alert
        alert: AlertResult = generate_alert(symbol, list(data), index)

        # 2. Regime classification
        regime_cls: RegimeClassification = self._detector.classify(data, index)

        # 3. Map direction → action with regime filter
        action = _map_direction(alert.direction, regime_cls.regime)

        # 4. Composite confidence
        raw_confidence = alert.confluence_score
        regime_factor = self._regime_confidence_factor(
            alert.direction, regime_cls.regime, action,
        )
        wf_factor = self._walk_forward_factor()
        ml_factor = self._ml_confidence_factor(data, index, action)
        confidence = min(1.0, raw_confidence * regime_factor * wf_factor * ml_factor)

        # 5. Below min → force HOLD
        if confidence < cfg.min_confidence:
            action = AdvisoryAction.HOLD

        # 6. ATR-based stop / target
        atr = float(self._hub.atr(data, index, 14))
        entry_price = float(data[index].close)
        stop_loss, take_profit, risk_reward = _compute_levels(
            entry_price, atr, action, cfg,
        )

        # 7. Position sizing (% of capital)
        position_pct = _position_size(confidence, cfg.max_position_pct)

        # 8. Build reasoning
        reasoning = _build_reasoning(
            alert, regime_cls, self._wf_scores, self._mapper,
        )

        # 9. Convert options recommendations
        call_adv = _convert_options(alert.call_recommendation)
        put_adv = _convert_options(alert.put_recommendation)

        timestamp = alert.date or datetime.now(timezone.utc).isoformat()

        return SignalAdvisory(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            confidence=round(confidence, 4),
            regime=regime_cls.regime.value,
            regime_confidence=round(regime_cls.confidence, 4),
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            risk_reward=round(risk_reward, 2),
            position_size_pct=round(position_pct, 2),
            reasoning=reasoning,
            call_advisory=call_adv,
            put_advisory=put_adv,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hold_advisory(
        self,
        symbol: str,
        data: list[PriceData] | Sequence[PriceData],
        index: int,
        *,
        reason: str,
    ) -> SignalAdvisory:
        """Return a neutral HOLD advisory."""
        price = float(data[index].close) if data else 0.0
        return SignalAdvisory(
            symbol=symbol,
            timestamp=data[index].date if data else datetime.now(timezone.utc).isoformat(),
            action=AdvisoryAction.HOLD,
            confidence=0.0,
            regime="unknown",
            regime_confidence=0.0,
            entry_price=round(price, 2),
            stop_loss=round(price, 2),
            take_profit=round(price, 2),
            risk_reward=0.0,
            position_size_pct=0.0,
            reasoning=AdvisoryReasoning(
                signal_confluence=reason,
                regime_alignment="N/A",
                walk_forward_validated="N/A",
            ),
        )

    def _regime_confidence_factor(
        self,
        direction: AlertDirection,
        regime: MarketRegime,
        action: AdvisoryAction,
    ) -> float:
        """Return a multiplier based on regime alignment.

        * Aligned (no downgrade) → 1.0
        * Downgraded one step → (1 - penalty/2)
        * Forced HOLD → (1 - penalty)
        """
        if action == AdvisoryAction.HOLD and direction != AlertDirection.NEUTRAL:
            # Signal completely overridden by regime
            return 1.0 - self._cfg.regime_penalty
        if _is_downgraded(direction, action):
            return 1.0 - self._cfg.regime_penalty / 2
        return 1.0

    def _walk_forward_factor(self) -> float:
        """Compute confidence multiplier from walk-forward scores.

        If no WF scores are available, returns 1.0 (neutral).
        """
        if not self._wf_scores:
            return 1.0

        # Use the best strategy's degradation ratio
        best_ratio = 0.0
        for oos, is_score in self._wf_scores.values():
            if is_score > 0:
                ratio = oos / is_score
                best_ratio = max(best_ratio, ratio)

        if best_ratio >= 0.85:
            return 1.15   # validated → boost
        elif best_ratio >= 0.6:
            return 1.0    # acceptable
        else:
            return 0.7    # likely overfit → penalize

    def _ml_confidence_factor(
        self,
        data: list[PriceData] | Sequence[PriceData],
        index: int,
        action: AdvisoryAction,
    ) -> float:
        """Return a confidence multiplier based on ML prediction.

        * No model → 1.0 (neutral, preserves existing behavior).
        * ML agrees with signal direction → boost up to ``1 + ml_boost_max``.
        * ML disagrees → penalty down to ``1 - ml_penalty_max``.
        * Linear interpolation between neutral (0.5) and threshold.
        """
        if self._ml_predictor is None:
            return 1.0

        cfg = self._cfg
        prob = self._ml_predictor.predict_proba(data, index)  # type: ignore[union-attr]

        # Determine if signal is bullish or bearish
        is_bullish = action in (AdvisoryAction.BUY, AdvisoryAction.STRONG_BUY)
        is_bearish = action in (AdvisoryAction.SELL, AdvisoryAction.STRONG_SELL)

        if not (is_bullish or is_bearish):
            return 1.0  # HOLD → no ML adjustment

        if is_bullish:
            # ML predicts profitable (high prob) → boost BUY
            if prob >= cfg.ml_min_probability:
                strength = (prob - cfg.ml_min_probability) / (1.0 - cfg.ml_min_probability)
                return 1.0 + cfg.ml_boost_max * strength
            # ML predicts unprofitable (low prob) → penalty on BUY
            elif prob <= (1.0 - cfg.ml_min_probability):
                strength = ((1.0 - cfg.ml_min_probability) - prob) / (1.0 - cfg.ml_min_probability)
                return 1.0 - cfg.ml_penalty_max * strength
        elif is_bearish:
            # ML predicts unprofitable (low prob) → boost SELL (opposite)
            if prob <= (1.0 - cfg.ml_min_probability):
                strength = ((1.0 - cfg.ml_min_probability) - prob) / (1.0 - cfg.ml_min_probability)
                return 1.0 + cfg.ml_boost_max * strength
            # ML predicts profitable (high prob) → penalty on SELL
            elif prob >= cfg.ml_min_probability:
                strength = (prob - cfg.ml_min_probability) / (1.0 - cfg.ml_min_probability)
                return 1.0 - cfg.ml_penalty_max * strength

        return 1.0  # Neutral range

    def _load_ml_predictor(self) -> object | None:
        """Try to load the ML predictor from config path.

        Returns ``None`` (silently) if path is unset or loading fails.
        """
        if not self._cfg.ml_model_path:
            return None

        try:
            from stockdownloader.ml.predictor import MLPredictor
            return MLPredictor.from_path(self._cfg.ml_model_path, hub=self._hub)
        except (ImportError, OSError, ValueError, KeyError):
            import logging
            logging.getLogger(__name__).debug(
                "Failed to load ML model from %s",
                self._cfg.ml_model_path,
                exc_info=True,
            )
            return None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _map_direction(
    direction: AlertDirection,
    regime: MarketRegime,
) -> AdvisoryAction:
    """Map alert direction + regime to a regime-filtered advisory action."""
    if direction == AlertDirection.NEUTRAL:
        return AdvisoryAction.HOLD

    key = (direction, regime)
    if key in _REGIME_OVERRIDE:
        return _REGIME_OVERRIDE[key]

    # Fallback: direct mapping without regime filter
    _DIRECT: dict[AlertDirection, AdvisoryAction] = {
        AlertDirection.STRONG_BUY: AdvisoryAction.STRONG_BUY,
        AlertDirection.BUY: AdvisoryAction.BUY,
        AlertDirection.SELL: AdvisoryAction.SELL,
        AlertDirection.STRONG_SELL: AdvisoryAction.STRONG_SELL,
    }
    return _DIRECT.get(direction, AdvisoryAction.HOLD)


def _is_downgraded(direction: AlertDirection, action: AdvisoryAction) -> bool:
    """Check if the action was downgraded from the original direction."""
    _STRENGTH: dict[str, int] = {
        "STRONG_BUY": 2, "BUY": 1, "HOLD": 0, "SELL": -1, "STRONG_SELL": -2,
    }
    orig = _STRENGTH.get(direction.value, 0)
    final = _STRENGTH.get(action.value, 0)
    # Downgraded if moved towards zero (weaker conviction)
    return abs(final) < abs(orig)


def _compute_levels(
    entry: float,
    atr: float,
    action: AdvisoryAction,
    cfg: AdvisorConfig,
) -> tuple[float, float, float]:
    """Compute stop-loss, take-profit, and risk-reward from ATR.

    Returns ``(stop_loss, take_profit, risk_reward)``.
    """
    if atr <= 0 or action == AdvisoryAction.HOLD:
        return entry, entry, 0.0

    stop_dist = atr * cfg.atr_stop_multiplier
    target_dist = atr * cfg.atr_target_multiplier

    if action in (AdvisoryAction.STRONG_BUY, AdvisoryAction.BUY):
        stop_loss = entry - stop_dist
        take_profit = entry + target_dist
    elif action in (AdvisoryAction.STRONG_SELL, AdvisoryAction.SELL):
        stop_loss = entry + stop_dist
        take_profit = entry - target_dist
    else:
        return entry, entry, 0.0

    rr = target_dist / stop_dist if stop_dist > 0 else 0.0
    return stop_loss, take_profit, rr


def _position_size(confidence: float, max_pct: float) -> float:
    """Scale position size linearly with confidence, capped at *max_pct*."""
    return min(max_pct, confidence * max_pct)


def _build_reasoning(
    alert: AlertResult,
    regime: RegimeClassification,
    wf_scores: dict[str, tuple[float, float]],
    mapper: RegimeStrategyMapper | None,
) -> AdvisoryReasoning:
    """Construct :class:`AdvisoryReasoning` from alert and regime data."""
    bullish_count = len(alert.bullish_indicators)
    bearish_count = len(alert.bearish_indicators)
    total = alert.total_indicators or (bullish_count + bearish_count) or 1
    confluence_str = (
        f"{bullish_count}/{total} indicators bullish "
        f"({alert.confluence_score * 100:.0f}%)"
    )

    # Regime alignment description
    regime_str = f"Regime: {regime.regime.value} (confidence {regime.confidence:.0%})"
    if mapper:
        best = mapper.best_strategy_for_regime(regime.regime)
        if best:
            perf = mapper.get_performance(best, regime.regime)
            if perf:
                regime_str += f", best strategy: {best} (avg ${perf.avg_pnl:,.0f})"

    # Walk-forward description
    if wf_scores:
        best_name = ""
        best_ratio = 0.0
        best_oos = 0.0
        for name, (oos, is_s) in wf_scores.items():
            ratio = oos / is_s if is_s > 0 else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_oos = oos
                best_name = name
        wf_str = (
            f"Top strategy OOS score: {best_oos:.1f}, "
            f"degradation: {best_ratio:.2f}"
        )
    else:
        wf_str = "No walk-forward data available"

    support = [float(s) for s in alert.support_levels[:3]]
    resistance = [float(r) for r in alert.resistance_levels[:3]]

    return AdvisoryReasoning(
        signal_confluence=confluence_str,
        regime_alignment=regime_str,
        walk_forward_validated=wf_str,
        key_bullish=alert.bullish_indicators[:5],
        key_bearish=alert.bearish_indicators[:5],
        support_levels=support,
        resistance_levels=resistance,
    )


def _convert_options(rec: OptionsRecommendation) -> OptionsAdvisory | None:
    """Convert an :class:`OptionsRecommendation` to an :class:`OptionsAdvisory`.

    Returns ``None`` if the recommendation is a HOLD (no action).
    """
    if rec.action == Action.HOLD:
        return None

    return OptionsAdvisory(
        action=rec.action.value,
        strike=float(rec.suggested_strike),
        dte=rec.suggested_dte,
        delta=float(rec.target_delta),
        est_premium=float(rec.estimated_premium),
    )
