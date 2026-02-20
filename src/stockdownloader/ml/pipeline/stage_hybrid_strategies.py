"""Stage 4: ML-Informed Hybrid Strategy Construction.

Creates three flavours of hybrid strategy that wrap an existing
:class:`~stockdownloader.strategy.trading_strategy.TradingStrategy`
with ML confidence:

* **MLConfirmedStrategy** — only take strategy signal if ML agrees.
* **MLWeightedStrategy** — softer filter (lower threshold).
* **MLOverrideStrategy** — ML drives timing, strategy filters.

All three are proper :class:`TradingStrategy` subclasses so the existing
:class:`~stockdownloader.backtest.backtest_engine.BacktestEngine` runs
them without modification.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from stockdownloader.ml.pipeline.config import HybridConfig
from stockdownloader.ml.pipeline.results import (
    ConvergenceResult,
    DataResult,
    HybridStageResult,
    HybridStrategyEntry,
    ModelCandidate,
    TrainingStageResult,
)
from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy

if TYPE_CHECKING:
    from stockdownloader.model.price_data import PriceData

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Base mixin for ML-aware feature extraction
# ------------------------------------------------------------------


class _MLMixin:
    """Shared helper for ML feature extraction inside evaluate()."""

    def _init_ml(
        self,
        model: object,
        feature_names: tuple[str, ...],
    ) -> None:
        self._model = model
        self._feature_names = feature_names
        self._extractor: object | None = None
        self._prob_cache: dict[int, float] = {}

    def _get_ml_prob(self, data: list[PriceData], index: int) -> float:
        """Extract features at *index* and return P(class=1)."""
        if index in self._prob_cache:
            return self._prob_cache[index]
        try:
            if self._extractor is None:
                from stockdownloader.ml.feature_extractor import FeatureExtractor

                self._extractor = FeatureExtractor()
            fv = self._extractor.extract(data, index)  # type: ignore[union-attr]
            X = np.array([fv.values], dtype=np.float64)
            proba = self._model.predict_proba(X)  # type: ignore[union-attr]
            prob = float(proba[0, 1])
        except Exception:
            prob = 0.5
        self._prob_cache[index] = prob
        return prob


# ------------------------------------------------------------------
# MLConfirmedStrategy
# ------------------------------------------------------------------


class MLConfirmedStrategy(_MLMixin, TradingStrategy):
    """Only take strategy signal if ML probability exceeds threshold.

    * BUY requires ML prob > ``threshold``
    * SELL requires ML prob < ``1 − threshold``
    * Otherwise HOLD
    """

    def __init__(
        self,
        base_strategy: TradingStrategy,
        model: object,
        feature_names: tuple[str, ...],
        threshold: float = 0.6,
        label: str = "",
    ) -> None:
        self._base = base_strategy
        self._init_ml(model, feature_names)
        self._threshold = threshold
        self._label = label or f"{base_strategy.name} (ML-Confirmed)"
        self._warmup = max(base_strategy.warmup_period, 201)

    @property
    def name(self) -> str:
        return self._label

    @property
    def warmup_period(self) -> int:
        return self._warmup

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self._warmup:
            return Signal.HOLD

        base_signal = self._base.evaluate(data, current_index)
        if base_signal == Signal.HOLD:
            return Signal.HOLD

        prob = self._get_ml_prob(data, current_index)

        if base_signal == Signal.BUY and prob >= self._threshold:
            return Signal.BUY
        if base_signal == Signal.SELL and prob <= (1.0 - self._threshold):
            return Signal.SELL

        return Signal.HOLD


# ------------------------------------------------------------------
# MLWeightedStrategy
# ------------------------------------------------------------------


class MLWeightedStrategy(_MLMixin, TradingStrategy):
    """Softer ML filter — lower threshold than Confirmed.

    Passes strategy signal through if ML probability is not strongly
    *against* the trade direction.  This preserves more trades while
    still filtering out the worst signals.
    """

    def __init__(
        self,
        base_strategy: TradingStrategy,
        model: object,
        feature_names: tuple[str, ...],
        threshold: float = 0.5,
        label: str = "",
    ) -> None:
        self._base = base_strategy
        self._init_ml(model, feature_names)
        self._threshold = threshold
        self._label = label or f"{base_strategy.name} (ML-Weighted)"
        self._warmup = max(base_strategy.warmup_period, 201)

    @property
    def name(self) -> str:
        return self._label

    @property
    def warmup_period(self) -> int:
        return self._warmup

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self._warmup:
            return Signal.HOLD

        base_signal = self._base.evaluate(data, current_index)
        if base_signal == Signal.HOLD:
            return Signal.HOLD

        prob = self._get_ml_prob(data, current_index)

        if base_signal == Signal.BUY and prob >= self._threshold:
            return Signal.BUY
        if base_signal == Signal.SELL and prob <= (1.0 - self._threshold):
            return Signal.SELL

        return Signal.HOLD


# ------------------------------------------------------------------
# MLOverrideStrategy
# ------------------------------------------------------------------


class MLOverrideStrategy(_MLMixin, TradingStrategy):
    """ML drives timing; strategy acts as a safety filter.

    * BUY when ML prob > threshold AND strategy is NOT SELL
    * SELL when ML prob < (1 − threshold) AND strategy is NOT BUY
    """

    def __init__(
        self,
        base_strategy: TradingStrategy,
        model: object,
        feature_names: tuple[str, ...],
        threshold: float = 0.55,
        label: str = "",
    ) -> None:
        self._base = base_strategy
        self._init_ml(model, feature_names)
        self._threshold = threshold
        self._label = label or f"{base_strategy.name} (ML-Override)"
        self._warmup = max(base_strategy.warmup_period, 201)

    @property
    def name(self) -> str:
        return self._label

    @property
    def warmup_period(self) -> int:
        return self._warmup

    def evaluate(self, data: list[PriceData], current_index: int) -> Signal:
        if current_index < self._warmup:
            return Signal.HOLD

        base_signal = self._base.evaluate(data, current_index)
        prob = self._get_ml_prob(data, current_index)

        # ML says buy and strategy is not actively selling
        if prob >= self._threshold and base_signal != Signal.SELL:
            return Signal.BUY
        # ML says sell and strategy is not actively buying
        if prob <= (1.0 - self._threshold) and base_signal != Signal.BUY:
            return Signal.SELL

        return Signal.HOLD


# ------------------------------------------------------------------
# HybridStage
# ------------------------------------------------------------------


_STRATEGY_CLASS = {
    "confirmed": MLConfirmedStrategy,
    "weighted": MLWeightedStrategy,
    "override": MLOverrideStrategy,
}

_THRESHOLD_ATTR = {
    "confirmed": "confirmed_threshold",
    "weighted": "weighted_threshold",
    "override": "override_threshold",
}


class HybridStage:
    """Create hybrid strategies from top convergence pairs.

    Parameters
    ----------
    config:
        Hybrid construction configuration.
    print_fn:
        Callable for progress output.
    """

    def __init__(
        self,
        config: HybridConfig,
        print_fn: Callable[..., None] = print,
    ) -> None:
        self._cfg = config
        self._out = print_fn

    def run(
        self,
        data_result: DataResult,
        training_result: TrainingStageResult,
        convergence_result: ConvergenceResult,
    ) -> HybridStageResult:
        """Build hybrid strategies from the top convergence pairs."""
        from stockdownloader.strategy.registration_loader import ensure_registered
        from stockdownloader.strategy.registry import StrategyRegistry

        ensure_registered()

        entries: list[HybridStrategyEntry] = []
        top_pairs = convergence_result.pairs[: self._cfg.top_pairs]

        # De-duplicate by (model_id, strategy_name) to avoid redundant hybrids
        seen: set[tuple[str, str]] = set()

        for pair in top_pairs:
            key = (pair.model_id, pair.strategy_name)
            if key in seen:
                continue
            seen.add(key)

            # Find the ModelCandidate
            candidate = next(
                (c for c in training_result.candidates
                 if c.model_id == pair.model_id),
                None,
            )
            if candidate is None:
                continue

            model = candidate.training_result.model
            feature_names = candidate.dataset.feature_names

            for mode in self._cfg.modes:
                strategy_cls = _STRATEGY_CLASS.get(mode)
                threshold_attr = _THRESHOLD_ATTR.get(mode)
                if strategy_cls is None or threshold_attr is None:
                    continue

                threshold = getattr(self._cfg, threshold_attr, 0.5)

                try:
                    base = StrategyRegistry.create(pair.strategy_name)
                except (KeyError, ValueError):
                    continue

                label = (
                    f"{base.name} (ML-{mode.title()}, "
                    f"{pair.model_id})"
                )
                hybrid = strategy_cls(
                    base_strategy=base,
                    model=model,
                    feature_names=feature_names,
                    threshold=threshold,
                    label=label,
                )

                entries.append(HybridStrategyEntry(
                    name=label,
                    mode=mode,
                    model_id=pair.model_id,
                    base_strategy_name=pair.strategy_name,
                    strategy=hybrid,
                    label_config=candidate.label_config,
                    model_config=candidate.model_config,
                    hybrid_threshold=threshold,
                ))

        self._out(f"Created {len(entries)} hybrid strategies")
        return HybridStageResult(hybrid_strategies=entries)
