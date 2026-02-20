"""Stage 3: Strategy-ML Convergence Analysis.

Measures how well ML predictions agree with existing trading strategy
signals.  The key insight: bars where ML *and* a strategy both agree on
direction should have higher forward win rates than bars where they
disagree.  The "lift" (agreement_win_rate − disagreement_win_rate) is the
core convergence metric.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from stockdownloader.ml.pipeline.config import ConvergenceConfig
from stockdownloader.ml.pipeline.results import (
    ConvergencePair,
    ConvergenceResult,
    DataResult,
    ModelCandidate,
    TrainingStageResult,
)

logger = logging.getLogger(__name__)


class ConvergenceStage:
    """Analyse agreement between ML predictions and strategy signals.

    Parameters
    ----------
    config:
        Convergence analysis configuration.
    print_fn:
        Callable for progress output.
    """

    def __init__(
        self,
        config: ConvergenceConfig,
        print_fn: Callable[..., None] = print,
    ) -> None:
        self._cfg = config
        self._out = print_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data_result: DataResult,
        training_result: TrainingStageResult,
    ) -> ConvergenceResult:
        """Run convergence analysis for top models × all daily strategies."""
        from stockdownloader.strategy.registrations import ensure_registered
        from stockdownloader.strategy.registry import StrategyRegistry
        from stockdownloader.strategy.trading_strategy import TradingStrategy

        ensure_registered()
        data = data_result.data

        # Gather daily strategies
        strategies: dict[str, TradingStrategy] = {}
        for cat in self._cfg.strategy_categories:
            for entry in StrategyRegistry.all_entries(category=cat):
                try:
                    strategies[entry.name] = StrategyRegistry.create(entry.name)
                except (KeyError, ValueError):
                    continue

        if not strategies:
            self._out("No strategies found for convergence analysis.")
            return ConvergenceResult()

        # Use the top N models
        top_models = training_result.candidates[: self._cfg.top_models]
        self._out(
            f"Convergence: {len(top_models)} models × "
            f"{len(strategies)} strategies"
        )

        pairs: list[ConvergencePair] = []

        for candidate in top_models:
            ml_preds = self._get_ml_predictions(candidate, data)
            if not ml_preds:
                continue

            for strat_name, strategy in strategies.items():
                pair = self._analyze_pair(
                    candidate, strat_name, strategy, data, ml_preds,
                )
                if pair is not None:
                    pairs.append(pair)
                    if pair.lift > 0.05:
                        self._out(
                            f"  {candidate.model_id} × {strat_name}: "
                            f"agree={pair.agreement_rate:.1%}, "
                            f"lift={pair.lift:+.1%}"
                        )

        pairs.sort(key=lambda p: p.lift, reverse=True)
        return ConvergenceResult(
            pairs=pairs,
            best_pair=pairs[0] if pairs else None,
        )

    # ------------------------------------------------------------------
    # ML prediction extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ml_predictions(
        candidate: ModelCandidate,
        data: list,
    ) -> dict[int, float]:
        """Run ML model predictions on all valid bar indices.

        Returns ``{bar_index: probability_of_class_1}``.
        """
        model = candidate.training_result.model
        dataset = candidate.dataset
        warmup = 201
        fp = candidate.label_config.forward_period
        last_valid = len(data) - fp - 1

        predictions: dict[int, float] = {}
        n_samples = len(dataset.X)

        for sample_i in range(n_samples):
            bar_i = warmup + sample_i
            if bar_i > last_valid:
                break
            try:
                X_row = dataset.X[sample_i: sample_i + 1]
                proba = model.predict_proba(X_row)
                predictions[bar_i] = float(proba[0, 1])
            except (ValueError, IndexError, TypeError):
                predictions[bar_i] = 0.5

        return predictions

    # ------------------------------------------------------------------
    # Pair analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _analyze_pair(
        candidate: ModelCandidate,
        strategy_name: str,
        strategy: object,
        data: list,
        ml_predictions: dict[int, float],
    ) -> ConvergencePair | None:
        """Compare ML vs strategy at each bar, compute convergence metrics."""
        from stockdownloader.strategy.trading_strategy import Signal

        threshold = candidate.training_result.optimal_threshold
        fp = candidate.label_config.forward_period
        pt = candidate.label_config.profit_threshold

        agree_wins = 0
        agree_total = 0
        disagree_wins = 0
        disagree_total = 0

        for bar_i, prob in ml_predictions.items():
            if bar_i + fp >= len(data):
                break

            # ML signal
            ml_bullish = prob > threshold

            # Strategy signal
            try:
                signal = strategy.evaluate(data, bar_i)  # type: ignore[attr-defined]
            except (ValueError, IndexError, KeyError, ZeroDivisionError):
                continue

            if signal == Signal.HOLD:
                continue

            strat_bullish = signal == Signal.BUY

            # Forward return (ground truth)
            current = float(data[bar_i].close)
            future = float(data[bar_i + fp].close)
            if current == 0:
                continue
            actual_win = (future - current) / current > pt

            if ml_bullish == strat_bullish:
                agree_total += 1
                if actual_win:
                    agree_wins += 1
            else:
                disagree_total += 1
                if actual_win:
                    disagree_wins += 1

        total = agree_total + disagree_total
        if total < 20:
            return None

        agree_wr = agree_wins / agree_total if agree_total > 0 else 0.0
        disagree_wr = (
            disagree_wins / disagree_total if disagree_total > 0 else 0.0
        )

        return ConvergencePair(
            model_id=candidate.model_id,
            strategy_name=strategy_name,
            agreement_rate=agree_total / total,
            agreement_win_rate=agree_wr,
            disagreement_win_rate=disagree_wr,
            lift=agree_wr - disagree_wr,
            n_agreement_bars=agree_total,
            n_disagreement_bars=disagree_total,
        )
