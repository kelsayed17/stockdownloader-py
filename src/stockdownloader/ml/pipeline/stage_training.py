"""Stage 2: Multi-Configuration ML Training.

Sweeps a grid of label/model configurations, trains each, and returns
ranked :class:`ModelCandidate` objects for downstream stages.

Optimizations
-------------
- **Dataset caching**: Datasets are keyed by ``(forward_period,
  profit_threshold, use_atr)`` and reused across model types / class-
  balance variants.  This avoids rebuilding the same feature matrix
  multiple times.
- **Parallel training**: Uses ``joblib.Parallel`` to train models
  concurrently (falls back to sequential if *joblib* is unavailable).
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Callable

from stockdownloader.ml.pipeline.config import TrainingGridConfig
from stockdownloader.ml.pipeline.results import (
    DataResult,
    ModelCandidate,
    TrainingStageResult,
)

logger = logging.getLogger(__name__)

try:
    from joblib import Parallel, delayed  # type: ignore[import-untyped]
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


class TrainingStage:
    """Train multiple models across a configuration grid.

    Parameters
    ----------
    config:
        Training grid configuration.
    print_fn:
        Callable for progress output.
    """

    def __init__(
        self,
        config: TrainingGridConfig,
        print_fn: Callable[..., None] = print,
    ) -> None:
        self._cfg = config
        self._out = print_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data_result: DataResult) -> TrainingStageResult:
        """Build datasets with different labels, train models, return ranked candidates."""
        from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig
        from stockdownloader.ml.feature_extractor import FeatureExtractor
        from stockdownloader.ml.trainer import MLModelConfig, MLTrainer
        from stockdownloader.util.indicators.hub import IndicatorHub

        data = data_result.data
        hub = IndicatorHub()
        extractor = FeatureExtractor(
            hub=hub,
            alt_data_store=data_result.alt_data_store,
            hmm_snapshots=data_result.hmm_snapshots,
        )

        # -------------------------------------------------------
        # Phase 1: Pre-build all unique datasets (by label config)
        # -------------------------------------------------------
        label_combos = list(product(
            self._cfg.forward_periods,
            self._cfg.profit_thresholds,
            self._cfg.use_atr_labels_options,
        ))

        dataset_cache: dict[tuple, object] = {}  # (fp, pt, atr) → dataset

        self._out(
            f"Building {len(label_combos)} unique datasets "
            f"(caching shared label configs)..."
        )

        for fp, pt, atr in label_combos:
            key = (fp, pt, atr)
            label_cfg = LabelConfig(
                forward_period=fp,
                profit_threshold=pt,
                use_atr_threshold=atr,
            )
            builder = DatasetBuilder(
                extractor, label_config=label_cfg, hub=hub,
            )
            try:
                dataset = builder.build(data)
                dataset_cache[key] = dataset
            except ValueError as exc:
                self._out(f"  SKIP dataset fp={fp} pt={pt} atr={atr}: {exc}")
                continue

        self._out(f"  {len(dataset_cache)} datasets built.\n")

        # -------------------------------------------------------
        # Phase 2: Train all model/balance combos (parallel)
        # -------------------------------------------------------
        training_combos = list(product(
            self._cfg.forward_periods,
            self._cfg.profit_thresholds,
            self._cfg.model_types,
            self._cfg.use_class_balance_options,
            self._cfg.use_atr_labels_options,
        ))

        self._out(f"Training {len(training_combos)} model configurations...")

        def _train_one(
            fp: int, pt: float, mt: str, bal: bool, atr: bool,
        ) -> ModelCandidate | None:
            key = (fp, pt, atr)
            dataset = dataset_cache.get(key)
            if dataset is None:
                return None

            label_cfg = LabelConfig(
                forward_period=fp,
                profit_threshold=pt,
                use_atr_threshold=atr,
            )
            model_cfg = MLModelConfig(
                model_type=mt,
                n_estimators=self._cfg.n_estimators,
                max_depth=self._cfg.max_depth,
                use_class_balance=bal,
            )
            trainer = MLTrainer(model_cfg)

            try:
                result = trainer.train(dataset)
            except (ValueError, Exception):
                return None

            model_id = self._make_model_id(fp, pt, mt, bal, atr)
            return ModelCandidate(
                model_id=model_id,
                training_result=result,
                label_config=label_cfg,
                model_config=model_cfg,
                dataset=dataset,
            )

        if _HAS_JOBLIB and len(training_combos) > 4:
            self._out("  (using parallel training)\n")
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(_train_one)(fp, pt, mt, bal, atr)
                for fp, pt, mt, bal, atr in training_combos
            )
            candidates: list[ModelCandidate] = [
                r for r in results if r is not None
            ]
        else:
            candidates = []
            for i, (fp, pt, mt, bal, atr) in enumerate(training_combos):
                model_id = self._make_model_id(fp, pt, mt, bal, atr)
                self._out(f"  [{i + 1}/{len(training_combos)}] {model_id}...")
                cand = _train_one(fp, pt, mt, bal, atr)
                if cand is not None:
                    self._out(
                        f"    ACC={cand.training_result.oos_accuracy:.3f}  "
                        f"AUC={cand.training_result.oos_roc_auc:.3f}  "
                        f"thresh={cand.training_result.optimal_threshold:.2f}"
                    )
                    candidates.append(cand)
                else:
                    self._out(f"    SKIP")

        # Rank by ROC-AUC
        candidates.sort(
            key=lambda c: c.training_result.oos_roc_auc, reverse=True,
        )

        best_auc = candidates[0] if candidates else None
        best_acc = (
            max(candidates, key=lambda c: c.training_result.oos_accuracy)
            if candidates else None
        )

        self._out(f"\nTrained {len(candidates)} models successfully.")
        if best_auc:
            self._out(
                f"Best by AUC: {best_auc.model_id} "
                f"(AUC={best_auc.training_result.oos_roc_auc:.3f})"
            )

        return TrainingStageResult(
            candidates=candidates,
            best_by_auc=best_auc,
            best_by_accuracy=best_acc,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_model_id(
        fp: int, pt: float, mt: str, bal: bool, atr: bool,
    ) -> str:
        """Generate a short human-readable model identifier."""
        mt_short = mt[:2]
        pt_str = str(pt).replace(".", "")
        bal_str = "bal" if bal else "nobal"
        atr_str = "atr" if atr else "abs"
        return f"{mt_short}_fp{fp}_pt{pt_str}__{bal_str}_{atr_str}"
