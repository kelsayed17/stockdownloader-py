"""Stage 6: Best Strategy Selection and PineScript Export.

Selects the top-performing strategy from the backtest results, prints
a ranked leaderboard, and optionally exports PineScript for TradingView.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from stockdownloader.ml.pipeline.config import SelectionConfig
from stockdownloader.ml.pipeline.results import (
    BacktestEntry,
    BacktestStageResult,
    ModelCandidate,
    TrainingStageResult,
)

logger = logging.getLogger(__name__)


class SelectionStage:
    """Select best performers and export PineScript.

    Parameters
    ----------
    config:
        Selection configuration.
    print_fn:
        Callable for progress output.
    """

    def __init__(
        self,
        config: SelectionConfig,
        print_fn: Callable[..., None] = print,
    ) -> None:
        self._cfg = config
        self._out = print_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        backtest_result: BacktestStageResult,
        training_result: TrainingStageResult,
        symbol: str,
    ) -> tuple[BacktestEntry | None, str | None, str | None]:
        """Select best strategy, export PineScript.

        Returns
        -------
        (best_entry, pine_code, pine_path) or (None, None, None)
        """
        ranked = backtest_result.ranked
        if not ranked:
            self._out("No strategies to select from!")
            return None, None, None

        # Print leaderboard
        self._print_leaderboard(ranked)

        best = ranked[0]
        self._out(
            f"\nBest: {best.strategy_name} "
            f"(score={best.composite_score:.1f})"
        )

        # Export PineScript
        pine_code: str | None = None
        pine_path: str | None = None

        if self._cfg.export_pine and training_result.candidates:
            candidate = self._find_candidate(best, training_result)
            if candidate is not None:
                pine_code, pine_path = self._export_pine(symbol, candidate)

        return best, pine_code, pine_path

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def _print_leaderboard(self, ranked: list[BacktestEntry]) -> None:
        top_n = self._cfg.top_n
        self._out(f"\nTOP {top_n} STRATEGIES:")
        self._out("-" * 100)
        self._out(
            f"{'Rank':<5} {'Strategy':<50} "
            f"{'P/L':>12} {'WR':>6} {'Sharpe':>7} "
            f"{'Score':>7} {'WF':>6}"
        )
        self._out("-" * 100)

        for i, entry in enumerate(ranked[:top_n], 1):
            r = entry.result
            pnl = float(r.total_pnl)
            wr = float(r.win_rate)
            sharpe = float(r.sharpe_ratio(252))
            wf = (
                f"{entry.walk_forward_degradation:.2f}"
                if entry.walk_forward_degradation is not None
                else "N/A"
            )
            mode_tag = f"[{entry.mode}]" if entry.is_hybrid else ""
            name = f"{entry.strategy_name[:48]}{mode_tag}"
            self._out(
                f"{i:<5} {name:<50} "
                f"${pnl:>10,.2f}  {wr:>5.1f}%  {sharpe:>6.2f}  "
                f"{entry.composite_score:>6.1f}  {wf:>6}"
            )

        self._out("-" * 100)

    # ------------------------------------------------------------------
    # PineScript export
    # ------------------------------------------------------------------

    @staticmethod
    def _find_candidate(
        best: BacktestEntry,
        training_result: TrainingStageResult,
    ) -> ModelCandidate | None:
        """Find the ModelCandidate for the best strategy."""
        if best.model_id:
            for c in training_result.candidates:
                if c.model_id == best.model_id:
                    return c
        # Fallback to best overall model
        return training_result.best_by_auc

    def _export_pine(
        self,
        symbol: str,
        candidate: ModelCandidate,
    ) -> tuple[str, str]:
        """Train surrogate tree and generate Pine Script.

        When the dataset includes non-Pine-compatible features (alt data,
        HMM regimes), the surrogate tree is trained only on the base 63
        Pine-compatible features — alt data and HMM features have no
        TradingView equivalent.
        """
        from stockdownloader.ml.feature_extractor import (
            _ALT_DATA_NAMES,
            _BASE_FEATURE_NAMES,
            _HMM_FEATURE_NAMES,
        )
        from stockdownloader.pinescript import PineScriptGenerator
        from stockdownloader.pinescript.ml_export import (
            DecisionTreeExporter,
            ml_signal_strategy,
        )

        dataset = candidate.dataset
        feature_importances = candidate.training_result.feature_importances

        # Check if dataset has non-Pine features (alt data and/or HMM)
        n_base = len(_BASE_FEATURE_NAMES)
        total_features = len(dataset.feature_names)
        non_pine_names = set(_ALT_DATA_NAMES) | set(_HMM_FEATURE_NAMES)

        if total_features > n_base:
            # Trim dataset to base features only for surrogate tree
            from stockdownloader.ml.dataset_builder import MLDataset

            base_name_set = set(_BASE_FEATURE_NAMES)

            # Find indices of base features (they are always the first n_base)
            trimmed_X = dataset.X[:, :n_base]
            trimmed_names = dataset.feature_names[:n_base]
            n_excluded = total_features - n_base

            # feature_importances is dict[str, float] — filter to base only
            trimmed_importances = {
                k: v for k, v in feature_importances.items()
                if k in base_name_set
            }

            # Re-create a trimmed dataset for the surrogate tree
            dataset = MLDataset(
                X=trimmed_X,
                y=dataset.y,
                feature_names=trimmed_names,
                dates=dataset.dates,
                label_config=dataset.label_config,
            )
            feature_importances = trimmed_importances

            self._out(
                f"  Surrogate tree uses {n_base} Pine-compatible features "
                f"(excluded {n_excluded} non-Pine features)"
            )

        exporter = DecisionTreeExporter(
            max_depth=self._cfg.pine_depth,
            min_samples_leaf=20,
        )
        exporter.train_surrogate(
            dataset,
            feature_importances,
            top_n=self._cfg.pine_top_features,
        )

        strategy_def = ml_signal_strategy(symbol, exporter)
        gen = PineScriptGenerator()
        pine_code = gen.generate(strategy_def)

        # Save
        output_dir = Path(self._cfg.output_dir)
        pine_dir = output_dir / "pinescript"
        pine_dir.mkdir(parents=True, exist_ok=True)
        pine_path = pine_dir / f"{symbol.lower()}_ml_pipeline.pine"
        pine_path.write_text(pine_code)

        self._out(f"PineScript saved: {pine_path}")
        return pine_code, str(pine_path)
