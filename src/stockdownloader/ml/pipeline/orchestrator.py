"""Pipeline Orchestrator — chains all 6 stages into a single run.

Usage::

    from stockdownloader.ml.pipeline.config import PipelineConfig
    from stockdownloader.ml.pipeline.orchestrator import MLPipelineOrchestrator

    pipeline = MLPipelineOrchestrator(PipelineConfig())
    result = pipeline.run()
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from stockdownloader.ml.pipeline.config import PipelineConfig
from stockdownloader.ml.pipeline.results import PipelineResult
from stockdownloader.ml.pipeline.stage_backtest import BacktestStage
from stockdownloader.ml.pipeline.stage_convergence import ConvergenceStage
from stockdownloader.ml.pipeline.stage_data import DataStage
from stockdownloader.ml.pipeline.stage_hybrid_strategies import HybridStage
from stockdownloader.ml.pipeline.stage_selection import SelectionStage
from stockdownloader.ml.pipeline.stage_training import TrainingStage

logger = logging.getLogger(__name__)


class MLPipelineOrchestrator:
    """End-to-end ML-driven trading pipeline.

    Runs 6 stages sequentially, passing typed results forward::

        Data → Training → Convergence → Hybrid → Backtest → Selection

    Parameters
    ----------
    config:
        Pipeline configuration.  Defaults are sensible for SPY daily.
    print_fn:
        Output function for progress (default: :func:`print`).
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        print_fn: Callable[..., None] = print,
        hmm_snapshots: list | None = None,
    ) -> None:
        self._cfg = config or PipelineConfig()
        self._out = print_fn
        self._hmm_snapshots = hmm_snapshots

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """Execute the full pipeline and return the combined result."""
        result = PipelineResult(config=self._cfg)
        total_start = time.time()

        self._out("=" * 100)
        self._out("  ML-DRIVEN TRADING PIPELINE")
        self._out(
            f"  Symbol: {self._cfg.data.symbol}  "
            f"Range: {self._cfg.data.range_}"
        )
        self._out("=" * 100)
        self._out()

        # ---- Stage 1: Data ----
        self._out("STAGE 1: DATA ACQUISITION")
        self._out("-" * 50)
        t0 = time.time()
        data_stage = DataStage(self._cfg.data, self._out)
        result.data_result = data_stage.run()

        # Inject HMM snapshots if provided
        if self._hmm_snapshots is not None:
            result.data_result.hmm_snapshots = self._hmm_snapshots
            self._out(
                f"  HMM regimes: {len(self._hmm_snapshots)} snapshots injected"
            )

        self._out(f"  Completed in {time.time() - t0:.1f}s\n")

        # ---- Stage 2: Training ----
        self._out("STAGE 2: MULTI-CONFIGURATION ML TRAINING")
        self._out("-" * 50)
        t0 = time.time()
        training_stage = TrainingStage(self._cfg.training, self._out)
        result.training_result = training_stage.run(result.data_result)
        self._out(
            f"  {len(result.training_result.candidates)} models "
            f"trained in {time.time() - t0:.1f}s\n"
        )

        if not result.training_result.candidates:
            self._out(
                "ERROR: No models trained successfully. Aborting pipeline."
            )
            return result

        # ---- Stage 3: Convergence ----
        self._out("STAGE 3: STRATEGY-ML CONVERGENCE ANALYSIS")
        self._out("-" * 50)
        t0 = time.time()
        conv_stage = ConvergenceStage(self._cfg.convergence, self._out)
        result.convergence_result = conv_stage.run(
            result.data_result, result.training_result,
        )
        self._out(
            f"  {len(result.convergence_result.pairs)} pairs "
            f"analyzed in {time.time() - t0:.1f}s\n"
        )

        # ---- Stage 4: Hybrid Construction ----
        self._out("STAGE 4: ML-INFORMED STRATEGY CONSTRUCTION")
        self._out("-" * 50)
        t0 = time.time()
        hybrid_stage = HybridStage(self._cfg.hybrid, self._out)
        result.hybrid_result = hybrid_stage.run(
            result.data_result,
            result.training_result,
            result.convergence_result,
        )
        self._out(
            f"  {len(result.hybrid_result.hybrid_strategies)} hybrids "
            f"created in {time.time() - t0:.1f}s\n"
        )

        # ---- Stage 5: Backtesting ----
        self._out("STAGE 5: COMPREHENSIVE BACKTESTING")
        self._out("-" * 50)
        t0 = time.time()
        bt_stage = BacktestStage(self._cfg.backtest, self._out)
        result.backtest_result = bt_stage.run(
            result.data_result, result.hybrid_result,
        )
        self._out(
            f"  {len(result.backtest_result.entries)} backtests "
            f"in {time.time() - t0:.1f}s\n"
        )

        # ---- Stage 6: Selection + Export ----
        self._out("STAGE 6: SELECTION + PINESCRIPT EXPORT")
        self._out("-" * 50)
        t0 = time.time()
        sel_stage = SelectionStage(self._cfg.selection, self._out)
        best, pine_code, pine_path = sel_stage.run(
            result.backtest_result,
            result.training_result,
            self._cfg.data.symbol,
        )
        result.best_strategy = best
        result.pine_script = pine_code
        result.pine_path = pine_path
        self._out(f"  Completed in {time.time() - t0:.1f}s\n")

        # ---- Summary ----
        total = time.time() - total_start
        self._out("=" * 100)
        self._out(f"  PIPELINE COMPLETE  ({total:.1f}s)")
        self._out(
            f"  Models trained: "
            f"{len(result.training_result.candidates)}"
        )
        self._out(
            f"  Convergence pairs: "
            f"{len(result.convergence_result.pairs)}"
        )
        self._out(
            f"  Hybrid strategies: "
            f"{len(result.hybrid_result.hybrid_strategies)}"
        )
        self._out(
            f"  Backtests run: "
            f"{len(result.backtest_result.entries)}"
        )
        if result.best_strategy:
            self._out(
                f"  Best: {result.best_strategy.strategy_name} "
                f"(score={result.best_strategy.composite_score:.1f})"
            )
        if result.pine_path:
            self._out(f"  PineScript: {result.pine_path}")
        self._out("=" * 100)

        return result
