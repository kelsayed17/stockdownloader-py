"""ML-driven trading pipeline — end-to-end orchestration.

Stages::

    Data → Training → Convergence → Hybrid Strategies → Backtesting → Selection+Export

Exports
-------
Orchestrator:
    MLPipelineOrchestrator

Config:
    PipelineConfig, DataConfig, TrainingGridConfig, ConvergenceConfig,
    HybridConfig, BacktestConfig, SelectionConfig, AltDataConfig

Results:
    PipelineResult, DataResult, ModelCandidate, TrainingStageResult,
    ConvergencePair, ConvergenceResult, HybridStrategyEntry,
    HybridStageResult, WalkForwardWindowResult, BacktestEntry,
    BacktestStageResult

Stages:
    DataStage, TrainingStage, ConvergenceStage, HybridStage,
    BacktestStage, SelectionStage
"""

from stockdownloader.ml.pipeline.config import (
    AltDataConfig,
    BacktestConfig,
    ConvergenceConfig,
    DataConfig,
    HybridConfig,
    PipelineConfig,
    SelectionConfig,
    TrainingGridConfig,
)
from stockdownloader.ml.pipeline.orchestrator import MLPipelineOrchestrator
from stockdownloader.ml.pipeline.results import (
    BacktestEntry,
    BacktestStageResult,
    ConvergencePair,
    ConvergenceResult,
    DataResult,
    HybridStageResult,
    HybridStrategyEntry,
    ModelCandidate,
    PipelineResult,
    TrainingStageResult,
    WalkForwardWindowResult,
)
from stockdownloader.ml.pipeline.stage_backtest import BacktestStage
from stockdownloader.ml.pipeline.stage_convergence import ConvergenceStage
from stockdownloader.ml.pipeline.stage_data import DataStage
from stockdownloader.ml.pipeline.stage_hybrid import HybridStage
from stockdownloader.ml.pipeline.stage_selection import SelectionStage
from stockdownloader.ml.pipeline.stage_training import TrainingStage

__all__ = [
    # Orchestrator
    "MLPipelineOrchestrator",
    # Config
    "PipelineConfig",
    "DataConfig",
    "TrainingGridConfig",
    "ConvergenceConfig",
    "HybridConfig",
    "BacktestConfig",
    "SelectionConfig",
    "AltDataConfig",
    # Results
    "PipelineResult",
    "DataResult",
    "ModelCandidate",
    "TrainingStageResult",
    "ConvergencePair",
    "ConvergenceResult",
    "HybridStrategyEntry",
    "HybridStageResult",
    "WalkForwardWindowResult",
    "BacktestEntry",
    "BacktestStageResult",
    # Stages
    "DataStage",
    "TrainingStage",
    "ConvergenceStage",
    "HybridStage",
    "BacktestStage",
    "SelectionStage",
]
