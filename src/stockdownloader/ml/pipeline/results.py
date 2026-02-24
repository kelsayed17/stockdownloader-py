"""Typed result dataclasses for inter-stage data flow.

Each pipeline stage produces a result that the next stage consumes.
All results are plain data containers with no business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stockdownloader.backtest.backtest_result import BacktestResult
    from stockdownloader.ml.dataset_builder import LabelConfig, MLDataset
    from stockdownloader.ml.trainer import MLModelConfig, TrainingResult
    from stockdownloader.core.models.price import PriceData
    from stockdownloader.strategy.trading_strategy import TradingStrategy


# ------------------------------------------------------------------
# Stage 1
# ------------------------------------------------------------------


@dataclass(slots=True)
class DataResult:
    """Output of the data acquisition stage."""

    symbol: str
    data: list[PriceData]
    date_range: tuple[str, str]
    bar_count: int
    cached: bool = False
    alt_data_store: Any = None  # AlternativeDataStore | None
    hmm_snapshots: list[Any] | None = None  # list[RegimeSnapshot] | None


# ------------------------------------------------------------------
# Stage 2
# ------------------------------------------------------------------


@dataclass(slots=True)
class ModelCandidate:
    """One trained model with its configuration and metrics."""

    model_id: str
    training_result: TrainingResult
    label_config: LabelConfig
    model_config: MLModelConfig
    dataset: MLDataset


@dataclass(slots=True)
class TrainingStageResult:
    """Output of the multi-config training stage."""

    candidates: list[ModelCandidate] = field(default_factory=list)
    best_by_auc: ModelCandidate | None = None
    best_by_accuracy: ModelCandidate | None = None


# ------------------------------------------------------------------
# Stage 3
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConvergencePair:
    """Agreement metrics for one (model × strategy) pair."""

    model_id: str
    strategy_name: str
    agreement_rate: float
    agreement_win_rate: float
    disagreement_win_rate: float
    lift: float
    n_agreement_bars: int
    n_disagreement_bars: int


@dataclass(slots=True)
class ConvergenceResult:
    """Output of the convergence analysis stage."""

    pairs: list[ConvergencePair] = field(default_factory=list)
    best_pair: ConvergencePair | None = None


# ------------------------------------------------------------------
# Stage 4
# ------------------------------------------------------------------


@dataclass(slots=True)
class HybridStrategyEntry:
    """One ML-informed strategy variant."""

    name: str
    mode: str  # "confirmed", "weighted", "override"
    model_id: str
    base_strategy_name: str
    strategy: TradingStrategy
    # Training metadata for true walk-forward retraining
    label_config: Any = None  # LabelConfig
    model_config: Any = None  # MLModelConfig
    hybrid_threshold: float = 0.5


@dataclass(slots=True)
class HybridStageResult:
    """Output of the hybrid strategy construction stage."""

    hybrid_strategies: list[HybridStrategyEntry] = field(default_factory=list)


# ------------------------------------------------------------------
# Stage 5
# ------------------------------------------------------------------


@dataclass(slots=True)
class WalkForwardWindowResult:
    """Per-window metrics from true walk-forward validation."""

    window_idx: int
    is_bars: int
    oos_bars: int
    is_score: float
    oos_score: float
    oos_trades: int
    oos_pnl: float
    oos_win_rate: float
    model_auc: float  # OOS AUC of the retrained model


@dataclass(slots=True)
class BacktestEntry:
    """Backtest result for one strategy variant."""

    strategy_name: str
    is_hybrid: bool
    mode: str  # "baseline", "confirmed", "weighted", "override"
    model_id: str | None
    result: BacktestResult
    composite_score: float
    walk_forward_degradation: float | None = None
    _strategy_ref: TradingStrategy | None = None  # kept for WF reuse
    # Training metadata (for true walk-forward retraining)
    _label_config: Any = None  # LabelConfig
    _model_config: Any = None  # MLModelConfig
    _base_strategy_name: str | None = None  # registry name for base strategy
    _hybrid_mode: str | None = None  # confirmed/weighted/override
    _hybrid_threshold: float | None = None  # threshold used
    # True walk-forward per-window results
    wf_window_results: list[WalkForwardWindowResult] | None = None
    wf_avg_oos_score: float | None = None


@dataclass(slots=True)
class BacktestStageResult:
    """Output of the backtesting stage."""

    entries: list[BacktestEntry] = field(default_factory=list)
    ranked: list[BacktestEntry] = field(default_factory=list)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------


@dataclass(slots=True)
class PipelineResult:
    """Complete pipeline output aggregating all stage results."""

    config: Any  # PipelineConfig
    data_result: DataResult | None = None
    training_result: TrainingStageResult | None = None
    convergence_result: ConvergenceResult | None = None
    hybrid_result: HybridStageResult | None = None
    backtest_result: BacktestStageResult | None = None
    best_strategy: BacktestEntry | None = None
    pine_script: str | None = None
    pine_path: str | None = None
