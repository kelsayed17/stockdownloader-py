"""Pipeline configuration dataclasses.

Each stage has its own frozen config; :class:`PipelineConfig` aggregates them
so the orchestrator can be initialised from a single object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stockdownloader.util.config_loader import DEFAULT_ML_PIPELINE_DIR


# ------------------------------------------------------------------
# Alternative data configuration
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AltDataConfig:
    """Configuration for alternative data sources (FTD, SI, dark pool, etc.)."""

    enable_ftd: bool = True
    enable_short_interest: bool = True
    enable_dark_pool: bool = True
    enable_ownership: bool = True
    enable_borrow_rate: bool = True
    ftd_start_year: int = 2004
    cache_dir: str = "data/cache"
    user_agent: str = "StockDownloader admin@example.com"
    finra_client_id: str = ""
    finra_client_secret: str = ""


# ------------------------------------------------------------------
# Stage 1 — Data acquisition
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Configuration for the data acquisition stage."""

    symbol: str = "SPY"
    range_: str = "10y"
    interval: str = "1d"
    cache_dir: str = "data/cache"
    use_cache: bool = True
    csv_file: str | None = None  # Path to CSV file (authoritative source)
    alt_data: AltDataConfig | None = None


# ------------------------------------------------------------------
# Stage 2 — Multi-configuration ML training
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingGridConfig:
    """Defines the grid of model configurations to sweep."""

    forward_periods: tuple[int, ...] = (5, 10, 20)
    profit_thresholds: tuple[float, ...] = (0.003, 0.005, 0.01)
    model_types: tuple[str, ...] = ("gradient_boosting", "logistic_regression")
    use_class_balance_options: tuple[bool, ...] = (False, True)
    use_atr_labels_options: tuple[bool, ...] = (False, True)
    use_feature_selection_options: tuple[bool, ...] = (False,)
    n_estimators: int = 200
    max_depth: int = 4
    random_seed: int = 42


# ------------------------------------------------------------------
# Stage 3 — Convergence analysis
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConvergenceConfig:
    """Configuration for the strategy-ML convergence analysis stage."""

    strategy_categories: tuple[str, ...] = ("daily",)
    min_agreement_rate: float = 0.3
    top_models: int = 10


# ------------------------------------------------------------------
# Stage 4 — Hybrid strategy construction
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Configuration for ML-informed strategy construction."""

    modes: tuple[str, ...] = ("confirmed", "weighted", "override")
    confirmed_threshold: float = 0.6
    weighted_threshold: float = 0.5
    override_threshold: float = 0.55
    top_pairs: int = 15


# ------------------------------------------------------------------
# Stage 5 — Backtesting
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for the backtesting stage."""

    initial_capital: float = 100_000.0
    commission: float = 10.0
    slippage_pct: float = 0.0  # per-trade slippage as fraction (e.g. 0.001 = 0.1%)
    walk_forward_windows: int = 5
    top_for_walk_forward: int = 20
    max_workers: int | None = None  # None → os.cpu_count()
    true_walk_forward: bool = False  # retrain ML model per window (slower but rigorous)
    wf_min_train_bars: int = 500  # minimum IS bars for model training per window


# ------------------------------------------------------------------
# Stage 6 — Selection & export
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Configuration for strategy selection and PineScript export."""

    top_n: int = 5
    pine_depth: int = 6
    pine_top_features: int = 15
    output_dir: str = str(DEFAULT_ML_PIPELINE_DIR)
    export_pine: bool = True


# ------------------------------------------------------------------
# Top-level pipeline config
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Aggregates all stage configs into a single pipeline config."""

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingGridConfig = field(default_factory=TrainingGridConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    verbose: bool = True
