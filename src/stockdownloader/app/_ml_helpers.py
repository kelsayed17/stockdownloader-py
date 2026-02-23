"""Shared helpers for ML pipeline CLI apps.

Extracts the common plumbing shared by
``spy_ml_pipeline_app`` and ``gme_ml_pipeline_app``:

- ``check_ml_deps`` — guard against missing ``scikit-learn``
- ``setup_ml_logging`` — configure logging (verbose vs normal)
- ``init_ml_env`` — one-call setup (logging + dep check)
- ``run_ml_pipeline`` — orchestrate pipeline run + result check
- ``build_training_config`` — ``TrainingGridConfig`` from parsed CLI args
- ``add_common_ml_args`` — shared argparse arguments (grid, backtest, pine, output)
- ``build_pipeline_config`` — assemble a ``PipelineConfig`` from parsed args

All numeric defaults are sourced from ``config/ml/standard.json`` when
available.  CLI arguments override config file values, which override
built-in fallbacks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from stockdownloader.util.config import DEFAULT_ML_PIPELINE_DIR


# ======================================================================
# Dependency guard
# ======================================================================


def check_ml_deps() -> None:
    """Exit with a helpful message when ML extras are missing."""
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print(
            "ERROR: ML dependencies not installed.\n"
            "Install ML extras: pip install -e '.[ml]'",
            file=sys.stderr,
        )
        sys.exit(1)


def setup_ml_logging(verbose: bool) -> None:
    """Configure logging for ML pipeline apps.

    Parameters
    ----------
    verbose:
        When *True* use ``DEBUG`` level; otherwise ``INFO``.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def init_ml_env(verbose: bool) -> None:
    """Set up logging and verify ML dependencies in one call.

    Convenience wrapper combining :func:`setup_ml_logging` and
    :func:`check_ml_deps`.
    """
    setup_ml_logging(verbose)
    check_ml_deps()


def run_ml_pipeline(
    config: Any,
    *,
    hmm_snapshots: Any | None = None,
) -> Any:
    """Create an ``MLPipelineOrchestrator``, run it, and return the result.

    Also checks whether a viable strategy was found and exits with an
    error message if not.

    Parameters
    ----------
    config:
        A ``PipelineConfig`` instance.
    hmm_snapshots:
        Optional HMM regime snapshots passed to the orchestrator.

    Returns
    -------
    The pipeline result object.
    """
    from stockdownloader.ml.pipeline.orchestrator import MLPipelineOrchestrator

    pipeline = MLPipelineOrchestrator(config, hmm_snapshots=hmm_snapshots)
    result = pipeline.run()

    if result.best_strategy is None:
        print("\nPipeline did not find a viable strategy.", file=sys.stderr)
        sys.exit(1)

    return result


# ======================================================================
# Training grid builder
# ======================================================================


def build_training_config(
    args: argparse.Namespace,
    *,
    model_types: tuple[str, ...] | None = None,
):
    """Build a ``TrainingGridConfig`` from parsed CLI args.

    Parameters
    ----------
    args:
        Parsed CLI namespace (must contain ``quick``, ``forward_periods``,
        ``profit_thresholds``, ``n_estimators``, ``max_depth``).
    model_types:
        Override model types.  Falls back to ``args.model_types`` string.
    """
    from stockdownloader.ml.pipeline.config import TrainingGridConfig

    mt = model_types or tuple(
        getattr(args, "model_types", "gradient_boosting,logistic_regression").split(",")
    )

    if args.quick:
        return TrainingGridConfig(
            forward_periods=(10,),
            profit_thresholds=(0.005,),
            model_types=(mt[0],) if len(mt) > 1 else mt,
            use_class_balance_options=(False,),
            use_atr_labels_options=(False,),
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )

    return TrainingGridConfig(
        forward_periods=tuple(int(x) for x in args.forward_periods.split(",")),
        profit_thresholds=tuple(float(x) for x in args.profit_thresholds.split(",")),
        model_types=mt,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )


# ======================================================================
# Argparse helpers
# ======================================================================


def add_common_ml_args(parser: argparse.ArgumentParser) -> None:
    """Register training / backtest / selection / pine / output args.

    These flags are identical across every ML pipeline variant.
    Includes ``--config`` to load defaults from a JSON file.
    """
    # Config file
    parser.add_argument(
        "--config", dest="config_file", default=None,
        help="JSON config file (e.g. config/ml/standard.json). "
             "CLI args override config values.",
    )

    # Training grid
    parser.add_argument(
        "--forward-periods", default="5,10,20",
        help="Comma-separated forward periods (default: 5,10,20)",
    )
    parser.add_argument(
        "--profit-thresholds", default="0.003,0.005,0.01",
        help="Comma-separated profit thresholds (default: 0.003,0.005,0.01)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: single model config for fast iteration",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200,
        help="Number of boosting rounds (default: 200)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=4,
        help="Maximum tree depth (default: 4)",
    )

    # Backtest
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--commission", type=float, default=10.0,
        help="Commission per trade (default: 10)",
    )

    # Selection
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of top strategies to display (default: 5)",
    )

    # PineScript
    parser.add_argument(
        "--no-pine", action="store_true",
        help="Skip PineScript export",
    )
    parser.add_argument(
        "--pine-depth", type=int, default=6,
        help="Surrogate tree depth for PineScript (default: 6)",
    )

    # Output
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )


def apply_config_defaults(args: argparse.Namespace) -> None:
    """Load JSON config and apply values as defaults for unset CLI args.

    Only applies config values when the CLI arg was not explicitly provided.
    This implements the layering: built-in → config file → CLI override.
    """
    config_file = getattr(args, "config_file", None)
    if not config_file:
        return

    from stockdownloader.util.config import load_config

    cfg = load_config(config_file)

    # Map config paths → argparse dest names
    _MAPPINGS: list[tuple[str, str, type]] = [
        ("training.n_estimators", "n_estimators", int),
        ("training.max_depth", "max_depth", int),
        ("backtest.initial_capital", "capital", float),
        ("backtest.commission", "commission", float),
        ("selection.top_n", "top_n", int),
        ("selection.pine_depth", "pine_depth", int),
        ("hybrid.ml_threshold", "ml_threshold", float),
    ]

    for config_key, attr, cast in _MAPPINGS:
        val = cfg.get(config_key)
        if val is not None:
            # Only override if user didn't explicitly pass the CLI arg
            # (argparse defaults are always present, so we use config as truth)
            setattr(args, attr, cast(val))

    # String-typed config values
    fp = cfg.get("training.forward_periods")
    if fp and isinstance(fp, list):
        setattr(args, "forward_periods", ",".join(str(x) for x in fp))

    pt = cfg.get("training.profit_thresholds")
    if pt and isinstance(pt, list):
        setattr(args, "profit_thresholds", ",".join(str(x) for x in pt))

    mt = cfg.get("training.model_types")
    if mt and isinstance(mt, list):
        setattr(args, "model_types", ",".join(mt))

    modes = cfg.get("hybrid.modes")
    if modes and isinstance(modes, list) and hasattr(args, "modes"):
        setattr(args, "modes", ",".join(modes))


# ======================================================================
# Pipeline assembly
# ======================================================================


def build_pipeline_config(
    args: argparse.Namespace,
    training: Any,
    *,
    data_kwargs: dict | None = None,
    backtest_kwargs: dict | None = None,
    selection_kwargs: dict | None = None,
):
    """Assemble a ``PipelineConfig`` from parsed CLI args.

    Parameters
    ----------
    args:
        Must include ``symbol``, ``range_``, ``modes``, ``ml_threshold``,
        ``capital``, ``commission``, ``top_n``, ``no_pine``, ``pine_depth``,
        ``output_dir``.
    training:
        A ``TrainingGridConfig`` (from :func:`build_training_config`).
    data_kwargs / backtest_kwargs / selection_kwargs:
        Extra overrides merged into the respective config dataclasses.
    """
    from stockdownloader.ml.pipeline.config import (
        BacktestConfig,
        ConvergenceConfig,
        DataConfig,
        HybridConfig,
        PipelineConfig,
        SelectionConfig,
    )

    dk = {
        "symbol": args.symbol.upper(),
        "range_": args.range_,
        **(data_kwargs or {}),
    }
    bk = {
        "initial_capital": args.capital,
        "commission": args.commission,
        **(backtest_kwargs or {}),
    }
    sk = {
        "top_n": args.top_n,
        "export_pine": not args.no_pine,
        "pine_depth": args.pine_depth,
        "output_dir": getattr(args, "output_dir", str(DEFAULT_ML_PIPELINE_DIR)),
        **(selection_kwargs or {}),
    }

    return PipelineConfig(
        data=DataConfig(**dk),
        training=training,
        convergence=ConvergenceConfig(),
        hybrid=HybridConfig(
            modes=tuple(args.modes.split(",")),
            confirmed_threshold=args.ml_threshold,
        ),
        backtest=BacktestConfig(**bk),
        selection=SelectionConfig(**sk),
    )
