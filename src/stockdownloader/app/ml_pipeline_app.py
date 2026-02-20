"""Standard end-to-end ML trading pipeline.

Runs the 6-stage pipeline (data → train → converge → hybrid → backtest
→ select/export) on daily price data for any symbol.

Usage::

    ml-pipeline SPY --quick
    ml-pipeline AAPL --range 10y --forward-periods 5,10
    ml-pipeline GME --model-types gradient_boosting --no-pine
"""

from __future__ import annotations

import argparse

from stockdownloader.app._ml_helpers import (
    add_common_ml_args,
    apply_config_defaults,
    build_pipeline_config,
    build_training_config,
    init_ml_env,
    run_ml_pipeline,
)
from stockdownloader.util.constants import DEFAULT_ML_PIPELINE_DIR


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``ml-pipeline``."""
    parser = argparse.ArgumentParser(
        prog="ml-pipeline",
        description="End-to-end ML-driven trading pipeline.",
    )
    parser.add_argument(
        "symbol", nargs="?", default="SPY",
        help="Ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--range", dest="range_", default="10y",
        help="Data range (default: 10y)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable data caching",
    )
    parser.add_argument(
        "--model-types",
        default="gradient_boosting,logistic_regression",
        help="Model types (default: gradient_boosting,logistic_regression)",
    )
    parser.add_argument(
        "--ml-threshold", type=float, default=0.6,
        help="ML confirmation threshold (default: 0.6)",
    )
    parser.add_argument(
        "--modes", default="confirmed,weighted,override",
        help="Hybrid modes (default: confirmed,weighted,override)",
    )
    parser.add_argument(
        "--pine-top-features", type=int, default=15,
        help="Number of features in PineScript (default: 15)",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_ML_PIPELINE_DIR),
        help="Output directory (default: output/ml_pipeline)",
    )
    add_common_ml_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    """``ml-pipeline`` — standard end-to-end ML pipeline."""
    args = _build_parser().parse_args(argv)
    apply_config_defaults(args)
    init_ml_env(args.verbose)

    training = build_training_config(args)
    config = build_pipeline_config(
        args,
        training,
        data_kwargs={"use_cache": not args.no_cache},
        selection_kwargs={"pine_top_features": args.pine_top_features},
    )

    run_ml_pipeline(config)


if __name__ == "__main__":
    main()
