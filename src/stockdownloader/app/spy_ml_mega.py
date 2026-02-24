"""SPY ML Mega pipeline — exhaustive algorithm search.

Trains all available ML algorithms (12 model types, 18 configs with
class balance variants), selects the best ensemble via diversity-aware
selection, optionally compares soft-voting vs. stacking, then exports
to PineScript through the deep surrogate.

Stages::

    [1/8] Download SPY daily data (10y)
    [2/8] Build feature dataset
    [3/8] Train full model grid (18 configs; 6 for --quick)
    [4/8] Rank & report leaderboard
    [5/8] Build diverse ensemble (diversity-aware selection)
    [6/8] Stacking comparison (optional)
    [7/8] Train deep surrogate
    [8/8] Export PineScript + tournament backtest

Usage::

    spy-ml-mega                        # Full pipeline (18 configs)
    spy-ml-mega --quick                # Quick mode (6 configs)
    spy-ml-mega --ensemble-method both # Compare soft-vote vs stacking
    spy-ml-mega --diversity-weight 0.6 # Heavier diversity weighting
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from stockdownloader.app.ml_helpers import add_common_ml_args, init_ml_env
from stockdownloader.core.config import DEFAULT_ML_PIPELINE_DIR


# ======================================================================
# Model grid definitions
# ======================================================================

_FULL_GRID: list[tuple[str, bool]] = [
    # (model_type, use_class_balance)
    # -- sklearn tree ensembles --
    ("gradient_boosting", False),
    ("gradient_boosting", True),
    ("random_forest", False),
    ("random_forest", True),
    ("extra_trees", False),
    ("extra_trees", True),
    ("hist_gradient_boosting", False),
    ("hist_gradient_boosting", True),
    # -- external boosters --
    ("xgboost", False),
    ("xgboost", True),
    ("lightgbm", False),
    ("lightgbm", True),
    ("catboost", False),
    ("catboost", True),
    # -- linear / non-tree --
    ("logistic_regression", False),  # always balanced internally
    ("svm", False),
    ("mlp", False),
    ("knn", False),
]

_QUICK_GRID: list[tuple[str, bool]] = [
    ("xgboost", False),
    ("lightgbm", False),
    ("catboost", False),
    ("random_forest", False),
    ("logistic_regression", False),
    ("svm", False),
]


# ======================================================================
# Argparse
# ======================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``spy-ml-mega``."""
    parser = argparse.ArgumentParser(
        prog="spy-ml-mega",
        description=(
            "SPY ML Mega pipeline: exhaustive algorithm search -> "
            "diversity-aware ensemble -> deep surrogate -> PineScript."
        ),
    )

    add_common_ml_args(parser)

    # Ensemble args
    parser.add_argument(
        "--top-models", type=int, default=5,
        help="Number of top models for ensemble (default: 5)",
    )
    parser.add_argument(
        "--ensemble-method",
        choices=["soft_vote", "stacking", "both"],
        default="both",
        help="Ensemble method (default: both — compares and picks best)",
    )
    parser.add_argument(
        "--diversity-weight", type=float, default=0.4,
        help="Weight for diversity vs accuracy in selection (default: 0.4)",
    )

    # Surrogate args
    parser.add_argument(
        "--depth", type=int, default=10,
        help="Max depth for deep surrogate tree (default: 10)",
    )
    parser.add_argument(
        "--top-features", type=int, default=25,
        help="Number of top features for surrogate (default: 25)",
    )
    parser.add_argument(
        "--min-leaf", type=int, default=10,
        help="Minimum samples per leaf in surrogate (default: 10)",
    )

    # Trading args
    parser.add_argument(
        "--buy-thresh", type=float, default=0.55,
        help="Buy threshold (default: 0.55)",
    )
    parser.add_argument(
        "--sell-thresh", type=float, default=0.45,
        help="Sell threshold (default: 0.45)",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="Initial capital for backtest (default: 100000)",
    )
    parser.add_argument(
        "--min-r2", type=float, default=0.70,
        help="Minimum R-squared for surrogate (default: 0.70)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Output directory "
            f"(default: {DEFAULT_ML_PIPELINE_DIR / 'spy_mega'})"
        ),
    )
    parser.add_argument(
        "--no-tournament", action="store_true",
        help="Skip tournament backtest",
    )

    # Walk-forward backtest
    parser.add_argument(
        "--walk-forward-windows", type=int, default=5,
        help="Number of expanding windows for walk-forward backtest (default: 5)",
    )
    parser.add_argument(
        "--no-walk-forward", action="store_true",
        help="Use in-sample backtest instead of walk-forward (faster but biased)",
    )

    return parser


# ======================================================================
# Main pipeline
# ======================================================================


def main(argv: list[str] | None = None) -> None:
    """Run the SPY ML Mega pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_ML_PIPELINE_DIR / "spy_mega"
    )
    init_ml_env(args.verbose)

    import numpy as np

    from stockdownloader.data.market.yahoo_data_client import YahooDataClient
    from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig
    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter
    from stockdownloader.ml.ensemble import EnsembleBuilder, EnsemblePredictor
    from stockdownloader.ml.feature_extractor import FeatureExtractor
    from stockdownloader.ml.trainer import MLModelConfig, MLTrainer
    from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
        spy_ml_ensemble_strategy,
    )
    from stockdownloader.pinescript.generator import PineScriptGenerator
    from stockdownloader.app.spy_ml_ensemble import (
        _run_tournament_insample,
        _run_tournament_walk_forward,
    )

    t0 = time.time()
    grid = _QUICK_GRID if args.quick else _FULL_GRID

    print("=" * 60)
    print("SPY ML MEGA PIPELINE")
    print("=" * 60)
    print(f"  Mode:             {'quick' if args.quick else 'full'}")
    print(f"  Model configs:    {len(grid)}")
    print(f"  Top models:       {args.top_models}")
    print(f"  Ensemble method:  {args.ensemble_method}")
    print(f"  Diversity weight: {args.diversity_weight}")
    print(f"  Surrogate:        depth={args.depth}, "
          f"features={args.top_features}, min_leaf={args.min_leaf}")
    print(f"  Thresholds:       buy={args.buy_thresh}, sell={args.sell_thresh}")
    print(f"  Capital:          ${args.initial_capital:,.0f}")
    if not args.no_tournament:
        bt_mode = "in-sample" if args.no_walk_forward else (
            f"walk-forward ({args.walk_forward_windows} windows)"
        )
        print(f"  Backtest:         {bt_mode}")
    print(f"  Output:           {output_dir}")
    print()

    # [1/8] Download
    print("[1/8] Downloading SPY daily data (10y)...")
    t1 = time.time()
    client = YahooDataClient()
    daily_data = client.fetch_price_data("SPY", range_="10y")
    if len(daily_data) < 500:
        print(f"\nERROR: Insufficient data — got {len(daily_data)} bars.",
              file=sys.stderr)
        sys.exit(1)
    print(f"  Downloaded {len(daily_data)} bars "
          f"({daily_data[0].date} to {daily_data[-1].date})")
    print(f"  [{time.time() - t1:.1f}s]")

    # [2/8] Features
    print("\n[2/8] Building feature dataset...")
    t2 = time.time()
    label_config = LabelConfig(
        forward_period=int(
            args.forward_periods.split(",")[0]
        ) if hasattr(args, "forward_periods") else 10,
        profit_threshold=float(
            args.profit_thresholds.split(",")[0]
        ) if hasattr(args, "profit_thresholds") else 0.005,
    )
    extractor = FeatureExtractor()
    builder = DatasetBuilder(extractor, label_config)
    dataset = builder.build(daily_data)
    print(f"  Samples:  {dataset.X.shape[0]}")
    print(f"  Features: {dataset.X.shape[1]}")
    print(f"  Class 1:  {int(np.sum(dataset.y == 1))} "
          f"({np.mean(dataset.y == 1):.1%})")
    print(f"  Class 0:  {int(np.sum(dataset.y == 0))} "
          f"({np.mean(dataset.y == 0):.1%})")
    print(f"  [{time.time() - t2:.1f}s]")

    # [3/8] Train all models
    print(f"\n[3/8] Training model grid ({len(grid)} configs)...")
    t3 = time.time()
    results = []
    for idx, (model_type, use_balance) in enumerate(grid, 1):
        label = f"{model_type}" + ("+balanced" if use_balance else "")
        print(f"  [{idx}/{len(grid)}] {label}...", end=" ", flush=True)
        try:
            config = MLModelConfig(
                model_type=model_type,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                use_class_balance=use_balance,
            )
            trainer = MLTrainer(config)
            result = trainer.train(dataset)
            results.append(result)
            print(f"acc={result.oos_accuracy:.4f}  auc={result.oos_roc_auc:.4f}")
        except (ImportError, Exception) as exc:
            print(f"SKIP ({exc})")
    print(f"  [{time.time() - t3:.1f}s]")

    if len(results) < 2:
        print("\nERROR: Need at least 2 successful models.", file=sys.stderr)
        sys.exit(1)

    # [4/8] Rank & report
    print(f"\n[4/8] Model leaderboard ({len(results)} models)...")
    ranked = sorted(results, key=lambda r: r.oos_accuracy, reverse=True)
    for i, r in enumerate(ranked, 1):
        bal = "+bal" if r.config.use_class_balance else ""
        print(f"  #{i:2d}  {r.config.model_type}{bal:5s}  "
              f"acc={r.oos_accuracy:.4f}  auc={r.oos_roc_auc:.4f}")

    # [5/8] Build ensemble (diversity-aware)
    print(f"\n[5/8] Building diverse ensemble (top {args.top_models})...")
    t5 = time.time()
    ensemble_builder = EnsembleBuilder(results)
    ensemble = ensemble_builder.select_diverse(
        n=args.top_models,
        diversity_weight=args.diversity_weight,
    )
    probs = ensemble.predict_proba(dataset.X)
    importances = ensemble.averaged_feature_importances()
    print(f"  Ensemble models: {ensemble.n_models}")
    print(f"  Prob range: [{float(np.min(probs)):.4f}, "
          f"{float(np.max(probs)):.4f}]")
    selected_types = [r.config.model_type for r in ensemble.results]
    print(f"  Selected: {selected_types}")
    print(f"  [{time.time() - t5:.1f}s]")

    # [6/8] Optional stacking comparison
    if args.ensemble_method in ("stacking", "both"):
        print("\n[6/8] Stacking ensemble comparison...")
        t6 = time.time()
        stacking = EnsemblePredictor(
            list(ensemble.results),
            ensemble_method="stacking",
        )
        stacking.fit_stacking(dataset.X, dataset.y)
        stacking_probs = stacking.predict_proba(dataset.X)
        print(f"  Stacking prob range: [{float(np.min(stacking_probs)):.4f}, "
              f"{float(np.max(stacking_probs)):.4f}]")

        if args.ensemble_method == "both":
            sv_spread = float(np.std(probs))
            st_spread = float(np.std(stacking_probs))
            if st_spread > sv_spread:
                print(f"  Stacking wins (spread {st_spread:.4f} > {sv_spread:.4f})")
                probs = stacking_probs
            else:
                print(f"  Soft-vote wins (spread {sv_spread:.4f} >= {st_spread:.4f})")
        elif args.ensemble_method == "stacking":
            probs = stacking_probs
        print(f"  [{time.time() - t6:.1f}s]")
    else:
        print("\n[6/8] Stacking skipped (--ensemble-method=soft_vote).")

    # [7/8] Deep surrogate
    print(f"\n[7/8] Training deep surrogate (depth={args.depth})...")
    t7 = time.time()
    exporter = DeepSurrogateExporter(
        max_depth=args.depth,
        min_samples_leaf=args.min_leaf,
        top_n=args.top_features,
    )
    exporter.train_surrogate(dataset, importances, ensemble_probs=probs)
    r_squared = exporter.fidelity_r_squared(dataset.X, probs)
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  Features used: {len(exporter.feature_names)}")
    if r_squared < args.min_r2:
        print(f"\n  WARNING: R² ({r_squared:.4f}) < {args.min_r2:.2f}",
              file=sys.stderr)
    print(f"  [{time.time() - t7:.1f}s]")

    # [8/8] PineScript export
    if args.no_pine:
        print("\n[8/8] PineScript export skipped.")
    else:
        print("\n[8/8] Exporting PineScript strategy...")
        t8 = time.time()
        strategy_def = spy_ml_ensemble_strategy(
            exporter,
            buy_threshold=args.buy_thresh,
            sell_threshold=args.sell_thresh,
        )
        pine_code = PineScriptGenerator().generate(strategy_def)
        output_dir.mkdir(parents=True, exist_ok=True)
        pine_path = output_dir / "spy_ml_mega.pine"
        pine_path.write_text(pine_code, encoding="utf-8")
        print(f"  PineScript: {pine_path} ({len(pine_code.splitlines())} lines)")
        print(f"  [{time.time() - t8:.1f}s]")

    # Tournament
    if not args.no_tournament:
        if args.no_walk_forward:
            # In-sample backtest (fast but has look-ahead bias)
            _run_tournament_insample(
                dataset.X, dataset.dates, daily_data, exporter,
                args.buy_thresh, args.sell_thresh,
                initial_capital=args.initial_capital,
            )
        else:
            # Walk-forward backtest (proper OOS — default)
            _run_tournament_walk_forward(
                dataset, daily_data, grid,
                args.n_estimators, args.max_depth,
                args.buy_thresh, args.sell_thresh,
                initial_capital=args.initial_capital,
                top_models=args.top_models,
                diversity_weight=args.diversity_weight,
                use_select_diverse=True,
                surrogate_depth=args.depth,
                surrogate_min_leaf=args.min_leaf,
                surrogate_top_features=args.top_features,
                n_windows=args.walk_forward_windows,
            )

    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("MEGA PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time:      {elapsed:.1f}s")
    print(f"  Models trained:  {len(results)}")
    print(f"  Ensemble size:   {ensemble.n_models}")
    print(f"  Surrogate R²:    {r_squared:.4f}")
    if not args.no_pine:
        print(f"  PineScript:      {output_dir / 'spy_ml_mega.pine'}")
    print()


if __name__ == "__main__":
    main()
