"""CLI entry point for training ML models on historical stock data.

Fetches data from Yahoo Finance, extracts features, trains a
GradientBoosting or LogisticRegression classifier, prints metrics
and top feature importances, and persists the model for later use
with ``signal-monitor --ml-model``.

Usage::

    # Train on SPY with defaults
    ml-train SPY

    # Custom forward period and model type
    ml-train AAPL --forward-period 10 --model-type logistic_regression

    # Custom output directory
    ml-train SPY --output-dir output/models/spy
"""

from __future__ import annotations

import argparse
import logging
import sys

from stockdownloader.util.constants import DEFAULT_MODELS_DIR

try:
    from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig
    from stockdownloader.ml.feature_extractor import FeatureExtractor
    from stockdownloader.ml.model_store import ModelMetadata, ModelStore
    from stockdownloader.ml.trainer import MLModelConfig, MLTrainer

    _HAS_ML = True
except ImportError:
    _HAS_ML = False

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-train",
        description="Train an ML classifier on historical stock data.",
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="SPY",
        help="Ticker symbol to train on (default: SPY)",
    )
    parser.add_argument(
        "--forward-period",
        type=int,
        default=10,
        help="Number of forward bars for label construction (default: 10)",
    )
    parser.add_argument(
        "--profit-threshold",
        type=float,
        default=0.005,
        help="Minimum return for positive label (default: 0.005)",
    )
    parser.add_argument(
        "--model-type",
        choices=["gradient_boosting", "logistic_regression"],
        default="gradient_boosting",
        help="Model type (default: gradient_boosting)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting rounds (default: 200)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR / "spy"),
        help="Directory to save trained model (default: output/models/spy)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Yahoo Finance look-back period (default: 5y)",
    )
    # Phase 2: Training improvements
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (grid search with temporal CV)",
    )
    parser.add_argument(
        "--feature-select",
        action="store_true",
        dest="feature_select",
        help="Enable feature selection (drop low-importance features)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use class-balanced sample weights",
    )
    parser.add_argument(
        "--use-atr-labels",
        action="store_true",
        dest="use_atr_labels",
        help="Use ATR-relative labeling threshold",
    )
    # Phase 3: PineScript export
    parser.add_argument(
        "--export-pine",
        action="store_true",
        dest="export_pine",
        help="Generate Pine Script indicator after training",
    )
    parser.add_argument(
        "--pine-depth",
        type=int,
        default=6,
        help="Surrogate decision tree depth for Pine export (default: 6)",
    )
    parser.add_argument(
        "--pine-top-features",
        type=int,
        default=15,
        dest="pine_top_features",
        help="Number of features in Pine Script surrogate (default: 15)",
    )
    return parser


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if not _HAS_ML:
        print(
            "ERROR: ML dependencies not installed.\n"
            "Install ML extras: pip install -e '.[ml]'",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Fetch data
    from stockdownloader.app.app_helpers import fetch_daily_data

    symbol = args.symbol.upper()
    data = fetch_daily_data(symbol, period=args.period)
    if not data:
        print(f"ERROR: Could not fetch data for {symbol}", file=sys.stderr)
        sys.exit(1)

    print(f"Using {len(data)} daily bars ({data[0].date} to {data[-1].date})")
    print()

    # 2. Build dataset
    label_cfg = LabelConfig(
        forward_period=args.forward_period,
        profit_threshold=args.profit_threshold,
        use_atr_threshold=getattr(args, "use_atr_labels", False),
    )
    extractor = FeatureExtractor()
    builder = DatasetBuilder(extractor, label_config=label_cfg)

    try:
        dataset = builder.build(data)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset: {dataset.X.shape[0]} samples, {dataset.X.shape[1]} features")
    unique, counts = __import__("numpy").unique(dataset.y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        pct = cnt / len(dataset.y) * 100
        print(f"  Class {int(cls)}: {cnt} ({pct:.1f}%)")
    print()

    # 3. Train model
    model_cfg = MLModelConfig(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        use_class_balance=getattr(args, "balanced", False),
        use_feature_selection=getattr(args, "feature_select", False),
    )
    trainer = MLTrainer(model_cfg)

    if getattr(args, "tune", False):
        print(f"Training {args.model_type} model with hyperparameter tuning...")
        result = trainer.train_with_tuning(dataset)
    else:
        print(f"Training {args.model_type} model...")
        result = trainer.train(dataset)

    # 4. Print metrics
    print()
    print("=" * 50)
    print(f"  TRAINING RESULTS: {symbol}")
    print("=" * 50)
    print(f"  OOS Accuracy:    {result.oos_accuracy:.3f}")
    print(f"  OOS ROC-AUC:     {result.oos_roc_auc:.3f}")
    print(f"  Train samples:   {result.train_size}")
    print(f"  Test samples:    {result.test_size}")

    if result.cv_scores:
        avg_cv = sum(result.cv_scores) / len(result.cv_scores)
        print(f"  CV scores:       {[f'{s:.3f}' for s in result.cv_scores]}")
        print(f"  CV mean:         {avg_cv:.3f}")

    if result.ensemble_accuracy is not None:
        print(f"  Ensemble Acc:    {result.ensemble_accuracy:.3f}")
        print(f"  Ensemble AUC:    {result.ensemble_roc_auc:.3f}")

    if result.optimal_threshold != 0.5:
        print(f"  Optimal thresh:  {result.optimal_threshold:.2f}")

    if result.selected_features is not None:
        print(f"  Selected feats:  {len(result.selected_features)}")

    # Top 15 feature importances
    print()
    print("  Top 15 Feature Importances:")
    sorted_feats = sorted(
        result.feature_importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, imp in sorted_feats[:15]:
        bar = "█" * int(imp * 100)
        print(f"    {name:<30s} {imp:.4f} {bar}")

    print()
    print("=" * 50)

    # 5. Persist model
    from datetime import datetime, timezone

    metadata = ModelMetadata(
        symbol=symbol,
        model_type=args.model_type,
        feature_names=dataset.feature_names,
        training_date_range=(dataset.dates[0], dataset.dates[-1]),
        oos_accuracy=result.oos_accuracy,
        oos_roc_auc=result.oos_roc_auc,
        class_distribution={str(k): v for k, v in result.class_distribution.items()},
        trained_at=datetime.now(tz=timezone.utc).isoformat(),
        config={
            "model_type": args.model_type,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "forward_period": args.forward_period,
            "profit_threshold": args.profit_threshold,
        },
    )

    store = ModelStore(base_dir=args.output_dir)
    model_path = store.save(result.model, metadata)
    print(f"\nModel saved to: {model_path}")
    print(f"Use with: signal-monitor {symbol} --ml-model {model_path}")

    # 6. Export Pine Script (optional)
    if getattr(args, "export_pine", False):
        _export_pinescript(
            symbol, dataset, result,
            pine_depth=args.pine_depth,
            pine_top_features=args.pine_top_features,
            output_dir=args.output_dir,
        )


def _export_pinescript(
    symbol: str,
    dataset: object,
    result: object,
    pine_depth: int = 6,
    pine_top_features: int = 15,
    output_dir: str = "output/models/spy",
) -> None:
    """Generate and save Pine Script indicator from training results."""
    from pathlib import Path

    from stockdownloader.util.pinescript_generator import PineScriptGenerator
    from stockdownloader.util.pinescript_ml_strategy import (
        DecisionTreeExporter,
        ml_signal_strategy,
    )

    print("\nGenerating Pine Script indicator...")

    exporter = DecisionTreeExporter(
        max_depth=pine_depth,
        min_samples_leaf=20,
    )
    exporter.train_surrogate(
        dataset,  # type: ignore[arg-type]
        result.feature_importances,  # type: ignore[attr-defined]
        top_n=pine_top_features,
    )

    strategy = ml_signal_strategy(symbol, exporter)
    gen = PineScriptGenerator()
    pine_code = gen.generate(strategy)

    pine_dir = Path(output_dir).parent / "pinescript"
    pine_dir.mkdir(parents=True, exist_ok=True)
    pine_path = pine_dir / f"{symbol.lower()}_ml_signal.pine"
    pine_path.write_text(pine_code)

    print(f"Pine Script saved to: {pine_path}")
    print(f"Open in TradingView: Indicators → My Scripts → {symbol} ML Signal")


if __name__ == "__main__":
    main()
