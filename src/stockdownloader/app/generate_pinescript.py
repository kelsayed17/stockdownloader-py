"""Generate Pine Script v6 indicators for any strategy.

Usage::

    # List all available strategies
    python -m stockdownloader.app.generate_pinescript --list

    # Generate for a specific strategy
    python -m stockdownloader.app.generate_pinescript macd_obv

    # Generate with custom params
    python -m stockdownloader.app.generate_pinescript sma_crossover \\
        --param short_period=20 --param long_period=21

    # Generate all strategies at once
    python -m stockdownloader.app.generate_pinescript --all

    # Output to file
    python -m stockdownloader.app.generate_pinescript macd_obv -o output/macd_obv.pine

    # List composite strategies
    python -m stockdownloader.app.generate_pinescript --composite --list

    # Generate a composite strategy
    python -m stockdownloader.app.generate_pinescript --composite vwap_composite

    # Generate all composites
    python -m stockdownloader.app.generate_pinescript --composite --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stockdownloader.util.config_loader import DEFAULT_PINESCRIPT_DIR
from stockdownloader.app.pinescript_catalog.catalogs import (
    COMPOSITE_STRATEGY_CATALOG,
    STRATEGY_CATALOG,
)
from stockdownloader.util.pinescript import PineScriptGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Pine Script v6 TradingView indicators"
    )
    parser.add_argument(
        "strategy",
        nargs="?",
        help="Strategy name (use --list to see available)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available strategies",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate Pine Script for all strategies",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Work with composite strategies (multi-mode, toggleable)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for --all mode",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override parameter: --param key=value",
    )
    # ML signal mode
    parser.add_argument(
        "--ml-signal",
        dest="ml_signal",
        metavar="SYMBOL",
        help="Generate ML signal Pine Script for SYMBOL (fetches data & trains)",
    )
    parser.add_argument(
        "--ml-model",
        dest="ml_model",
        metavar="PATH",
        help="Path to trained model (for --ml-signal mode)",
    )
    parser.add_argument(
        "--pine-depth",
        type=int,
        default=6,
        help="Surrogate tree depth for --ml-signal (default: 6)",
    )
    parser.add_argument(
        "--pine-top-features",
        type=int,
        default=15,
        dest="pine_top_features",
        help="Number of features for --ml-signal (default: 15)",
    )
    args = parser.parse_args()

    gen = PineScriptGenerator()

    # ---- ML signal mode ----
    if args.ml_signal:
        _handle_ml_signal(args, gen)
        return

    # ---- Composite mode ----
    if args.composite:
        if args.list:
            print("Available composite strategies:")
            print()
            for name, factory in sorted(COMPOSITE_STRATEGY_CATALOG.items()):
                defn = factory()
                modes = ", ".join(m.short_name.upper() for m in defn.modes)
                print(f"  {name:25s} {defn.name}")
                print(f"  {'':25s} Modes: {modes}")
            print()
            print("Usage: python -m stockdownloader.app.generate_pinescript "
                  "--composite <name>")
            return

        if args.all:
            out_dir = (Path(args.output_dir) if args.output_dir
                       else DEFAULT_PINESCRIPT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)

            for name, factory in sorted(COMPOSITE_STRATEGY_CATALOG.items()):
                defn = factory()
                pine = gen.generate_composite(defn)
                out_path = out_dir / f"{name}.pine"
                out_path.write_text(pine)
                print(f"  {out_path}")

            print(f"\nGenerated {len(COMPOSITE_STRATEGY_CATALOG)} composite "
                  f"Pine Script files in {out_dir}/")
            return

        if not args.strategy:
            parser.error(
                "Specify a composite name or use --composite --list / --all"
            )

        if args.strategy not in COMPOSITE_STRATEGY_CATALOG:
            print(f"Unknown composite: {args.strategy}", file=sys.stderr)
            print(
                f"Available: "
                f"{', '.join(sorted(COMPOSITE_STRATEGY_CATALOG.keys()))}",
                file=sys.stderr,
            )
            sys.exit(1)

        factory = COMPOSITE_STRATEGY_CATALOG[args.strategy]
        defn = factory()
        pine = gen.generate_composite(defn)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(pine)
            print(f"Written to {out_path}")
        else:
            print(pine)
        return

    # ---- Standard mode ----
    if args.list:
        print("Available strategies:")
        print()
        for name, factory in sorted(STRATEGY_CATALOG.items()):
            defn = factory()
            print(f"  {name:25s} {defn.name}")
        print()
        print("Available composite strategies (use --composite):")
        print()
        for name, factory in sorted(COMPOSITE_STRATEGY_CATALOG.items()):
            defn = factory()
            print(f"  {name:25s} {defn.name}")
        print()
        print("Usage: python -m stockdownloader.app.generate_pinescript <name>")
        return

    if args.all:
        out_dir = (Path(args.output_dir) if args.output_dir
                   else DEFAULT_PINESCRIPT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, factory in sorted(STRATEGY_CATALOG.items()):
            defn = factory()
            pine = gen.generate(defn)
            out_path = out_dir / f"{name}.pine"
            out_path.write_text(pine)
            print(f"  {out_path}")

        print(f"\nGenerated {len(STRATEGY_CATALOG)} Pine Script files "
              f"in {out_dir}/")
        return

    if not args.strategy:
        parser.error("Specify a strategy name or use --list / --all")

    if args.strategy not in STRATEGY_CATALOG:
        print(f"Unknown strategy: {args.strategy}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(STRATEGY_CATALOG.keys()))}",
              file=sys.stderr)
        sys.exit(1)

    # Parse parameter overrides
    params: dict[str, int | float | str] = {}
    for p in args.param:
        if "=" not in p:
            print(f"Invalid param format: {p} (expected key=value)",
                  file=sys.stderr)
            sys.exit(1)
        key, val = p.split("=", 1)
        # Try to parse as int, then float, then string
        try:
            params[key] = int(val)
        except ValueError:
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val

    factory = STRATEGY_CATALOG[args.strategy]
    defn = factory(**params)

    pine = gen.generate(defn)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(pine)
        print(f"Written to {out_path}")
    else:
        print(pine)


def _handle_ml_signal(args: argparse.Namespace, gen: PineScriptGenerator) -> None:
    """Generate ML signal Pine Script from trained model or fresh training."""
    try:
        from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig
        from stockdownloader.ml.feature_extractor import FeatureExtractor
        from stockdownloader.ml.trainer import MLModelConfig, MLTrainer
        from stockdownloader.util.pinescript_ml_strategy import (
            DecisionTreeExporter,
            ml_signal_strategy,
        )
    except ImportError:
        print(
            "ERROR: ML dependencies not installed.\n"
            "Install ML extras: pip install -e '.[ml]'",
            file=sys.stderr,
        )
        sys.exit(1)

    symbol = args.ml_signal.upper()

    # Fetch data and build dataset
    from stockdownloader.app.app_helpers import fetch_daily_data

    print(f"Fetching {symbol} data...")
    data = fetch_daily_data(symbol, period="5y")
    if not data:
        print(f"ERROR: Could not fetch data for {symbol}", file=sys.stderr)
        sys.exit(1)

    print(f"Building dataset from {len(data)} bars...")
    extractor = FeatureExtractor()
    builder = DatasetBuilder(extractor)
    dataset = builder.build(data)

    # Get feature importances (train quick model or load existing)
    print("Training model for feature importances...")
    cfg = MLModelConfig(n_estimators=100, max_depth=4)
    trainer = MLTrainer(cfg)
    result = trainer.train(dataset)

    print(f"Model accuracy: {result.oos_accuracy:.3f}")

    # Train surrogate and export
    exporter = DecisionTreeExporter(
        max_depth=args.pine_depth,
        min_samples_leaf=20,
    )
    exporter.train_surrogate(
        dataset, result.feature_importances,
        top_n=args.pine_top_features,
    )

    strategy = ml_signal_strategy(symbol, exporter)
    pine_code = gen.generate(strategy)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(pine_code)
        print(f"Pine Script written to {out_path}")
    else:
        out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_PINESCRIPT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{symbol.lower()}_ml_signal.pine"
        out_path.write_text(pine_code)
        print(f"Pine Script written to {out_path}")


if __name__ == "__main__":
    main()
