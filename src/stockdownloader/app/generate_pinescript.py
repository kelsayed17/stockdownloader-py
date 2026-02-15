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

from stockdownloader.util.pinescript_composites import COMPOSITE_STRATEGY_CATALOG
from stockdownloader.util.pinescript_generator import PineScriptGenerator
from stockdownloader.util.pinescript_strategies import STRATEGY_CATALOG


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
    args = parser.parse_args()

    gen = PineScriptGenerator()

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
                       else Path("output/pinescript"))
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
                   else Path("output/pinescript"))
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


if __name__ == "__main__":
    main()
