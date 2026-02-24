"""GME alternative-data ML pipeline.

Extends the standard pipeline with SEC/FINRA alternative data sources
(FTD, short interest, dark pool, ownership, borrow rates).

Usage::

    gme-ml-pipeline                          # Full pipeline, all alt data
    gme-ml-pipeline --quick                  # Single config, fast iteration
    gme-ml-pipeline --no-ftd --no-dark-pool  # Selective alt data
    gme-ml-pipeline --no-alt-data            # Standard 63-feature mode
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from stockdownloader.app.ml_helpers import (
    add_common_ml_args,
    build_pipeline_config,
    build_training_config,
    init_ml_env,
    run_ml_pipeline,
)
from stockdownloader.core.config import AppConfig
from stockdownloader.core.config import DEFAULT_ML_PIPELINE_DIR

logger = logging.getLogger(__name__)


# ======================================================================
# Alt-data feature importance report
# ======================================================================


def _print_alt_feature_importance(result: object) -> None:
    """Print alternative data feature importance breakdown."""
    from stockdownloader.ml.feature_extractor import _ALT_DATA_NAMES

    try:
        training_result = result.training_result  # type: ignore[attr-defined]
        if training_result is None or not training_result.candidates:
            return

        best = training_result.best_by_auc
        if best is None:
            return

        importances = best.training_result.feature_importances
        feature_names = best.dataset.feature_names

        if len(feature_names) <= 63:
            print("\n(No alternative data features in this run)")
            return

        print("\n" + "=" * 60)
        print("ALTERNATIVE DATA FEATURE IMPORTANCE")
        print("=" * 60)

        alt_pairs: list[tuple[str, float]] = []
        for name in _ALT_DATA_NAMES:
            if name in importances:
                alt_pairs.append((name, importances[name]))

        if not alt_pairs:
            print("  No alternative data features found in model.")
            return

        alt_pairs.sort(key=lambda x: x[1], reverse=True)
        total_imp = sum(imp for _, imp in alt_pairs)

        print(f"{'Feature':<35} {'Importance':>12} {'Share':>8}")
        print("-" * 60)
        for name, imp in alt_pairs:
            share = imp / total_imp * 100 if total_imp > 0 else 0.0
            print(f"  {name:<33} {imp:>10.4f}   {share:>5.1f}%")
        print("-" * 60)
        print(f"  {'TOTAL (alt data)':<33} {total_imp:>10.4f}   100.0%")

        total_all = sum(importances.values())
        alt_pct = total_imp / total_all * 100 if total_all > 0 else 0.0
        print(f"\n  Alt data share of total importance: {alt_pct:.1f}%")
        print("=" * 60)

    except (AttributeError, IndexError, TypeError) as exc:
        logger.debug("Could not print alt feature importance: %s", exc)


# ======================================================================
# CLI
# ======================================================================


def main(argv: list[str] | None = None) -> None:
    """``gme-ml-pipeline`` — GME alt-data ML pipeline."""
    parser = argparse.ArgumentParser(
        prog="gme-ml-pipeline",
        description="GME Alternative Data ML Pipeline.",
    )
    # Data
    parser.add_argument("symbol", nargs="?", default="GME", help="Ticker symbol (default: GME)")
    parser.add_argument("--range", dest="range_", default="max", help="Data range (default: max)")
    parser.add_argument("--csv-file", default=None, help="Path to CSV with daily bars")
    parser.add_argument("--no-cache", action="store_true", help="Disable data caching")
    parser.add_argument("--polygon-key", default=None, help="Polygon.io API key")
    parser.add_argument("--use-polygon", action="store_true", help="Use Polygon.io for daily data")
    parser.add_argument("--ftd-start-year", type=int, default=2020, help="FTD start year (default: 2020)")
    # FINRA
    parser.add_argument("--finra-client-id", default=None, help="FINRA API client ID")
    parser.add_argument("--finra-client-secret", default=None, help="FINRA API client secret")
    # Alt data toggles
    parser.add_argument("--no-ftd", action="store_true", help="Disable FTD data")
    parser.add_argument("--no-short-interest", action="store_true", help="Disable short interest")
    parser.add_argument("--no-dark-pool", action="store_true", help="Disable dark pool")
    parser.add_argument("--no-ownership", action="store_true", help="Disable 13F ownership")
    parser.add_argument("--no-borrow-rate", action="store_true", help="Disable borrow rate")
    parser.add_argument("--no-alt-data", action="store_true", help="Disable ALL alternative data")
    # ML / hybrid
    parser.add_argument("--model-types", default="gradient_boosting,logistic_regression", help="Model types")
    parser.add_argument("--ml-threshold", type=float, default=0.6, help="ML threshold (default: 0.6)")
    parser.add_argument("--modes", default="confirmed,weighted,override", help="Hybrid modes")
    parser.add_argument("--pine-top-features", type=int, default=15, help="Features in PineScript (default: 15)")
    parser.add_argument("--output-dir", default=str(DEFAULT_ML_PIPELINE_DIR), help="Output directory")
    add_common_ml_args(parser)
    args = parser.parse_args(argv)
    init_ml_env(args.verbose)

    from stockdownloader.ml.pipeline.config import AltDataConfig

    # Build centralized config — CLI flags override env vars
    app_cfg = AppConfig.from_env(
        polygon_api_key=args.polygon_key or "",
        finra_client_id=args.finra_client_id or "",
        finra_client_secret=args.finra_client_secret or "",
    )

    # Polygon fetch
    if args.use_polygon:
        if not app_cfg.polygon_api_key:
            print("ERROR: Polygon API key required.", file=sys.stderr)
            sys.exit(1)
        print("Fetching daily data from Polygon.io...")
        try:
            from datetime import date as _date
            from stockdownloader.data.market.polygon_client import PolygonDataClient

            client = PolygonDataClient(api_key=app_cfg.polygon_api_key)
            today = _date.today()
            range_map = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "max": 9000}
            days = range_map.get(args.range_, 3650)
            from_date = today.replace(year=today.year - (days // 365))
            client.fetch_intraday_data(
                args.symbol.upper(), from_date=str(from_date),
                to_date=str(today), multiplier=1, timespan="day",
            )
        except Exception as exc:
            print(f"  Polygon fetch failed: {exc}", file=sys.stderr)
            print("  Falling back to Yahoo Finance...")

    # Alt data config
    alt_data = None
    if not args.no_alt_data:
        alt_data = AltDataConfig(
            enable_ftd=not args.no_ftd,
            enable_short_interest=not args.no_short_interest,
            enable_dark_pool=not args.no_dark_pool,
            enable_ownership=not args.no_ownership,
            enable_borrow_rate=not args.no_borrow_rate,
            ftd_start_year=args.ftd_start_year,
            finra_client_id=app_cfg.finra_client_id,
            finra_client_secret=app_cfg.finra_client_secret,
        )

    # Auto-detect CSV
    csv_file = args.csv_file
    if csv_file is None:
        auto = f"data/{args.symbol.upper()}/daily_bars.csv"
        if os.path.exists(auto):
            csv_file = auto
            print(f"  Auto-detected CSV: {csv_file}")

    # Build config
    training = build_training_config(args)
    config = build_pipeline_config(
        args,
        training,
        data_kwargs={
            "use_cache": not args.no_cache,
            "csv_file": csv_file,
            "alt_data": alt_data,
        },
        selection_kwargs={"pine_top_features": args.pine_top_features},
    )

    # Banner
    symbol = args.symbol.upper()
    print("=" * 70)
    print(f"  GME Alternative Data ML Pipeline — {symbol}")
    print("=" * 70)
    print(f"  Range:       {args.range_}")
    if alt_data:
        sources = [
            label for flag, label in [
                (alt_data.enable_ftd, "FTD"),
                (alt_data.enable_short_interest, "Short Interest"),
                (alt_data.enable_dark_pool, "Dark Pool"),
                (alt_data.enable_ownership, "Ownership"),
                (alt_data.enable_borrow_rate, "Borrow Rate"),
            ] if flag
        ]
        print(f"  Alt Data:    {', '.join(sources)}")
        print(f"  Features:    80 (63 base + 17 alt data)")
    else:
        print("  Alt Data:    Disabled")
        print("  Features:    63 (base only)")
    print(f"  Quick Mode:  {'Yes' if args.quick else 'No'}")
    print("=" * 70)
    print()

    start_time = time.time()
    result = run_ml_pipeline(config)
    elapsed = time.time() - start_time

    _print_alt_feature_importance(result)

    best = result.best_strategy
    print(f"\nPipeline completed in {elapsed:.1f}s")
    print(
        f"Best: {best.strategy_name} "
        f"(score={best.composite_score:.1f}, "
        f"P/L=${float(best.result.total_pnl):,.2f}, "
        f"WR={float(best.result.win_rate):.1f}%)"
    )
    if result.pine_path:
        print(f"PineScript: {result.pine_path}")


if __name__ == "__main__":
    main()
