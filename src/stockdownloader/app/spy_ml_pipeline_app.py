"""ML pipeline CLI apps.

Contains two entry points:

``main_generic``
    Standard end-to-end ML trading pipeline.  Runs the 6-stage pipeline
    (data -> train -> converge -> hybrid -> backtest -> select/export) on
    daily price data for any symbol.

    Usage::

        ml-pipeline SPY --quick
        ml-pipeline AAPL --range 10y --forward-periods 5,10
        ml-pipeline GME --model-types gradient_boosting --no-pine

``main``
    SPY multi-timeframe ML pipeline.  Runs the 6-stage ML pipeline on
    daily data and intraday timeframes (15m, 30m, 1h, 4h) resampled from
    5-minute bars.  Includes optional HMM regime detection as additional
    ML features.

    Usage::

        spy-ml-pipeline --quick              # Fast single-config sweep
        spy-ml-pipeline                      # Full sweep, all timeframes
        spy-ml-pipeline --daily-only         # Daily only
        spy-ml-pipeline --timeframes 1d,1h   # Specific timeframes
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from stockdownloader.app._ml_helpers import (
    add_common_ml_args,
    apply_config_defaults,
    build_pipeline_config,
    build_training_config,
    init_ml_env,
    run_ml_pipeline,
)
from stockdownloader.util.config_loader import DEFAULT_ML_PIPELINE_DIR


# ======================================================================
# Generic ML pipeline (formerly ml_pipeline_app.py)
# ======================================================================


def _build_generic_parser() -> argparse.ArgumentParser:
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


def main_generic(argv: list[str] | None = None) -> None:
    """``ml-pipeline`` -- standard end-to-end ML pipeline."""
    args = _build_generic_parser().parse_args(argv)
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


# ======================================================================
# SPY multi-timeframe pipeline
# ======================================================================


@dataclass
class _TimeframeResult:
    """Result from running the ML pipeline on one timeframe."""

    timeframe: str
    bar_count: int
    models_trained: int = 0
    convergence_pairs: int = 0
    hybrid_strategies: int = 0
    backtests_run: int = 0
    best_strategy_name: str = ""
    best_score: float = 0.0
    best_pnl: float = 0.0
    best_win_rate: float = 0.0
    best_sharpe: float = 0.0
    pine_path: str | None = None
    elapsed: float = 0.0


def _extract_tf_result(
    tf_label: str, bar_count: int, result: object, elapsed: float,
) -> _TimeframeResult:
    """Populate a ``_TimeframeResult`` from a pipeline result."""
    r = _TimeframeResult(timeframe=tf_label, bar_count=bar_count, elapsed=elapsed)
    if result.training_result:
        r.models_trained = len(result.training_result.candidates)
    if result.convergence_result:
        r.convergence_pairs = len(result.convergence_result.pairs)
    if result.hybrid_result:
        r.hybrid_strategies = len(result.hybrid_result.hybrid_strategies)
    if result.backtest_result:
        r.backtests_run = len(result.backtest_result.entries)
    if result.best_strategy:
        r.best_strategy_name = result.best_strategy.strategy_name
        r.best_score = result.best_strategy.composite_score
        r.best_pnl = float(result.best_strategy.result.total_pnl)
        r.best_win_rate = float(result.best_strategy.result.win_rate)
        r.best_sharpe = float(result.best_strategy.result.sharpe_ratio(252))
    if result.pine_path:
        r.pine_path = result.pine_path
    return r


def _print_comparison(results: list[_TimeframeResult], total_elapsed: float) -> None:
    """Print cross-timeframe comparison table."""
    print()
    print("=" * 100)
    print("  MULTI-TIMEFRAME COMPARISON")
    print("=" * 100)
    print()
    print(
        f"{'TF':<6} {'Bars':>7} {'Models':>7} {'Hybrids':>8} "
        f"{'Best Strategy':<42} {'P/L':>14} {'WR':>7} "
        f"{'Sharpe':>7} {'Score':>7} {'Time':>7}"
    )
    print("-" * 120)

    best_overall = None
    for r in results:
        pnl_str = f"${r.best_pnl:>11,.2f}" if r.best_pnl else "$0.00"
        wr_str = f"{r.best_win_rate:.1f}%" if r.best_win_rate else "N/A"
        name = r.best_strategy_name[:40] if r.best_strategy_name else "N/A"
        print(
            f"{r.timeframe:<6} {r.bar_count:>7,} {r.models_trained:>7} "
            f"{r.hybrid_strategies:>8} "
            f"{name:<42} {pnl_str:>14} {wr_str:>7} "
            f"{r.best_sharpe:>7.2f} {r.best_score:>7.1f} "
            f"{r.elapsed:>6.1f}s"
        )
        if best_overall is None or r.best_score > best_overall.best_score:
            best_overall = r

    print("-" * 120)
    print()

    if best_overall:
        print(
            f"BEST OVERALL: [{best_overall.timeframe}] "
            f"{best_overall.best_strategy_name} "
            f"(score={best_overall.best_score:.1f}, "
            f"P/L=${best_overall.best_pnl:,.2f}, "
            f"WR={best_overall.best_win_rate:.1f}%)"
        )
        if best_overall.pine_path:
            print(f"PineScript: {best_overall.pine_path}")

    pine_files = [r for r in results if r.pine_path]
    if pine_files:
        print("\nPineScript exports:")
        for r in pine_files:
            print(f"  [{r.timeframe}] {r.pine_path}")

    print(f"\nTotal pipeline time: {total_elapsed:.1f}s")


# ======================================================================
# CLI
# ======================================================================


def main(argv: list[str] | None = None) -> None:
    """``spy-ml-pipeline`` — SPY multi-timeframe ML pipeline."""
    parser = argparse.ArgumentParser(
        prog="spy-ml-pipeline",
        description="SPY Multi-Timeframe ML Pipeline.",
    )
    # Data
    parser.add_argument("symbol", nargs="?", default="SPY", help="Ticker (default: SPY)")
    parser.add_argument("--daily-csv", default=None, help="Path to daily CSV")
    parser.add_argument("--intraday-csv", default=None, help="Path to 5-min CSV")
    parser.add_argument("--range", dest="range_", default="max", help="Daily data range (default: max)")
    # Timeframes
    parser.add_argument("--timeframes", default="1d,15m,30m,1h,4h", help="Timeframes (default: 1d,15m,30m,1h,4h)")
    parser.add_argument("--daily-only", action="store_true", help="Daily timeframe only")
    # Models
    parser.add_argument("--models", default=None, help="ML model types (comma-separated)")
    parser.add_argument("--model-types", default="gradient_boosting,logistic_regression", help=argparse.SUPPRESS)
    # HMM
    parser.add_argument("--no-hmm", action="store_true", help="Disable HMM regime features")
    parser.add_argument("--hmm-regimes", type=int, default=3, help="HMM regimes (default: 3)")
    # Backtest extras
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage fraction (default: 0.0005)")
    parser.add_argument("--true-wf", action="store_true", help="Walk-forward validation")
    parser.add_argument("--wf-windows", type=int, default=5, help="WF windows (default: 5)")
    parser.add_argument("--wf-min-train", type=int, default=500, help="Min IS bars per WF window (default: 500)")
    # Hybrid (suppressed — defaults differ from standard pipeline)
    parser.add_argument("--ml-threshold", type=float, default=0.6, help=argparse.SUPPRESS)
    parser.add_argument("--modes", default="confirmed,weighted", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=str(DEFAULT_ML_PIPELINE_DIR), help="Output directory")
    add_common_ml_args(parser)
    args = parser.parse_args(argv)
    init_ml_env(args.verbose)

    from stockdownloader.ml.pipeline.config import (
        BacktestConfig,
        ConvergenceConfig,
        DataConfig,
        HybridConfig,
        PipelineConfig,
        SelectionConfig,
        TrainingGridConfig,
    )
    from stockdownloader.ml.pipeline.orchestrator import MLPipelineOrchestrator

    symbol = args.symbol.upper()

    # Auto-detect data files
    daily_csv = args.daily_csv
    if daily_csv is None:
        auto = f"data/{symbol.lower()}_daily_bars.csv"
        if os.path.exists(auto):
            daily_csv = auto

    intraday_csv = args.intraday_csv
    if intraday_csv is None:
        auto = f"data/{symbol.lower()}_5m_bars.csv"
        if os.path.exists(auto):
            intraday_csv = auto

    # Parse timeframes
    if args.daily_only:
        timeframes = ["1d"]
    else:
        timeframes = [tf.strip() for tf in args.timeframes.split(",")]

    has_daily = "1d" in timeframes
    intraday_tfs = [tf for tf in timeframes if tf != "1d"]

    if intraday_tfs and not intraday_csv:
        print(
            f"WARNING: No 5-min CSV found. Skipping: {', '.join(intraday_tfs)}",
            file=sys.stderr,
        )
        intraday_tfs = []

    # Model types
    if args.models:
        model_types = tuple(m.strip() for m in args.models.split(","))
    elif args.quick:
        model_types = ("gradient_boosting",)
    else:
        model_types = ("gradient_boosting", "random_forest", "hist_gradient_boosting")

    training = build_training_config(args, model_types=model_types)

    # Banner
    all_tfs = (["1d"] if has_daily else []) + intraday_tfs
    print("=" * 80)
    print(f"  {symbol} Multi-Timeframe ML Pipeline")
    print("=" * 80)
    print(f"  Daily CSV:     {daily_csv or 'N/A'}")
    print(f"  Intraday CSV:  {intraday_csv or 'N/A'}")
    print(f"  Timeframes:    {', '.join(all_tfs)}")
    print(f"  ML Models:     {', '.join(model_types)}")
    print(f"  HMM Regimes:   {'Disabled' if args.no_hmm else args.hmm_regimes}")
    print(f"  Slippage:      {args.slippage * 100:.3f}%")
    print(f"  True WF:       {'Yes' if args.true_wf else 'No'}")
    print(f"  Quick Mode:    {'Yes' if args.quick else 'No'}")
    print("=" * 80)
    print()

    total_start = time.time()
    results: list[_TimeframeResult] = []

    # Shared backtest kwargs
    bt_kwargs = dict(
        initial_capital=args.capital,
        commission=args.commission,
        slippage_pct=args.slippage,
        true_walk_forward=args.true_wf,
        walk_forward_windows=args.wf_windows,
        wf_min_train_bars=args.wf_min_train,
    )

    # ── HMM regime detection ────────────────────────────────────────
    hmm_snapshots = None
    if not args.no_hmm and daily_csv:
        print("FITTING HMM REGIME DETECTOR...")
        print("-" * 50)
        try:
            from collections import Counter
            from stockdownloader.ml.hmm_regime_detector import HMMRegimeDetector
            from stockdownloader.ml.pipeline.stage_data import DataStage

            daily_data = DataStage._load_csv(Path(daily_csv))
            if daily_data:
                closes = [float(d.close) for d in daily_data]
                detector = HMMRegimeDetector(n_regimes=args.hmm_regimes)
                hmm_snapshots = detector.fit_predict(closes)
                print(f"  HMM fitted on {len(closes)} daily bars")
                print(f"  Regime labels: {detector.regime_labels}")
                for rid, count in sorted(Counter(s.regime_id for s in hmm_snapshots).items()):
                    label = detector.regime_labels.get(rid, f"regime_{rid}")
                    print(f"    {label}: {count} bars ({count / len(hmm_snapshots) * 100:.1f}%)")
                print()
        except Exception as exc:
            print(f"  HMM fitting failed: {exc}\n")

    # ── Daily timeframe ─────────────────────────────────────────────
    if has_daily and daily_csv:
        print("=" * 80)
        print("  TIMEFRAME: DAILY (1d)")
        print("=" * 80 + "\n")

        tf_start = time.time()
        config = PipelineConfig(
            data=DataConfig(symbol=symbol, range_=args.range_, csv_file=daily_csv),
            training=training,
            convergence=ConvergenceConfig(),
            hybrid=HybridConfig(modes=("confirmed", "weighted")),
            backtest=BacktestConfig(**bt_kwargs),
            selection=SelectionConfig(
                top_n=args.top_n, export_pine=not args.no_pine,
                pine_depth=args.pine_depth,
                output_dir=str(Path(args.output_dir) / "1d"),
            ),
        )
        pipeline = MLPipelineOrchestrator(config, hmm_snapshots=hmm_snapshots)
        result = pipeline.run()
        bar_count = result.data_result.bar_count if result.data_result else 0
        results.append(_extract_tf_result("1d", bar_count, result, time.time() - tf_start))
        print()

    # ── Intraday timeframes (resampled from 5m) ────────────────────
    if intraday_tfs and intraday_csv:
        from stockdownloader.data.intraday_csv import IntradayCsvLoader
        from stockdownloader.util.timeframe_aggregator import Timeframe, TimeframeAggregator

        print("Loading 5-minute data for timeframe aggregation...")
        raw_5m = IntradayCsvLoader.load_from_file(intraday_csv)
        if not raw_5m:
            print(f"ERROR: Could not load {intraday_csv}", file=sys.stderr)
        else:
            print(f"  Loaded {len(raw_5m)} 5-minute bars")
            agg = TimeframeAggregator(raw_5m)

            tf_enum_map = {
                "5m": Timeframe.M5, "15m": Timeframe.M15,
                "30m": Timeframe.M30, "1h": Timeframe.H1, "4h": Timeframe.H4,
            }
            bars_per_day = {"5m": 78, "15m": 26, "30m": 13, "1h": 7, "4h": 2}

            for tf_label in intraday_tfs:
                tf_enum = tf_enum_map.get(tf_label)
                if tf_enum is None:
                    print(f"  Skipping unknown timeframe: {tf_label}")
                    continue

                print(f"\n{'=' * 80}\n  TIMEFRAME: {tf_label.upper()}\n{'=' * 80}\n")
                tf_start = time.time()

                price_data = agg.as_price_data(tf_enum)
                if len(price_data) < 300:
                    print(f"  Insufficient data: {len(price_data)} bars. Skipping.")
                    continue

                print(f"  Aggregated to {len(price_data)} {tf_label} bars")

                # Write temp CSV for pipeline
                tf_csv = Path(args.output_dir) / f"{symbol.lower()}_{tf_label}_bars.csv"
                tf_csv.parent.mkdir(parents=True, exist_ok=True)
                with open(tf_csv, "w", newline="") as f:
                    writer = csv_mod.writer(f)
                    writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
                    for d in price_data:
                        writer.writerow([d.date, str(d.open), str(d.high), str(d.low),
                                         str(d.close), str(d.adj_close), d.volume])

                # Scale forward periods to intraday bar count
                bpd = bars_per_day.get(tf_label, 10)
                if args.quick:
                    itd_training = TrainingGridConfig(
                        forward_periods=(bpd,), profit_thresholds=(0.005,),
                        model_types=model_types,
                        use_class_balance_options=(False,), use_atr_labels_options=(False,),
                        n_estimators=args.n_estimators, max_depth=args.max_depth,
                    )
                else:
                    itd_training = TrainingGridConfig(
                        forward_periods=(bpd // 2, bpd, bpd * 2),
                        profit_thresholds=(0.003, 0.005, 0.01),
                        model_types=model_types,
                        n_estimators=args.n_estimators, max_depth=args.max_depth,
                    )

                config = PipelineConfig(
                    data=DataConfig(symbol=symbol, range_=args.range_, csv_file=str(tf_csv)),
                    training=itd_training,
                    convergence=ConvergenceConfig(),
                    hybrid=HybridConfig(modes=("confirmed", "weighted")),
                    backtest=BacktestConfig(**bt_kwargs),
                    selection=SelectionConfig(
                        top_n=args.top_n, export_pine=not args.no_pine,
                        pine_depth=args.pine_depth,
                        output_dir=str(Path(args.output_dir) / tf_label),
                    ),
                )
                pipeline = MLPipelineOrchestrator(config)
                result = pipeline.run()
                results.append(
                    _extract_tf_result(tf_label, len(price_data), result, time.time() - tf_start)
                )
                print()

    # ── Cross-timeframe comparison ──────────────────────────────────
    _print_comparison(results, time.time() - total_start)


if __name__ == "__main__":
    main()
