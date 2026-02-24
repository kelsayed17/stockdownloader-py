"""SPY ML Ensemble CLI pipeline.

End-to-end pipeline that downloads SPY data, trains a model grid,
builds a soft-voting ensemble, distills it into a deep surrogate
decision tree, and exports a PineScript v6 strategy.

Stages::

    [1/6] Download SPY daily data (10y) via YahooDataClient
    [2/6] Build feature dataset via DatasetBuilder
    [3/6] Train model grid (5 configs; 3 for --quick)
    [4/6] Build ensemble (top N by accuracy)
    [5/6] Train deep surrogate, measure R-squared
    [6/6] Export to PineScript
    [Optional] Tournament comparison via simple backtest

Usage::

    spy-ml-ensemble                       # Full pipeline
    spy-ml-ensemble --quick               # Quick mode (3 configs)
    spy-ml-ensemble --top-models 5        # Top 5 in ensemble
    spy-ml-ensemble --no-pine             # Skip PineScript export
    spy-ml-ensemble --no-tournament       # Skip tournament backtest
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

_FULL_GRID = [
    # (model_type, use_class_balance)
    ("gradient_boosting", False),
    ("random_forest", False),
    ("logistic_regression", False),
    ("gradient_boosting", True),
    ("random_forest", True),
]

_QUICK_GRID = [
    ("gradient_boosting", False),
    ("random_forest", False),
    ("logistic_regression", False),
]


# ======================================================================
# Argparse
# ======================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``spy-ml-ensemble``."""
    parser = argparse.ArgumentParser(
        prog="spy-ml-ensemble",
        description=(
            "SPY ML Ensemble pipeline: download -> features -> train grid "
            "-> ensemble -> deep surrogate -> PineScript."
        ),
    )

    # Common ML args (--verbose, --quick, --no-pine, etc.)
    add_common_ml_args(parser)

    # Ensemble-specific args
    parser.add_argument(
        "--top-models", type=int, default=3,
        help="Number of top models to include in ensemble (default: 3)",
    )
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
    parser.add_argument(
        "--buy-thresh", type=float, default=0.55,
        help="Buy threshold for ML probability (default: 0.55)",
    )
    parser.add_argument(
        "--sell-thresh", type=float, default=0.45,
        help="Sell threshold for ML probability (default: 0.45)",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="Initial capital for tournament backtest (default: 100000)",
    )
    parser.add_argument(
        "--min-r2", type=float, default=0.70,
        help="Minimum R-squared for surrogate fidelity warning (default: 0.70)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Output directory for pipeline artifacts "
            f"(default: {DEFAULT_ML_PIPELINE_DIR / 'spy_ensemble'})"
        ),
    )
    parser.add_argument(
        "--no-tournament", action="store_true",
        help="Skip tournament backtest comparison",
    )

    return parser


# ======================================================================
# Tournament (simple backtest)
# ======================================================================


def _run_tournament(
    dataset_X,  # numpy ndarray
    dataset_dates: tuple[str, ...],
    daily_data: list,
    surrogate,  # DeepSurrogateExporter
    buy_thresh: float,
    sell_thresh: float,
    initial_capital: float = 100_000.0,
) -> None:
    """Run a portfolio-level walk-through backtest on surrogate predictions.

    Uses proper position sizing: ``shares = floor(capital / price)``.
    Goes long when surrogate prob > *buy_thresh*, short when < *sell_thresh*.
    Exits to flat when probability returns to neutral zone (between thresholds).
    Tracks portfolio equity, return %, win rate, max drawdown, and Sharpe ratio.

    Parameters
    ----------
    dataset_X:
        Feature matrix (n_samples, n_features).
    dataset_dates:
        Tuple of date strings aligned with *dataset_X*.
    daily_data:
        List of price bars (each with ``.date`` and ``.close``).
    surrogate:
        Trained :class:`DeepSurrogateExporter`.
    buy_thresh:
        Go long when prob > buy_thresh (default 0.55).
    sell_thresh:
        Go short when prob < sell_thresh (default 0.45).
    initial_capital:
        Starting portfolio value in dollars (default 100,000).
    """
    import math

    import numpy as np

    print("\n" + "=" * 60)
    print("TOURNAMENT: Portfolio Backtest")
    print("=" * 60)
    print(f"  Initial capital: ${initial_capital:,.0f}")
    print(f"  Thresholds:      buy>{buy_thresh:.2f}  sell<{sell_thresh:.2f}")

    preds = surrogate.predict(dataset_X)

    # Build a date-to-close map from daily_data
    date_close: dict[str, float] = {}
    for bar in daily_data:
        date_close[bar.date] = float(bar.close)

    # Portfolio state
    capital = initial_capital  # cash available
    position = 0  # 1 = long, -1 = short, 0 = flat
    shares = 0  # shares held (positive for long, negative for short)
    entry_price = 0.0
    trades_pnl: list[float] = []  # dollar PnL per trade
    equity_curve: list[float] = [initial_capital]

    for i in range(len(preds) - 1):
        prob = float(preds[i])
        dt = dataset_dates[i]
        close = date_close.get(dt)
        if close is None:
            continue

        next_dt = dataset_dates[i + 1]
        next_close = date_close.get(next_dt)
        if next_close is None:
            continue

        if position == 0:
            # -- Entry from flat --
            if prob > buy_thresh:
                shares = math.floor(capital / close)
                if shares > 0:
                    entry_price = close
                    capital -= shares * close
                    position = 1
            elif prob < sell_thresh:
                shares = math.floor(capital / close)
                if shares > 0:
                    entry_price = close
                    capital += shares * close  # short sale proceeds
                    position = -1
        else:
            # -- Exit to flat when signal leaves threshold zone --
            should_exit_flat = False
            if position == 1 and prob <= buy_thresh:
                should_exit_flat = True
            elif position == -1 and prob >= sell_thresh:
                should_exit_flat = True

            if should_exit_flat:
                if position == 1:
                    pnl = shares * (next_close - entry_price)
                    capital += shares * next_close
                else:
                    pnl = shares * (entry_price - next_close)
                    capital -= shares * next_close  # buy back short
                trades_pnl.append(pnl)
                position = 0
                shares = 0
                entry_price = 0.0

                # Record equity after closing
                equity_curve.append(capital)

                # -- Immediately check for reversal entry --
                if prob > buy_thresh:
                    new_shares = math.floor(capital / next_close)
                    if new_shares > 0:
                        entry_price = next_close
                        capital -= new_shares * next_close
                        shares = new_shares
                        position = 1
                elif prob < sell_thresh:
                    new_shares = math.floor(capital / next_close)
                    if new_shares > 0:
                        entry_price = next_close
                        capital += new_shares * next_close
                        shares = new_shares
                        position = -1
            else:
                # Still in position — update equity mark-to-market
                if position == 1:
                    mark = capital + shares * next_close
                else:
                    mark = capital - shares * next_close
                equity_curve.append(mark)

    # Close final position at last available price
    if position != 0 and len(dataset_dates) > 0:
        last_dt = dataset_dates[-1]
        last_close = date_close.get(last_dt)
        if last_close is not None:
            if position == 1:
                pnl = shares * (last_close - entry_price)
                capital += shares * last_close
            else:
                pnl = shares * (entry_price - last_close)
                capital -= shares * last_close
            trades_pnl.append(pnl)
            equity_curve.append(capital)
            position = 0
            shares = 0

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    if not trades_pnl:
        print("  No trades generated.")
        return

    trades_arr = np.array(trades_pnl)
    n_trades = len(trades_pnl)
    wins = int(np.sum(trades_arr > 0))
    win_rate = wins / n_trades if n_trades > 0 else 0.0

    final_equity = equity_curve[-1]
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100.0
    total_return_dollar = final_equity - initial_capital

    # Max drawdown from equity curve
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_drawdown_pct = float(np.min(drawdowns)) * 100.0
    max_drawdown_dollar = float(np.min(equity_arr - running_max))

    # Annualized Sharpe (daily equity returns → annualized)
    if len(equity_arr) > 2:
        daily_returns = np.diff(equity_arr) / equity_arr[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = (
                float(np.mean(daily_returns))
                / float(np.std(daily_returns))
                * np.sqrt(252)
            )
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Avg trade duration (approximate from # trades and # bars)
    n_bars = len(equity_curve) - 1
    avg_hold = n_bars / n_trades if n_trades > 0 else 0

    print(f"\n  {'─' * 40}")
    print(f"  Total trades:    {n_trades}")
    print(f"  Win rate:        {win_rate:.1%}")
    print(f"  Avg hold (bars): {avg_hold:.0f}")
    print(f"  {'─' * 40}")
    print(f"  Initial capital: ${initial_capital:>12,.2f}")
    print(f"  Final equity:    ${final_equity:>12,.2f}")
    print(f"  Total return:    ${total_return_dollar:>+12,.2f}"
          f"  ({total_return_pct:+.1f}%)")
    print(f"  {'─' * 40}")
    print(f"  Max drawdown:    ${max_drawdown_dollar:>12,.2f}"
          f"  ({max_drawdown_pct:.1f}%)")
    print(f"  Sharpe ratio:    {sharpe:>12.2f}")
    print(f"  {'─' * 40}")
    print(f"  Avg trade PnL:   ${float(np.mean(trades_arr)):>+12,.2f}")
    print(f"  Best trade:      ${float(np.max(trades_arr)):>+12,.2f}")
    print(f"  Worst trade:     ${float(np.min(trades_arr)):>+12,.2f}")


# ======================================================================
# Main pipeline
# ======================================================================


def main(argv: list[str] | None = None) -> None:
    """Run the SPY ML Ensemble pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve output directory
    output_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_ML_PIPELINE_DIR / "spy_ensemble"
    )

    # Initialize environment
    init_ml_env(args.verbose)

    # Lazy imports for heavy dependencies
    import numpy as np

    from stockdownloader.data.market.yahoo_data_client import YahooDataClient
    from stockdownloader.ml.dataset_builder import (
        DatasetBuilder,
        LabelConfig,
    )
    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter
    from stockdownloader.ml.ensemble import EnsembleBuilder
    from stockdownloader.ml.feature_extractor import FeatureExtractor
    from stockdownloader.ml.trainer import MLModelConfig, MLTrainer
    from stockdownloader.app.pinescript_catalog.spy_ml_ensemble import (
        spy_ml_ensemble_strategy,
    )
    from stockdownloader.pinescript.generator import PineScriptGenerator

    t0 = time.time()

    # Select model grid
    grid = _QUICK_GRID if args.quick else _FULL_GRID

    print("=" * 60)
    print("SPY ML ENSEMBLE PIPELINE")
    print("=" * 60)
    print(f"  Mode:          {'quick' if args.quick else 'full'}")
    print(f"  Model configs: {len(grid)}")
    print(f"  Top models:    {args.top_models}")
    print(f"  Surrogate:     depth={args.depth}, "
          f"features={args.top_features}, min_leaf={args.min_leaf}")
    print(f"  Thresholds:    buy={args.buy_thresh}, sell={args.sell_thresh}")
    print(f"  Capital:       ${args.initial_capital:,.0f}")
    print(f"  Output:        {output_dir}")
    print()

    # ------------------------------------------------------------------
    # [1/6] Download SPY daily data
    # ------------------------------------------------------------------
    print("[1/6] Downloading SPY daily data (10y)...")
    t1 = time.time()

    client = YahooDataClient()
    daily_data = client.fetch_price_data("SPY", range_="10y")

    if len(daily_data) < 500:
        print(
            f"\nERROR: Insufficient data — got {len(daily_data)} bars "
            f"(need at least 500).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Downloaded {len(daily_data)} daily bars "
          f"({daily_data[0].date} to {daily_data[-1].date})")
    print(f"  [{time.time() - t1:.1f}s]")

    # ------------------------------------------------------------------
    # [2/6] Build feature dataset
    # ------------------------------------------------------------------
    print("\n[2/6] Building feature dataset...")
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

    # ------------------------------------------------------------------
    # [3/6] Train model grid
    # ------------------------------------------------------------------
    print(f"\n[3/6] Training model grid ({len(grid)} configs)...")
    t3 = time.time()

    results = []
    for idx, (model_type, use_balance) in enumerate(grid, 1):
        label = f"{model_type}" + ("+balanced" if use_balance else "")
        print(f"  [{idx}/{len(grid)}] {label}...", end=" ", flush=True)

        config = MLModelConfig(
            model_type=model_type,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            use_class_balance=use_balance,
        )
        trainer = MLTrainer(config)
        result = trainer.train(dataset)
        results.append(result)

        print(f"acc={result.oos_accuracy:.4f}  "
              f"auc={result.oos_roc_auc:.4f}")

    print(f"  [{time.time() - t3:.1f}s]")

    # ------------------------------------------------------------------
    # [4/6] Build ensemble
    # ------------------------------------------------------------------
    print(f"\n[4/6] Building ensemble (top {args.top_models})...")
    t4 = time.time()

    ensemble = EnsembleBuilder(results).select_top(n=args.top_models)
    probs = ensemble.predict_proba(dataset.X)
    importances = ensemble.averaged_feature_importances()

    print(f"  Ensemble models: {ensemble.n_models}")
    print(f"  Prob range: [{float(np.min(probs)):.4f}, "
          f"{float(np.max(probs)):.4f}]")
    print(f"  Prob mean:  {float(np.mean(probs)):.4f}")

    # Show top features
    top_feat = list(importances.items())[:5]
    print("  Top features:")
    for name, imp in top_feat:
        print(f"    {name:30s} {imp:.4f}")

    print(f"  [{time.time() - t4:.1f}s]")

    # ------------------------------------------------------------------
    # [5/6] Train deep surrogate
    # ------------------------------------------------------------------
    print(f"\n[5/6] Training deep surrogate (depth={args.depth})...")
    t5 = time.time()

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
        print(
            f"\n  WARNING: Surrogate R-squared ({r_squared:.4f}) is below "
            f"threshold ({args.min_r2:.2f}).",
            file=sys.stderr,
        )
        print(
            "  The surrogate tree may not faithfully represent the ensemble.",
            file=sys.stderr,
        )
        print(
            "  Consider increasing --depth or --top-features.\n",
            file=sys.stderr,
        )

    print(f"  [{time.time() - t5:.1f}s]")

    # ------------------------------------------------------------------
    # [6/6] Export to PineScript
    # ------------------------------------------------------------------
    if args.no_pine:
        print("\n[6/6] PineScript export skipped (--no-pine).")
    else:
        print("\n[6/6] Exporting PineScript strategy...")
        t6 = time.time()

        strategy_def = spy_ml_ensemble_strategy(
            exporter,
            buy_threshold=args.buy_thresh,
            sell_threshold=args.sell_thresh,
        )
        pine_code = PineScriptGenerator().generate(strategy_def)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        pine_path = output_dir / "spy_ml_ensemble.pine"
        pine_path.write_text(pine_code, encoding="utf-8")

        print(f"  PineScript written to: {pine_path}")
        print(f"  Lines: {len(pine_code.splitlines())}")
        print(f"  [{time.time() - t6:.1f}s]")

    # ------------------------------------------------------------------
    # [Optional] Tournament
    # ------------------------------------------------------------------
    if not args.no_tournament:
        _run_tournament(
            dataset.X,
            dataset.dates,
            daily_data,
            exporter,
            args.buy_thresh,
            args.sell_thresh,
            initial_capital=args.initial_capital,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time:    {elapsed:.1f}s")
    print(f"  Models trained: {len(results)}")
    print(f"  Ensemble size:  {ensemble.n_models}")
    print(f"  Surrogate R2:   {r_squared:.4f}")
    if not args.no_pine:
        print(f"  PineScript:     {output_dir / 'spy_ml_ensemble.pine'}")
    print()


if __name__ == "__main__":
    main()
