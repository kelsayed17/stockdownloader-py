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

    # Walk-forward backtest
    parser.add_argument(
        "--walk-forward-windows", type=int, default=5,
        help="Number of expanding windows for walk-forward backtest (default: 5)",
    )
    parser.add_argument(
        "--no-walk-forward", action="store_true",
        help="Use in-sample backtest instead of walk-forward (faster but biased)",
    )
    parser.add_argument(
        "--direct-ensemble", action="store_true", default=True,
        help=(
            "Use ensemble predict_proba directly instead of surrogate "
            "(default: True — preserves full signal)"
        ),
    )
    parser.add_argument(
        "--use-surrogate", action="store_true",
        help="Use surrogate distillation for walk-forward predictions (slower, lossy)",
    )
    parser.add_argument(
        "--long-only", action="store_true", default=True,
        help="Long-only trading mode — no short positions (default: True)",
    )
    parser.add_argument(
        "--allow-shorts", action="store_true",
        help="Allow short positions in backtest (overrides --long-only)",
    )

    # Crash avoidance mode
    parser.add_argument(
        "--crash-avoidance", action="store_true",
        help=(
            "Crash avoidance mode: start fully invested and only exit "
            "to cash on strong bearish signals. Overrides --long-only."
        ),
    )
    parser.add_argument(
        "--crash-exit-thresh", type=float, default=0.35,
        help=(
            "Exit to cash when prob < this threshold "
            "(crash avoidance mode only, default: 0.35)"
        ),
    )
    parser.add_argument(
        "--re-entry-thresh", type=float, default=0.50,
        help=(
            "Re-enter long when prob > this threshold "
            "(crash avoidance mode only, default: 0.50)"
        ),
    )

    return parser


# ======================================================================
# Portfolio backtest engine
# ======================================================================


def _run_portfolio_backtest(
    predictions,  # numpy 1-D ndarray of probabilities
    dates: tuple[str, ...],
    daily_data: list,
    buy_thresh: float,
    sell_thresh: float,
    initial_capital: float = 100_000.0,
    label: str = "Portfolio Backtest",
    long_only: bool = False,
    crash_avoidance: bool = False,
    crash_exit_thresh: float = 0.35,
    re_entry_thresh: float = 0.50,
) -> dict[str, float]:
    """Run portfolio backtest on pre-generated predictions.

    **Active mode** (default): Uses proper position sizing.  Goes long
    when prob > *buy_thresh*, short when < *sell_thresh* (unless
    *long_only*).  Exits when probability returns to neutral zone.

    **Crash avoidance mode**: Starts fully invested on day 1.  Stays
    long unless prob < *crash_exit_thresh* (strong bearish).  Re-enters
    when prob > *re_entry_thresh*.  Captures SPY's natural upward drift
    and only exits to avoid the worst drawdowns.

    Parameters
    ----------
    predictions:
        1-D array of probability predictions aligned with *dates*.
    dates:
        Tuple of date strings aligned with *predictions*.
    daily_data:
        List of price bars (each with ``.date`` and ``.close``).
    buy_thresh:
        Go long when prob > buy_thresh (active mode).
    sell_thresh:
        Go short when prob < sell_thresh (active mode, ignored when
        *long_only*).
    initial_capital:
        Starting portfolio value in dollars.
    label:
        Banner label for printed output.
    long_only:
        If ``True``, never enter short positions (active mode only).
    crash_avoidance:
        If ``True``, start fully invested and only exit to cash when
        prob < *crash_exit_thresh*.  Overrides active entry/exit logic.
    crash_exit_thresh:
        Exit to cash when prob < this (crash avoidance only, default 0.35).
    re_entry_thresh:
        Re-enter long when prob > this (crash avoidance only, default 0.50).

    Returns
    -------
    dict[str, float]
        Metrics dict with keys: n_trades, win_rate, initial_capital,
        final_equity, total_return_pct, total_return_dollar,
        max_drawdown_pct, max_drawdown_dollar, sharpe,
        avg_hold_bars, avg_trade_pnl, best_trade, worst_trade.
    """
    import math

    import numpy as np

    print("\n" + "=" * 60)
    print(f"TOURNAMENT: {label}")
    print("=" * 60)
    print(f"  Initial capital: ${initial_capital:,.0f}")
    if crash_avoidance:
        print(f"  Mode:            CRASH-AVOIDANCE")
        print(f"  Exit threshold:  prob < {crash_exit_thresh:.2f}")
        print(f"  Re-entry:        prob > {re_entry_thresh:.2f}")
    else:
        mode_str = "LONG-ONLY" if long_only else "long/short"
        print(f"  Mode:            {mode_str}")
        print(f"  Thresholds:      buy>{buy_thresh:.2f}  sell<{sell_thresh:.2f}")
    print(f"  Predictions:     {len(predictions)} samples")

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

    preds = predictions

    if crash_avoidance:
        # ==============================================================
        # CRASH AVOIDANCE: start fully invested, exit only on danger
        # ==============================================================

        # Buy immediately at first available price
        for dt in dates:
            first_close = date_close.get(dt)
            if first_close is not None and first_close > 0:
                shares = math.floor(capital / first_close)
                if shares > 0:
                    capital -= shares * first_close
                    entry_price = first_close
                    position = 1
                break

        for i in range(len(preds) - 1):
            prob = float(preds[i])
            dt = dates[i]
            close = date_close.get(dt)
            if close is None:
                continue

            next_dt = dates[i + 1]
            next_close = date_close.get(next_dt)
            if next_close is None:
                continue

            if position == 1 and prob < crash_exit_thresh:
                # EXIT to cash — strong bearish signal
                pnl = shares * (next_close - entry_price)
                capital += shares * next_close
                trades_pnl.append(pnl)
                position = 0
                shares = 0
                entry_price = 0.0
                equity_curve.append(capital)

            elif position == 0 and prob > re_entry_thresh:
                # RE-ENTER — danger has passed
                shares = math.floor(capital / next_close)
                if shares > 0:
                    entry_price = next_close
                    capital -= shares * next_close
                    position = 1
                equity_curve.append(
                    capital + shares * next_close if position == 1
                    else capital
                )

            else:
                # Hold current position — mark to market
                if position == 1:
                    equity_curve.append(capital + shares * next_close)
                else:
                    equity_curve.append(capital)

    else:
        # ==============================================================
        # ACTIVE TRADING: enter/exit based on threshold crossings
        # ==============================================================
        for i in range(len(preds) - 1):
            prob = float(preds[i])
            dt = dates[i]
            close = date_close.get(dt)
            if close is None:
                continue

            next_dt = dates[i + 1]
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
                elif prob < sell_thresh and not long_only:
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
                    elif prob < sell_thresh and not long_only:
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
    if position != 0 and len(dates) > 0:
        last_dt = dates[-1]
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
    _EMPTY_METRICS: dict[str, float] = {
        "n_trades": 0, "win_rate": 0.0,
        "initial_capital": initial_capital, "final_equity": initial_capital,
        "total_return_pct": 0.0, "total_return_dollar": 0.0,
        "max_drawdown_pct": 0.0, "max_drawdown_dollar": 0.0,
        "sharpe": 0.0, "avg_hold_bars": 0.0,
        "avg_trade_pnl": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
    }

    if not trades_pnl:
        print("  No trades generated.")
        return _EMPTY_METRICS

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

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "total_return_dollar": total_return_dollar,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_dollar": max_drawdown_dollar,
        "sharpe": sharpe,
        "avg_hold_bars": avg_hold,
        "avg_trade_pnl": float(np.mean(trades_arr)),
        "best_trade": float(np.max(trades_arr)),
        "worst_trade": float(np.min(trades_arr)),
    }


# ======================================================================
# In-sample tournament (has look-ahead bias)
# ======================================================================


def _run_tournament_insample(
    dataset_X,  # numpy ndarray
    dataset_dates: tuple[str, ...],
    daily_data: list,
    surrogate,  # DeepSurrogateExporter
    buy_thresh: float,
    sell_thresh: float,
    initial_capital: float = 100_000.0,
) -> dict[str, float]:
    """Run in-sample tournament backtest (has look-ahead bias).

    WARNING: This generates predictions from the surrogate trained on the
    *full* dataset, then backtests on that same data.  Results are inflated
    and not representative of live trading performance.  Use
    :func:`_run_tournament_walk_forward` for realistic OOS results.

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
        Go long when prob > buy_thresh.
    sell_thresh:
        Go short when prob < sell_thresh.
    initial_capital:
        Starting portfolio value in dollars.

    Returns
    -------
    dict[str, float]
        Metrics dict from :func:`_run_portfolio_backtest`.
    """
    preds = surrogate.predict(dataset_X)
    return _run_portfolio_backtest(
        preds, dataset_dates, daily_data,
        buy_thresh, sell_thresh,
        initial_capital=initial_capital,
        label="In-sample Backtest (WARNING: has look-ahead bias)",
    )


# Backward-compatibility alias — will be removed once all callers migrate.
_run_tournament = _run_tournament_insample


# ======================================================================
# Walk-forward prediction generator
# ======================================================================


def _generate_walk_forward_predictions(
    dataset,  # MLDataset
    daily_data: list,
    grid: list[tuple[str, bool]],
    n_estimators: int,
    max_depth: int,
    *,
    top_models: int = 3,
    diversity_weight: float = 0.4,
    use_select_diverse: bool = False,
    use_direct_ensemble: bool = False,
    surrogate_depth: int = 10,
    surrogate_min_leaf: int = 10,
    surrogate_top_features: int = 25,
    n_windows: int = 5,
    min_train_ratio: float = 0.5,
):
    """Generate truly out-of-sample predictions via expanding-window walk-forward.

    For each window the entire pipeline is re-run from scratch: train model
    grid on past data only, build ensemble, then predict on the unseen future
    window.  No future data ever leaks into predictions.

    Parameters
    ----------
    dataset:
        Full :class:`MLDataset` to split temporally.
    daily_data:
        Raw price bars (for date lookups).
    grid:
        Model grid as ``(model_type, use_class_balance)`` tuples.
    n_estimators:
        Boosting rounds for tree models.
    max_depth:
        Maximum tree depth.
    top_models:
        How many models to include in each window's ensemble.
    diversity_weight:
        Diversity weight for :meth:`EnsembleBuilder.select_diverse`.
    use_select_diverse:
        If ``True`` use diversity-aware selection; otherwise ``select_top``.
    use_direct_ensemble:
        If ``True`` skip surrogate distillation and use ensemble
        ``predict_proba`` directly.  This preserves the full ensemble
        signal — no R² loss from surrogate approximation.
    surrogate_depth:
        Max depth for the per-window surrogate tree.
    surrogate_min_leaf:
        Min samples per leaf for the surrogate.
    surrogate_top_features:
        Number of top features to feed the surrogate.
    n_windows:
        Number of expanding windows (default 5).
    min_train_ratio:
        Minimum fraction of data for the first training window.

    Returns
    -------
    tuple[numpy.ndarray, tuple[str, ...], int]
        ``(oos_predictions, oos_dates, n_windows_used)`` — the concatenated
        out-of-sample probability predictions, their aligned dates, and the
        number of windows actually used.
    """
    import numpy as np

    from stockdownloader.ml.dataset_builder import MLDataset
    from stockdownloader.ml.deep_surrogate import DeepSurrogateExporter
    from stockdownloader.ml.ensemble import EnsembleBuilder
    from stockdownloader.ml.trainer import MLModelConfig, MLTrainer
    from stockdownloader.ml.trainer_tuning import TimeSeriesExpandingCV

    n_samples = dataset.X.shape[0]
    cv = TimeSeriesExpandingCV(n_splits=n_windows, min_train_ratio=min_train_ratio)
    folds = cv.split(n_samples)

    all_oos_preds: list = []
    all_oos_dates: list[str] = []
    n_windows_used = 0

    for fold_idx, (train_range, test_range) in enumerate(folds, 1):
        train_idx = list(train_range)
        test_idx = list(test_range)

        print(f"\n  [Window {fold_idx}/{len(folds)}] "
              f"Train 0..{train_idx[-1]} ({len(train_idx)} samples), "
              f"Test {test_idx[0]}..{test_idx[-1]} ({len(test_idx)} samples)")

        # -- Create train-only sub-dataset --
        train_dataset = MLDataset(
            X=dataset.X[train_idx],
            y=dataset.y[train_idx],
            dates=tuple(dataset.dates[i] for i in train_idx),
            feature_names=dataset.feature_names,
            label_config=dataset.label_config,
        )

        # -- Train full model grid on train data --
        results = []
        for model_type, use_balance in grid:
            try:
                config = MLModelConfig(
                    model_type=model_type,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    use_class_balance=use_balance,
                )
                trainer = MLTrainer(config)
                result = trainer.train(train_dataset)
                results.append(result)
            except Exception:
                continue

        if len(results) < 1:
            print(f"    WARNING: No models trained — skipping window {fold_idx}")
            continue

        # -- Build ensemble from train results --
        builder = EnsembleBuilder(results)
        if use_select_diverse and len(results) >= 2:
            ensemble = builder.select_diverse(
                n=top_models, diversity_weight=diversity_weight,
            )
        else:
            ensemble = builder.select_top(n=top_models)

        # -- Predict on test portion only (truly OOS) --
        test_X = dataset.X[test_idx]

        if use_direct_ensemble:
            # Use ensemble predict_proba directly — no surrogate signal loss
            oos_preds = ensemble.predict_proba(test_X)
            pred_mode = "direct-ensemble"
        else:
            # Train surrogate on train data, predict via surrogate
            train_probs = ensemble.predict_proba(train_dataset.X)
            importances = ensemble.averaged_feature_importances()

            exporter = DeepSurrogateExporter(
                max_depth=surrogate_depth,
                min_samples_leaf=surrogate_min_leaf,
                top_n=surrogate_top_features,
            )
            exporter.train_surrogate(
                train_dataset, importances, ensemble_probs=train_probs,
            )
            oos_preds = exporter.predict(test_X)
            pred_mode = "surrogate"

        oos_dates = [dataset.dates[i] for i in test_idx]

        all_oos_preds.append(oos_preds)
        all_oos_dates.extend(oos_dates)
        n_windows_used += 1

        print(f"    Ensemble: {ensemble.n_models} models ({pred_mode}), "
              f"OOS preds range: [{float(np.min(oos_preds)):.4f}, "
              f"{float(np.max(oos_preds)):.4f}]")

    if not all_oos_preds:
        raise RuntimeError(
            "Walk-forward produced no predictions — all windows failed"
        )

    concatenated = np.concatenate(all_oos_preds)
    return concatenated, tuple(all_oos_dates), n_windows_used


# ======================================================================
# Walk-forward tournament orchestrator
# ======================================================================


def _run_tournament_walk_forward(
    dataset,  # MLDataset
    daily_data: list,
    grid: list[tuple[str, bool]],
    n_estimators: int,
    max_depth: int,
    buy_thresh: float,
    sell_thresh: float,
    *,
    initial_capital: float = 100_000.0,
    top_models: int = 3,
    diversity_weight: float = 0.4,
    use_select_diverse: bool = False,
    use_direct_ensemble: bool = False,
    long_only: bool = False,
    crash_avoidance: bool = False,
    crash_exit_thresh: float = 0.35,
    re_entry_thresh: float = 0.50,
    surrogate_depth: int = 10,
    surrogate_min_leaf: int = 10,
    surrogate_top_features: int = 25,
    n_windows: int = 5,
    min_train_ratio: float = 0.5,
) -> dict[str, float]:
    """Run walk-forward tournament backtest with proper OOS predictions.

    Generates truly out-of-sample predictions via expanding-window
    walk-forward, then runs the portfolio backtest on those predictions.
    This eliminates look-ahead bias present in the in-sample backtest.

    Parameters
    ----------
    dataset:
        Full :class:`MLDataset`.
    daily_data:
        Raw price bars.
    grid:
        Model grid as ``(model_type, use_class_balance)`` tuples.
    n_estimators:
        Boosting rounds for tree models.
    max_depth:
        Maximum tree depth.
    buy_thresh:
        Go long when prob > buy_thresh.
    sell_thresh:
        Go short when prob < sell_thresh.
    initial_capital:
        Starting portfolio value in dollars.
    top_models:
        Models per window ensemble.
    diversity_weight:
        Diversity weight for diversity-aware ensemble selection.
    use_select_diverse:
        Use diversity-aware selection instead of top-N.
    surrogate_depth:
        Surrogate tree depth per window.
    surrogate_min_leaf:
        Surrogate min samples per leaf.
    surrogate_top_features:
        Surrogate feature count.
    n_windows:
        Number of expanding windows.
    min_train_ratio:
        Minimum fraction of data for first training window.

    Returns
    -------
    dict[str, float]
        Metrics dict from :func:`_run_portfolio_backtest`.
    """
    pred_mode = "direct-ensemble" if use_direct_ensemble else "surrogate"
    if crash_avoidance:
        trade_mode = "CRASH-AVOIDANCE"
    elif long_only:
        trade_mode = "LONG-ONLY"
    else:
        trade_mode = "long/short"

    print("\n" + "=" * 60)
    print("WALK-FORWARD TOURNAMENT")
    print("=" * 60)
    print(f"  Windows:       {n_windows}")
    print(f"  Min train:     {min_train_ratio:.0%} of data")
    print(f"  Grid size:     {len(grid)} configs per window")
    print(f"  Selection:     {'diverse' if use_select_diverse else 'top-N'} "
          f"(n={top_models})")
    print(f"  Prediction:    {pred_mode}")
    print(f"  Trading mode:  {trade_mode}")
    if crash_avoidance:
        print(f"  Exit thresh:   prob < {crash_exit_thresh:.2f}")
        print(f"  Re-entry:      prob > {re_entry_thresh:.2f}")

    oos_preds, oos_dates, n_used = _generate_walk_forward_predictions(
        dataset, daily_data, grid, n_estimators, max_depth,
        top_models=top_models,
        diversity_weight=diversity_weight,
        use_select_diverse=use_select_diverse,
        use_direct_ensemble=use_direct_ensemble,
        surrogate_depth=surrogate_depth,
        surrogate_min_leaf=surrogate_min_leaf,
        surrogate_top_features=surrogate_top_features,
        n_windows=n_windows,
        min_train_ratio=min_train_ratio,
    )

    print(f"\n  Walk-forward complete: {n_used}/{n_windows} windows, "
          f"{len(oos_preds)} OOS predictions")

    return _run_portfolio_backtest(
        oos_preds, oos_dates, daily_data,
        buy_thresh, sell_thresh,
        initial_capital=initial_capital,
        label=f"Walk-Forward OOS ({n_used} windows, {pred_mode}, {trade_mode})",
        long_only=long_only,
        crash_avoidance=crash_avoidance,
        crash_exit_thresh=crash_exit_thresh,
        re_entry_thresh=re_entry_thresh,
    )


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

    # Resolve effective flags (--use-surrogate overrides --direct-ensemble)
    use_direct_ensemble = args.direct_ensemble and not args.use_surrogate
    long_only = args.long_only and not args.allow_shorts
    crash_avoidance = args.crash_avoidance

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
    if not args.no_tournament:
        bt_mode = "in-sample" if args.no_walk_forward else (
            f"walk-forward ({args.walk_forward_windows} windows)"
        )
        pred_mode = "direct-ensemble" if use_direct_ensemble else "surrogate"
        trade_mode = "LONG-ONLY" if long_only else "long/short"
        print(f"  Backtest:      {bt_mode}")
        print(f"  Prediction:    {pred_mode}")
        print(f"  Trading:       {trade_mode}")
        if crash_avoidance:
            print(f"  Strategy:      CRASH-AVOIDANCE "
                  f"(exit<{args.crash_exit_thresh}, "
                  f"re-entry>{args.re_entry_thresh})")
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
        if args.no_walk_forward:
            # In-sample backtest (fast but has look-ahead bias)
            _run_tournament_insample(
                dataset.X,
                dataset.dates,
                daily_data,
                exporter,
                args.buy_thresh,
                args.sell_thresh,
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
                use_direct_ensemble=use_direct_ensemble,
                long_only=long_only,
                crash_avoidance=crash_avoidance,
                crash_exit_thresh=args.crash_exit_thresh,
                re_entry_thresh=args.re_entry_thresh,
                surrogate_depth=args.depth,
                surrogate_min_leaf=args.min_leaf,
                surrogate_top_features=args.top_features,
                n_windows=args.walk_forward_windows,
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
