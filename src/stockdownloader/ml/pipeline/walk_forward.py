"""Walk-forward validation helpers for the backtesting stage.

Provides simple temporal walk-forward, parallel walk-forward, and true
walk-forward with ML model retraining.  These were extracted from
:class:`BacktestStage` to keep the stage class thin and focused on
orchestration.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from typing import TYPE_CHECKING, Callable

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.ml.pipeline.results import (
    BacktestEntry,
    WalkForwardWindowResult,
)

if TYPE_CHECKING:
    from stockdownloader.ml.pipeline.config import BacktestConfig
    from stockdownloader.ml.pipeline.results import DataResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Strategy recreation
# ------------------------------------------------------------------


def recreate_strategy(entry: BacktestEntry) -> object | None:
    """Re-create a baseline strategy from its registry name."""
    if entry.is_hybrid:
        return None  # Can't recreate hybrids here
    try:
        from stockdownloader.strategies.loader import ensure_registered
        from stockdownloader.strategies.registry import StrategyRegistry

        ensure_registered()
        # Try to find the matching registry name
        for reg in StrategyRegistry.all_entries(category="daily"):
            strat = StrategyRegistry.create(reg.name)
            if strat.name == entry.strategy_name:
                return strat
    except (KeyError, ValueError, ImportError):
        pass
    return None


# ------------------------------------------------------------------
# Simple temporal walk-forward
# ------------------------------------------------------------------


def walk_forward(
    config: BacktestConfig,
    entry: BacktestEntry,
    data: list,
    initial_capital: Decimal,
    commission: Decimal,
    slippage_pct: Decimal,
    compute_score_fn: Callable[..., float],
) -> float | None:
    """Simple temporal walk-forward: split data into windows.

    Returns the degradation ratio (IS_score / OOS_score).
    A value close to 1.0 means the strategy generalises well.
    Values > 1.5 suggest overfitting.

    For hybrids, the ML model was trained on full historical data so
    IS/OOS windows don't retrain it.  However, running the hybrid on
    temporal slices still tests whether the *combined* signal (base
    strategy + ML filter) generalises across different market regimes.
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage_pct=slippage_pct,
    )

    n = len(data)
    n_windows = config.walk_forward_windows
    if n < n_windows * 100:
        return None  # Not enough data

    window_size = n // (n_windows + 1)
    is_scores: list[float] = []
    oos_scores: list[float] = []

    for w in range(n_windows):
        is_end = (w + 1) * window_size
        oos_start = is_end
        oos_end = min(oos_start + window_size, n)
        if oos_end - oos_start < 50:
            continue

        is_data = data[:is_end]
        oos_data = data[oos_start:oos_end]

        try:
            if entry.is_hybrid:
                # Use the retained strategy reference for hybrids
                strat = entry._strategy_ref
                if strat is None:
                    return None
                # Clear probability cache so features are re-extracted
                # per window (cache keys are bar indices, which differ)
                if hasattr(strat, "_prob_cache"):
                    strat._prob_cache.clear()
                is_result = engine.run(strat, is_data)
                is_score = compute_score_fn(is_result)

                if hasattr(strat, "_prob_cache"):
                    strat._prob_cache.clear()
                oos_result = engine.run(strat, oos_data)
                oos_score = compute_score_fn(oos_result)

                is_scores.append(is_score)
                oos_scores.append(oos_score)
            else:
                # For baselines, run both IS and OOS
                strat = recreate_strategy(entry)
                if strat is None:
                    return None
                is_result = engine.run(strat, is_data)
                is_score = compute_score_fn(is_result)

                strat2 = recreate_strategy(entry)
                if strat2 is None:
                    return None
                oos_result = engine.run(strat2, oos_data)
                oos_score = compute_score_fn(oos_result)

                is_scores.append(is_score)
                oos_scores.append(oos_score)
        except (ValueError, ZeroDivisionError, ArithmeticError, KeyError):
            continue

    if not oos_scores:
        return None

    avg_is = sum(is_scores) / len(is_scores) if is_scores else 0
    avg_oos = sum(oos_scores) / len(oos_scores)

    if avg_oos == 0:
        return 999.0 if avg_is > 0 else 1.0

    return avg_is / avg_oos if avg_oos != 0 else 1.0


# ------------------------------------------------------------------
# Parallel walk-forward
# ------------------------------------------------------------------


def parallel_walk_forward(
    config: BacktestConfig,
    top_entries: list[BacktestEntry],
    data: list,
    initial_capital: Decimal,
    commission: Decimal,
    slippage_pct: Decimal,
    max_workers: int,
    compute_score_fn: Callable[..., float],
    print_fn: Callable[..., None],
) -> None:
    """Run walk-forward validation for all top entries concurrently."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for entry in top_entries:
            fut = pool.submit(
                walk_forward,
                config,
                entry,
                data,
                initial_capital,
                commission,
                slippage_pct,
                compute_score_fn,
            )
            futures[fut] = entry

        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                deg = fut.result()
                entry.walk_forward_degradation = deg
                if deg is not None:
                    print_fn(
                        f"  {entry.strategy_name}: "
                        f"WF degradation={deg:.2f}"
                    )
            except Exception as exc:
                logger.warning(
                    "Walk-forward failed for %s: %s",
                    entry.strategy_name, exc,
                )


# ------------------------------------------------------------------
# True walk-forward with ML model retraining
# ------------------------------------------------------------------


def true_walk_forward(
    config: BacktestConfig,
    entry: BacktestEntry,
    data: list,
    initial_capital: Decimal,
    commission: Decimal,
    slippage_pct: Decimal,
    compute_score_fn: Callable[..., float],
    print_fn: Callable[..., None],
    data_result: DataResult,
) -> None:
    """True walk-forward: retrain ML model on each IS window, test on OOS.

    For each window:
    1. Build dataset from IS data using the same LabelConfig
    2. Train a fresh model with the same MLModelConfig
    3. Construct a new hybrid strategy using the fresh model
    4. Backtest the hybrid on OOS data
    5. Record per-window metrics

    Updates ``entry.walk_forward_degradation``,
    ``entry.wf_window_results``, and ``entry.wf_avg_oos_score`` in-place.
    """
    from stockdownloader.ml.dataset_builder import DatasetBuilder
    from stockdownloader.ml.feature_extractor import FeatureExtractor
    from stockdownloader.ml.pipeline.stage_hybrid import (
        _STRATEGY_CLASS,
    )
    from stockdownloader.ml.trainer import MLTrainer
    from stockdownloader.strategies.loader import ensure_registered
    from stockdownloader.strategies.registry import StrategyRegistry
    from stockdownloader.indicators.hub import IndicatorHub

    ensure_registered()

    label_cfg = entry._label_config
    model_cfg = entry._model_config
    base_name = entry._base_strategy_name
    hybrid_mode = entry._hybrid_mode or entry.mode
    threshold = entry._hybrid_threshold or 0.5

    if label_cfg is None or model_cfg is None or base_name is None:
        logger.warning(
            "Cannot run true WF for %s: missing training metadata",
            entry.strategy_name,
        )
        return

    n = len(data)
    n_windows = config.walk_forward_windows
    min_train = config.wf_min_train_bars

    if n < min_train + 200:
        logger.info(
            "Not enough data for true WF: %d bars (need %d+200)",
            n, min_train,
        )
        return

    # Expanding window: each window adds more IS data
    # OOS is always the next chunk after IS
    oos_size = max(n // (n_windows + 1), 100)
    window_results: list[WalkForwardWindowResult] = []
    is_scores: list[float] = []
    oos_scores: list[float] = []

    strategy_cls = _STRATEGY_CLASS.get(hybrid_mode)
    if strategy_cls is None:
        logger.warning("Unknown hybrid mode: %s", hybrid_mode)
        return

    for w in range(n_windows):
        is_end = min_train + w * oos_size
        if is_end >= n:
            break
        oos_start = is_end
        oos_end = min(oos_start + oos_size, n)
        if oos_end - oos_start < 50:
            continue

        is_data = data[:is_end]
        oos_data = data[oos_start:oos_end]

        try:
            # 1. Build dataset on IS data
            hub = IndicatorHub()
            extractor = FeatureExtractor(
                hub=hub,
                alt_data_store=data_result.alt_data_store,
                hmm_snapshots=data_result.hmm_snapshots,
            )
            builder = DatasetBuilder(extractor, label_config=label_cfg, hub=hub)

            try:
                dataset = builder.build(is_data)
            except ValueError:
                # Not enough data in this window
                continue

            if dataset.X.shape[0] < 30:
                continue  # Too few samples

            # 2. Train fresh model on IS data
            trainer = MLTrainer(model_cfg)
            try:
                training_result = trainer.train(dataset)
            except (ValueError, Exception) as exc:
                logger.debug(
                    "WF window %d training failed: %s", w, exc,
                )
                continue

            model_auc = training_result.oos_roc_auc
            fresh_model = training_result.model
            feature_names = dataset.feature_names

            # 3. Create fresh hybrid strategy
            base_strategy = StrategyRegistry.create(base_name)
            label = f"{base_strategy.name} (WF-{w})"
            hybrid = strategy_cls(
                base_strategy=base_strategy,
                model=fresh_model,
                feature_names=feature_names,
                threshold=threshold,
                label=label,
            )

            # 4. Backtest on IS and OOS
            engine = BacktestEngine(
                initial_capital=initial_capital,
                commission=commission,
                slippage_pct=slippage_pct,
            )

            is_result = engine.run(hybrid, is_data)
            is_score = compute_score_fn(is_result)

            # Clear cache before OOS run
            if hasattr(hybrid, "_prob_cache"):
                hybrid._prob_cache.clear()

            oos_result = engine.run(hybrid, oos_data)
            oos_score = compute_score_fn(oos_result)

            is_scores.append(is_score)
            oos_scores.append(oos_score)

            wr = WalkForwardWindowResult(
                window_idx=w,
                is_bars=len(is_data),
                oos_bars=len(oos_data),
                is_score=is_score,
                oos_score=oos_score,
                oos_trades=oos_result.total_trades,
                oos_pnl=float(oos_result.total_pnl),
                oos_win_rate=float(oos_result.win_rate),
                model_auc=model_auc,
            )
            window_results.append(wr)

            print_fn(
                f"  WF[{w}] {entry.strategy_name}: "
                f"IS={is_score:.1f} OOS={oos_score:.1f} "
                f"trades={oos_result.total_trades} "
                f"P/L=${float(oos_result.total_pnl):,.0f} "
                f"AUC={model_auc:.3f}"
            )

        except Exception as exc:
            logger.debug("WF window %d failed: %s", w, exc)
            continue

    # Aggregate results
    entry.wf_window_results = window_results

    if oos_scores:
        avg_oos = sum(oos_scores) / len(oos_scores)
        avg_is = sum(is_scores) / len(is_scores) if is_scores else 0
        entry.wf_avg_oos_score = avg_oos

        if avg_oos == 0:
            entry.walk_forward_degradation = 999.0 if avg_is > 0 else 1.0
        else:
            entry.walk_forward_degradation = avg_is / avg_oos

        print_fn(
            f"  → {entry.strategy_name}: TRUE WF "
            f"IS={avg_is:.1f} OOS={avg_oos:.1f} "
            f"deg={entry.walk_forward_degradation:.2f} "
            f"({len(window_results)} windows)"
        )
    else:
        print_fn(
            f"  → {entry.strategy_name}: TRUE WF — no valid windows"
        )
