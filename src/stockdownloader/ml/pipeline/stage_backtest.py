"""Stage 5: Comprehensive Backtesting (parallelised).

Backtests all baseline daily strategies and hybrid strategies, scores
each by a composite metric, runs walk-forward validation on the top
performers, and returns a ranked list.

Baseline and hybrid backtests are run concurrently using a thread pool
(strategies contain ML model references that are not pickle-safe, so
process-pool parallelism is impractical).  Walk-forward windows are
also parallelised per strategy.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from typing import Callable

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.ml.pipeline.config import BacktestConfig
from stockdownloader.ml.pipeline.results import (
    BacktestEntry,
    BacktestStageResult,
    DataResult,
    HybridStageResult,
    WalkForwardWindowResult,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Module-level helpers (avoids pickling method references)
# ------------------------------------------------------------------


def _run_baseline_backtest(
    reg_name: str,
    data: list,
    initial_capital: Decimal,
    commission: Decimal,
    slippage_pct: Decimal = Decimal("0"),
) -> BacktestEntry | None:
    """Run a single baseline backtest (thread target)."""
    from stockdownloader.strategy.registration_loader import ensure_registered
    from stockdownloader.strategy.base_registry import StrategyRegistry

    ensure_registered()
    try:
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            slippage_pct=slippage_pct,
        )
        strategy = StrategyRegistry.create(reg_name)
        result = engine.run(strategy, data)
        score = BacktestStage._compute_score(result)
        return BacktestEntry(
            strategy_name=strategy.name,
            is_hybrid=False,
            mode="baseline",
            model_id=None,
            result=result,
            composite_score=score,
        )
    except Exception as exc:
        logger.warning("Baseline backtest %s failed: %s", reg_name, exc)
        return None


def _run_hybrid_backtest(
    hybrid_entry: object,
    data: list,
    initial_capital: Decimal,
    commission: Decimal,
    slippage_pct: Decimal = Decimal("0"),
) -> BacktestEntry | None:
    """Run a single hybrid backtest (thread target)."""
    try:
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            slippage_pct=slippage_pct,
        )
        result = engine.run(hybrid_entry.strategy, data)  # type: ignore[attr-defined]
        score = BacktestStage._compute_score(result)
        return BacktestEntry(
            strategy_name=hybrid_entry.name,  # type: ignore[attr-defined]
            is_hybrid=True,
            mode=hybrid_entry.mode,  # type: ignore[attr-defined]
            model_id=hybrid_entry.model_id,  # type: ignore[attr-defined]
            result=result,
            composite_score=score,
            _strategy_ref=hybrid_entry.strategy,  # type: ignore[attr-defined]
            _label_config=getattr(hybrid_entry, "label_config", None),
            _model_config=getattr(hybrid_entry, "model_config", None),
            _base_strategy_name=getattr(hybrid_entry, "base_strategy_name", None),
            _hybrid_mode=getattr(hybrid_entry, "mode", None),
            _hybrid_threshold=getattr(hybrid_entry, "hybrid_threshold", None),
        )
    except Exception as exc:
        logger.warning(
            "Hybrid backtest %s failed: %s",
            getattr(hybrid_entry, "name", "?"),
            exc,
        )
        return None


class BacktestStage:
    """Backtest all baseline and hybrid strategies.

    Parameters
    ----------
    config:
        Backtesting configuration.
    print_fn:
        Callable for progress output.
    """

    def __init__(
        self,
        config: BacktestConfig,
        print_fn: Callable[..., None] = print,
    ) -> None:
        self._cfg = config
        self._out = print_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data_result: DataResult,
        hybrid_result: HybridStageResult,
    ) -> BacktestStageResult:
        """Backtest all strategies and return ranked results."""
        from stockdownloader.strategy.registration_loader import ensure_registered
        from stockdownloader.strategy.base_registry import StrategyRegistry

        ensure_registered()
        data = data_result.data
        initial_capital = Decimal(str(self._cfg.initial_capital))
        commission = Decimal(str(self._cfg.commission))
        slippage_pct = Decimal(str(self._cfg.slippage_pct))

        max_workers = self._cfg.max_workers or os.cpu_count() or 4
        entries: list[BacktestEntry] = []

        # 1 & 2. Submit all backtests concurrently
        baseline_entries = list(StrategyRegistry.all_entries(category="daily"))
        hybrid_entries = list(hybrid_result.hybrid_strategies)
        total = len(baseline_entries) + len(hybrid_entries)

        self._out(
            f"Running {len(baseline_entries)} baseline + "
            f"{len(hybrid_entries)} hybrid backtests "
            f"({max_workers} workers)..."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}

            # Submit baselines
            for reg_entry in baseline_entries:
                fut = pool.submit(
                    _run_baseline_backtest,
                    reg_entry.name,
                    data,
                    initial_capital,
                    commission,
                    slippage_pct,
                )
                futures[fut] = ("baseline", reg_entry.name)

            # Submit hybrids
            for hybrid_entry in hybrid_entries:
                fut = pool.submit(
                    _run_hybrid_backtest,
                    hybrid_entry,
                    data,
                    initial_capital,
                    commission,
                    slippage_pct,
                )
                futures[fut] = ("hybrid", hybrid_entry.name)

            # Collect results as they complete
            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                kind, name = futures[fut]
                try:
                    entry = fut.result()
                    if entry is not None:
                        entries.append(entry)
                        self._out(
                            f"  [{done_count}/{total}] {entry.strategy_name}: "
                            f"P/L=${float(entry.result.total_pnl):,.2f}  "
                            f"WR={float(entry.result.win_rate):.1f}%  "
                            f"Score={entry.composite_score:.1f}"
                        )
                    else:
                        self._out(f"  [{done_count}/{total}] {name}: SKIPPED")
                except Exception as exc:
                    self._out(f"  [{done_count}/{total}] {name}: ERROR ({exc})")

        # 3. Rank all by composite score
        entries.sort(key=lambda e: e.composite_score, reverse=True)

        # 4. Walk-forward validation on top performers (also parallelised)
        top_for_wf = [
            e for e in entries
            if e.result.total_trades >= 5
        ][: self._cfg.top_for_walk_forward]

        if top_for_wf:
            # True WF with model retraining for hybrids (if enabled)
            if self._cfg.true_walk_forward:
                hybrid_wf = [e for e in top_for_wf if e.is_hybrid and e._label_config is not None]
                baseline_wf = [e for e in top_for_wf if not e.is_hybrid]
                non_retrain_hybrid = [e for e in top_for_wf if e.is_hybrid and e._label_config is None]

                if hybrid_wf:
                    self._out(
                        f"\nTrue walk-forward (ML retraining) on "
                        f"{len(hybrid_wf)} hybrid strategies..."
                    )
                    for entry in hybrid_wf:
                        try:
                            self._true_walk_forward(
                                entry, data, initial_capital, commission,
                                slippage_pct, data_result,
                            )
                        except Exception as exc:
                            logger.warning(
                                "True WF failed for %s: %s",
                                entry.strategy_name, exc,
                            )

                # Standard WF for baselines + hybrids without training metadata
                std_wf = baseline_wf + non_retrain_hybrid
                if std_wf:
                    self._out(
                        f"\nStandard walk-forward on {len(std_wf)} "
                        f"baseline strategies ({max_workers} workers)..."
                    )
                    self._parallel_walk_forward(
                        std_wf, data, initial_capital, commission,
                        slippage_pct, max_workers,
                    )
            else:
                self._out(
                    f"\nWalk-forward validation on top "
                    f"{len(top_for_wf)} strategies ({max_workers} workers)..."
                )
                self._parallel_walk_forward(
                    top_for_wf, data, initial_capital, commission,
                    slippage_pct, max_workers,
                )

        # Re-sort after walk-forward (penalise high degradation)
        entries.sort(
            key=lambda e: self._adjusted_score(e), reverse=True,
        )

        return BacktestStageResult(entries=entries, ranked=entries)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(result: object) -> float:
        """Composite score: Sharpe + win rate + PF − drawdown."""
        sharpe = float(result.sharpe_ratio(252))  # type: ignore[attr-defined]
        wr = float(result.win_rate) / 100.0  # type: ignore[attr-defined]
        pf = min(float(result.profit_factor), 5.0)  # type: ignore[attr-defined]
        dd = float(result.max_drawdown) / 100.0  # type: ignore[attr-defined]
        trades = result.total_trades  # type: ignore[attr-defined]

        score = (sharpe * 40.0) + (wr * 20.0) + (pf * 20.0) - (dd * 20.0)

        # Penalise very few trades
        if trades < 5:
            score -= 100.0
        elif trades < 20:
            score -= (20 - trades) * 3.0

        return score

    @staticmethod
    def _adjusted_score(entry: BacktestEntry) -> float:
        """Score adjusted for walk-forward degradation."""
        base = entry.composite_score
        if entry.walk_forward_degradation is not None:
            # Penalise if OOS is much worse than IS (degradation > 1.0)
            deg = entry.walk_forward_degradation
            if deg > 1.5:
                base -= (deg - 1.0) * 20.0
        return base

    # ------------------------------------------------------------------
    # Walk-forward validation (parallelised)
    # ------------------------------------------------------------------

    def _parallel_walk_forward(
        self,
        top_entries: list[BacktestEntry],
        data: list,
        initial_capital: Decimal,
        commission: Decimal,
        slippage_pct: Decimal,
        max_workers: int,
    ) -> None:
        """Run walk-forward validation for all top entries concurrently."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for entry in top_entries:
                fut = pool.submit(
                    self._walk_forward,
                    entry,
                    data,
                    initial_capital,
                    commission,
                    slippage_pct,
                )
                futures[fut] = entry

            for fut in as_completed(futures):
                entry = futures[fut]
                try:
                    deg = fut.result()
                    entry.walk_forward_degradation = deg
                    if deg is not None:
                        self._out(
                            f"  {entry.strategy_name}: "
                            f"WF degradation={deg:.2f}"
                        )
                except Exception as exc:
                    logger.warning(
                        "Walk-forward failed for %s: %s",
                        entry.strategy_name, exc,
                    )

    def _walk_forward(
        self,
        entry: BacktestEntry,
        data: list,
        initial_capital: Decimal,
        commission: Decimal,
        slippage_pct: Decimal = Decimal("0"),
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
        n_windows = self._cfg.walk_forward_windows
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
                    is_score = self._compute_score(is_result)

                    if hasattr(strat, "_prob_cache"):
                        strat._prob_cache.clear()
                    oos_result = engine.run(strat, oos_data)
                    oos_score = self._compute_score(oos_result)

                    is_scores.append(is_score)
                    oos_scores.append(oos_score)
                else:
                    # For baselines, run both IS and OOS
                    strat = self._recreate_strategy(entry)
                    if strat is None:
                        return None
                    is_result = engine.run(strat, is_data)
                    is_score = self._compute_score(is_result)

                    strat2 = self._recreate_strategy(entry)
                    if strat2 is None:
                        return None
                    oos_result = engine.run(strat2, oos_data)
                    oos_score = self._compute_score(oos_result)

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
    # True walk-forward with ML model retraining
    # ------------------------------------------------------------------

    def _true_walk_forward(
        self,
        entry: BacktestEntry,
        data: list,
        initial_capital: Decimal,
        commission: Decimal,
        slippage_pct: Decimal,
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
        from stockdownloader.ml.pipeline.stage_hybrid_strategies import (
            _STRATEGY_CLASS,
        )
        from stockdownloader.ml.trainer import MLTrainer
        from stockdownloader.strategy.registration_loader import ensure_registered
        from stockdownloader.strategy.base_registry import StrategyRegistry
        from stockdownloader.util.indicator_hub import IndicatorHub

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
        n_windows = self._cfg.walk_forward_windows
        min_train = self._cfg.wf_min_train_bars

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
                is_score = self._compute_score(is_result)

                # Clear cache before OOS run
                if hasattr(hybrid, "_prob_cache"):
                    hybrid._prob_cache.clear()

                oos_result = engine.run(hybrid, oos_data)
                oos_score = self._compute_score(oos_result)

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

                self._out(
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

            self._out(
                f"  → {entry.strategy_name}: TRUE WF "
                f"IS={avg_is:.1f} OOS={avg_oos:.1f} "
                f"deg={entry.walk_forward_degradation:.2f} "
                f"({len(window_results)} windows)"
            )
        else:
            self._out(
                f"  → {entry.strategy_name}: TRUE WF — no valid windows"
            )

    @staticmethod
    def _recreate_strategy(entry: BacktestEntry) -> object | None:
        """Re-create a baseline strategy from its registry name."""
        if entry.is_hybrid:
            return None  # Can't recreate hybrids here
        try:
            from stockdownloader.strategy.registration_loader import ensure_registered
            from stockdownloader.strategy.base_registry import StrategyRegistry

            ensure_registered()
            # Try to find the matching registry name
            for reg in StrategyRegistry.all_entries(category="daily"):
                strat = StrategyRegistry.create(reg.name)
                if strat.name == entry.strategy_name:
                    return strat
        except (KeyError, ValueError, ImportError):
            pass
        return None
