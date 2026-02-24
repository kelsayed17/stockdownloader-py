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
from stockdownloader.ml.pipeline.walk_forward import (
    parallel_walk_forward as _parallel_walk_forward_impl,
    recreate_strategy,
    true_walk_forward as _true_walk_forward_impl,
    walk_forward as _walk_forward_impl,
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
        _parallel_walk_forward_impl(
            self._cfg, top_entries, data, initial_capital, commission,
            slippage_pct, max_workers, self._compute_score, self._out,
        )

    def _walk_forward(
        self,
        entry: BacktestEntry,
        data: list,
        initial_capital: Decimal,
        commission: Decimal,
        slippage_pct: Decimal = Decimal("0"),
    ) -> float | None:
        """Simple temporal walk-forward: split data into windows."""
        return _walk_forward_impl(
            self._cfg, entry, data, initial_capital, commission,
            slippage_pct, self._compute_score,
        )

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
        """True walk-forward: retrain ML model on each IS window, test on OOS."""
        _true_walk_forward_impl(
            self._cfg, entry, data, initial_capital, commission,
            slippage_pct, self._compute_score, self._out, data_result,
        )

    @staticmethod
    def _recreate_strategy(entry: BacktestEntry) -> object | None:
        """Re-create a baseline strategy from its registry name."""
        return recreate_strategy(entry)
