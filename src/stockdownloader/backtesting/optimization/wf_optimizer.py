"""Walk-forward aware optimizer — searches IS data, validates on OOS.

Prevents overfitting by splitting data into in-sample (IS) and
out-of-sample (OOS) windows.  Parameters are optimized on IS data
only, then validated on OOS.  Only params that improve OOS performance
over the baseline are accepted.

Usage::

    from stockdownloader.backtesting.optimization.wf_optimizer import (
        WalkForwardOptimizer, WFOptResult,
    )

    wf_opt = WalkForwardOptimizer(data, split_ratio=0.7)
    results = wf_opt.optimize_all()
    for r in results:
        print(f"{r.display_name}: accepted={r.accepted}, OOS P/L={r.optimized_oos.total_pnl}")
"""
from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, TextIO

from stockdownloader.backtesting.results.result import BacktestResult
from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.backtesting.optimization.base import OptimizerBase
from stockdownloader.backtesting.optimization.scoring import score_v2 as _score
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.strategies.loader import ensure_registered
from stockdownloader.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class WFOptResult:
    """Result of walk-forward optimization for one strategy."""

    name: str
    display_name: str
    category: str
    baseline_is: BacktestResult | None = None
    baseline_oos: BacktestResult | None = None
    optimized_kwargs: dict[str, Any] = field(default_factory=dict)
    optimized_is: BacktestResult | None = None
    optimized_oos: BacktestResult | None = None
    accepted: bool = False


def _run_backtest(
    strategy: Any,
    data: list[IntradayPriceData],
    initial_capital: Decimal,
    risk_per_trade: Decimal,
) -> BacktestResult:
    """Run a single backtest with an already-constructed strategy."""
    engine = IntradayBacktestEngine(
        initial_capital, risk_per_trade,
        vol_scale=False, dd_throttle=True,
    )
    return engine.run(strategy, data)


class WalkForwardOptimizer(OptimizerBase):
    """Optimizer that searches on IS data, validates on OOS.

    Prevents overfitting by:
    1. Splitting data into IS (first ``split_ratio``) and OOS (remainder)
    2. Running greedy param search on IS data only
    3. Validating optimized params on OOS data
    4. Accepting params only if OOS score improves over baseline

    Parameters
    ----------
    data:
        Full intraday bar data (5-minute).
    split_ratio:
        Fraction of data for in-sample (default 0.7 = 70/30 split).
    initial_capital:
        Starting capital.
    risk_per_trade:
        Risk per trade as a decimal (e.g. 0.01 = 1%).
    verbose:
        Print progress to stdout.
    log_file:
        Optional writable file object for tailable output.
    """

    def __init__(
        self,
        data: list[IntradayPriceData],
        split_ratio: float = 0.7,
        initial_capital: Decimal = Decimal("100000"),
        risk_per_trade: Decimal = Decimal("0.01"),
        verbose: bool = True,
        log_file: TextIO | None = None,
    ) -> None:
        split = int(len(data) * split_ratio)
        self._is_data = data[:split]
        self._oos_data = data[split:]
        self._full_data = data

        # Base class uses IS data for _trading_days and _data
        super().__init__(
            data=self._is_data,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            verbose=verbose,
            log_file=log_file,
        )
        ensure_registered()

        self._oos_trading_days = len(
            {bar.date[:10] for bar in self._oos_data}
        ) if self._oos_data else 0

        is_days = self._trading_days
        oos_days = self._oos_trading_days
        self._print(f"  IS: {len(self._is_data):,} bars ({is_days} days)")
        self._print(f"  OOS: {len(self._oos_data):,} bars ({oos_days} days)")

    def optimize_all(
        self,
        categories: list[str] | None = None,
        strategy_filter: str | None = None,
    ) -> list[WFOptResult]:
        """Optimize all registered strategies with OOS gating.

        Parameters
        ----------
        categories:
            List of categories to optimize (default: intraday + daily).
        strategy_filter:
            If set, optimize only this strategy by name.

        Returns
        -------
        list[WFOptResult]
            Results for each strategy, with ``accepted=True`` if OOS improved.
        """
        cats = categories or ["intraday", "daily"]
        results: list[WFOptResult] = []

        self._print()
        self._print("=" * 70)
        self._print("  WALK-FORWARD OPTIMIZER")
        self._print("  Searching IS data only, validating on OOS")
        self._print("=" * 70)
        self._print()

        t_start = time.time()

        for cat in cats:
            entries = StrategyRegistry.all_entries(category=cat)
            if strategy_filter:
                entries = [e for e in entries if e.name == strategy_filter]

            for entry in entries:
                if not entry.param_space:
                    self._print(f"  {entry.display_name}: no param_space, skipping")
                    continue

                self._print("-" * 70)
                self._print(f"  Optimizing: {entry.display_name} ({cat})")
                self._print("-" * 70)

                t0 = time.time()

                if cat == "intraday":
                    result = self._optimize_intraday(entry)
                elif cat == "daily":
                    result = self._optimize_daily(entry)
                else:
                    continue

                elapsed = time.time() - t0
                status = "ACCEPTED" if result.accepted else "REJECTED (OOS worse)"
                self._print(f"  => {status} ({elapsed:.1f}s)")
                self._print()
                results.append(result)

        total = time.time() - t_start
        accepted = sum(1 for r in results if r.accepted)
        self._print("=" * 70)
        self._print(f"  WALK-FORWARD OPTIMIZATION COMPLETE")
        self._print(f"  {accepted}/{len(results)} strategies accepted")
        self._print(f"  Time: {total:.1f}s")
        self._print("=" * 70)

        return results

    # ------------------------------------------------------------------
    # Intraday strategy optimization
    # ------------------------------------------------------------------

    def _optimize_intraday(self, entry) -> WFOptResult:
        """Optimize an intraday strategy on IS, validate on OOS."""
        result = WFOptResult(
            name=entry.name,
            display_name=entry.display_name,
            category="intraday",
        )

        # Baseline on IS and OOS
        baseline_strategy = entry.factory(**entry.default_kwargs)
        default_config = baseline_strategy._infra._c

        result.baseline_is = _run_backtest(
            entry.factory(**entry.default_kwargs),
            self._is_data, self._capital, self._risk,
        )
        result.baseline_oos = _run_backtest(
            entry.factory(**entry.default_kwargs),
            self._oos_data, self._capital, self._risk,
        )

        baseline_is_score = _score(result.baseline_is, self._trading_days)
        baseline_oos_score = _score(result.baseline_oos, self._oos_trading_days)

        self._print(
            f"  Baseline IS:  ${result.baseline_is.total_pnl:>9,.2f}  "
            f"({result.baseline_is.total_trades} trades)  Score: {baseline_is_score:.1f}"
        )
        self._print(
            f"  Baseline OOS: ${result.baseline_oos.total_pnl:>9,.2f}  "
            f"({result.baseline_oos.total_trades} trades)  Score: {baseline_oos_score:.1f}"
        )

        # Greedy search on IS data only
        self._best_score = baseline_is_score
        self._best_result = result.baseline_is

        def _run_fn(overrides: dict[str, Any]) -> BacktestResult | None:
            try:
                trial_config = dataclasses.replace(default_config, **overrides)
                strategy = entry.factory(config=trial_config)
            except (TypeError, ValueError, AttributeError):
                return None
            return _run_backtest(strategy, self._is_data, self._capital, self._risk)

        winners = self._greedy_search(
            param_space=entry.param_space,
            current_best={},
            run_fn=_run_fn,
            phase_label=entry.display_name,
        )

        if not winners:
            self._print("  No improvement found on IS data")
            return result

        result.optimized_kwargs = winners

        # Validate on OOS
        try:
            opt_config = dataclasses.replace(default_config, **winners)
            opt_strategy_is = entry.factory(config=opt_config)
            result.optimized_is = _run_backtest(
                opt_strategy_is, self._is_data, self._capital, self._risk,
            )
            opt_strategy_oos = entry.factory(config=opt_config)
            result.optimized_oos = _run_backtest(
                opt_strategy_oos, self._oos_data, self._capital, self._risk,
            )
        except (ValueError, ZeroDivisionError, ArithmeticError) as e:
            self._print(f"  OOS validation failed: {e}")
            return result

        opt_oos_score = _score(result.optimized_oos, self._oos_trading_days)

        self._print(
            f"  Optimized IS:  ${result.optimized_is.total_pnl:>9,.2f}  "
            f"({result.optimized_is.total_trades} trades)"
        )
        self._print(
            f"  Optimized OOS: ${result.optimized_oos.total_pnl:>9,.2f}  "
            f"({result.optimized_oos.total_trades} trades)  Score: {opt_oos_score:.1f}"
        )

        # Accept only if OOS improved
        result.accepted = opt_oos_score > baseline_oos_score
        return result

    # ------------------------------------------------------------------
    # Daily strategy optimization
    # ------------------------------------------------------------------

    def _optimize_daily(self, entry) -> WFOptResult:
        """Optimize a daily strategy (via adapter) on IS, validate on OOS."""
        from stockdownloader.strategies.intraday.daily_adapter import (
            DailyToIntradayAdapter,
        )

        result = WFOptResult(
            name=entry.name,
            display_name=entry.display_name,
            category="daily",
        )

        default_kwargs = dict(entry.default_kwargs)
        adapter_kwargs = {
            "sl_atr_mult": Decimal("1.5"),
            "rr": Decimal("1.5"),
            "sl_cap": Decimal("2.00"),
            "allow_shorts": False,
        }

        adapter_param_space = {
            "sl_atr_mult": [Decimal("0.8"), Decimal("1.0"), Decimal("1.3"),
                            Decimal("1.5"), Decimal("1.8")],
            "rr": [Decimal("1.0"), Decimal("1.2"), Decimal("1.5"), Decimal("2.0")],
            "sl_cap": [Decimal("1.00"), Decimal("1.50"), Decimal("2.00"),
                       Decimal("2.50")],
            "allow_shorts": [True, False],
        }

        def _build_and_run_is(
            strat_kw: dict[str, Any], adapt_kw: dict[str, Any],
        ) -> BacktestResult | None:
            try:
                strategy = entry.factory(**strat_kw)
            except (ValueError, TypeError):
                return None
            adapter = DailyToIntradayAdapter(strategy, **adapt_kw)
            return _run_backtest(adapter, self._is_data, self._capital, self._risk)

        def _build_and_run_oos(
            strat_kw: dict[str, Any], adapt_kw: dict[str, Any],
        ) -> BacktestResult | None:
            try:
                strategy = entry.factory(**strat_kw)
            except (ValueError, TypeError):
                return None
            adapter = DailyToIntradayAdapter(strategy, **adapt_kw)
            return _run_backtest(adapter, self._oos_data, self._capital, self._risk)

        # Baseline
        result.baseline_is = _build_and_run_is(default_kwargs, adapter_kwargs)
        result.baseline_oos = _build_and_run_oos(default_kwargs, adapter_kwargs)

        if result.baseline_is is None or result.baseline_oos is None:
            self._print("  Baseline failed")
            return result

        baseline_is_score = _score(result.baseline_is, self._trading_days)
        baseline_oos_score = _score(result.baseline_oos, self._oos_trading_days)

        self._print(
            f"  Baseline IS:  ${result.baseline_is.total_pnl:>9,.2f}  "
            f"({result.baseline_is.total_trades} trades)  Score: {baseline_is_score:.1f}"
        )
        self._print(
            f"  Baseline OOS: ${result.baseline_oos.total_pnl:>9,.2f}  "
            f"({result.baseline_oos.total_trades} trades)  Score: {baseline_oos_score:.1f}"
        )

        # Phase 1: Strategy params on IS
        self._best_score = baseline_is_score
        self._best_result = result.baseline_is
        best_strat_kw = dict(default_kwargs)

        if entry.param_space:
            strat_winners = self._greedy_search(
                param_space=entry.param_space,
                current_best=best_strat_kw,
                run_fn=lambda kw: _build_and_run_is(kw, adapter_kwargs),
                phase_label="Phase 1 (strategy)",
            )
            if strat_winners:
                best_strat_kw = best_strat_kw | strat_winners

        # Phase 2: Adapter params on IS
        best_adapt_kw = dict(adapter_kwargs)
        adapt_winners = self._greedy_search(
            param_space=adapter_param_space,
            current_best=best_adapt_kw,
            run_fn=lambda kw: _build_and_run_is(best_strat_kw, kw),
            phase_label="Phase 2 (adapter)",
        )
        if adapt_winners:
            best_adapt_kw = best_adapt_kw | adapt_winners

        combined_kwargs = best_strat_kw | best_adapt_kw
        if combined_kwargs == (default_kwargs | adapter_kwargs):
            self._print("  No improvement found on IS data")
            return result

        result.optimized_kwargs = combined_kwargs

        # Validate on OOS
        result.optimized_is = _build_and_run_is(best_strat_kw, best_adapt_kw)
        result.optimized_oos = _build_and_run_oos(best_strat_kw, best_adapt_kw)

        if result.optimized_oos is None:
            self._print("  OOS validation failed")
            return result

        opt_oos_score = _score(result.optimized_oos, self._oos_trading_days)

        self._print(
            f"  Optimized IS:  ${result.optimized_is.total_pnl:>9,.2f}  "
            f"({result.optimized_is.total_trades} trades)"
        )
        self._print(
            f"  Optimized OOS: ${result.optimized_oos.total_pnl:>9,.2f}  "
            f"({result.optimized_oos.total_trades} trades)  Score: {opt_oos_score:.1f}"
        )

        # Accept only if OOS improved
        result.accepted = opt_oos_score > baseline_oos_score
        return result
