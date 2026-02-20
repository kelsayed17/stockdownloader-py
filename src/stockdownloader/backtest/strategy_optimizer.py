"""Automated strategy parameter optimizer for intraday backtesting.

Optimises each standalone intraday strategy individually by searching
over its ``param_space`` (registered in :mod:`registrations`).  Also
runs daily strategies via :class:`DailyToIntradayAdapter` for an
all-strategy comparison.

The optimizer uses the greedy sequential search from
:class:`OptimizerBase`: for each parameter, try every candidate value
while holding the others at their current best, then combine the
per-parameter winners.

Usage::

    from stockdownloader.backtest.strategy_optimizer import StrategyOptimizer

    optimizer = StrategyOptimizer(data, initial_capital=Decimal("100000"))
    results = optimizer.optimize()          # per-strategy optimisation
    ranked  = optimizer.optimize_all_strategies()  # + daily comparison
"""
from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from decimal import Decimal
from operator import itemgetter
from typing import Any, TextIO

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_base import OptimizerBase
from stockdownloader.backtest.optimizer_scoring import score_v2 as _score
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.strategy.registrations import ensure_registered
from stockdownloader.strategy.registry import StrategyRegistry

logger = logging.getLogger(__name__)


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


class StrategyOptimizer(OptimizerBase):
    """Searches for optimal parameters across all intraday strategies.

    For each intraday strategy registered in :class:`StrategyRegistry`,
    runs a greedy sequential parameter search using its ``param_space``.

    Parameters
    ----------
    data:
        Intraday bar data (5-minute).
    initial_capital:
        Starting capital (default: $100,000).
    risk_per_trade:
        Risk per trade as a decimal (default: 0.01 = 1%).
    verbose:
        Print progress to stdout (default: True).
    log_file:
        Optional writable file object for tailable output.
    """

    def __init__(
        self,
        data: list[IntradayPriceData],
        initial_capital: Decimal = Decimal("100000"),
        risk_per_trade: Decimal = Decimal("0.01"),
        verbose: bool = True,
        log_file: TextIO | None = None,
    ) -> None:
        super().__init__(data, initial_capital, risk_per_trade, verbose, log_file)
        ensure_registered()
        self._history: list[dict[str, Any]] = []

    def _make_run_fn(
        self,
        default_config: Any,
        strategy_name: str,
    ) -> Callable[[dict[str, Any]], BacktestResult | None]:
        """Build a ``run_fn`` callback for :meth:`_greedy_search`.

        The returned callable constructs a strategy by replacing fields on
        *default_config* with the trial overrides, runs a backtest, and
        records the trial to ``self._history``.
        """
        # Capture *entry* from the calling scope via *default_config*.
        # We need a stable reference to self._data, self._capital, etc.
        # but those are instance attributes so ``self`` suffices.
        registry_entry = next(
            e for e in StrategyRegistry.all_entries(category="intraday")
            if e.display_name == strategy_name
        )

        def run_fn(overrides: dict[str, Any]) -> BacktestResult | None:
            try:
                trial_config = dataclasses.replace(default_config, **overrides)
                strategy = registry_entry.factory(config=trial_config)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.debug("Invalid config for %s: %s", strategy_name, exc)
                return None
            result = _run_backtest(
                strategy, self._data, self._capital, self._risk,
            )
            # Record to optimisation history
            score = _score(result, trading_days=self._trading_days)
            for param, val in overrides.items():
                self._history.append({
                    "run": self._run_count,
                    "strategy": strategy_name,
                    "param": param,
                    "value": val,
                    "score": score,
                    "pnl": float(result.total_pnl),
                })
            return result

        return run_fn

    def optimize(
        self,
    ) -> list[tuple[str, BacktestResult, float]]:
        """Run per-strategy optimisation for all intraday strategies.

        Returns
        -------
        list[tuple[str, BacktestResult, float]]
            ``(strategy_display_name, best_result, best_score)`` for each
            intraday strategy, sorted by score descending.
        """
        start = time.time()

        self._print_banner("STRATEGY OPTIMIZER — per-strategy mode")

        results: list[tuple[str, BacktestResult, float]] = []

        for entry in StrategyRegistry.all_entries(category="intraday"):
            if not entry.param_space:
                # Nothing to optimise — just run baseline
                self._print(f"  {entry.display_name}: no param_space, running baseline")
                strategy = entry.factory(**entry.default_kwargs)
                result = _run_backtest(
                    strategy, self._data, self._capital, self._risk,
                )
                score = _score(result, trading_days=self._trading_days)
                results.append((entry.display_name, result, score))
                self._print(
                    f"    P/L: ${result.total_pnl:>9,.2f}  "
                    f"WR: {result.win_rate:>5.1f}%  "
                    f"Trades: {result.total_trades:>3d}  "
                    f"Score: {score:>7.2f}"
                )
                self._print()
                continue

            self._print("-" * 70)
            self._print(f"  Optimising: {entry.display_name}")
            self._print("-" * 70)

            # Baseline — sets _best_score / _best_result
            baseline_strategy = entry.factory(**entry.default_kwargs)
            baseline_result = _run_backtest(
                baseline_strategy, self._data, self._capital, self._risk,
            )
            self._set_baseline(baseline_result)

            # Build run_fn that constructs strategy via config replacement
            default_config = entry.factory()._infra._c
            run_fn = self._make_run_fn(default_config, entry.display_name)

            # Delegate to base class greedy search
            self._greedy_search(
                param_space=entry.param_space,
                current_best={},
                run_fn=run_fn,
                phase_label=entry.display_name,
            )

            self._print_comparison(baseline_result, self._best_result)
            results.append((entry.display_name, self._best_result, self._best_score))
            self._print()

        self._print_summary(time.time() - start)

        results.sort(key=itemgetter(2), reverse=True)
        return results

    def optimize_all_strategies(
        self,
    ) -> list[tuple[str, BacktestResult, float]]:
        """Run all strategies and return ranked results.

        First optimises each intraday strategy, then runs each daily
        strategy via :class:`DailyToIntradayAdapter`, and returns
        results sorted by score (highest first).

        Returns
        -------
        list[tuple[str, BacktestResult, float]]
            ``(strategy_name, result, score)`` sorted best-first.
        """
        from stockdownloader.strategy.intraday.daily_to_intraday_adapter import (
            DailyToIntradayAdapter,
        )

        # Step 1: Optimise intraday strategies
        results = self.optimize()

        # Step 2: Run daily strategies via adapter
        self._print()
        self._print("=" * 70)
        self._print("  ALL-STRATEGY COMPARISON")
        self._print("=" * 70)
        self._print()

        engine = IntradayBacktestEngine(
            self._capital, self._risk,
            vol_scale=False, dd_throttle=True,
        )

        for entry in StrategyRegistry.all_entries(category="daily"):
            name = entry.display_name
            self._print(f"  Running: {name}...")
            try:
                daily = StrategyRegistry.create(entry.name)
                adapter = DailyToIntradayAdapter(daily)
                result = engine.run(adapter, self._data)
                score = _score(result, trading_days=self._trading_days)
                results.append((name, result, score))

                pnl = result.total_pnl
                sign = "+" if pnl >= 0 else ""
                self._print(
                    f"    P/L: {sign}${pnl:>9,.2f}  WR: {result.win_rate:>5.1f}%  "
                    f"Trades: {result.total_trades:>3d}  Score: {score:>7.2f}"
                )
            except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
                logger.warning("Strategy %s failed: %s", name, exc, exc_info=True)
                self._print(f"    ERROR: {exc}")

        # Sort by score descending
        results.sort(key=itemgetter(2), reverse=True)

        # Print ranked table
        self._print()
        self._print("-" * 70)
        self._print(
            f"  {'Rank':<5s} {'Strategy':<35s} {'P/L':>12s}  {'WR':>6s}  "
            f"{'Trades':>6s}  {'Score':>7s}"
        )
        self._print("-" * 70)

        for rank, (name, result, score) in enumerate(results, 1):
            pnl = result.total_pnl
            sign = "+" if pnl >= 0 else ""
            self._print(
                f"  {rank:<5d} {name:<35s} {sign}${pnl:>10,.2f}  "
                f"{result.win_rate:>5.1f}%  {result.total_trades:>6d}  "
                f"{score:>7.2f}"
            )

        self._print("-" * 70)
        self._print()

        return results
