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
from decimal import Decimal
from operator import itemgetter
from typing import Any, TextIO

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_base import OptimizerBase
from stockdownloader.backtest.optimizer_scoring import score as _score
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
    engine = IntradayBacktestEngine(initial_capital, risk_per_trade)
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

        self._print("=" * 70)
        self._print("  STRATEGY OPTIMIZER — per-strategy mode")
        self._print("=" * 70)
        self._print()

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

            # Baseline
            baseline_strategy = entry.factory(**entry.default_kwargs)
            baseline_result = _run_backtest(
                baseline_strategy, self._data, self._capital, self._risk,
            )
            baseline_score = _score(baseline_result, trading_days=self._trading_days)
            self._run_count += 1

            self._print(
                f"  Baseline: P/L: ${baseline_result.total_pnl:>9,.2f}  "
                f"WR: {baseline_result.win_rate:>5.1f}%  "
                f"Trades: {baseline_result.total_trades:>3d}  "
                f"Score: {baseline_score:>7.2f}"
            )

            # Reset best tracking for this strategy
            self._best_score = baseline_score
            self._best_result = baseline_result
            best_overrides: dict[str, Any] = {}

            # Greedy sequential search
            for param, values in entry.param_space.items():
                param_best_score = self._best_score
                param_best_val = None  # current default

                for val in values:
                    trial_overrides = best_overrides | {param: val}
                    try:
                        default_strategy = entry.factory()
                        default_config = default_strategy._infra._c
                        trial_config = dataclasses.replace(
                            default_config, **trial_overrides,
                        )
                        strategy = entry.factory(config=trial_config)
                    except (TypeError, ValueError, AttributeError) as exc:
                        logger.debug(
                            "Skipping %s=%s for %s: %s",
                            param, val, entry.display_name, exc,
                        )
                        continue

                    result = _run_backtest(
                        strategy, self._data, self._capital, self._risk,
                    )
                    self._run_count += 1
                    score = _score(result, trading_days=self._trading_days)

                    improved = ""
                    if score > param_best_score:
                        param_best_score = score
                        param_best_val = val
                        improved = " *"

                    pnl = result.total_pnl
                    sign = "+" if pnl >= 0 else ""
                    self._print(
                        f"    {param}={str(val):<15s} "
                        f"P/L: {sign}${pnl:>9,.2f}  WR: {result.win_rate:>5.1f}%  "
                        f"Trades: {result.total_trades:>3d}  "
                        f"Score: {score:>7.2f}{improved}"
                    )

                    self._history.append({
                        "run": self._run_count,
                        "strategy": entry.display_name,
                        "param": param,
                        "value": val,
                        "score": score,
                        "pnl": float(pnl),
                    })

                if param_best_val is not None:
                    best_overrides[param] = param_best_val
                    self._print(f"    >>> Winner: {param}={param_best_val}")

            # Apply combined winners
            if best_overrides:
                try:
                    default_strategy = entry.factory()
                    default_config = default_strategy._infra._c
                    combined_config = dataclasses.replace(
                        default_config, **best_overrides,
                    )
                    combined_strategy = entry.factory(config=combined_config)
                    combined_result = _run_backtest(
                        combined_strategy, self._data, self._capital, self._risk,
                    )
                    combined_score = _score(
                        combined_result, trading_days=self._trading_days,
                    )
                    self._run_count += 1

                    if combined_score >= self._best_score:
                        self._best_score = combined_score
                        self._best_result = combined_result
                        self._print(f"  >>> Combined improved: {best_overrides}")
                    else:
                        self._print(
                            f"  Combined ({combined_score:.2f}) did not beat "
                            f"baseline ({self._best_score:.2f})"
                        )
                except (TypeError, ValueError) as exc:
                    self._print(f"  Combined config failed: {exc}")

            self._print_comparison(baseline_result, self._best_result)
            results.append((entry.display_name, self._best_result, self._best_score))
            self._print()

        elapsed = time.time() - start

        self._print("=" * 70)
        self._print("  OPTIMISATION COMPLETE")
        self._print("=" * 70)
        self._print(f"  Total configurations tested: {self._run_count}")
        self._print(f"  Time elapsed: {elapsed:.1f}s")
        self._print()

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
        from stockdownloader.strategy.daily_to_intraday_adapter import (
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

        engine = IntradayBacktestEngine(self._capital, self._risk)

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
            except Exception as exc:
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
