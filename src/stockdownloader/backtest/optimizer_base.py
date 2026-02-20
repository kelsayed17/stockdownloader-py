"""Shared base class for strategy optimizers.

Provides common infrastructure for greedy sequential parameter search,
result tracking, logging, and comparison output used by both
:class:`StrategyOptimizer` (VWAP intraday) and
:class:`DailyStrategyOptimizer` (registry-based daily strategies).
"""
from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any, TextIO

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.optimizer_scoring import score_v2 as _score
from stockdownloader.model.intraday_price_data import IntradayPriceData


class OptimizerBase:
    """Common infrastructure for strategy optimizers.

    Provides: ``_print()``, ``_print_comparison()``, ``_run_count``,
    ``_best_score``, ``_best_result``, ``_trading_days``, and
    ``_evaluate_and_track()`` for score/best-update bookkeeping.

    Parameters
    ----------
    data:
        Intraday bar data (5-minute).
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
        initial_capital: Decimal = Decimal("100000"),
        risk_per_trade: Decimal = Decimal("0.01"),
        verbose: bool = True,
        log_file: TextIO | None = None,
    ) -> None:
        self._data = data
        self._capital = initial_capital
        self._risk = risk_per_trade
        self._verbose = verbose
        self._log_file = log_file
        self._trading_days = len({bar.date[:10] for bar in data}) if data else 0
        self._run_count = 0
        self._best_score: float = float("-inf")
        self._best_result: BacktestResult | None = None

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _print(self, msg: str = "") -> None:
        if self._verbose:
            print(msg, flush=True)
            if self._log_file is not None:
                self._log_file.write(msg + "\n")
                self._log_file.flush()

    def _print_comparison(
        self,
        baseline: BacktestResult,
        best: BacktestResult,
        extra_metrics: list[tuple[str, Decimal, Decimal, str]] | None = None,
    ) -> None:
        """Print before/after comparison table.

        Parameters
        ----------
        baseline:
            The baseline backtest result.
        best:
            The optimized backtest result.
        extra_metrics:
            Additional (name, base_val, best_val, unit) rows to append.
        """
        self._print(f"  {'':30s} {'BASELINE':>15s}  {'OPTIMIZED':>15s}  {'CHANGE':>15s}")
        self._print(f"  {'':30s} {'─' * 15}  {'─' * 15}  {'─' * 15}")

        metrics: list[tuple[str, Decimal, Decimal, str]] = [
            ("Total P/L", baseline.total_pnl, best.total_pnl, "$"),
            ("Total Return", baseline.total_return, best.total_return, "%"),
            ("Win Rate", baseline.win_rate, best.win_rate, "%"),
            ("Sharpe Ratio", baseline.sharpe_ratio(252 * 78), best.sharpe_ratio(252 * 78), ""),
            ("Profit Factor", baseline.profit_factor, best.profit_factor, ""),
            ("Max Drawdown", baseline.max_drawdown, best.max_drawdown, "%"),
            ("Total Trades", Decimal(str(baseline.total_trades)), Decimal(str(best.total_trades)), ""),
        ]
        if extra_metrics:
            metrics.extend(extra_metrics)

        for name, base_val, best_val, unit in metrics:
            diff = best_val - base_val
            sign = "+" if diff > 0 else ""
            match unit:
                case "$":
                    self._print(
                        f"  {name:<30s} ${base_val:>14,.2f}  ${best_val:>14,.2f}  {sign}${diff:>13,.2f}"
                    )
                case "%":
                    self._print(
                        f"  {name:<30s} {base_val:>14.2f}%  {best_val:>14.2f}%  {sign}{diff:>13.2f}%"
                    )
                case _:
                    self._print(
                        f"  {name:<30s} {base_val:>15.2f}  {best_val:>15.2f}  {sign}{diff:>14.2f}"
                    )
        self._print()

    def _print_param_diff(
        self,
        defaults: dict[str, Any],
        best: dict[str, Any],
        label: str,
    ) -> None:
        """Print only the parameters that changed."""
        self._print(f"  {label.upper()} PARAMETER CHANGES:")
        self._print(f"  {'Parameter':<25s} {'Default':>15s}  {'Optimized':>15s}")
        self._print(f"  {'─' * 25} {'─' * 15}  {'─' * 15}")

        changed = False
        for param in sorted(set(defaults) | set(best)):
            default_val = defaults.get(param, "—")
            best_val = best.get(param, "—")
            if str(default_val) != str(best_val):
                changed = True
                self._print(f"  {param:<25s} {str(default_val):>15s}  {str(best_val):>15s}")

        if not changed:
            self._print("  (no changes — defaults are already optimal)")
        self._print()

    # ------------------------------------------------------------------
    # Optimization lifecycle helpers
    # ------------------------------------------------------------------

    def _print_banner(self, title: str) -> None:
        """Print standard optimizer banner."""
        self._print("=" * 70)
        self._print(f"  {title}")
        self._print("=" * 70)
        self._print()

    def _set_baseline(self, result: BacktestResult) -> float:
        """Score *result*, set it as the best, print summary, return score."""
        score = _score(result, trading_days=self._trading_days)
        self._best_result = result
        self._best_score = score
        self._run_count += 1
        self._print(
            f"  Baseline: P/L: ${result.total_pnl:>9,.2f}  "
            f"WR: {result.win_rate:>5.1f}%  "
            f"Trades: {result.total_trades:>3d}  "
            f"Score: {score:>7.2f}"
        )
        return score

    def _print_summary(self, elapsed: float) -> None:
        """Print standard optimization-complete footer."""
        self._print()
        self._print("=" * 70)
        self._print("  OPTIMIZATION COMPLETE")
        self._print("=" * 70)
        self._print(f"  Total configurations tested: {self._run_count}")
        self._print(f"  Time elapsed: {elapsed:.1f}s")
        self._print()

    # ------------------------------------------------------------------
    # Greedy search helpers
    # ------------------------------------------------------------------

    def _greedy_search(
        self,
        param_space: dict[str, list[Any]],
        current_best: dict[str, Any],
        run_fn: Callable[[dict[str, Any]], BacktestResult | None],
        phase_label: str = "Phase",
    ) -> dict[str, Any]:
        """Greedy sequential search over a parameter grid.

        Tests each parameter independently against *current_best*, keeps
        the best value per parameter, then tries combining all winners.

        Parameters
        ----------
        param_space:
            ``{param_name: [val1, val2, ...]}`` to search.
        current_best:
            Current best parameter values (dict or will be read via callback).
        run_fn:
            Callable ``(overrides: dict) -> BacktestResult | None``.
            Receives the full kwargs dict and returns a result, or ``None``
            for invalid combos.
        phase_label:
            Label for output.

        Returns
        -------
        dict[str, Any]
            The winning parameter overrides from this search.
        """
        winners: dict[str, Any] = {}

        for param, values in param_space.items():
            param_best_score = self._best_score
            param_best_val = current_best.get(param)

            for val in values:
                if val == current_best.get(param):
                    continue

                trial_kwargs = current_best | winners | {param: val}
                result = run_fn(trial_kwargs)
                self._run_count += 1
                if result is None:
                    self._print(f"  {param}={val:<15s}  (invalid combo — skipped)")
                    continue

                s = _score(result, trading_days=self._trading_days)

                improved = ""
                if s > param_best_score:
                    param_best_score = s
                    param_best_val = val
                    improved = " ★"

                pnl = result.total_pnl
                sign = "+" if pnl >= 0 else ""
                self._print(
                    f"  [{self._run_count:3d}] {param}={str(val):<15s} "
                    f"P/L: {sign}${pnl:>9,.2f}  WR: {result.win_rate:>5.1f}%  "
                    f"Trades: {result.total_trades:>3d}  "
                    f"Score: {s:>7.2f}{improved}"
                )

            if param_best_val != current_best.get(param):
                winners[param] = param_best_val
                self._print(f"  >>> Winner: {param}={param_best_val}")

        # Apply all winners
        if winners:
            combined = current_best | winners
            result = run_fn(combined)
            self._run_count += 1
            if result is not None:
                s = _score(result, trading_days=self._trading_days)
                if s > self._best_score:
                    self._best_score = s
                    self._best_result = result
                    self._print(f"  >>> {phase_label} improved: {winners}")
                else:
                    # Individual winners don't combine well; try them one at a time
                    for param, val in winners.items():
                        trial = current_best | {param: val}
                        result = run_fn(trial)
                        self._run_count += 1
                        if result is not None:
                            s = _score(result, trading_days=self._trading_days)
                            if s > self._best_score:
                                self._best_score = s
                                self._best_result = result

        self._print()
        return winners
