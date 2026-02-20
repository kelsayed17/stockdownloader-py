"""Per-strategy optimizer for any registry-registered daily strategy.

Uses the same greedy sequential search as :class:`StrategyOptimizer`, but
works with constructor kwargs instead of dataclass field replacement.  Each
daily strategy is wrapped in a :class:`DailyToIntradayAdapter` before
backtesting on intraday data.

Two optimization phases:

1. **Strategy-specific params** — from the registry ``param_space``
   (e.g. ``period``, ``oversold``, ``overbought`` for RSI).
2. **Adapter params** — ``sl_atr_mult``, ``rr``, ``sl_cap`` that control
   stop-loss and take-profit behavior.

Usage::

    from stockdownloader.backtest.daily_strategy_optimizer import DailyStrategyOptimizer

    opt = DailyStrategyOptimizer("rsi", data, capital, risk)
    best_kwargs, best_result = opt.optimize()
"""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, TextIO

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_base import OptimizerBase
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.base_registry import StrategyRegistry, StrategyEntry


# Default search space for adapter parameters.
_ADAPTER_PARAM_SPACE = {
    "sl_atr_mult": [Decimal("0.8"), Decimal("1.0"), Decimal("1.3"), Decimal("1.5"), Decimal("1.8")],
    "rr": [Decimal("1.0"), Decimal("1.2"), Decimal("1.5"), Decimal("2.0")],
    "sl_cap": [Decimal("1.00"), Decimal("1.50"), Decimal("2.00"), Decimal("2.50")],
    "allow_shorts": [True, False],
}

_DEFAULT_ADAPTER_KWARGS: dict[str, Any] = {
    "sl_atr_mult": Decimal("1.5"),
    "rr": Decimal("1.5"),
    "sl_cap": Decimal("2.00"),
    "allow_shorts": False,
}


class DailyStrategyOptimizer(OptimizerBase):
    """Optimizes parameters for any registry-registered daily strategy.

    Uses greedy sequential search (one parameter at a time, keeping the
    best value from each) across two phases.

    Parameters
    ----------
    strategy_name:
        Registry CLI name (e.g. ``"rsi"``, ``"macd"``).
    data:
        Intraday bar data (5-minute).
    initial_capital:
        Starting capital.
    risk_per_trade:
        Risk per trade as a decimal (e.g. 0.01 = 1%).
    adapter_kwargs:
        Optional overrides for adapter defaults (``sl_atr_mult``, ``rr``, etc.).
    allow_shorts:
        Initial setting for short-selling (default: False).  The optimizer
        will also test toggling this as part of the adapter param search.
    verbose:
        Print progress to stdout.
    log_file:
        Optional writable file object for tailable output.
    """

    def __init__(
        self,
        strategy_name: str,
        data: list[IntradayPriceData],
        initial_capital: Decimal = Decimal("100000"),
        risk_per_trade: Decimal = Decimal("0.01"),
        adapter_kwargs: dict[str, Any] | None = None,
        allow_shorts: bool = False,
        verbose: bool = True,
        log_file: TextIO | None = None,
    ) -> None:
        super().__init__(data, initial_capital, risk_per_trade, verbose, log_file)

        self._entry: StrategyEntry = StrategyRegistry.get(strategy_name)
        if self._entry.category != "daily":
            raise ValueError(
                f"Strategy '{strategy_name}' is category '{self._entry.category}', "
                "expected 'daily'."
            )

        # Current best state
        self._best_strategy_kwargs: dict[str, Any] = dict(self._entry.default_kwargs)
        self._best_adapter_kwargs: dict[str, Any] = adapter_kwargs or (
            _DEFAULT_ADAPTER_KWARGS | {"allow_shorts": allow_shorts}
        )

    def optimize(self) -> tuple[dict[str, Any], BacktestResult]:
        """Run the full two-phase optimization.

        Returns
        -------
        tuple[dict[str, Any], BacktestResult]
            The best combined kwargs and the corresponding backtest result.
        """
        start = time.time()

        self._print_banner(f"DAILY STRATEGY OPTIMIZER: {self._entry.display_name}")

        # Baseline
        baseline_result = self._build_and_run(
            self._best_strategy_kwargs, self._best_adapter_kwargs
        )
        if baseline_result is None:
            self._print("ERROR: Baseline config produced no result.")
            raise RuntimeError("Cannot optimize — baseline config is invalid.")

        self._set_baseline(baseline_result)
        self._print()

        # Phase 1: Strategy-specific params
        if self._entry.param_space:
            self._print("-" * 70)
            self._print("  PHASE 1: Strategy Parameter Optimization")
            self._print("-" * 70)
            winners = self._greedy_search(
                param_space=self._entry.param_space,
                current_best=self._best_strategy_kwargs,
                run_fn=lambda kw: self._build_and_run(kw, self._best_adapter_kwargs),
                phase_label="Phase 1",
            )
            if winners:
                self._best_strategy_kwargs = self._best_strategy_kwargs | winners
        else:
            self._print("  (no strategy param_space — skipping Phase 1)")
            self._print()

        # Phase 2: Adapter params
        self._print("-" * 70)
        self._print("  PHASE 2: Adapter Parameter Optimization")
        self._print("-" * 70)
        winners = self._greedy_search(
            param_space=_ADAPTER_PARAM_SPACE,
            current_best=self._best_adapter_kwargs,
            run_fn=lambda kw: self._build_and_run(self._best_strategy_kwargs, kw),
            phase_label="Phase 2",
        )
        if winners:
            self._best_adapter_kwargs = self._best_adapter_kwargs | winners

        self._print_summary(time.time() - start)

        self._print_comparison(baseline_result, self._best_result)
        self._print_param_diff(self._entry.default_kwargs, self._best_strategy_kwargs, "Strategy")
        self._print_param_diff(
            _DEFAULT_ADAPTER_KWARGS,
            self._best_adapter_kwargs,
            "Adapter",
        )

        combined_kwargs = self._best_strategy_kwargs | self._best_adapter_kwargs
        return combined_kwargs, self._best_result

    # ------------------------------------------------------------------
    # Build + run
    # ------------------------------------------------------------------

    def _build_and_run(
        self,
        strategy_kwargs: dict[str, Any],
        adapter_kwargs: dict[str, Any],
    ) -> BacktestResult | None:
        """Create strategy + adapter, run backtest, return result or None."""
        try:
            strategy = self._entry.factory(**strategy_kwargs)
        except (ValueError, TypeError):
            return None  # invalid param combo (e.g. SMA short > long)

        adapter = DailyToIntradayAdapter(strategy, **adapter_kwargs)
        engine = IntradayBacktestEngine(self._capital, self._risk)
        return engine.run(adapter, self._data)
