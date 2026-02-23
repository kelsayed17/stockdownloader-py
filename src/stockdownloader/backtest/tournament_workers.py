"""Process-safe worker functions for tournament backtest execution.

Each function is designed to run safely in a multiprocessing pool:
no shared state, deferred imports for non-picklable objects, self-contained
error handling returning result tuples.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.backtest.tournament_models import ComboKey, ComboResult
from stockdownloader.backtest.walk_forward import WalkForwardResult, WalkForwardValidator
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.registration_loader import ensure_registered
from stockdownloader.strategy.base_registry import StrategyRegistry
from stockdownloader.util.config import INITIAL_CAPITAL, RISK_PER_TRADE


def run_combo_backtest(
    key: ComboKey,
    data: list[IntradayPriceData],
) -> ComboResult:
    """Run baseline backtest for one combo (process-safe).

    Parameters
    ----------
    key:
        The strategy × timeframe combo to test.
    data:
        The price data for this timeframe (already resampled).
    """
    ensure_registered()
    t0 = time.time()

    try:
        entry = StrategyRegistry.get(key.strategy_name)
    except KeyError:
        return ComboResult(
            key=key,
            error=f"Strategy {key.strategy_name!r} not found in registry",
            elapsed=time.time() - t0,
        )

    engine = IntradayBacktestEngine(
        INITIAL_CAPITAL, RISK_PER_TRADE,
        vol_scale=False, dd_throttle=True,
    )

    try:
        if entry.category == "intraday":
            strategy = entry.factory(**entry.default_kwargs)
            result = engine.run(strategy, data)
        elif entry.category == "daily":
            daily_strategy = entry.factory(**entry.default_kwargs)
            adapter = DailyToIntradayAdapter(daily_strategy)
            result = engine.run(adapter, data)
        else:
            return ComboResult(
                key=key,
                display_name=entry.display_name,
                category=entry.category,
                error=f"Unsupported category: {entry.category}",
                elapsed=time.time() - t0,
            )

        trading_days = len({d.date[:10] for d in data})
        combo = ComboResult(
            key=key,
            display_name=entry.display_name,
            category=entry.category,
            baseline=result,
            elapsed=time.time() - t0,
        )
        combo.compute_tournament_score(trading_days)
        return combo

    except Exception as e:
        return ComboResult(
            key=key,
            display_name=entry.display_name,
            category=entry.category,
            error=str(e),
            elapsed=time.time() - t0,
        )


def run_combo_optimize(
    key: ComboKey,
    category: str,
    data: list[IntradayPriceData],
) -> tuple[ComboKey, Any | None, float, str | None]:
    """Run walk-forward optimization for one combo (process-safe).

    Creates a fresh WalkForwardOptimizer instance per process, searches
    IS data only, accepts if OOS improves.

    Parameters
    ----------
    key:
        The strategy × timeframe combo to optimize.
    category:
        Strategy category (``"intraday"`` or ``"daily"``).
    data:
        The price data for this timeframe.

    Returns
    -------
    tuple of (key, WFOptResult | None, elapsed, error)
    """
    from stockdownloader.backtest.walk_forward_optimizer import (
        WalkForwardOptimizer,
    )

    ensure_registered()
    t0 = time.time()

    try:
        wf_opt = WalkForwardOptimizer(
            data,
            split_ratio=0.7,
            initial_capital=INITIAL_CAPITAL,
            risk_per_trade=RISK_PER_TRADE,
            verbose=False,
        )
        results = wf_opt.optimize_all(
            categories=[category],
            strategy_filter=key.strategy_name,
        )
        wf_result = results[0] if results else None
        return key, wf_result, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)


def run_combo_rebacktest(
    key: ComboKey,
    category: str,
    optimized_kwargs: dict[str, Any],
    data: list[IntradayPriceData],
) -> tuple[ComboKey, BacktestResult | None, float, str | None]:
    """Re-backtest one combo with optimized params on full data (process-safe).

    Parameters
    ----------
    key:
        The strategy × timeframe combo.
    category:
        Strategy category (``"intraday"`` or ``"daily"``).
    optimized_kwargs:
        Optimized parameter overrides from walk-forward optimization.
    data:
        The *full* price data for this timeframe.

    Returns
    -------
    tuple of (key, result, elapsed, error)
    """
    ensure_registered()
    t0 = time.time()

    engine = IntradayBacktestEngine(
        INITIAL_CAPITAL, RISK_PER_TRADE,
        vol_scale=False, dd_throttle=True,
    )

    try:
        entry = StrategyRegistry.get(key.strategy_name)

        if category == "intraday":
            baseline_strategy = entry.factory(**entry.default_kwargs)
            default_config = baseline_strategy._infra._c
            opt_config = dataclasses.replace(default_config, **optimized_kwargs)
            strategy = entry.factory(config=opt_config)
            result = engine.run(strategy, data)
        elif category == "daily":
            adapter_keys = {"sl_atr_mult", "rr", "sl_cap", "allow_shorts"}
            strat_kwargs = {
                k: v for k, v in optimized_kwargs.items()
                if k not in adapter_keys
            }
            adapt_kwargs = {
                k: v for k, v in optimized_kwargs.items()
                if k in adapter_keys
            }
            daily_strategy = entry.factory(**strat_kwargs)
            adapter = DailyToIntradayAdapter(daily_strategy, **adapt_kwargs)
            result = engine.run(adapter, data)
        else:
            return key, None, time.time() - t0, f"Unsupported category: {category}"

        return key, result, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)


def run_combo_walkforward(
    key: ComboKey,
    data: list[IntradayPriceData],
    optimized_kwargs: dict[str, Any] | None = None,
) -> tuple[ComboKey, WalkForwardResult | None, float, str | None]:
    """Run walk-forward validation for one combo (process-safe).

    Parameters
    ----------
    key:
        The strategy × timeframe combo.
    data:
        The price data for this timeframe.
    optimized_kwargs:
        If provided, validate using these optimized params instead
        of default kwargs.

    Returns
    -------
    tuple of (key, wf_result, elapsed, error)
    """
    ensure_registered()
    t0 = time.time()

    try:
        entry = StrategyRegistry.get(key.strategy_name)
    except KeyError:
        return key, None, time.time() - t0, f"Strategy not found: {key.strategy_name}"

    try:
        validator = WalkForwardValidator(data, n_windows=5, is_ratio=0.7)
        engine = IntradayBacktestEngine(
            INITIAL_CAPITAL, RISK_PER_TRADE,
            vol_scale=False, dd_throttle=True,
        )

        if entry.category == "intraday":
            if optimized_kwargs:
                def _factory(e=entry, kw=optimized_kwargs):
                    baseline = e.factory(**e.default_kwargs)
                    default_config = baseline._infra._c
                    opt_config = dataclasses.replace(default_config, **kw)
                    return e.factory(config=opt_config)
            else:
                def _factory(e=entry):
                    return e.factory(**e.default_kwargs)
        elif entry.category == "daily":
            if optimized_kwargs:
                adapter_keys = {"sl_atr_mult", "rr", "sl_cap", "allow_shorts"}
                strat_kw = {
                    k: v for k, v in optimized_kwargs.items()
                    if k not in adapter_keys
                }
                adapt_kw = {
                    k: v for k, v in optimized_kwargs.items()
                    if k in adapter_keys
                }

                def _factory(e=entry, sk=strat_kw, ak=adapt_kw):
                    daily = e.factory(**sk)
                    return DailyToIntradayAdapter(daily, **ak)
            else:
                def _factory(e=entry):
                    daily = e.factory(**e.default_kwargs)
                    return DailyToIntradayAdapter(daily)
        else:
            return key, None, time.time() - t0, "unsupported category"

        wf = validator.validate(
            strategy_factory=_factory,
            engine=engine,
            strategy_name=entry.display_name,
        )
        return key, wf, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)
