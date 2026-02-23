"""Pipeline stage functions — baseline, optimize, re-backtest, walk-forward."""
from __future__ import annotations

import dataclasses
import logging
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from stockdownloader.app.app_helpers import status_label
from stockdownloader.app.pipeline.helpers import unique_days
from stockdownloader.app.pipeline.models import SlotResult
from stockdownloader.backtest.backtest_result import BaseBacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.walk_forward import WalkForwardResult, WalkForwardValidator
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.registration_loader import ensure_registered
from stockdownloader.strategy.base_registry import StrategyRegistry
from stockdownloader.util.config import INITIAL_CAPITAL, OPTIONS_COMMISSION, RISK_PER_TRADE

logger = logging.getLogger(__name__)

_MP_CTX = multiprocessing.get_context("fork")


# ======================================================================
# Stage 1: Baseline Backtest
# ======================================================================


def _baseline_one_slot(
    slot: SlotResult,
    intraday_data: list[IntradayPriceData],
    daily_data: list,
) -> tuple[SlotResult, BaseBacktestResult | None, float, str | None]:
    """Run baseline backtest for one slot (process-safe)."""
    ensure_registered()
    t0 = time.time()
    engine = IntradayBacktestEngine(
        INITIAL_CAPITAL, RISK_PER_TRADE,
        vol_scale=False, dd_throttle=True,
    )
    try:
        if slot.category == "intraday":
            strategy = StrategyRegistry.create(slot.name)
            result = engine.run(strategy, intraday_data)
        elif slot.category == "daily":
            entry = StrategyRegistry.get(slot.name)
            daily_strategy = entry.factory(**entry.default_kwargs)
            adapter = DailyToIntradayAdapter(daily_strategy)
            result = engine.run(adapter, intraday_data)
        elif slot.category == "options":
            if not daily_data:
                return slot, None, time.time() - t0, "SKIPPED (no daily data)"
            from stockdownloader.backtest.options_backtest_engine import (
                OptionsBacktestEngine,
            )
            opt_engine = OptionsBacktestEngine(INITIAL_CAPITAL, OPTIONS_COMMISSION)
            entry = StrategyRegistry.get(slot.name)
            strategy = entry.factory(**entry.default_kwargs)
            result = opt_engine.run(strategy, daily_data)
        else:
            return slot, None, time.time() - t0, None
        return slot, result, time.time() - t0, None
    except Exception as e:
        return slot, None, time.time() - t0, str(e)


def _run_baseline(
    slots: list[SlotResult],
    intraday_data: list[IntradayPriceData],
    daily_data: list,
    out,
) -> None:
    """Run baseline backtests for all slots (parallelized)."""
    out()
    out("=" * 78)
    out("  STAGE 1: BASELINE BACKTEST")
    out("=" * 78)
    out()

    max_workers = min(len(slots), os.cpu_count() or 4)
    results_map: dict[str, tuple[BaseBacktestResult | None, float, str | None]] = {}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {
            pool.submit(_baseline_one_slot, slot, intraday_data, daily_data): slot
            for slot in slots
        }
        for future in as_completed(futures):
            slot_orig = futures[future]
            _slot, result, elapsed, error = future.result()
            results_map[slot_orig.name] = (result, elapsed, error)

    # Print results in original slot order for deterministic output
    for slot in slots:
        result, elapsed, error = results_map[slot.name]
        out(f"  [{slot.category:>8s}] {slot.display_name}...")
        if error:
            out(f"    FAILED: {error}")
            logger.warning("Baseline failed for %s: %s", slot.name, error)
            continue
        slot.baseline = result
        if result is not None:
            pnl = result.total_pnl
            sign = "+" if pnl >= 0 else ""
            out(
                f"    {sign}${pnl:>9,.2f}  "
                f"WR: {result.win_rate:>5.1f}%  "
                f"Trades: {result.total_trades:>3d}  "
                f"({elapsed:.1f}s)"
            )
        else:
            out(f"    No result ({elapsed:.1f}s)")


# ======================================================================
# Stage 2: Optimize
# ======================================================================


def _run_optimize(
    slots: list[SlotResult],
    intraday_data: list[IntradayPriceData],
    optimize_mode: str,
    out,
) -> None:
    """Run optimizer for eligible strategies.

    Parameters
    ----------
    optimize_mode:
        ``"wf"`` — walk-forward optimizer (IS/OOS split, anti-overfit).
        ``"full"`` — old full-data optimizer (for comparison).
    """
    if optimize_mode == "wf":
        _run_optimize_wf(slots, intraday_data, out)
    else:
        _run_optimize_full(slots, intraday_data, out)


def _optimize_one_strategy(
    strategy_name: str,
    category: str,
    intraday_data: list[IntradayPriceData],
) -> list:
    """Run walk-forward optimization for a single strategy (thread-safe).

    Each call creates its own WalkForwardOptimizer instance so greedy
    search mutable state (``_best_score``, ``_best_result``) is isolated.
    """
    from stockdownloader.backtest.walk_forward_optimizer import (
        WalkForwardOptimizer,
    )

    wf_opt = WalkForwardOptimizer(
        intraday_data,
        split_ratio=0.7,
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        verbose=False,
    )
    return wf_opt.optimize_all(categories=[category], strategy_filter=strategy_name)


def _run_optimize_wf(
    slots: list[SlotResult],
    intraday_data: list[IntradayPriceData],
    out,
) -> None:
    """Walk-forward optimizer: search IS, validate OOS (parallelized)."""
    from stockdownloader.backtest.walk_forward_optimizer import (
        WalkForwardOptimizer,
    )

    out()
    out("=" * 78)
    out("  STAGE 2: OPTIMIZE (walk-forward mode — IS/OOS split)")
    out("=" * 78)
    out()

    # Print data split info once
    split = int(len(intraday_data) * 0.7)
    is_days = len({bar.date[:10] for bar in intraday_data[:split]})
    oos_days = len({bar.date[:10] for bar in intraday_data[split:]})
    out(f"  IS: {split:,} bars ({is_days} days)")
    out(f"  OOS: {len(intraday_data) - split:,} bars ({oos_days} days)")
    out()

    # Collect eligible strategies
    cats = list({s.category for s in slots if s.category in ("intraday", "daily")})
    eligible: list[tuple[str, str]] = []  # (name, category)

    for cat in cats:
        entries = StrategyRegistry.all_entries(category=cat)
        for entry in entries:
            if entry.param_space:
                # Only include if it's in our slots
                if any(s.name == entry.name for s in slots):
                    eligible.append((entry.name, cat))

    if not eligible:
        out("  No strategies eligible for optimization")
        return

    out(f"  Optimizing {len(eligible)} strategies in parallel...")
    out()

    # Run each strategy's optimization in its own process with its own
    # WalkForwardOptimizer instance (avoids shared mutable state issues).
    # ProcessPoolExecutor with fork context bypasses GIL for true parallelism.
    max_workers = min(len(eligible), os.cpu_count() or 4)
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {
            pool.submit(
                _optimize_one_strategy, name, cat, intraday_data,
            ): (name, cat)
            for name, cat in eligible
        }
        for future in as_completed(futures):
            name, cat = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                out(f"  {name}: FAILED ({e})")
                logger.warning("Optimization failed for %s: %s", name, e, exc_info=True)

    # Map results back to slots (print in deterministic order)
    for name, cat in eligible:
        matching_results = [r for r in all_results if r.name == name]
        if not matching_results:
            continue
        wf_result = matching_results[0]
        matching_slots = [s for s in slots if s.name == wf_result.name]
        if not matching_slots:
            continue
        slot = matching_slots[0]
        if wf_result.accepted and wf_result.optimized_kwargs:
            slot.optimized_kwargs = wf_result.optimized_kwargs
            out(f"  {slot.display_name}: ACCEPTED (OOS improved)")
        else:
            out(f"  {slot.display_name}: REJECTED (OOS not improved)")

    out()
    accepted = sum(1 for r in all_results if r.accepted)
    out(f"  Walk-forward optimization: {accepted}/{len(all_results)} accepted")


def _run_optimize_full(
    slots: list[SlotResult],
    intraday_data: list[IntradayPriceData],
    out,
) -> None:
    """Full-data optimizer (original behavior)."""
    out()
    out("=" * 78)
    out("  STAGE 2: OPTIMIZE (full-data mode)")
    out("=" * 78)
    out()

    eligible = [
        s for s in slots
        if s.baseline is not None
        and s.baseline.total_trades >= 3
        and s.category in ("intraday", "daily")
    ]
    out(f"  Eligible strategies: {len(eligible)} "
        f"(need >= 3 baseline trades, intraday or daily)")
    out()

    for slot in eligible:
        out(f"  Optimizing: {slot.display_name} ({slot.category})...")
        t0 = time.time()

        try:
            if slot.category == "intraday":
                from stockdownloader.backtest.strategy_optimizer import (
                    StrategyOptimizer,
                )
                optimizer = StrategyOptimizer(
                    intraday_data,
                    initial_capital=INITIAL_CAPITAL,
                    risk_per_trade=RISK_PER_TRADE,
                    verbose=False,
                )
                entry = StrategyRegistry.get(slot.name)
                if entry.param_space:
                    from stockdownloader.backtest.optimizer_scoring import score_v2 as _score

                    baseline_strategy = entry.factory(**entry.default_kwargs)
                    default_config = baseline_strategy._infra._c
                    run_fn = optimizer._make_run_fn(default_config, entry.display_name)

                    trading_days = unique_days(intraday_data)
                    baseline_result = slot.baseline
                    baseline_score = _score(baseline_result, trading_days=trading_days)
                    optimizer._best_score = baseline_score
                    optimizer._best_result = baseline_result

                    winners = optimizer._greedy_search(
                        param_space=entry.param_space,
                        current_best={},
                        run_fn=run_fn,
                        phase_label=entry.display_name,
                    )
                    if winners and optimizer._best_score > baseline_score:
                        slot.optimized_kwargs = winners
                        out(f"    Found improvement: {winners}")
                    else:
                        out(f"    No improvement found")

            elif slot.category == "daily":
                from stockdownloader.backtest.daily_strategy_optimizer import (
                    DailyStrategyOptimizer,
                )
                daily_opt = DailyStrategyOptimizer(
                    slot.name,
                    intraday_data,
                    initial_capital=INITIAL_CAPITAL,
                    risk_per_trade=RISK_PER_TRADE,
                    verbose=False,
                )
                best_kwargs, best_result = daily_opt.optimize()
                trading_days = unique_days(intraday_data)
                from stockdownloader.backtest.optimizer_scoring import score_v2 as _score
                baseline_score = _score(slot.baseline, trading_days=trading_days)
                opt_score = _score(best_result, trading_days=trading_days)
                if opt_score > baseline_score:
                    slot.optimized_kwargs = best_kwargs
                    out(f"    Found improvement (score {baseline_score:.1f} -> {opt_score:.1f})")
                else:
                    out(f"    No improvement found")

            elapsed = time.time() - t0
            out(f"    ({elapsed:.1f}s)")

        except Exception as e:
            out(f"    FAILED: {e}")
            logger.warning("Optimize failed for %s: %s", slot.name, e, exc_info=True)

    out()
    improved = sum(1 for s in slots if s.optimized_kwargs is not None)
    out(f"  Optimization complete: {improved}/{len(eligible)} strategies improved")


# ======================================================================
# Stage 3: Re-backtest with optimized params
# ======================================================================


def _rebacktest_one_slot(
    slot: SlotResult,
    intraday_data: list[IntradayPriceData],
) -> tuple[SlotResult, BaseBacktestResult | None, float, str | None]:
    """Re-backtest one slot with optimized params (process-safe)."""
    import dataclasses

    ensure_registered()
    t0 = time.time()
    engine = IntradayBacktestEngine(
        INITIAL_CAPITAL, RISK_PER_TRADE,
        vol_scale=False, dd_throttle=True,
    )
    try:
        if slot.category == "intraday":
            entry = StrategyRegistry.get(slot.name)
            baseline_strategy = entry.factory(**entry.default_kwargs)
            default_config = baseline_strategy._infra._c
            opt_config = dataclasses.replace(default_config, **slot.optimized_kwargs)
            strategy = entry.factory(config=opt_config)
            result = engine.run(strategy, intraday_data)
        elif slot.category == "daily":
            entry = StrategyRegistry.get(slot.name)
            adapter_keys = {"sl_atr_mult", "rr", "sl_cap", "allow_shorts"}
            strat_kwargs = {k: v for k, v in slot.optimized_kwargs.items()
                            if k not in adapter_keys}
            adapt_kwargs = {k: v for k, v in slot.optimized_kwargs.items()
                            if k in adapter_keys}
            daily_strategy = entry.factory(**strat_kwargs)
            adapter = DailyToIntradayAdapter(daily_strategy, **adapt_kwargs)
            result = engine.run(adapter, intraday_data)
        else:
            return slot, None, time.time() - t0, "unsupported category"
        return slot, result, time.time() - t0, None
    except Exception as e:
        return slot, None, time.time() - t0, str(e)


def _run_rebacktest(
    slots: list[SlotResult],
    intraday_data: list[IntradayPriceData],
    out,
) -> None:
    """Re-backtest strategies that got optimized params (parallelized)."""
    out()
    out("=" * 78)
    out("  STAGE 3: RE-BACKTEST WITH OPTIMIZED PARAMS")
    out("=" * 78)
    out()

    optimized_slots = [s for s in slots if s.optimized_kwargs is not None]
    if not optimized_slots:
        out("  No strategies were improved by optimizer — skipping re-backtest.")
        return

    max_workers = min(len(optimized_slots), os.cpu_count() or 4)
    results_map: dict[str, tuple[BaseBacktestResult | None, float, str | None]] = {}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {
            pool.submit(_rebacktest_one_slot, slot, intraday_data): slot
            for slot in optimized_slots
        }
        for future in as_completed(futures):
            slot_orig = futures[future]
            _slot, result, elapsed, error = future.result()
            results_map[slot_orig.name] = (result, elapsed, error)

    # Print results in original order
    for slot in optimized_slots:
        result, elapsed, error = results_map[slot.name]
        out(f"  Re-testing: {slot.display_name}...")

        if error:
            out(f"    FAILED: {error}")
            logger.warning("Re-backtest failed for %s: %s", slot.name, error)
            continue

        slot.optimized = result
        baseline = slot.baseline
        if result is not None and baseline is not None:
            b_pnl = baseline.total_pnl
            o_pnl = result.total_pnl
            delta = o_pnl - b_pnl
            sign = "+" if delta >= 0 else ""
            out(
                f"    Baseline: ${b_pnl:>9,.2f} ({baseline.total_trades} trades)  "
                f"Optimized: ${o_pnl:>9,.2f} ({result.total_trades} trades)  "
                f"Delta: {sign}${delta:>,.2f}"
            )


# ======================================================================
# Stage 4: Walk-Forward Validation
# ======================================================================


def _validate_one_slot(
    slot: SlotResult,
    intraday_data: list[IntradayPriceData],
) -> tuple[SlotResult, WalkForwardResult | None, float, str | None]:
    """Run walk-forward validation for one slot (process-safe)."""
    import dataclasses

    ensure_registered()
    t0 = time.time()
    try:
        validator = WalkForwardValidator(intraday_data, n_windows=5, is_ratio=0.7)
        engine = IntradayBacktestEngine(
            INITIAL_CAPITAL, RISK_PER_TRADE,
            vol_scale=False, dd_throttle=True,
        )

        if slot.category == "intraday":
            if slot.optimized_kwargs:
                entry = StrategyRegistry.get(slot.name)

                def _factory(e=entry, kw=slot.optimized_kwargs):
                    baseline_strat = e.factory(**e.default_kwargs)
                    default_cfg = baseline_strat._infra._c
                    opt_cfg = dataclasses.replace(default_cfg, **kw)
                    return e.factory(config=opt_cfg)
            else:
                def _factory(name=slot.name):
                    return StrategyRegistry.create(name)

        elif slot.category == "daily":
            entry = StrategyRegistry.get(slot.name)
            if slot.optimized_kwargs:
                adapter_keys = {"sl_atr_mult", "rr", "sl_cap", "allow_shorts"}
                strat_kw = {k: v for k, v in slot.optimized_kwargs.items()
                            if k not in adapter_keys}
                adapt_kw = {k: v for k, v in slot.optimized_kwargs.items()
                            if k in adapter_keys}

                def _factory(e=entry, skw=strat_kw, akw=adapt_kw):
                    daily = e.factory(**skw)
                    return DailyToIntradayAdapter(daily, **akw)
            else:
                def _factory(e=entry):
                    daily = e.factory(**e.default_kwargs)
                    return DailyToIntradayAdapter(daily)
        else:
            return slot, None, time.time() - t0, "unsupported category"

        wf = validator.validate(
            strategy_factory=_factory,
            engine=engine,
            strategy_name=slot.display_name,
        )
        return slot, wf, time.time() - t0, None
    except Exception as e:
        return slot, None, time.time() - t0, str(e)


def _run_walkforward(
    slots: list[SlotResult],
    intraday_data: list[IntradayPriceData],
    out,
) -> None:
    """Run walk-forward validation for intraday and adapted-daily strategies."""
    out()
    out("=" * 78)
    out("  STAGE 4: WALK-FORWARD VALIDATION")
    out("=" * 78)
    out()

    wf_slots = [
        s for s in slots
        if s.category in ("intraday", "daily")
        and s.best_result is not None
        and s.best_result.total_trades >= 1
    ]

    out(f"  Validating {len(wf_slots)} strategies (5 windows, 70/30 IS/OOS)")
    out()

    max_workers = min(len(wf_slots), os.cpu_count() or 4)
    results_map: dict[str, tuple[WalkForwardResult | None, float, str | None]] = {}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {
            pool.submit(_validate_one_slot, slot, intraday_data): slot
            for slot in wf_slots
        }
        for future in as_completed(futures):
            slot_orig = futures[future]
            _slot, wf, elapsed, error = future.result()
            results_map[slot_orig.name] = (wf, elapsed, error)

    # Print results in original slot order for deterministic output
    for slot in wf_slots:
        wf, elapsed, error = results_map[slot.name]
        out(f"  Validating: {slot.display_name}...")
        if error:
            out(f"    FAILED: {error}")
            logger.warning("Walk-forward failed for %s: %s", slot.name, error)
            continue
        if wf is not None:
            slot.wf_result = wf
            status = status_label(wf.degradation_ratio)
            out(
                f"    IS: {wf.in_sample_score:>+7.2f}  "
                f"OOS: {wf.out_of_sample_score:>+7.2f}  "
                f"Degradation: {wf.degradation_ratio:>5.2f}  "
                f"[{status}]  ({elapsed:.1f}s)"
            )
