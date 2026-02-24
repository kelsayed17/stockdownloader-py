"""Tournament stage functions — round-robin, optimization, walk-forward, etc."""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from stockdownloader.app.tournament.helpers import (
    _box_title,
    _build_skip_set,
    _status_label,
)
from stockdownloader.backtesting.tournament.engine import (
    ComboKey,
    ComboResult,
    apply_cross_timeframe_bonus,
    run_combo_backtest,
    run_combo_optimize,
    run_combo_rebacktest,
    run_combo_walkforward,
)
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.strategies.loader import ensure_registered
from stockdownloader.strategies.registry import StrategyRegistry

_MP_CTX = multiprocessing.get_context("fork")


# ======================================================================
# Stage 1: Round-Robin (parallelized baseline backtests)
# ======================================================================


def _run_round_robin(
    tf_data: dict[str, list[IntradayPriceData]],
    category_filter: str | None,
    out,
) -> list[ComboResult]:
    """Run baseline backtest for all strategy x timeframe combos."""
    ensure_registered()

    out(_box_title("STAGE 1: ROUND-ROBIN \u2014 ALL COMBOS"))
    out()

    # Build combo keys
    categories = ["intraday", "daily"]
    if category_filter:
        categories = [category_filter]

    combos: list[tuple[ComboKey, list[IntradayPriceData]]] = []
    for cat in categories:
        entries = StrategyRegistry.all_entries(category=cat)
        for entry in entries:
            for tf_label, data in tf_data.items():
                key = ComboKey(
                    strategy_name=entry.name,
                    timeframe=tf_label,
                    data_source="resampled",
                )
                combos.append((key, data))

    out(f"  {len(combos)} strategy \u00d7 timeframe combinations")
    out()

    # Parallelize
    max_workers = min(len(combos), os.cpu_count() or 4)
    results: list[ComboResult] = []

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {
            pool.submit(run_combo_backtest, key, data): key
            for key, data in combos
        }
        for future in as_completed(futures):
            combo_result = future.result()
            results.append(combo_result)

    # Sort by tournament score descending
    results.sort(key=lambda c: c.tournament_score, reverse=True)

    # Print per-timeframe tables
    for tf_label in tf_data:
        tf_results = [r for r in results if r.key.timeframe == tf_label]
        if not tf_results:
            continue
        tf_results.sort(key=lambda c: c.tournament_score, reverse=True)

        bar_count = len(tf_data[tf_label])
        out(f"\n  TIMEFRAME: {tf_label} ({bar_count:,} bars)")
        out("  " + "\u2500" * 98)
        out(
            f"  {'Rank':<5s} {'Strategy':<35s} {'P/L':>12s}  "
            f"{'WR':>6s}  {'Trades':>6s}  {'Score':>7s}"
        )
        out("  " + "\u2500" * 98)

        for rank, r in enumerate(tf_results, 1):
            if r.error:
                out(f"  {rank:<5d} {r.display_name:<35s}  ERROR: {r.error}")
                continue
            br = r.best_result
            if br is None:
                out(f"  {rank:<5d} {r.display_name:<35s}  No result")
                continue
            pnl = br.total_pnl
            sign = "+" if pnl >= 0 else ""
            out(
                f"  {rank:<5d} {r.display_name:<35s} "
                f"{sign}${pnl:>10,.2f}  "
                f"{br.win_rate:>5.1f}%  "
                f"{br.total_trades:>6d}  "
                f"{r.tournament_score:>7.1f}"
            )

    # Grand ranking
    out()
    out(_box_title("GRAND RANKING \u2014 TOP 20 BY TOURNAMENT SCORE"))
    out()
    out(
        f"  {'Rank':<5s} {'TF':<5s} {'Strategy':<35s} {'P/L':>12s}  "
        f"{'WR':>6s}  {'Trades':>6s}  {'Score':>7s}  {'Opt':>3s}"
    )
    out("  " + "\u2500" * 102)

    for rank, r in enumerate(results[:20], 1):
        if r.error or r.best_result is None:
            continue
        br = r.best_result
        pnl = br.total_pnl
        sign = "+" if pnl >= 0 else ""
        opt_marker = " *" if r.optimized is not None else ""
        out(
            f"  {rank:<5d} {r.key.timeframe:<5s} {r.display_name:<35s} "
            f"{sign}${pnl:>10,.2f}  "
            f"{br.win_rate:>5.1f}%  "
            f"{br.total_trades:>6d}  "
            f"{r.tournament_score:>7.1f}  {opt_marker}"
        )

    # Win rate ranking
    qualifying = [
        r for r in results
        if r.best_result is not None
        and r.best_result.total_trades >= 10
    ]
    qualifying.sort(key=lambda c: float(c.best_result.win_rate), reverse=True)

    out()
    out(_box_title('WIN RATE RANKING \u2014 TOP 15 (\u226510 trades)'))
    out()
    out(
        f"  {'Rank':<5s} {'TF':<5s} {'Strategy':<35s} {'WR':>6s}  "
        f"{'Trades':>6s}  {'P/L':>12s}  {'Score':>7s}"
    )
    out("  " + "\u2500" * 98)

    for rank, r in enumerate(qualifying[:15], 1):
        br = r.best_result
        pnl = br.total_pnl
        sign = "+" if pnl >= 0 else ""
        out(
            f"  {rank:<5d} {r.key.timeframe:<5s} {r.display_name:<35s} "
            f"{br.win_rate:>5.1f}%  "
            f"{br.total_trades:>6d}  "
            f"{sign}${pnl:>10,.2f}  "
            f"{r.tournament_score:>7.1f}"
        )

    # Best per timeframe
    out()
    out(_box_title('BEST STRATEGY PER TIMEFRAME'))
    out()
    out(
        f"  {'TF':<5s} {'Best by Score':<35s} {'Score':>7s}  "
        f"{'Best by WR':<35s} {'WR':>6s}"
    )
    out("  " + "\u2500" * 98)

    for tf_label in tf_data:
        tf_r = [r for r in results if r.key.timeframe == tf_label and r.best_result is not None]
        if not tf_r:
            continue
        best_score = max(tf_r, key=lambda c: c.tournament_score)
        best_wr = max(tf_r, key=lambda c: float(c.best_result.win_rate))
        out(
            f"  {tf_label:<5s} {best_score.display_name:<35s} "
            f"{best_score.tournament_score:>7.1f}  "
            f"{best_wr.display_name:<35s} "
            f"{best_wr.best_result.win_rate:>5.1f}%"
        )

    out()
    return results


# ======================================================================
# Stage 1.5: Walk-Forward Optimization (parallelized)
# ======================================================================


def _run_optimization(
    results: list[ComboResult],
    tf_data: dict[str, list[IntradayPriceData]],
    out,
) -> list[ComboResult]:
    """Run walk-forward optimization on qualifying combos."""
    out()
    out(_box_title("STAGE 1.5: WALK-FORWARD OPTIMIZATION"))
    out()

    ensure_registered()

    # Build skip set for zero-trade combos at higher timeframes
    skip_set = _build_skip_set(results)

    # Filter qualifying combos
    qualifying: list[ComboResult] = []
    for r in results:
        if r.best_result is None or r.best_result.total_trades < 1:
            continue
        if (r.key.strategy_name, r.key.timeframe) in skip_set:
            continue
        try:
            entry = StrategyRegistry.get(r.key.strategy_name)
            if not entry.param_space:
                continue
        except (KeyError, ValueError):
            continue
        qualifying.append(r)

    skipped = len(results) - len(qualifying)
    out(f"  Optimizing {len(qualifying)} combos (skipped {skipped} — "
        f"no trades, no param_space, or higher TF skip)")
    out()

    if not qualifying:
        out("  No combos qualify for optimization")
        return results

    # Phase 1: Run WF optimization in parallel
    max_workers = min(len(qualifying), os.cpu_count() or 4)
    accepted_combos: list[ComboResult] = []

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {
            pool.submit(
                run_combo_optimize,
                combo.key,
                combo.category,
                tf_data[combo.key.timeframe],
            ): combo
            for combo in qualifying
        }

        for future in as_completed(futures):
            combo = futures[future]
            try:
                _key, wf_result, elapsed, error = future.result()
            except Exception as e:
                out(f"  {combo.key.label}: ERROR ({e})")
                continue

            if error:
                out(f"  {combo.key.label}: FAILED ({error})")
            elif wf_result is not None and wf_result.accepted:
                combo.optimized_kwargs = wf_result.optimized_kwargs
                accepted_combos.append(combo)
                out(f"  {combo.key.label}: ACCEPTED ({elapsed:.1f}s)")
            else:
                reason = "baseline optimal" if wf_result else "no result"
                out(f"  {combo.key.label}: REJECTED — {reason} ({elapsed:.1f}s)")

    out()

    # Phase 2: Re-backtest accepted combos on FULL data
    if accepted_combos:
        out(f"  Re-backtesting {len(accepted_combos)} accepted combos on full data...")
        out()

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
            futures = {
                pool.submit(
                    run_combo_rebacktest,
                    combo.key,
                    combo.category,
                    combo.optimized_kwargs,
                    tf_data[combo.key.timeframe],
                ): combo
                for combo in accepted_combos
            }

            for future in as_completed(futures):
                combo = futures[future]
                try:
                    _key, result, elapsed, error = future.result()
                except Exception as e:
                    out(f"  {combo.key.label}: re-backtest ERROR ({e})")
                    combo.optimized_kwargs = None
                    continue

                if error:
                    out(f"  {combo.key.label}: re-backtest FAILED ({error})")
                    combo.optimized_kwargs = None
                elif result is not None:
                    combo.optimized = result
                    trading_days = len(
                        {d.date[:10] for d in tf_data[combo.key.timeframe]}
                    )
                    combo.compute_tournament_score(trading_days)
                    baseline_pnl = float(
                        combo.baseline.total_pnl if combo.baseline else 0
                    )
                    opt_pnl = float(result.total_pnl)
                    delta = opt_pnl - baseline_pnl
                    sign = "+" if delta >= 0 else ""
                    out(
                        f"  {combo.key.label}: "
                        f"${opt_pnl:>9,.2f} "
                        f"(delta: {sign}${delta:>,.2f}) "
                        f"({elapsed:.1f}s)"
                    )

    # Re-sort after optimization
    results.sort(key=lambda c: c.tournament_score, reverse=True)

    # Summary
    out()
    total_accepted = sum(1 for r in results if r.optimized is not None)
    out(f"  Optimization complete: {total_accepted}/{len(qualifying)} combos improved")

    # Per-timeframe acceptance rates
    for tf_label in tf_data:
        tf_q = [r for r in qualifying if r.key.timeframe == tf_label]
        tf_a = [r for r in tf_q if r.optimized is not None]
        if tf_q:
            out(f"    {tf_label}: {len(tf_a)}/{len(tf_q)} accepted")

    out()
    return results


# ======================================================================
# Stage 2: Walk-Forward Validation (parallelized)
# ======================================================================


def _run_walkforward(
    results: list[ComboResult],
    tf_data: dict[str, list[IntradayPriceData]],
    out,
) -> list[ComboResult]:
    """Run walk-forward validation on qualifying combos."""
    out()
    out(_box_title('STAGE 2: WALK-FORWARD VALIDATION'))
    out()

    # Filter: only combos with enough trades
    qualifying = [
        r for r in results
        if r.best_result is not None
        and r.best_result.total_trades >= 10
    ]

    out(f"  Validating {len(qualifying)} combos (5 windows, 70/30 IS/OOS)")
    out()

    if not qualifying:
        out("  No combos qualify for walk-forward validation")
        return results

    max_workers = min(len(qualifying), os.cpu_count() or 4)
    wf_map: dict[str, tuple[ComboResult, ...]] = {}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {}
        for combo in qualifying:
            data = tf_data.get(combo.key.timeframe, [])
            if data:
                future = pool.submit(
                    run_combo_walkforward,
                    combo.key,
                    data,
                    combo.optimized_kwargs,
                )
                futures[future] = combo

        for future in as_completed(futures):
            combo = futures[future]
            key, wf_result, elapsed, error = future.result()
            if error:
                out(f"  {combo.key.label}: FAILED ({error})")
            elif wf_result is not None:
                combo.wf_result = wf_result
                trading_days = len({d.date[:10] for d in tf_data.get(combo.key.timeframe, [])})
                combo.compute_tournament_score(trading_days)
                status = _status_label(wf_result.degradation_ratio)
                out(
                    f"  {combo.key.label}: "
                    f"IS={wf_result.in_sample_score:>+7.1f}  "
                    f"OOS={wf_result.out_of_sample_score:>+7.1f}  "
                    f"Degrade={wf_result.degradation_ratio:>.2f}  "
                    f"[{status}]  ({elapsed:.1f}s)"
                )

    # Re-sort after WF scoring
    results.sort(key=lambda c: c.tournament_score, reverse=True)
    out()
    return results
