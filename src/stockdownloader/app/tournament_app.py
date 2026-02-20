"""Multi-timeframe strategy tournament — round-robin, bracket, portfolio.

Runs all registered strategies across multiple timeframes, ranks them
using composite scoring, runs elimination brackets, and builds a
diversified multi-strategy portfolio.

Usage::

    tournament                                          # default: resample from spy/5m_bars.csv
    tournament --extra-csv 15m:data/spy_15m.csv         # add external CSV for a timeframe
    tournament --mode all                               # all stages (default)
    tournament --mode roundrobin                        # ranking only
    tournament --mode bracket                           # elimination only
    tournament --mode portfolio                         # portfolio analysis only
    tournament --bracket-size 16                        # top N for bracket (default 16)
    tournament --top-k 5                                # portfolio size (default 5)
    tournament --no-walk-forward                        # skip WF validation
    tournament --timeframes 5m,15m,1h                   # subset of timeframes
    tournament --category intraday                      # filter strategies
"""
from __future__ import annotations

import argparse
import logging
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import TextIO

from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.backtest.portfolio_analyzer import (
    correlation_matrix,
    portfolio_equity_curve,
    portfolio_metrics,
    select_portfolio,
)
from stockdownloader.backtest.tournament_engine import (
    INITIAL_CAPITAL,
    RISK_PER_TRADE,
    ComboKey,
    ComboResult,
    MatchResult,
    TournamentResult,
    apply_cross_timeframe_bonus,
    classify_timeframe_bars,
    run_combo_backtest,
    run_combo_optimize,
    run_combo_rebacktest,
    run_combo_walkforward,
    run_elimination_bracket,
    run_monte_carlo,
    run_regime_analysis,
)
from stockdownloader.strategy.regime.regime_detector import MarketRegime
from stockdownloader.strategy.regime.regime_strategy_map import RegimeStrategyMapper
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.strategy.registrations import ensure_registered
from stockdownloader.strategy.registry import StrategyRegistry
from stockdownloader.util.file_helper import TeeWriter
from stockdownloader.util.timeframe_aggregator import Timeframe, TimeframeAggregator

logger = logging.getLogger(__name__)

from stockdownloader.app.app_helpers import (
    DEFAULT_DATA_FILE as _DATA_FILE,
    DEFAULT_OUTPUT_DIR as _OUTPUT_DIR,
    STANDARD_TIMEFRAMES as _TIMEFRAMES,
)

_MP_CTX = multiprocessing.get_context("fork")

_MODES = ("all", "roundrobin", "bracket", "portfolio")


# ======================================================================
# Output helpers
# ======================================================================


def _box_title(title: str, width: int = 100) -> str:
    lines = [
        "\u2554" + "\u2550" * width + "\u2557",
        "\u2551" + title.center(width) + "\u2551",
        "\u255a" + "\u2550" * width + "\u255d",
    ]
    return "\n".join(lines)


def _status_label(degradation: float) -> str:
    if degradation >= 0.8:
        return "ROBUST"
    if degradation >= 0.5:
        return "ACCEPTABLE"
    return "OVERFIT"


# ======================================================================
# Data loading
# ======================================================================


def _load_data(
    csv_file: Path,
    extra_csvs: dict[str, Path],
    timeframe_filter: list[str] | None,
    out,
) -> dict[str, list[IntradayPriceData]]:
    """Load price data for each timeframe.

    Returns dict mapping timeframe label -> price data.
    """
    out(f"Loading base 5m data from {csv_file}...")
    raw_data = IntradayCsvLoader.load_from_file(csv_file)
    if not raw_data:
        out(f"ERROR: Could not load data from {csv_file}")
        return {}

    out(f"Loaded {len(raw_data):,} 5-minute bars")
    out(f"Date range: {raw_data[0].date} to {raw_data[-1].date}")
    trading_days = len({d.date[:10] for d in raw_data})
    out(f"Trading days: {trading_days}")

    # Determine which timeframes to test
    all_tfs = [(tf, label) for tf, label in _TIMEFRAMES]
    if timeframe_filter:
        all_tfs = [(tf, label) for tf, label in all_tfs if label in timeframe_filter]

    if not all_tfs:
        out("ERROR: No valid timeframes selected")
        return {}

    out(f"Timeframes: {', '.join(label for _, label in all_tfs)}")
    out()

    # Build data for each timeframe
    agg = TimeframeAggregator(raw_data)
    tf_data: dict[str, list[IntradayPriceData]] = {}

    for tf_enum, tf_label in all_tfs:
        if tf_label in extra_csvs:
            # Load from external CSV
            out(f"  Loading {tf_label} from {extra_csvs[tf_label]}...")
            ext_data = IntradayCsvLoader.load_from_file(extra_csvs[tf_label])
            if ext_data:
                tf_data[tf_label] = ext_data
                out(f"    {len(ext_data):,} bars loaded")
            else:
                out(f"    WARNING: Could not load {extra_csvs[tf_label]}")
        else:
            # Resample from 5m bars
            data = agg.as_intraday_price_data(tf_enum)
            tf_data[tf_label] = data
            out(f"  {tf_label}: {len(data):,} bars (resampled)")

    out()
    return tf_data


# ======================================================================
# Stage 1: Round-Robin (parallelized baseline backtests)
# ======================================================================


def _run_round_robin(
    tf_data: dict[str, list[IntradayPriceData]],
    category_filter: str | None,
    out,
) -> list[ComboResult]:
    """Run baseline backtest for all strategy × timeframe combos."""
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


def _build_skip_set(results: list[ComboResult]) -> set[tuple[str, str]]:
    """Build set of (strategy_name, timeframe) pairs to skip optimization.

    If a strategy has 0 trades on a timeframe, skip it and all higher TFs.
    Timeframes are ordered: 5m < 15m < 30m < 1h < 4h < 1d.
    """
    tf_order = ["5m", "15m", "30m", "1h", "4h", "1d"]
    skip: set[tuple[str, str]] = set()

    # Group by strategy
    by_strategy: dict[str, dict[str, ComboResult]] = {}
    for r in results:
        by_strategy.setdefault(r.key.strategy_name, {})[r.key.timeframe] = r

    for strat_name, tf_map in by_strategy.items():
        found_zero = False
        for tf in tf_order:
            if found_zero:
                skip.add((strat_name, tf))
                continue
            combo = tf_map.get(tf)
            if combo and combo.best_result is not None and combo.best_result.total_trades == 0:
                found_zero = True
                skip.add((strat_name, tf))

    return skip


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


# ======================================================================
# Stage 2.5: Regime Analysis
# ======================================================================


def _run_regime_analysis(
    results: list[ComboResult],
    tf_data: dict[str, list[IntradayPriceData]],
    out,
) -> list[ComboResult]:
    """Run regime-aware analysis on all combos with trades."""
    out()
    out(_box_title("STAGE 2.5: REGIME ANALYSIS"))
    out()

    # Step 1: Classify bars per timeframe (sequential — IndicatorHub not process-safe)
    out("  Classifying bars by market regime...")
    regime_maps: dict[str, dict[str, MarketRegime]] = {}

    for tf_label, data in tf_data.items():
        regime_map = classify_timeframe_bars(data)
        regime_maps[tf_label] = regime_map

        # Count regime distribution
        regime_counts: dict[MarketRegime, int] = {}
        for regime in regime_map.values():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        active_regimes = sum(1 for c in regime_counts.values() if c >= 10)
        out(f"    {tf_label}: {len(data):,} bars classified ({active_regimes} regimes active)")

    out()

    # Step 2: Extract trade data and run parallel regime analysis
    qualifying = [
        r for r in results
        if r.best_result is not None and r.best_result.total_trades >= 1
    ]

    out(f"  Analysing regime performance for {len(qualifying)} combos...")
    out()

    if not qualifying:
        out("  No combos qualify for regime analysis")
        return results

    max_workers = min(len(qualifying), os.cpu_count() or 4)
    mapper = RegimeStrategyMapper()

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {}
        for combo in qualifying:
            trades = [
                (t.entry_date, float(t.profit_loss), t.is_win())
                for t in combo.best_result.closed_trades
            ]
            regime_map = regime_maps.get(combo.key.timeframe, {})
            future = pool.submit(
                run_regime_analysis,
                combo.key,
                trades,
                regime_map,
            )
            futures[future] = combo

        for future in as_completed(futures):
            combo = futures[future]
            try:
                _key, analysis, elapsed, error = future.result()
            except Exception as e:
                out(f"  {combo.key.label}: ERROR ({e})")
                continue

            if error:
                out(f"  {combo.key.label}: FAILED ({error})")
            elif analysis is not None:
                combo.regime_analysis = analysis
                # Recompute score with regime bonus
                trading_days = len(
                    {d.date[:10] for d in tf_data.get(combo.key.timeframe, [])}
                )
                combo.compute_tournament_score(trading_days)

                # Record trades in global mapper
                for regime, stats in analysis.per_regime.items():
                    for _ in range(stats.trade_count):
                        mapper.record(
                            combo.key.label,
                            regime,
                            pnl=stats.avg_pnl,
                            is_win=None,
                        )

    # Re-sort after regime scoring
    results.sort(key=lambda c: c.tournament_score, reverse=True)

    # Print regime summary table
    regime_combos = [r for r in results if r.regime_analysis is not None]
    if regime_combos:
        out()
        out(_box_title("REGIME ANALYSIS SUMMARY"))
        out()
        out(
            f"  {'Strategy @ TF':<40s} {'Coverage':>8s}  "
            f"{'Worst Regime':<20s} {'Consistency':>11s}  {'Bonus':>6s}"
        )
        out("  " + "\u2500" * 92)

        for r in regime_combos[:20]:
            ra = r.regime_analysis
            # Find worst regime name
            active = {
                reg: s for reg, s in ra.per_regime.items() if s.trade_count >= 3
            }
            worst_name = ""
            if active:
                worst = min(active.values(), key=lambda s: s.avg_pnl)
                worst_name = f"{worst.regime.value[:15]} (${worst.avg_pnl:+.0f})"

            out(
                f"  {r.key.label:<40s} "
                f"{ra.regime_coverage:>4d}/5    "
                f"{worst_name:<20s} "
                f"{ra.regime_consistency:>9.1f}  "
                f"{ra.regime_bonus:>+6.1f}"
            )

    # Best strategy per regime
    out()
    out("  BEST STRATEGY PER REGIME:")
    out("  " + "\u2500" * 60)
    for regime in MarketRegime:
        best = mapper.best_strategy_for_regime(regime, min_trades=3)
        perf = mapper.get_performance(best, regime) if best else None
        if best and perf:
            out(
                f"    {regime.value:<20s} \u2192 {best:<30s} "
                f"(avg ${perf.avg_pnl:+.0f}, {perf.trade_count} trades)"
            )
        else:
            out(f"    {regime.value:<20s} \u2192 (no data)")

    out()
    return results


# ======================================================================
# Stage 2.75: Monte Carlo Robustness Testing
# ======================================================================


def _run_monte_carlo_stage(
    results: list[ComboResult],
    tf_data: dict[str, list[IntradayPriceData]],
    top_n: int,
    n_simulations: int,
    out,
) -> list[ComboResult]:
    """Run Monte Carlo robustness tests on top combos."""
    out()
    out(_box_title("STAGE 2.75: MONTE CARLO ROBUSTNESS"))
    out()

    # Select top N combos with enough trades
    qualifying = [
        r for r in results
        if r.best_result is not None and r.best_result.total_trades >= 10
    ]
    qualifying.sort(key=lambda c: c.tournament_score, reverse=True)
    qualifying = qualifying[:top_n]

    out(f"  Testing {len(qualifying)} top combos ({n_simulations} simulations each)")
    out()

    if not qualifying:
        out("  No combos qualify for Monte Carlo testing")
        return results

    max_workers = min(len(qualifying), os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CTX) as pool:
        futures = {}
        for combo in qualifying:
            pnls = [float(t.profit_loss) for t in combo.best_result.closed_trades]
            future = pool.submit(
                run_monte_carlo,
                combo.key,
                pnls,
                float(INITIAL_CAPITAL),
                n_simulations,
            )
            futures[future] = combo

        for future in as_completed(futures):
            combo = futures[future]
            try:
                _key, mc_result, elapsed, error = future.result()
            except Exception as e:
                out(f"  {combo.key.label}: ERROR ({e})")
                continue

            if error:
                out(f"  {combo.key.label}: FAILED ({error})")
            elif mc_result is not None:
                combo.monte_carlo = mc_result
                # Recompute score with MC penalty
                td = tf_data.get(combo.key.timeframe, [])
                trading_days = len({d.date[:10] for d in td}) if td else 100
                combo.compute_tournament_score(trading_days)

    # Re-sort after MC scoring
    results.sort(key=lambda c: c.tournament_score, reverse=True)

    # Print MC summary table
    mc_combos = [r for r in results if r.monte_carlo is not None]
    if mc_combos:
        out()
        out(_box_title("MONTE CARLO ROBUSTNESS RESULTS"))
        out()
        out(
            f"  {'Strategy @ TF':<35s} {'Trades':>6s}  "
            f"{'DD p50':>6s} {'DD p95':>6s}  "
            f"{'Ret p5':>7s} {'Ret p50':>7s} {'Ret p95':>7s}  "
            f"{'Boot p5':>7s}  {'Robust':>6s}"
        )
        out("  " + "\u2500" * 105)

        for r in mc_combos:
            mc = r.monte_carlo
            robust_str = "YES" if mc.is_robust else "NO"
            out(
                f"  {r.key.label:<35s} {mc.n_trades:>6d}  "
                f"{mc.max_drawdown.p50:>5.1f}% {mc.max_drawdown.p95:>5.1f}%  "
                f"{mc.total_return.p5:>+6.1f}% {mc.total_return.p50:>+6.1f}% "
                f"{mc.total_return.p95:>+6.1f}%  "
                f"{mc.bootstrap_return.p5:>+6.1f}%  "
                f"{'  ' + robust_str:>6s}"
            )

        # Summary counts
        robust_count = sum(1 for r in mc_combos if r.monte_carlo.is_robust)
        fragile_count = len(mc_combos) - robust_count
        out()
        out(f"  Summary: {robust_count} ROBUST, {fragile_count} FRAGILE")

    # Interpretation guide
    out()
    out("  MONTE CARLO INTERPRETATION:")
    out("    DD p95:    Max drawdown in 95% of random trade orderings")
    out("    Ret p5:    5th percentile return (worst-case shuffle)")
    out("    Boot p5:   5th percentile return from bootstrap resampling")
    out("    ROBUST:    Bootstrap return p5 > 0 (edge is statistically significant)")

    out()
    return results


# ======================================================================
# Stage 3: Elimination Bracket
# ======================================================================


def _run_bracket(
    results: list[ComboResult],
    bracket_size: int,
    out,
) -> tuple[list[MatchResult], ComboKey | None]:
    """Run single-elimination bracket."""
    out()
    out(_box_title('STAGE 3: ELIMINATION BRACKET'))
    out()

    # Use only combos with valid results
    valid = [r for r in results if r.best_result is not None]
    valid.sort(key=lambda c: c.tournament_score, reverse=True)

    matches, champion = run_elimination_bracket(valid, bracket_size)

    if not matches:
        out("  Not enough combos for a bracket")
        return [], champion

    # Print bracket visualization
    rounds: dict[int, list[MatchResult]] = {}
    for m in matches:
        rounds.setdefault(m.round_num, []).append(m)

    for round_num in sorted(rounds):
        round_matches = rounds[round_num]
        rname = round_matches[0].round_name if round_matches else f"Round {round_num + 1}"
        out(f"  \u2500\u2500\u2500 {rname} \u2500\u2500\u2500")
        out()

        for m in round_matches:
            w_label = f"{m.winner.strategy_name} @ {m.winner.timeframe}"
            l_label = f"{m.loser.strategy_name} @ {m.loser.timeframe}"
            out(
                f"    \u2714 {w_label:<40s} ({m.winner_score:>+7.1f})"
            )
            out(
                f"    \u2718 {l_label:<40s} ({m.loser_score:>+7.1f})"
            )
            out()

    # Champion
    if champion:
        champ_combo = next(
            (r for r in results if r.key == champion), None,
        )
        out()
        out(_box_title('BRACKET CHAMPION'))
        out()
        out(f"  \U0001f3c6  {champion.strategy_name} @ {champion.timeframe}")
        if champ_combo and champ_combo.best_result:
            br = champ_combo.best_result
            pnl = br.total_pnl
            sign = "+" if pnl >= 0 else ""
            out(f"     P/L: {sign}${pnl:,.2f}")
            out(f"     Win Rate: {br.win_rate:.1f}%")
            out(f"     Trades: {br.total_trades}")
            out(f"     Score: {champ_combo.tournament_score:.1f}")
            if champ_combo.wf_result:
                deg = champ_combo.wf_result.degradation_ratio
                out(f"     Walk-Forward: {_status_label(deg)} ({deg:.2f})")
        out()

    return matches, champion


# ======================================================================
# Stage 4: Portfolio Analysis
# ======================================================================


def _run_portfolio(
    results: list[ComboResult],
    top_k: int,
    out,
) -> tuple[list[ComboResult], dict[tuple[str, str], float]]:
    """Run portfolio analysis: correlation + diversification."""
    out()
    out(_box_title('STAGE 4: PORTFOLIO ANALYSIS'))
    out()

    # Only use combos with valid results
    valid = [r for r in results if r.best_result is not None]

    # Correlation matrix
    out("  Computing pairwise correlation matrix...")
    corr = correlation_matrix(valid)

    # Print top correlations (most correlated pairs)
    if corr:
        sorted_corr = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)

        out()
        out(_box_title('CORRELATION MATRIX \u2014 TOP 10 MOST CORRELATED PAIRS'))
        out()
        out(f"  {'Strategy A':<40s} {'Strategy B':<40s} {'Corr':>6s}")
        out("  " + "\u2500" * 88)

        for (a, b), c_val in sorted_corr[:10]:
            out(f"  {a:<40s} {b:<40s} {c_val:>+6.3f}")

        # Least correlated
        out()
        out("  TOP 10 LEAST CORRELATED PAIRS:")
        out("  " + "\u2500" * 88)

        for (a, b), c_val in sorted_corr[-10:]:
            out(f"  {a:<40s} {b:<40s} {c_val:>+6.3f}")

    out()

    # Portfolio selection
    out("  Selecting diversified portfolio...")
    portfolio = select_portfolio(
        valid, corr,
        max_strategies=top_k,
        max_correlation=0.4,
        min_trades=10,
        min_win_rate=50.0,
    )

    if not portfolio:
        out("  No strategies qualify for portfolio")
        return [], corr

    out()
    out(_box_title('RECOMMENDED PORTFOLIO'))
    out()
    out(
        f"  {'#':<3s} {'TF':<5s} {'Strategy':<35s} {'P/L':>12s}  "
        f"{'WR':>6s}  {'Trades':>6s}  {'Score':>7s}  {'WF':>10s}"
    )
    out("  " + "\u2500" * 98)

    for i, combo in enumerate(portfolio, 1):
        br = combo.best_result
        pnl = br.total_pnl
        sign = "+" if pnl >= 0 else ""
        wf_str = ""
        if combo.wf_result:
            wf_str = _status_label(combo.wf_result.degradation_ratio)
        out(
            f"  {i:<3d} {combo.key.timeframe:<5s} {combo.display_name:<35s} "
            f"{sign}${pnl:>10,.2f}  "
            f"{br.win_rate:>5.1f}%  "
            f"{br.total_trades:>6d}  "
            f"{combo.tournament_score:>7.1f}  "
            f"{wf_str:>10s}"
        )

    # Pairwise correlations within portfolio
    if len(portfolio) > 1:
        out(f"\n  PORTFOLIO INTERNAL CORRELATIONS:")
        out("  " + "\u2500" * 60)
        for i, a in enumerate(portfolio):
            for b in portfolio[i + 1:]:
                a_label = a.key.label
                b_label = b.key.label
                key = (min(a_label, b_label), max(a_label, b_label))
                c_val = corr.get(key, 0.0)
                out(f"  {a_label:<30s} vs {b_label:<30s}: {c_val:>+.3f}")

    # Portfolio equity simulation
    eq_curve = portfolio_equity_curve(portfolio, initial_capital=float(INITIAL_CAPITAL))
    if eq_curve:
        metrics = portfolio_metrics(eq_curve, initial_capital=float(INITIAL_CAPITAL))

        out()
        out(_box_title('PORTFOLIO PERFORMANCE (equal-weight simulation)'))
        out()
        out(f"  Total Return:  {metrics['total_return']:>+.2f}%")
        out(f"  Total P/L:     ${metrics['total_pnl']:>+,.2f}")
        out(f"  Max Drawdown:  {metrics['max_drawdown']:.2f}%")
        out(f"  Sharpe Ratio:  {metrics['sharpe']:.2f}")
        out(f"  Sortino Ratio: {metrics['sortino']:.2f}")
        out(f"  Strategies:    {len(portfolio)}")
        out()

    return portfolio, corr


# ======================================================================
# Main
# ======================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-timeframe strategy tournament",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=_DATA_FILE,
        help="Path to base 5-minute bar CSV file",
    )
    parser.add_argument(
        "--extra-csv",
        action="append",
        default=[],
        help="Extra CSV for a timeframe: TF:path (e.g., 15m:data/spy_15m.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=_MODES,
        default="all",
        help="Tournament mode (default: all)",
    )
    parser.add_argument(
        "--bracket-size",
        type=int,
        default=16,
        help="Number of combos in elimination bracket (default: 16)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max strategies in portfolio (default: 5)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip walk-forward optimization (faster, uses baseline params)",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Skip walk-forward validation",
    )
    parser.add_argument(
        "--no-regime",
        action="store_true",
        help="Skip regime-aware analysis",
    )
    parser.add_argument(
        "--no-monte-carlo",
        action="store_true",
        help="Skip Monte Carlo robustness testing",
    )
    parser.add_argument(
        "--mc-top-n",
        type=int,
        default=20,
        help="Number of top strategies for Monte Carlo testing (default: 20)",
    )
    parser.add_argument(
        "--mc-simulations",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations (default: 1000)",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default=None,
        help="Comma-separated timeframes to test (e.g., 5m,15m,1h)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter strategies by category (intraday or daily)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file path (default: output/tournament_results.log)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the multi-timeframe strategy tournament."""
    args = _parse_args(argv)
    ensure_registered()

    # Set up output
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = args.log or (_OUTPUT_DIR / "tournament_results.log")
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    tee = TeeWriter(log_file)

    def out(msg: str = "") -> None:
        tee.write(msg + "\n")

    start_time = time.time()

    out(_box_title("MULTI-TIMEFRAME STRATEGY TOURNAMENT"))
    out()

    # Parse extra CSVs
    extra_csvs: dict[str, Path] = {}
    for spec in args.extra_csv:
        if ":" in spec:
            tf, path = spec.split(":", 1)
            extra_csvs[tf] = Path(path)

    # Parse timeframe filter
    tf_filter = None
    if args.timeframes:
        tf_filter = [t.strip() for t in args.timeframes.split(",")]

    # Load data
    tf_data = _load_data(args.csv_file, extra_csvs, tf_filter, out)
    if not tf_data:
        log_file.close()
        return

    mode = args.mode
    tournament = TournamentResult()

    # Stage 1: Round-Robin (always runs)
    if mode in ("all", "roundrobin", "bracket", "portfolio"):
        tournament.combos = _run_round_robin(tf_data, args.category, out)

    # Stage 1.5: Walk-Forward Optimization (optional)
    if not args.no_optimize and mode in ("all",):
        tournament.combos = _run_optimization(tournament.combos, tf_data, out)

    # Stage 2: Walk-Forward Validation (optional)
    if not args.no_walk_forward and mode in ("all",):
        tournament.combos = _run_walkforward(tournament.combos, tf_data, out)

    # Stage 2.5: Regime Analysis (optional)
    if not args.no_regime and mode in ("all",):
        tournament.combos = _run_regime_analysis(tournament.combos, tf_data, out)

    # Stage 2.75: Monte Carlo Robustness (optional)
    if not args.no_monte_carlo and mode in ("all",):
        tournament.combos = _run_monte_carlo_stage(
            tournament.combos, tf_data, args.mc_top_n, args.mc_simulations, out,
        )

    # Cross-timeframe consistency bonus (before bracket)
    if mode in ("all", "bracket", "portfolio") and len(tf_data) > 1:
        apply_cross_timeframe_bonus(tournament.combos)
        tournament.combos.sort(key=lambda c: c.tournament_score, reverse=True)

    # Stage 3: Elimination Bracket
    if mode in ("all", "bracket"):
        matches, champion = _run_bracket(
            tournament.combos, args.bracket_size, out,
        )
        tournament.matches = matches
        tournament.bracket_champion = champion

    # Stage 4: Portfolio Analysis
    if mode in ("all", "portfolio"):
        portfolio, corr = _run_portfolio(
            tournament.combos, args.top_k, out,
        )
        tournament.portfolio = portfolio
        tournament.correlation_matrix = corr

    # Final summary
    elapsed = time.time() - start_time
    out()
    out("=" * 100)
    out(f"  Tournament completed in {elapsed:.1f}s")
    out(f"  Total combinations tested: {len(tournament.combos)}")

    valid = [c for c in tournament.combos if c.best_result is not None]
    profitable = [c for c in valid if c.best_result.total_pnl > 0]
    out(f"  Profitable: {len(profitable)}/{len(valid)}")

    opt_count = sum(1 for c in tournament.combos if c.optimized is not None)
    if opt_count:
        out(f"  Optimized: {opt_count} combos improved via WF optimization")

    if tournament.bracket_champion:
        out(f"  Bracket Champion: {tournament.bracket_champion.label}")

    if tournament.portfolio:
        out(f"  Portfolio: {len(tournament.portfolio)} strategies")

    regime_count = sum(1 for c in tournament.combos if c.regime_analysis is not None)
    if regime_count:
        versatile = sum(
            1 for c in tournament.combos
            if c.regime_analysis and c.regime_analysis.regime_coverage >= 3
        )
        narrow = regime_count - versatile
        out(f"  Regime: {versatile} VERSATILE, {narrow} NARROW")

    mc_count = sum(1 for c in tournament.combos if c.monte_carlo is not None)
    if mc_count:
        mc_robust = sum(
            1 for c in tournament.combos
            if c.monte_carlo and c.monte_carlo.is_robust
        )
        mc_fragile = mc_count - mc_robust
        out(f"  Monte Carlo: {mc_robust} ROBUST, {mc_fragile} FRAGILE")

    wf_count = sum(1 for c in tournament.combos if c.wf_result is not None)
    if wf_count:
        robust = sum(
            1 for c in tournament.combos
            if c.wf_result and c.wf_result.degradation_ratio >= 0.8
        )
        acceptable = sum(
            1 for c in tournament.combos
            if c.wf_result and 0.5 <= c.wf_result.degradation_ratio < 0.8
        )
        overfit = sum(
            1 for c in tournament.combos
            if c.wf_result and c.wf_result.degradation_ratio < 0.5
        )
        out(f"  Walk-forward: {robust} ROBUST, {acceptable} ACCEPTABLE, {overfit} OVERFIT")

    out(f"  Results logged to: {log_path}")
    out("=" * 100)
    out()

    log_file.close()


# ======================================================================
# Backward-compatible entry points for consolidated CLI commands
# ======================================================================


def main_walkforward(argv: list[str] | None = None) -> None:
    """``grand-tournament`` — walk-forward validated tournament.

    Equivalent to: ``tournament --no-optimize --no-regime --no-monte-carlo``
    """
    # Build default args that mimic the old grand_tournament behaviour:
    # round-robin + walk-forward validation only, no optimization stages.
    extra = [
        "--no-optimize",
        "--no-regime",
        "--no-monte-carlo",
    ]
    combined = extra + (argv or sys.argv[1:])
    main(combined)


def main_baseline(argv: list[str] | None = None) -> None:
    """``multi-timeframe-tournament`` — baseline ranking (no optimisation).

    Equivalent to: ``tournament --mode roundrobin --no-optimize --no-walk-forward
    --no-regime --no-monte-carlo``
    """
    extra = [
        "--mode", "roundrobin",
        "--no-optimize",
        "--no-walk-forward",
        "--no-regime",
        "--no-monte-carlo",
    ]
    combined = extra + (argv or sys.argv[1:])
    main(combined)


def main_greedy(argv: list[str] | None = None) -> None:
    """``multi-timeframe-optimizer`` — optimisation + ranking.

    Equivalent to: ``tournament --no-walk-forward --no-regime --no-monte-carlo``
    (runs round-robin + walk-forward optimization, skips validation stages)
    """
    extra = [
        "--no-walk-forward",
        "--no-regime",
        "--no-monte-carlo",
    ]
    combined = extra + (argv or sys.argv[1:])
    main(combined)


if __name__ == "__main__":
    main()
