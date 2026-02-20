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
from stockdownloader.backtest.tournament_engine import (
    INITIAL_CAPITAL,
    ComboKey,
    ComboResult,
    MatchResult,
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
from stockdownloader.backtest.portfolio_analyzer import (
    correlation_matrix,
    portfolio_equity_curve,
    portfolio_metrics,
    select_portfolio,
)
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.strategy.regime.regime_detector import MarketRegime
from stockdownloader.strategy.regime.regime_strategy_map import RegimeStrategyMapper
from stockdownloader.strategy.registration_loader import ensure_registered
from stockdownloader.strategy.base_registry import StrategyRegistry

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
