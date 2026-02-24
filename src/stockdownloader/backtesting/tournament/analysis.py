"""Tournament analysis functions — regime, Monte Carlo, and cross-timeframe.

Provides regime-aware trade analysis, Monte Carlo robustness testing, and
cross-timeframe consistency scoring for strategy tournament evaluation.
These functions are process-safe (except ``classify_timeframe_bars``) and
are designed to run in parallel across strategy x timeframe combos.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from collections import defaultdict

from stockdownloader.backtesting.tournament.models import (
    ComboKey,
    ComboResult,
    MonteCarloPercentiles,
    MonteCarloResult,
    RegimeAnalysis,
    RegimeTradeStats,
)
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.strategies.regime.detector import MarketRegime


# ======================================================================
# Regime analysis
# ======================================================================


def classify_timeframe_bars(
    data: list[IntradayPriceData],
) -> dict[str, MarketRegime]:
    """Classify every bar's regime for one timeframe.

    Runs in the main process (IndicatorHub is not process-safe).
    Creates a fresh detector and classifies every bar, mapping
    ``data[i].date -> MarketRegime``.  Bars within the warmup
    period are mapped to ``WEAK_TREND``.

    Parameters
    ----------
    data:
        Price data for one timeframe.

    Returns
    -------
    Dict mapping bar date string to MarketRegime.
    """
    from stockdownloader.indicators.hub import IndicatorHub
    from stockdownloader.strategies.regime.detector import MarketRegimeDetector

    hub = IndicatorHub()
    detector = MarketRegimeDetector(hub)
    warmup = detector.warmup_period

    regime_map: dict[str, MarketRegime] = {}

    for i, bar in enumerate(data):
        if i < warmup:
            regime_map[bar.date] = MarketRegime.WEAK_TREND
        else:
            try:
                rc = detector.classify(data, i)
                regime_map[bar.date] = rc.regime
            except (ValueError, KeyError, IndexError):
                regime_map[bar.date] = MarketRegime.WEAK_TREND

    return regime_map


def _compute_regime_bonus(
    per_regime: dict[MarketRegime, RegimeTradeStats],
) -> float:
    """Compute regime-aware scoring bonus/penalty.

    Rewards coverage (trades in many regimes), consistency (similar
    avg_pnl across regimes), and penalises regime collapse (one regime
    with large negative avg_pnl) and single-regime dependency.
    """
    # Filter regimes with meaningful trade count
    active = {r: s for r, s in per_regime.items() if s.trade_count >= 3}

    if not active:
        return 0.0

    # 1. Coverage bonus: +1.0 per regime with >= 3 trades (max +5.0)
    coverage_bonus = len(active) * 1.0

    # 2. Consistency bonus: low stdev of avg_pnl across regimes
    avg_pnls = [s.avg_pnl for s in active.values()]
    if len(avg_pnls) >= 2:
        stdev = statistics.stdev(avg_pnls)
        consistency_bonus = 3.0 / (1.0 + stdev / 50.0)
    else:
        consistency_bonus = 0.0

    # 3. Regime collapse penalty: worst regime drags score down
    worst_avg = min(avg_pnls)
    collapse_penalty = 0.0
    if worst_avg < -50.0:
        collapse_penalty = abs(worst_avg + 50.0) * 0.1

    # 4. Single-regime penalty
    single_regime_penalty = 0.0
    if len(active) == 1:
        single_regime_penalty = 2.0

    return coverage_bonus + consistency_bonus - collapse_penalty - single_regime_penalty


def run_regime_analysis(
    key: ComboKey,
    trades: list[tuple[str, float, bool]],
    regime_at_bar: dict[str, MarketRegime],
) -> tuple[ComboKey, RegimeAnalysis | None, float, str | None]:
    """Compute regime-aware stats for one combo (process-safe).

    Parameters
    ----------
    key:
        The strategy x timeframe combo.
    trades:
        List of ``(entry_date_str, pnl_float, is_win)`` tuples.
    regime_at_bar:
        Pre-computed mapping from bar date string to MarketRegime.

    Returns
    -------
    tuple of (key, RegimeAnalysis | None, elapsed, error)
    """
    t0 = time.time()

    if not trades:
        return key, None, time.time() - t0, None

    try:
        # Accumulate stats per regime
        regime_data: dict[MarketRegime, dict] = defaultdict(
            lambda: {"count": 0, "pnl": 0.0, "wins": 0},
        )

        for entry_date, pnl, is_win in trades:
            # Look up regime — try exact match, then prefix match
            regime = regime_at_bar.get(entry_date)
            if regime is None:
                # Fallback: match on first 19 chars (YYYY-MM-DD HH:MM:SS)
                prefix = entry_date[:19]
                for bar_date, bar_regime in regime_at_bar.items():
                    if bar_date[:19] == prefix:
                        regime = bar_regime
                        break
            if regime is None:
                regime = MarketRegime.WEAK_TREND

            d = regime_data[regime]
            d["count"] += 1
            d["pnl"] += pnl
            if is_win:
                d["wins"] += 1

        # Build RegimeTradeStats per regime
        per_regime: dict[MarketRegime, RegimeTradeStats] = {}
        for regime, d in regime_data.items():
            count = d["count"]
            per_regime[regime] = RegimeTradeStats(
                regime=regime,
                trade_count=count,
                total_pnl=d["pnl"],
                win_count=d["wins"],
                avg_pnl=d["pnl"] / count if count else 0.0,
                win_rate=d["wins"] / count if count else 0.0,
            )

        # Compute aggregate stats
        active = {r: s for r, s in per_regime.items() if s.trade_count >= 3}
        regime_coverage = len(active)

        active_avg_pnls = [s.avg_pnl for s in active.values()]
        worst_regime_pnl = min(active_avg_pnls) if active_avg_pnls else 0.0

        if len(active_avg_pnls) >= 2:
            regime_consistency = statistics.stdev(active_avg_pnls)
        else:
            regime_consistency = 0.0

        regime_bonus = _compute_regime_bonus(per_regime)

        analysis = RegimeAnalysis(
            per_regime=per_regime,
            regime_coverage=regime_coverage,
            worst_regime_pnl=worst_regime_pnl,
            regime_consistency=regime_consistency,
            regime_bonus=regime_bonus,
        )
        return key, analysis, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)


# ======================================================================
# Monte Carlo robustness testing
# ======================================================================


def _compute_percentiles(
    values: list[float],
) -> MonteCarloPercentiles:
    """Compute 5th/25th/50th/75th/95th percentiles from a list of values."""
    values.sort()
    n = len(values)
    if n == 0:
        return MonteCarloPercentiles(0.0, 0.0, 0.0, 0.0, 0.0)

    def _pctl(p: float) -> float:
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    return MonteCarloPercentiles(
        p5=_pctl(0.05),
        p25=_pctl(0.25),
        p50=_pctl(0.50),
        p75=_pctl(0.75),
        p95=_pctl(0.95),
    )


def _equity_max_drawdown(equity: list[float]) -> float:
    """Compute max drawdown percentage from an equity curve."""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for eq in equity:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def run_monte_carlo(
    key: ComboKey,
    trade_pnls: list[float],
    initial_capital: float,
    n_simulations: int = 1000,
) -> tuple[ComboKey, MonteCarloResult | None, float, str | None]:
    """Run Monte Carlo robustness tests for one combo (process-safe).

    Two independent simulation types:

    1. **Trade shuffling** -- randomly permute trade order, rebuild equity
       curve, measure drawdown and return distribution.
    2. **Bootstrap resampling** -- sample trades WITH replacement, build
       equity curve, measure return and Sharpe distributions.

    Parameters
    ----------
    key:
        The strategy x timeframe combo.
    trade_pnls:
        List of float P&L values from closed trades.
    initial_capital:
        Starting capital for equity curve construction.
    n_simulations:
        Number of Monte Carlo iterations (default 1000).

    Returns
    -------
    tuple of (key, MonteCarloResult | None, elapsed, error)
    """
    t0 = time.time()

    if not trade_pnls or len(trade_pnls) < 2:
        return key, None, time.time() - t0, None

    try:
        rng = random.Random(42 + hash(key))
        n_trades = len(trade_pnls)

        # ---- Trade shuffling ----
        shuffle_dds: list[float] = []
        shuffle_finals: list[float] = []
        shuffle_returns: list[float] = []

        for _ in range(n_simulations):
            pnls = trade_pnls[:]
            rng.shuffle(pnls)

            equity = [initial_capital]
            for pnl in pnls:
                equity.append(equity[-1] + pnl)

            final = equity[-1]
            shuffle_dds.append(_equity_max_drawdown(equity))
            shuffle_finals.append(final)
            shuffle_returns.append(
                (final - initial_capital) / initial_capital * 100.0
            )

        # ---- Bootstrap resampling ----
        boot_returns: list[float] = []
        boot_sharpes: list[float] = []

        for _ in range(n_simulations):
            sampled = rng.choices(trade_pnls, k=n_trades)

            equity = [initial_capital]
            for pnl in sampled:
                equity.append(equity[-1] + pnl)

            final = equity[-1]
            ret = (final - initial_capital) / initial_capital * 100.0
            boot_returns.append(ret)

            # Per-trade returns for Sharpe approximation
            if len(sampled) >= 2:
                mean_pnl = sum(sampled) / len(sampled)
                std_pnl = (
                    sum((p - mean_pnl) ** 2 for p in sampled) / (len(sampled) - 1)
                ) ** 0.5
                if std_pnl > 0:
                    boot_sharpes.append(
                        mean_pnl / std_pnl * math.sqrt(252)
                    )
                else:
                    boot_sharpes.append(0.0)
            else:
                boot_sharpes.append(0.0)

        # ---- Compute percentiles ----
        dd_pctl = _compute_percentiles(shuffle_dds)
        final_pctl = _compute_percentiles(shuffle_finals)
        return_pctl = _compute_percentiles(shuffle_returns)
        boot_ret_pctl = _compute_percentiles(boot_returns)
        boot_sharpe_pctl = _compute_percentiles(boot_sharpes)

        # ---- Robustness assessment ----
        is_robust = boot_ret_pctl.p5 > 0.0

        penalty = 0.0
        if boot_ret_pctl.p5 < 0:
            penalty += abs(boot_ret_pctl.p5) * 2.0
        if dd_pctl.p95 > 20:
            penalty += (dd_pctl.p95 - 20) * 0.5

        mc_result = MonteCarloResult(
            n_simulations=n_simulations,
            n_trades=n_trades,
            max_drawdown=dd_pctl,
            final_equity=final_pctl,
            total_return=return_pctl,
            bootstrap_return=boot_ret_pctl,
            bootstrap_sharpe=boot_sharpe_pctl,
            is_robust=is_robust,
            mc_penalty=penalty,
        )
        return key, mc_result, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)


# ======================================================================
# Cross-timeframe consistency
# ======================================================================


def apply_cross_timeframe_bonus(
    combos: list[ComboResult],
    min_profitable_tfs: int = 3,
) -> None:
    """Add a consistency bonus for strategies profitable across multiple TFs.

    Strategies that perform well in many timeframes are more likely to be
    genuinely robust rather than curve-fit to a single TF.  The bonus is
    larger when the score variance across timeframes is lower (more
    consistent).

    Parameters
    ----------
    combos:
        All combo results (mutated in-place).
    min_profitable_tfs:
        Minimum number of profitable timeframes to qualify for bonus.
    """
    # Group combos by strategy name
    by_strategy: dict[str, list[ComboResult]] = defaultdict(list)
    for c in combos:
        by_strategy[c.key.strategy_name].append(c)

    for _strat_name, group in by_strategy.items():
        # Count profitable timeframes
        profitable_tfs = [
            c for c in group
            if c.best_result is not None and c.best_result.total_pnl > 0
        ]

        if len(profitable_tfs) < min_profitable_tfs:
            continue

        # Compute score variance across profitable TFs
        scores = [c.tournament_score for c in profitable_tfs]
        if len(scores) < 2:
            continue

        variance = statistics.variance(scores)
        # Lower variance = more consistent = bigger bonus
        # Bonus capped at 5.0, decays with increasing variance
        bonus = 5.0 / (1.0 + variance / 100.0)

        for c in group:
            c.tournament_score += bonus
