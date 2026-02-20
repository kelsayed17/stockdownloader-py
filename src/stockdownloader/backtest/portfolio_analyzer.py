"""Portfolio analysis for tournament strategy selection.

Computes pairwise correlation between strategy equity curves, selects
a diversified portfolio using greedy diversification, and simulates
combined portfolio equity curves.

Usage::

    from stockdownloader.backtest.portfolio_analyzer import (
        correlation_matrix,
        select_portfolio,
        portfolio_equity_curve,
        portfolio_metrics,
    )

    corr = correlation_matrix(combo_results)
    portfolio = select_portfolio(combo_results, corr, max_strategies=5)
    eq_curve = portfolio_equity_curve(portfolio)
    metrics = portfolio_metrics(eq_curve, initial_capital=100_000)
"""
from __future__ import annotations

import math
import statistics
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.backtest.tournament_engine import ComboResult


def daily_returns(equity_curve: list[Decimal]) -> list[float]:
    """Extract daily percentage returns from an equity curve.

    Parameters
    ----------
    equity_curve:
        List of Decimal equity values (one per bar).

    Returns
    -------
    List of daily percentage returns.
    """
    if len(equity_curve) < 2:
        return []

    returns: list[float] = []
    prev = float(equity_curve[0])
    for val in equity_curve[1:]:
        curr = float(val)
        if prev != 0:
            returns.append((curr - prev) / abs(prev) * 100.0)
        else:
            returns.append(0.0)
        prev = curr
    return returns


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two series.

    Returns 0.0 if either series has zero variance or lengths differ.
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0

    x = x[:n]
    y = y[:n]

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    if var_x == 0 or var_y == 0:
        return 0.0

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    return cov / (math.sqrt(var_x) * math.sqrt(var_y))


def correlation_matrix(
    combos: list[ComboResult],
) -> dict[tuple[str, str], float]:
    """Compute pairwise Pearson correlation of daily returns.

    Parameters
    ----------
    combos:
        List of combo results with valid baseline results.

    Returns
    -------
    Dict mapping (label_a, label_b) -> correlation coefficient.
    Keys are sorted so (a, b) where a < b lexicographically.
    """
    # Extract daily returns for each combo
    returns_map: dict[str, list[float]] = {}
    for combo in combos:
        if combo.best_result is not None and combo.best_result.equity_curve:
            label = combo.key.label
            returns_map[label] = daily_returns(combo.best_result.equity_curve)

    labels = sorted(returns_map.keys())
    result: dict[tuple[str, str], float] = {}

    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            corr = _pearson_correlation(returns_map[a], returns_map[b])
            result[(a, b)] = round(corr, 4)

    return result


def _get_correlation(
    corr_matrix: dict[tuple[str, str], float],
    label_a: str,
    label_b: str,
) -> float:
    """Look up correlation between two labels (order-independent)."""
    if label_a == label_b:
        return 1.0
    key = (min(label_a, label_b), max(label_a, label_b))
    return corr_matrix.get(key, 0.0)


def select_portfolio(
    combos: list[ComboResult],
    corr_matrix: dict[tuple[str, str], float],
    max_strategies: int = 5,
    max_correlation: float = 0.4,
    min_trades: int = 10,
    min_win_rate: float = 50.0,
) -> list[ComboResult]:
    """Greedy diversification portfolio selection.

    Starts with the highest-scoring combo, then iteratively adds the
    next-best combo that has low correlation (< max_correlation) with
    all existing portfolio members.

    Parameters
    ----------
    combos:
        All combo results, sorted by tournament_score descending.
    corr_matrix:
        Pairwise correlation dict from :func:`correlation_matrix`.
    max_strategies:
        Maximum number of strategies in the portfolio.
    max_correlation:
        Maximum allowed correlation between any portfolio member pair.
    min_trades:
        Minimum trades required for inclusion.
    min_win_rate:
        Minimum win rate (%) required for inclusion.

    Returns
    -------
    List of selected combo results forming a diversified portfolio.
    """
    # Filter eligible combos
    eligible = [
        c for c in combos
        if c.best_result is not None
        and c.best_result.total_pnl > 0
        and c.best_result.total_trades >= min_trades
        and float(c.best_result.win_rate) >= min_win_rate
    ]

    if not eligible:
        # Relax constraints: just require profitability and some trades
        eligible = [
            c for c in combos
            if c.best_result is not None
            and c.best_result.total_pnl > 0
            and c.best_result.total_trades >= 3
        ]

    if not eligible:
        return []

    # Sort by tournament score descending
    eligible.sort(key=lambda c: c.tournament_score, reverse=True)

    portfolio: list[ComboResult] = [eligible[0]]

    for candidate in eligible[1:]:
        if len(portfolio) >= max_strategies:
            break

        # Check correlation with all existing portfolio members
        c_label = candidate.key.label
        too_correlated = False
        for member in portfolio:
            m_label = member.key.label
            corr = _get_correlation(corr_matrix, c_label, m_label)
            if abs(corr) > max_correlation:
                too_correlated = True
                break

        if not too_correlated:
            portfolio.append(candidate)

    return portfolio


def portfolio_equity_curve(
    combos: list[ComboResult],
    initial_capital: float = 100_000.0,
) -> list[float]:
    """Simulate combined equity curve with equal capital allocation.

    Each strategy gets ``initial_capital / len(combos)`` allocation.
    Returns are summed at each bar to produce a portfolio equity curve.

    Parameters
    ----------
    combos:
        Selected portfolio strategies with valid baseline results.
    initial_capital:
        Total starting capital across all strategies.

    Returns
    -------
    Combined equity curve as list of floats.
    """
    if not combos:
        return []

    # Get all equity curves
    curves: list[list[float]] = []
    for c in combos:
        if c.best_result is not None and c.best_result.equity_curve:
            per_strat_capital = initial_capital / len(combos)
            strat_initial = float(c.best_result.initial_capital)
            scale = per_strat_capital / strat_initial if strat_initial else 1.0

            # Scale equity curve to portfolio allocation
            scaled = [float(v) * scale for v in c.best_result.equity_curve]
            curves.append(scaled)

    if not curves:
        return []

    # Find max length and pad shorter curves with their last value
    max_len = max(len(c) for c in curves)
    for i, curve in enumerate(curves):
        if len(curve) < max_len:
            pad_val = curve[-1] if curve else initial_capital / len(combos)
            curves[i] = curve + [pad_val] * (max_len - len(curve))

    # Sum across all curves at each bar
    return [sum(curves[j][i] for j in range(len(curves))) for i in range(max_len)]


def portfolio_metrics(
    equity_curve: list[float],
    initial_capital: float = 100_000.0,
) -> dict[str, float]:
    """Compute portfolio-level performance metrics.

    Parameters
    ----------
    equity_curve:
        Combined portfolio equity curve.
    initial_capital:
        Starting capital.

    Returns
    -------
    Dict with keys: total_return, total_pnl, max_drawdown, sharpe, sortino.
    """
    if not equity_curve:
        return {
            "total_return": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    final = equity_curve[-1]
    total_pnl = final - initial_capital
    total_return = (final - initial_capital) / initial_capital * 100.0

    # Max drawdown
    peak = initial_capital
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Daily returns for Sharpe/Sortino
    returns = daily_returns([Decimal(str(v)) for v in equity_curve])

    sharpe = 0.0
    sortino = 0.0
    if len(returns) > 1:
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        if std_r > 0:
            # Annualize: sqrt(252 * 78) for 5-min bars
            sharpe = (mean_r / std_r) * math.sqrt(252 * 78)

        # Sortino: only downside deviation
        downside = [r for r in returns if r < 0]
        if len(downside) > 1:
            down_std = statistics.stdev(downside)
            if down_std > 0:
                sortino = (mean_r / down_std) * math.sqrt(252 * 78)

    return {
        "total_return": round(total_return, 2),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
    }
