"""Shared scoring function used by all strategy optimizers.

Provides a composite fitness score that balances Sharpe ratio, win rate,
profit factor, max drawdown, and trade frequency.  Both the VWAP
:class:`StrategyOptimizer` and the :class:`DailyStrategyOptimizer` import
from here so that all strategies are ranked on identical criteria.

Usage::

    from stockdownloader.backtest.optimizer_scoring import score, MIN_TRADES

    s = score(result, trading_days=200)
"""
from __future__ import annotations

import math

from stockdownloader.backtest.backtest_result import BacktestResult

# Minimum trade count for a reliable backtest result.  Configs below this
# threshold are penalized to discourage overly selective filters.
MIN_TRADES = 20


def score(result: BacktestResult, trading_days: int = 0) -> float:
    """Compute a composite fitness score for a backtest result.

    Higher is better.  Balances P&L, Sharpe, win rate, drawdown, and
    trade frequency.

    Parameters
    ----------
    result:
        Completed backtest result.
    trading_days:
        Number of unique trading days in the dataset.  Used to compute
        a trades-per-day bonus.  Pass 0 to skip that component.

    Formula::

        base  = (sharpe × 40) + (win_rate × 20) + (pf × 20) - (dd × 20)
        score = base - trade_penalty + trade_bonus + trades_per_day_bonus - overtrade_penalty
    """
    sharpe = float(result.sharpe_ratio(trading_days_per_year=252 * 78))
    win_rate = float(result.win_rate) / 100.0
    pf = min(float(result.profit_factor), 5.0)  # cap at 5 to avoid outliers
    dd = float(result.max_drawdown) / 100.0

    # --- Penalty for too few trades (unreliable stats) ---
    trade_penalty = 0.0
    if result.total_trades < 5:
        trade_penalty = 100.0  # effectively disqualify < 5 trades
    elif result.total_trades < MIN_TRADES:
        trade_penalty = (MIN_TRADES - result.total_trades) * 3.0

    # --- Bonus for higher trade counts (log-scaled, caps at ~5 pts) ---
    trade_bonus = 0.0
    if result.total_trades > 0:
        trade_bonus = min(math.log2(result.total_trades) * 1.5, 5.0)

    # --- Bonus for trades-per-day targeting ~0.15/day (1 every ~7 days) ---
    trades_per_day_bonus = 0.0
    overtrade_penalty = 0.0
    if trading_days > 0 and result.total_trades > 0:
        tpd = result.total_trades / trading_days
        # Ideal: 0.15 trades/day.  Score falls off as tpd deviates.
        # Gaussian-like: bonus = 5 * exp(-2 * (tpd - 0.15)^2)
        trades_per_day_bonus = 5.0 * math.exp(-2.0 * (tpd - 0.15) ** 2)
        # Overtrading penalty: penalize >1 trade/day
        if tpd > 1.0:
            overtrade_penalty = (tpd - 1.0) * 10.0

    base = (sharpe * 40) + (win_rate * 20) + (pf * 20) - (dd * 20)
    return base - trade_penalty + trade_bonus + trades_per_day_bonus - overtrade_penalty


# =========================================================================
# score_v2 — Institutional-grade composite fitness
# =========================================================================


def score_v2(result: BacktestResult, trading_days: int = 0) -> float:
    """Institutional-grade composite fitness score.

    Improvements over :func:`score`:

    * **P&L floor** — strategies that lose money get a steep penalty
      proportional to the loss.  The original ``score`` ignores P&L
      entirely, so a negative-return config can still rank highly on
      Sharpe/WR/PF from a few lucky trades.
    * **Sortino ratio** replaces Sharpe — penalises only *downside*
      volatility, allowing strategies with large upside swings to
      score higher.
    * **Calmar ratio** (return / max drawdown) as a separate component.
    * **Consecutive-loss penalty** — 3+ consecutive losses is
      penalised to discourage streaky strategies.
    * Reduced weight on trade frequency (less gaming-prone).

    Higher is better.

    Parameters
    ----------
    result:
        Completed backtest result.
    trading_days:
        Number of unique trading days.  Pass 0 to skip the
        trades-per-day bonus.
    """
    pnl_pct = float(result.total_return)
    sortino = min(float(result.sortino_ratio(trading_days_per_year=252 * 78)), 3.0)
    calmar = max(-5.0, min(float(result.calmar_ratio()), 5.0))
    win_rate = float(result.win_rate) / 100.0
    pf = min(float(result.profit_factor), 5.0)
    dd = float(result.max_drawdown) / 100.0

    # ---- P&L floor: steep penalty for losing money ----
    pnl_penalty = max(0.0, -pnl_pct) * 15.0  # steeper penalty

    # ---- P&L bonus: reward positive returns ----
    pnl_bonus = max(0.0, pnl_pct) * 5.0  # reward profitability

    # ---- Consecutive-loss penalty ----
    consec = result.max_consecutive_losses
    consec_penalty = max(0.0, consec - 3) * 2.0

    # ---- Trade-count penalty (unreliable stats) ----
    trade_penalty = 0.0
    if result.total_trades < 5:
        trade_penalty = 100.0  # effectively disqualify < 5 trades
    elif result.total_trades < MIN_TRADES:
        trade_penalty = (MIN_TRADES - result.total_trades) * 3.0

    # ---- Trade-count bonus (log-scaled, caps ~5) ----
    trade_bonus = 0.0
    if result.total_trades > 0:
        trade_bonus = min(math.log2(result.total_trades) * 1.5, 5.0)

    # ---- Trades-per-day bonus (~0.15/day ideal) ----
    tpd_bonus = 0.0
    overtrade_penalty = 0.0
    if trading_days > 0 and result.total_trades > 0:
        tpd = result.total_trades / trading_days
        tpd_bonus = 5.0 * math.exp(-2.0 * (tpd - 0.15) ** 2)
        # Overtrading penalty: penalize >1 trade/day
        if tpd > 1.0:
            overtrade_penalty = (tpd - 1.0) * 10.0

    base = (
        (sortino * 30)
        + (calmar * 15)
        + (win_rate * 20)
        + (pf * 15)
        - (dd * 20)
    )
    return (
        base
        + pnl_bonus
        - pnl_penalty
        - consec_penalty
        - trade_penalty
        + trade_bonus
        + tpd_bonus
        - overtrade_penalty
    )
