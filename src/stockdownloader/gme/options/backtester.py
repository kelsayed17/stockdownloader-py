"""GME Options Backtester with pluggable strategy interface.

Provides a daily-evaluation backtesting engine for GME options strategies.
Strategies implement the abstract ``GMEOptionsStrategy`` base class, and the
``GMEOptionsBacktester`` orchestrates position tracking, expiration handling,
mark-to-market, and metric computation.

Data classes
------------
- ``Trade`` -- immutable description of a single options trade.
- ``Position`` -- mutable tracking of an open (or closed) position.
- ``BacktestConfig`` -- immutable configuration for a backtest run.
- ``BacktestResult`` -- output container with metrics, trades, and equity curve.
"""
from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trade:
    """Immutable description of an options trade signal.

    Parameters
    ----------
    option_ticker:
        Polygon-style ticker, e.g. ``O:GME230721C00025000``.
    direction:
        ``"buy"`` or ``"sell"``.
    option_type:
        ``"call"`` or ``"put"``.
    strike:
        Strike price.
    expiration:
        Expiration date as an ISO string (``YYYY-MM-DD``).
    premium:
        Per-share premium (option price).
    contracts:
        Number of contracts (1 contract = 100 shares).
    """

    option_ticker: str
    direction: str  # "buy" / "sell"
    option_type: str  # "call" / "put"
    strike: float
    expiration: str
    premium: float
    contracts: int = 1


@dataclass(slots=True)
class Position:
    """Mutable tracking of an open or closed options position.

    Parameters
    ----------
    trade:
        The original ``Trade`` that opened this position.
    entry_date:
        Date the position was opened.
    entry_premium:
        Per-share premium at entry.
    contracts:
        Number of contracts.
    current_value:
        Current mark-to-market value of the position.
    closed:
        Whether the position has been closed.
    exit_date:
        Date the position was closed (``None`` if still open).
    exit_premium:
        Per-share premium at exit.
    """

    trade: Trade
    entry_date: date
    entry_premium: float
    contracts: int
    current_value: float = 0.0
    closed: bool = False
    exit_date: date | None = None
    exit_premium: float = 0.0

    @property
    def pnl(self) -> float:
        """Profit/loss for this position in dollars.

        For a *buy* position: ``(exit_premium - entry_premium) * contracts * 100``.
        For a *sell* position: ``(entry_premium - exit_premium) * contracts * 100``.
        """
        multiplier = self.contracts * 100
        if self.trade.direction == "buy":
            if self.closed:
                return (self.exit_premium - self.entry_premium) * multiplier
            return (self.current_value - self.entry_premium) * multiplier
        # sell direction
        if self.closed:
            return (self.entry_premium - self.exit_premium) * multiplier
        return (self.entry_premium - self.current_value) * multiplier


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Immutable configuration for a backtest run.

    Parameters
    ----------
    initial_capital:
        Starting cash in dollars.
    commission_per_contract:
        Dollar commission charged per contract on each option trade.
    max_positions:
        Maximum number of concurrent open positions.
    max_contracts_per_trade:
        Maximum contracts allowed per individual trade.
    """

    initial_capital: float = 100_000.0
    commission_per_contract: float = 0.65
    max_positions: int = 10
    max_contracts_per_trade: int = 5


@dataclass
class BacktestResult:
    """Output container for a single strategy's backtest results.

    Parameters
    ----------
    strategy_name:
        Name of the strategy that was backtested.
    metrics:
        Dictionary of computed performance metrics.
    trades:
        List of all closed trades (``Position`` objects).
    equity_curve:
        List of equity values at each evaluation point.
    """

    strategy_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    trades: list[Position] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


# ------------------------------------------------------------------
# Abstract strategy interface
# ------------------------------------------------------------------


class GMEOptionsStrategy(ABC):
    """Abstract base class for GME options strategies.

    Subclasses must implement :pyattr:`name`, :pymeth:`evaluate`, and
    :pymeth:`on_expiry`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame | None,
    ) -> list[Trade]:
        """Evaluate the strategy for a single trading day.

        Parameters
        ----------
        trade_date:
            Current trading date.
        state:
            State row (a pandas Series) with IV, GEX, OI metrics.
        chain:
            Options chain DataFrame for this day (may be ``None``).

        Returns
        -------
        List of ``Trade`` objects to open.
        """

    @abstractmethod
    def on_expiry(
        self,
        trade_date: date,
        positions: list[Position],
    ) -> list[Trade]:
        """Handle expiring positions.

        Called when positions expire on ``trade_date``. The strategy may
        return new trades to roll or replace the expiring positions.

        Parameters
        ----------
        trade_date:
            Current trading date.
        positions:
            List of positions expiring today.

        Returns
        -------
        List of ``Trade`` objects to open as replacements/rolls.
        """


# ------------------------------------------------------------------
# Backtester engine
# ------------------------------------------------------------------


class GMEOptionsBacktester:
    """Daily-evaluation backtesting engine for GME options strategies.

    Parameters
    ----------
    strategies:
        List of ``GMEOptionsStrategy`` instances to evaluate.
    config:
        Backtest configuration.  Uses defaults if ``None``.
    """

    def __init__(
        self,
        strategies: list[GMEOptionsStrategy],
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategies = strategies
        self.config = config or BacktestConfig()
        self.positions: list[Position] = []
        self.closed_trades: list[Position] = []
        self.equity_curve: list[float] = []
        self.cash: float = self.config.initial_capital

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single_day(
        self,
        trade_date: date,
        state_row: Any,
        chain: pd.DataFrame | None,
    ) -> list[Trade]:
        """Process a single trading day.

        1. Handle expirations (close expired positions).
        2. Evaluate each strategy to generate trade signals.
        3. Open new positions (subject to config limits).

        Parameters
        ----------
        trade_date:
            Current trading date.
        state_row:
            State row (pandas Series) with IV/GEX/OI metrics.
        chain:
            Options chain DataFrame for this day.

        Returns
        -------
        List of all ``Trade`` objects executed this day.
        """
        all_trades: list[Trade] = []

        # Step 1: handle expirations
        expiry_trades = self._handle_expirations(trade_date)
        all_trades.extend(expiry_trades)

        # Step 2: evaluate strategies
        for strategy in self.strategies:
            signals = strategy.evaluate(trade_date, state_row, chain)
            for trade in signals:
                # Enforce contract limit
                if trade.contracts > self.config.max_contracts_per_trade:
                    trade = Trade(
                        option_ticker=trade.option_ticker,
                        direction=trade.direction,
                        option_type=trade.option_type,
                        strike=trade.strike,
                        expiration=trade.expiration,
                        premium=trade.premium,
                        contracts=self.config.max_contracts_per_trade,
                    )

                # Enforce max positions
                open_count = len(
                    [p for p in self.positions if not p.closed]
                )
                if open_count >= self.config.max_positions:
                    break

                self._open_position(trade, trade_date)
                all_trades.append(trade)

        return all_trades

    def run(
        self,
        state_df: pd.DataFrame,
        chain_by_date: dict[str, pd.DataFrame],
    ) -> list[BacktestResult]:
        """Run a full backtest over the provided date range.

        Parameters
        ----------
        state_df:
            DataFrame with one row per trading day (must include a
            ``trade_date`` or ``date`` column).
        chain_by_date:
            Mapping of ISO date strings to options chain DataFrames.

        Returns
        -------
        List of ``BacktestResult`` objects, one per strategy.
        """
        date_col = "trade_date" if "trade_date" in state_df.columns else "date"

        for _, row in state_df.iterrows():
            trade_date = row[date_col]
            if isinstance(trade_date, str):
                trade_date = date.fromisoformat(trade_date)

            chain = chain_by_date.get(str(trade_date))
            self.run_single_day(trade_date, row, chain)

            # Record equity
            equity = self.cash + self._mark_to_market()
            self.equity_curve.append(equity)

        # Build per-strategy results
        metrics = self.compute_metrics()
        results: list[BacktestResult] = []
        for strategy in self.strategies:
            results.append(
                BacktestResult(
                    strategy_name=strategy.name,
                    metrics=metrics,
                    trades=list(self.closed_trades),
                    equity_curve=list(self.equity_curve),
                )
            )
        return results

    def compute_metrics(self) -> dict[str, Any]:
        """Compute summary performance metrics.

        Returns
        -------
        Dictionary with keys: ``total_return_pct``, ``max_drawdown_pct``,
        ``sharpe``, ``total_trades``, ``win_rate``, ``initial_capital``,
        ``final_equity``.
        """
        initial = self.config.initial_capital
        final = self.equity_curve[-1] if self.equity_curve else initial
        total_return_pct = ((final - initial) / initial) * 100.0 if initial > 0 else 0.0

        # Max drawdown
        max_dd_pct = 0.0
        peak = initial
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd_pct = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        # Sharpe ratio (daily returns, annualized with 252 trading days)
        sharpe = 0.0
        if len(self.equity_curve) >= 2:
            returns: list[float] = []
            prev = initial
            for eq in self.equity_curve:
                returns.append((eq - prev) / prev if prev > 0 else 0.0)
                prev = eq
            if returns:
                mean_r = statistics.mean(returns)
                std_r = statistics.pstdev(returns)
                if std_r > 0:
                    sharpe = (mean_r / std_r) * math.sqrt(252)

        total_trades = len(self.closed_trades)
        win_rate = self._win_rate()

        return {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd_pct,
            "sharpe": sharpe,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "initial_capital": initial,
            "final_equity": final,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_position(self, trade: Trade, trade_date: date) -> None:
        """Open a new position from a trade signal.

        For *buy* trades, deducts ``premium * contracts * 100 + commission``.
        For *sell* trades, adds ``premium * contracts * 100 - commission``.
        """
        multiplier = trade.contracts * 100
        commission = self.config.commission_per_contract * trade.contracts

        if trade.direction == "buy":
            self.cash -= trade.premium * multiplier + commission
        else:
            self.cash += trade.premium * multiplier - commission

        position = Position(
            trade=trade,
            entry_date=trade_date,
            entry_premium=trade.premium,
            contracts=trade.contracts,
            current_value=trade.premium,
        )
        self.positions.append(position)

    def _handle_expirations(self, trade_date: date) -> list[Trade]:
        """Close positions that expire on ``trade_date``.

        Calls each strategy's ``on_expiry`` method with the list of
        expiring positions and collects any replacement trades.

        Returns
        -------
        List of replacement/roll trades from strategies.
        """
        expiring: list[Position] = []

        for pos in self.positions:
            if pos.closed:
                continue
            exp_date_str = pos.trade.expiration
            try:
                exp_date = date.fromisoformat(exp_date_str)
            except (ValueError, TypeError):
                continue

            if exp_date <= trade_date:
                pos.closed = True
                pos.exit_date = trade_date
                pos.exit_premium = 0.0  # expired worthless by default
                expiring.append(pos)
                self.closed_trades.append(pos)

        # Notify strategies about expirations
        replacement_trades: list[Trade] = []
        if expiring:
            for strategy in self.strategies:
                replacements = strategy.on_expiry(trade_date, expiring)
                replacement_trades.extend(replacements)

        return replacement_trades

    def _mark_to_market(self) -> float:
        """Sum current values of all open positions.

        Returns
        -------
        Total mark-to-market value of open positions.
        """
        total = 0.0
        for pos in self.positions:
            if not pos.closed:
                multiplier = pos.contracts * 100
                if pos.trade.direction == "buy":
                    total += pos.current_value * multiplier
                else:
                    # Short position: value is premium received minus current
                    total += (pos.entry_premium - pos.current_value) * multiplier
        return total

    def _win_rate(self) -> float:
        """Compute win rate from closed trades.

        Returns
        -------
        Fraction of closed trades with positive PnL (0.0 if no trades).
        """
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for pos in self.closed_trades if pos.pnl > 0)
        return wins / len(self.closed_trades)
