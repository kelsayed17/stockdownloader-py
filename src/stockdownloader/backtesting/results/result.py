"""Backtest result containers and performance metrics.

Provides the base ABC with shared performance metrics (total return, win rate,
Sharpe ratio, max drawdown, profit factor, etc.) and concrete subclasses for
stock and options backtesting.
"""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_HALF_UP
from itertools import groupby, pairwise
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from stockdownloader.core.models import PriceData

from stockdownloader.core.models import OptionsTrade, OptionsTradeStatus, Trade, TradeStatus
from stockdownloader.core.math import HUNDRED


# ======================================================================
# Protocol & base class
# ======================================================================


@runtime_checkable
class _HasProfitLoss(Protocol):
    """Duck type for trade-like objects with profit_loss and is_win()."""

    profit_loss: Decimal

    def is_win(self) -> bool: ...


class BaseBacktestResult(ABC):
    """Abstract base for backtest results.

    Holds the equity curve, capital tracking, and lazily computed
    performance metrics that are identical across stock and options
    backtests.
    """

    def __init__(self, strategy_name: str, initial_capital: Decimal) -> None:
        if strategy_name is None:
            raise ValueError("strategy_name must not be None")
        if initial_capital is None:
            raise ValueError("initial_capital must not be None")

        self.strategy_name: str = strategy_name
        self.initial_capital: Decimal = initial_capital
        self.final_capital: Decimal = initial_capital
        self.equity_curve: list[Decimal] = []
        self.start_date: str | None = None
        self.end_date: str | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def closed_trades(self) -> list[_HasProfitLoss]:
        """Return the list of closed trades.

        Each element must expose ``profit_loss: Decimal`` and
        ``is_win() -> bool``.
        """

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    @property
    def total_return(self) -> Decimal:
        return (
            (self.final_capital - self.initial_capital)
            / self.initial_capital
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * HUNDRED

    @property
    def total_pnl(self) -> Decimal:
        return self.final_capital - self.initial_capital

    @property
    def total_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.closed_trades if t.is_win())

    @property
    def losing_trades(self) -> int:
        return self.total_trades - self.winning_trades

    @property
    def win_rate(self) -> Decimal:
        total = self.total_trades
        if total == 0:
            return Decimal("0")
        return (
            Decimal(str(self.winning_trades))
            / Decimal(str(total))
        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) * HUNDRED

    @property
    def average_win(self) -> Decimal:
        return self._average_profit_loss(winners=True)

    @property
    def average_loss(self) -> Decimal:
        return self._average_profit_loss(winners=False)

    def _average_profit_loss(self, winners: bool) -> Decimal:
        filtered = [
            t.profit_loss for t in self.closed_trades if t.is_win() == winners
        ]
        if not filtered:
            return Decimal("0")
        total = sum(filtered, Decimal("0"))
        return (total / Decimal(str(len(filtered)))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @property
    def profit_factor(self) -> Decimal:
        gross_profit = Decimal("0")
        gross_loss = Decimal("0")
        for t in self.closed_trades:
            if t.is_win():
                gross_profit += t.profit_loss
            else:
                gross_loss += abs(t.profit_loss)
        if gross_loss == Decimal("0"):
            return Decimal("999.99") if gross_profit > Decimal("0") else Decimal("0")
        return (gross_profit / gross_loss).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @property
    def max_drawdown(self) -> Decimal:
        if not self.equity_curve:
            return Decimal("0")

        peak = self.equity_curve[0]
        max_dd = Decimal("0")

        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            if peak <= 0:
                continue
            drawdown = (
                (peak - equity) / peak
            ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * HUNDRED
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd

    def sharpe_ratio(self, trading_days_per_year: int = 252) -> Decimal:
        """Annualised Sharpe ratio using *sample* standard deviation (n-1)."""
        if len(self.equity_curve) < 2:
            return Decimal("0")

        daily_returns: list[float] = [
            float((curr - prev) / prev)
            for prev, curr in pairwise(self.equity_curve)
            if prev != 0
        ]

        if len(daily_returns) < 2:
            return Decimal("0")

        mean_return = statistics.mean(daily_returns)
        std_dev = statistics.stdev(daily_returns)

        if std_dev == 0:
            return Decimal("0")

        sharpe = (mean_return / std_dev) * math.sqrt(trading_days_per_year)
        return Decimal(str(sharpe)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def sortino_ratio(
        self,
        trading_days_per_year: int = 252,
        risk_free: float = 0.0,
    ) -> Decimal:
        """Annualised Sortino ratio (downside deviation only).

        Like Sharpe but only penalises *negative* volatility.  A strategy
        with large upside swings (good) is not penalised.
        """
        if len(self.equity_curve) < 2:
            return Decimal("0")

        daily_returns: list[float] = [
            float((curr - prev) / prev)
            for prev, curr in pairwise(self.equity_curve)
            if prev != 0
        ]

        n = len(daily_returns)
        if n < 2:
            return Decimal("0")

        mean_return = statistics.mean(daily_returns)

        # Downside deviation: only consider returns below risk-free rate
        downside_sq = [
            (r - risk_free) ** 2 for r in daily_returns if r < risk_free
        ]
        if not downside_sq:
            # No downside -> infinite Sortino; cap at a large value
            return Decimal("99.99") if mean_return > 0 else Decimal("0")

        downside_dev = math.sqrt(sum(downside_sq) / (n - 1))
        if downside_dev == 0:
            return Decimal("0")

        sortino = ((mean_return - risk_free) / downside_dev) * math.sqrt(
            trading_days_per_year
        )
        return Decimal(str(sortino)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    def calmar_ratio(self, bars_per_year: int = 252 * 78) -> Decimal:
        """Annualised return divided by max drawdown.

        The return is annualised using the number of bars in the equity
        curve so that the ratio is comparable across different time spans.

        Parameters
        ----------
        bars_per_year:
            Number of bars in one trading year.  Default ``252 * 78``
            (= 19 656) assumes 5-minute bars, 78 bars/day, 252 days/year.
            For daily bars pass 252.

        Higher is better.  Undefined (returns 0) when max drawdown is zero
        or equity curve is too short.
        """
        dd = self.max_drawdown
        if dd == Decimal("0"):
            return Decimal("0")

        n_bars = len(self.equity_curve)
        if n_bars < 2 or bars_per_year < 1:
            return Decimal("0")

        # Annualise: (1 + total_return) ^ (bars_per_year / n_bars) - 1
        total_ret_frac = float(self.total_return) / 100.0
        years = n_bars / bars_per_year
        if years <= 0:
            return Decimal("0")

        if total_ret_frac <= -1.0:
            # Lost everything — cannot annualise
            ann_return_pct = -100.0
        else:
            try:
                ann_return_pct = ((1.0 + total_ret_frac) ** (1.0 / years) - 1.0) * 100.0
            except OverflowError:
                # Extremely short equity curve relative to bars_per_year
                # causes astronomical annualisation — cap at a large value.
                ann_return_pct = 9999.0 if total_ret_frac > 0 else -9999.0

        calmar = ann_return_pct / float(dd)
        # Guard against values too large for Decimal quantize
        calmar = max(-9999.99, min(9999.99, calmar))
        return Decimal(str(calmar)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @property
    def max_consecutive_losses(self) -> int:
        """Longest losing streak among closed trades."""
        return self._max_consecutive(win=False)

    @property
    def max_consecutive_wins(self) -> int:
        """Longest winning streak among closed trades."""
        return self._max_consecutive(win=True)

    def _max_consecutive(self, win: bool) -> int:
        """Count the longest consecutive streak of wins or losses."""
        trades = self.closed_trades
        if not trades:
            return 0
        return max(
            (sum(1 for _ in group)
             for key, group in groupby(trades, key=lambda t: t.is_win() == win)
             if key),
            default=0,
        )

    @property
    def avg_trade_duration_bars(self) -> float:
        """Average number of bars a trade is held open.

        Uses ISO datetime strings on entry_date/exit_date to compute
        real calendar duration.  Falls back to counting same-date trades
        as 1 bar.  Returns 0.0 when no closed trades have date info.
        """
        from datetime import datetime

        closed = self.closed_trades
        if not closed:
            return 0.0
        durations: list[float] = []
        for t in closed:
            if hasattr(t, "entry_date") and hasattr(t, "exit_date") and t.entry_date and t.exit_date:
                try:
                    entry_dt = datetime.fromisoformat(t.entry_date)
                    exit_dt = datetime.fromisoformat(t.exit_date)
                    delta = exit_dt - entry_dt
                    # For intraday (5-min bars): duration in minutes / 5
                    total_minutes = delta.total_seconds() / 60.0
                    if total_minutes > 0:
                        # Estimate bars -- default to 5-min bar size
                        durations.append(max(1.0, total_minutes / 5.0))
                    else:
                        durations.append(1.0)
                except (ValueError, TypeError):
                    # Date strings that don't parse as ISO -- treat as 1 bar
                    durations.append(1.0)
        return sum(durations) / len(durations) if durations else 0.0


# ======================================================================
# Stock backtest result
# ======================================================================


class BacktestResult(BaseBacktestResult):
    """Holds the output of a backtest run and lazily computes performance
    metrics from the trade list and equity curve."""

    def __init__(self, strategy_name: str, initial_capital: Decimal) -> None:
        super().__init__(strategy_name, initial_capital)
        self._trades: list[Trade] = []
        self._closed_trades_cache: list[Trade] | None = None
        self.trade_modes: list[str] = []
        self.trade_exit_reasons: list[str] = []

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------

    def add_trade(self, trade: Trade) -> None:
        if trade is None:
            raise ValueError("trade must not be None")
        self._trades.append(trade)
        self._closed_trades_cache = None  # invalidate cache

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    @property
    def closed_trades(self) -> list[Trade]:
        """Return closed trades, caching the filtered list for reuse.

        The cache is invalidated whenever :meth:`add_trade` is called.
        """
        if self._closed_trades_cache is None:
            self._closed_trades_cache = [
                t for t in self._trades if t.status == TradeStatus.CLOSED
            ]
        return self._closed_trades_cache

    # ------------------------------------------------------------------
    # Stock-specific metrics
    # ------------------------------------------------------------------

    def buy_and_hold_return(self, data: list[PriceData]) -> Decimal:
        if not data:
            return Decimal("0")
        first = data[0].close
        last = data[-1].close
        return (
            (last - first) / first
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * HUNDRED


# ======================================================================
# Options backtest result
# ======================================================================


class OptionsBacktestResult(BaseBacktestResult):
    """Holds the output of an options backtest run and computes performance
    metrics from the trade list and equity curve."""

    def __init__(self, strategy_name: str, initial_capital: Decimal) -> None:
        super().__init__(strategy_name, initial_capital)
        self._trades: list[OptionsTrade] = []
        self._total_volume_traded: int = 0

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------

    def add_trade(self, trade: OptionsTrade) -> None:
        if trade is None:
            raise ValueError("trade must not be None")
        self._trades.append(trade)
        self._total_volume_traded += trade.entry_volume

    @property
    def trades(self) -> list[OptionsTrade]:
        return list(self._trades)

    @property
    def closed_trades(self) -> list[OptionsTrade]:
        return [t for t in self._trades if t.status != OptionsTradeStatus.OPEN]

    # ------------------------------------------------------------------
    # Options-specific metrics
    # ------------------------------------------------------------------

    @property
    def average_premium_collected(self) -> Decimal:
        closed = self.closed_trades
        if not closed:
            return Decimal("0")
        total_premium = sum(
            (t.total_entry_cost() for t in closed), Decimal("0")
        )
        return (total_premium / Decimal(str(len(closed)))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @property
    def total_volume_traded(self) -> int:
        return self._total_volume_traded
