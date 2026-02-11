"""Stores backtest results and computes performance metrics including total
return, Sharpe ratio, max drawdown, win rate, and profit factor."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockdownloader.model import PriceData

from stockdownloader.model import Trade, TradeStatus

_HUNDRED = Decimal("100")


class BacktestResult:
    """Holds the output of a backtest run and lazily computes performance
    metrics from the trade list and equity curve."""

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
        self._trades: list[Trade] = []

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------

    def add_trade(self, trade: Trade) -> None:
        if trade is None:
            raise ValueError("trade must not be None")
        self._trades.append(trade)

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    def get_closed_trades(self) -> list[Trade]:
        return [t for t in self._trades if t.status == TradeStatus.CLOSED]

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    @property
    def total_return(self) -> Decimal:
        return (
            (self.final_capital - self.initial_capital)
            / self.initial_capital
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * _HUNDRED

    @property
    def total_pnl(self) -> Decimal:
        return self.final_capital - self.initial_capital

    @property
    def total_trades(self) -> int:
        return len(self.get_closed_trades())

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.get_closed_trades() if t.is_win())

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
        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) * _HUNDRED

    @property
    def average_win(self) -> Decimal:
        return self._average_profit_loss(winners=True)

    @property
    def average_loss(self) -> Decimal:
        return self._average_profit_loss(winners=False)

    def _average_profit_loss(self, winners: bool) -> Decimal:
        filtered = [
            t.profit_loss for t in self.get_closed_trades() if t.is_win() == winners
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
        for t in self.get_closed_trades():
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
            drawdown = (
                (peak - equity) / peak
            ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * _HUNDRED
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd

    def sharpe_ratio(self, trading_days_per_year: int = 252) -> Decimal:
        if len(self.equity_curve) < 2:
            return Decimal("0")

        daily_returns: list[float] = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1]
            curr = self.equity_curve[i]
            ret = float(
                (curr - prev)
                / prev
            )
            daily_returns.append(ret)

        n = len(daily_returns)
        mean_return = sum(daily_returns) / n

        sum_sq_diff = sum((r - mean_return) ** 2 for r in daily_returns)
        std_dev = math.sqrt(sum_sq_diff / n)

        if std_dev == 0:
            return Decimal("0")

        sharpe = (mean_return / std_dev) * math.sqrt(trading_days_per_year)
        return Decimal(str(sharpe)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def buy_and_hold_return(self, data: list[PriceData]) -> Decimal:
        if not data:
            return Decimal("0")
        first = data[0].close
        last = data[-1].close
        return (
            (last - first) / first
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * _HUNDRED
