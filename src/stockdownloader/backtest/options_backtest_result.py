"""Stores options backtest results and computes performance metrics including
total P/L from premiums, win rate, average premium captured, max drawdown,
and volume statistics."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.model import OptionsTrade, OptionsTradeStatus

_HUNDRED = Decimal("100")


class OptionsBacktestResult:
    """Holds the output of an options backtest run and computes performance
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

    def get_closed_trades(self) -> list[OptionsTrade]:
        return [t for t in self._trades if t.status != OptionsTradeStatus.OPEN]

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
        return self._average_pl(winners=True)

    @property
    def average_loss(self) -> Decimal:
        return self._average_pl(winners=False)

    def _average_pl(self, winners: bool) -> Decimal:
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
            dd = (
                (peak - equity) / peak
            ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * _HUNDRED
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def sharpe_ratio(self, trading_days_per_year: int = 252) -> Decimal:
        if len(self.equity_curve) < 2:
            return Decimal("0")

        daily_returns: list[float] = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1]
            curr = self.equity_curve[i]
            ret = float((curr - prev) / prev)
            daily_returns.append(ret)

        n = len(daily_returns)
        mean_return = sum(daily_returns) / n

        sum_sq_diff = sum((r - mean_return) ** 2 for r in daily_returns)
        std_dev = math.sqrt(sum_sq_diff / n)

        if std_dev == 0:
            return Decimal("0")

        sharpe = (mean_return / std_dev) * math.sqrt(trading_days_per_year)
        return Decimal(str(sharpe)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @property
    def average_premium_collected(self) -> Decimal:
        closed = self.get_closed_trades()
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
