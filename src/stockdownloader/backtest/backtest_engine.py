"""Core backtesting simulation engine that runs a strategy against historical
price data and produces a detailed result with trade log and equity curve."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.model import Trade, Direction, TradeStatus
from stockdownloader.model.price_data import PriceData
from stockdownloader.strategy.trading_strategy import TradingStrategy, Signal


class BacktestEngine:
    """Runs a trading strategy against historical price data and returns a
    :class:`BacktestResult` containing the trade log and equity curve."""

    def __init__(self, initial_capital: Decimal, commission: Decimal) -> None:
        if initial_capital is None:
            raise ValueError("initial_capital must not be None")
        if commission is None:
            raise ValueError("commission must not be None")
        self._initial_capital = initial_capital
        self._commission = commission

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, strategy: TradingStrategy, data: list[PriceData]) -> BacktestResult:
        """Execute *strategy* over *data* and return the backtest result."""
        if strategy is None:
            raise ValueError("strategy must not be None")
        if not data:
            raise ValueError("data must not be None or empty")

        result = BacktestResult(strategy.get_name(), self._initial_capital)
        cash: Decimal = self._initial_capital
        current_trade: Trade | None = None
        equity_curve: list[Decimal] = []

        result.start_date = data[0].date
        result.end_date = data[-1].date

        for i, bar in enumerate(data):
            signal = strategy.evaluate(data, i)

            # Compute current equity
            equity = cash
            if current_trade is not None and current_trade.status == TradeStatus.OPEN:
                position_value = bar.close * Decimal(str(current_trade.shares))
                equity = cash + position_value
            equity_curve.append(equity)

            if signal == Signal.BUY and current_trade is None:
                shares = int(
                    (cash - self._commission) / bar.close
                )

                if shares > 0:
                    cost = bar.close * Decimal(str(shares)) + self._commission
                    cash = cash - cost
                    current_trade = Trade(
                        direction=Direction.LONG,
                        entry_date=bar.date,
                        entry_price=bar.close,
                        shares=shares,
                    )

            elif (
                signal == Signal.SELL
                and current_trade is not None
                and current_trade.status == TradeStatus.OPEN
            ):
                cash = self._close_position(current_trade, bar, cash)
                result.add_trade(current_trade)
                current_trade = None

        # Force-close any remaining open position at the last bar
        if current_trade is not None and current_trade.status == TradeStatus.OPEN:
            last_bar = data[-1]
            cash = self._close_position(current_trade, last_bar, cash)
            result.add_trade(current_trade)

        result.final_capital = cash
        result.equity_curve = equity_curve

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_position(self, trade: Trade, bar: PriceData, cash: Decimal) -> Decimal:
        proceeds = bar.close * Decimal(str(trade.shares)) - self._commission
        trade.close(bar.date, bar.close)
        return cash + proceeds
