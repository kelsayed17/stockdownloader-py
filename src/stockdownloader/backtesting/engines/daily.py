"""Core backtesting simulation engine that runs a strategy against historical
price data and produces a detailed result with trade log and equity curve."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from stockdownloader.backtesting.results.result import BacktestResult
from stockdownloader.core.models import Trade, Direction, TradeStatus
from stockdownloader.core.models.price import PriceData
from stockdownloader.strategies.base import TradingStrategy, Signal


class BacktestEngine:
    """Runs a trading strategy against historical price data and returns a
    :class:`BacktestResult` containing the trade log and equity curve.

    Parameters
    ----------
    initial_capital:
        Starting cash.
    commission:
        Flat commission per trade (entry or exit).
    slippage_pct:
        Proportional slippage applied to the fill price.  E.g. ``0.001``
        means the buy price is 0.1 % *above* the bar close and the sell
        price is 0.1 % *below*.  Models bid-ask spread + market impact.
    """

    def __init__(
        self,
        initial_capital: Decimal,
        commission: Decimal,
        slippage_pct: Decimal | float = 0,
    ) -> None:
        if initial_capital is None:
            raise ValueError("initial_capital must not be None")
        if commission is None:
            raise ValueError("commission must not be None")
        self._initial_capital = initial_capital
        self._commission = commission
        self._slippage_pct = Decimal(str(slippage_pct))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, strategy: TradingStrategy, data: list[PriceData]) -> BacktestResult:
        """Execute *strategy* over *data* and return the backtest result."""
        if strategy is None:
            raise ValueError("strategy must not be None")
        if not data:
            raise ValueError("data must not be None or empty")

        result = BacktestResult(strategy.name, self._initial_capital)
        cash: Decimal = self._initial_capital
        current_trade: Trade | None = None
        equity_curve: list[Decimal] = []

        result.start_date = data[0].date
        result.end_date = data[-1].date

        for i, bar in enumerate(data):
            signal = strategy.evaluate(data, i)

            # Slippage: buys fill above close, sells fill below close.
            buy_price = bar.close * (1 + self._slippage_pct)
            sell_price = bar.close * (1 - self._slippage_pct)

            # Process signal first, then compute equity at bar close
            if signal == Signal.BUY and current_trade is None:
                shares = int(
                    (cash - self._commission) / buy_price
                )

                if shares > 0:
                    cost = buy_price * Decimal(str(shares)) + self._commission
                    cash = cash - cost
                    current_trade = Trade(
                        direction=Direction.LONG,
                        entry_date=bar.date,
                        entry_price=buy_price,
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

            # Compute equity *after* processing the signal so that
            # entry/exit on this bar's close is reflected immediately.
            equity = cash
            if current_trade is not None and current_trade.status == TradeStatus.OPEN:
                position_value = bar.close * Decimal(str(current_trade.shares))
                equity = cash + position_value
            equity_curve.append(equity)

        # Force-close any remaining open position at the last bar
        if current_trade is not None and current_trade.status == TradeStatus.OPEN:
            last_bar = data[-1]
            cash = self._close_position(current_trade, last_bar, cash)
            result.add_trade(current_trade)
            # Update last equity point to reflect the close
            equity_curve[-1] = cash

        result.final_capital = cash
        result.equity_curve = equity_curve

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_position(self, trade: Trade, bar: PriceData, cash: Decimal) -> Decimal:
        sell_price = bar.close * (1 - self._slippage_pct)
        proceeds = sell_price * Decimal(str(trade.shares)) - self._commission
        trade.close(bar.date, sell_price)
        return cash + proceeds
