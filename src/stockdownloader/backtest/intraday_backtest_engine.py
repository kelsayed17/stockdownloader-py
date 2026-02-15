"""Backtesting engine for intraday strategies operating on 5-minute bars.

Unlike :class:`BacktestEngine` which processes daily bars with simple BUY/SELL
signals, this engine handles :class:`IntradaySignal` objects with stop-loss,
take-profit, risk-per-share, and both LONG and SHORT positions.

Position accounting
-------------------
* **Long entry**: cash decreases by ``price * shares`` (buying stock).
* **Short entry**: cash increases by ``price * shares`` (selling borrowed
  stock).  A *margin hold* equal to the same notional is tracked separately
  so that ``_compute_shares`` limits sizing to available buying power.
* **Long close**: cash increases by ``exit_price * shares``.
* **Short close**: cash decreases by ``exit_price * shares`` (buying back).

Slippage model
--------------
A configurable *slippage_pct* (default 5 bps = 0.05%) is applied to every
fill, always working *against* the trader:

* **Buy fills** (long entry, short exit) at ``close × (1 + slippage_pct)``.
* **Sell fills** (short entry, long exit) at ``close × (1 − slippage_pct)``.

Set ``slippage_pct=0`` to disable.
"""
from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.model import Trade, Direction, TradeStatus
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import IntradayAction, IntradaySignal
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.util.big_decimal_math import ZERO

logger = logging.getLogger(__name__)

class IntradayBacktestEngine:
    """Runs an intraday strategy against historical 5-minute bar data and
    returns a :class:`BacktestResult` with trade log and equity curve."""

    def __init__(
        self,
        initial_capital: Decimal,
        risk_per_trade: Decimal = Decimal("0.01"),
        commission: Decimal = Decimal("0"),
        slippage_pct: Decimal = Decimal("0.0005"),
    ) -> None:
        if initial_capital is None:
            raise ValueError("initial_capital must not be None")
        self._initial_capital = initial_capital
        self._risk_per_trade = risk_per_trade
        self._commission = commission
        self._slippage_pct = slippage_pct

    # ------------------------------------------------------------------
    # Slippage helper
    # ------------------------------------------------------------------

    def _fill_price(self, price: Decimal, *, is_buy: bool) -> Decimal:
        """Apply slippage to *price*, always working against the trader.

        Buys fill higher, sells fill lower.
        """
        if self._slippage_pct == ZERO:
            return price
        if is_buy:
            return (price * (1 + self._slippage_pct)).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP,
            )
        return (price * (1 - self._slippage_pct)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )

    def run(
        self,
        strategy: IntradayTradingStrategy,
        data: list[IntradayPriceData],
    ) -> BacktestResult:
        """Execute *strategy* over intraday *data* and return the result."""
        if strategy is None:
            raise ValueError("strategy must not be None")
        if not data:
            raise ValueError("data must not be None or empty")

        result = BacktestResult(strategy.name, self._initial_capital)
        cash: Decimal = self._initial_capital
        current_trade: Trade | None = None
        # Margin hold for short positions -- deducted from buying power
        # but not from cash, so equity stays correct.
        margin_hold: Decimal = ZERO
        equity_curve: list[Decimal] = []
        trade_modes: list[str] = []

        result.start_date = data[0].date
        result.end_date = data[-1].date

        for i, bar in enumerate(data):
            signal = strategy.evaluate(data, i)

            # Handle signals
            if (
                signal.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT)
                and current_trade is None
            ):
                direction = (
                    Direction.LONG
                    if signal.action == IntradayAction.ENTER_LONG
                    else Direction.SHORT
                )
                is_buy = direction == Direction.LONG
                fill = self._fill_price(bar.close, is_buy=is_buy)
                buying_power = cash - margin_hold
                shares = self._compute_shares(
                    buying_power, fill, signal.risk_per_share
                )
                if shares > 0:
                    notional = fill * Decimal(str(shares))
                    if direction == Direction.LONG:
                        cash -= notional
                    else:
                        # Short: receive sale proceeds, set aside margin
                        cash += notional
                        margin_hold = notional
                    cash -= self._commission
                    current_trade = Trade(
                        direction=direction,
                        entry_date=bar.date,
                        entry_price=fill,
                        shares=shares,
                    )
                    trade_modes.append(signal.mode)
                    strategy.on_position_opened(
                        direction == Direction.LONG,
                    )

            elif (
                signal.action == IntradayAction.EXIT
                and current_trade is not None
                and current_trade.status == TradeStatus.OPEN
            ):
                exit_is_buy = current_trade.direction == Direction.SHORT
                exit_fill = self._fill_price(bar.close, is_buy=exit_is_buy)
                cash = self._close_position(
                    current_trade, bar.date, exit_fill, cash, self._commission,
                )
                margin_hold = ZERO
                result.add_trade(current_trade)
                current_trade = None
                strategy.on_position_closed()

            # Compute equity
            equity = cash - margin_hold  # available cash (excluding margin)
            if current_trade is not None and current_trade.status == TradeStatus.OPEN:
                shares_d = Decimal(str(current_trade.shares))
                if current_trade.direction == Direction.LONG:
                    equity += bar.close * shares_d
                else:
                    # Short: unrealized P/L = (entry - current) * shares.
                    # ``cash`` already includes the short-sale proceeds and
                    # ``margin_hold`` was subtracted above, so only add the
                    # unrealized gain/loss — NOT margin_hold.
                    unrealized = (
                        (current_trade.entry_price - bar.close) * shares_d
                    )
                    equity += unrealized
            equity_curve.append(equity)

        # Force-close any remaining open position
        if current_trade is not None and current_trade.status == TradeStatus.OPEN:
            last_bar = data[-1]
            force_is_buy = current_trade.direction == Direction.SHORT
            force_fill = self._fill_price(last_bar.close, is_buy=force_is_buy)
            cash = self._close_position(
                current_trade, last_bar.date, force_fill, cash, self._commission,
            )
            margin_hold = ZERO
            result.add_trade(current_trade)
            strategy.on_position_closed()

        result.final_capital = cash - margin_hold
        result.equity_curve = equity_curve

        result.trade_modes = trade_modes

        return result

    def _compute_shares(
        self,
        buying_power: Decimal,
        price: Decimal,
        risk_per_share: Decimal,
    ) -> int:
        """Compute position size based on risk.

        *buying_power* is the cash available for new positions (cash minus
        any existing margin holds).
        """
        if risk_per_share <= ZERO or price <= ZERO or buying_power <= ZERO:
            return 0

        risk_amount = buying_power * self._risk_per_trade
        shares = int(risk_amount / risk_per_share)

        # Cap to what we can afford
        max_shares = int(buying_power / price)
        return min(shares, max_shares)

    @staticmethod
    def _close_position(
        trade: Trade,
        exit_date: str,
        fill_price: Decimal,
        cash: Decimal,
        commission: Decimal = ZERO,
    ) -> Decimal:
        """Close *trade* at *fill_price* on *exit_date* and return updated cash.

        The caller is responsible for applying slippage to the bar's
        close before passing it here.
        """
        trade.close(exit_date, fill_price)
        notional = fill_price * Decimal(str(trade.shares))

        if trade.direction == Direction.LONG:
            cash += notional
        else:
            # Short close: buy back the shares
            cash -= notional

        return cash - commission
