"""Backtesting engine for options strategies.

Simulates options trading against historical underlying price data using
Black-Scholes pricing for synthetic option premiums.  Supports dated
expirations and strike price selection.

The engine:
- Evaluates the strategy signal at each bar
- Uses Black-Scholes to price synthetic options at the target strike/expiry
- Tracks time decay (theta) and premium changes as the underlying moves
- Handles expiration: closes positions when DTE reaches zero
- Captures volume from the underlying for each trade entry
- Maintains an equity curve accounting for premium flow
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from stockdownloader.backtest.options_backtest_result import OptionsBacktestResult
from stockdownloader.model import OptionsTrade, OptionsDirection, OptionsTradeStatus
from stockdownloader.model.price_data import PriceData
from stockdownloader.strategy.options_strategy import OptionsStrategy, OptionsSignal
from stockdownloader.util import black_scholes_calculator as bsc

_CONTRACT_MULTIPLIER = 100
_DEFAULT_RISK_FREE_RATE = Decimal("0.05")
_VOLATILITY_LOOKBACK = 20


class OptionsBacktestEngine:
    """Runs an options strategy against historical price data and returns an
    :class:`OptionsBacktestResult`."""

    def __init__(
        self,
        initial_capital: Decimal,
        commission: Decimal,
        risk_free_rate: Decimal | None = None,
    ) -> None:
        if initial_capital is None:
            raise ValueError("initial_capital must not be None")
        if commission is None:
            raise ValueError("commission must not be None")
        self._initial_capital = initial_capital
        self._commission = commission
        self._risk_free_rate = (
            risk_free_rate if risk_free_rate is not None else _DEFAULT_RISK_FREE_RATE
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, strategy: OptionsStrategy, data: list[PriceData]
    ) -> OptionsBacktestResult:
        """Execute *strategy* over *data* and return the options backtest result."""
        if strategy is None:
            raise ValueError("strategy must not be None")
        if not data:
            raise ValueError("data must not be None or empty")

        result = OptionsBacktestResult(strategy.get_name(), self._initial_capital)
        cash: Decimal = self._initial_capital
        current_trade: OptionsTrade | None = None
        trade_entry_bar: int = -1
        trade_strike: Decimal = Decimal("0")
        trade_dte: int = 0
        equity_curve: list[Decimal] = []

        result.start_date = data[0].date
        result.end_date = data[-1].date

        # Pre-compute close prices for volatility estimation
        close_prices: list[Decimal] = [bar.close for bar in data]

        for i, bar in enumerate(data):
            spot_price = bar.close

            # ---- Calculate current position value ----
            equity = cash
            if (
                current_trade is not None
                and current_trade.status == OptionsTradeStatus.OPEN
            ):
                bars_held = i - trade_entry_bar
                remaining_dte = max(trade_dte - bars_held, 0)
                time_to_expiry = Decimal(str(remaining_dte)) / Decimal("365")
                vol = bsc.estimate_volatility(
                    close_prices, min(i + 1, _VOLATILITY_LOOKBACK)
                )

                current_premium = bsc.price(
                    strategy.get_option_type(),
                    spot_price,
                    trade_strike,
                    time_to_expiry,
                    self._risk_free_rate,
                    vol,
                )

                position_value = current_premium * Decimal(
                    str(current_trade.contracts * _CONTRACT_MULTIPLIER)
                )

                if strategy.is_short():
                    # Short position: we received premium, owe the current value
                    equity = cash - position_value + current_trade.total_entry_cost()
                else:
                    # Long position: value is the current premium
                    equity = cash + position_value

            equity_curve.append(equity)

            # ---- Check expiration ----
            if (
                current_trade is not None
                and current_trade.status == OptionsTradeStatus.OPEN
            ):
                bars_held = i - trade_entry_bar
                if bars_held >= trade_dte:
                    # Option expired
                    intrinsic = bsc.intrinsic_value(
                        strategy.get_option_type(), spot_price, trade_strike
                    )

                    notional = intrinsic * Decimal(
                        str(current_trade.contracts * _CONTRACT_MULTIPLIER)
                    )

                    if strategy.is_short():
                        current_trade.expire(bar.date, intrinsic)
                        cash = cash - notional - self._commission
                    else:
                        current_trade.expire(bar.date, intrinsic)
                        cash = cash + notional - self._commission

                    result.add_trade(current_trade)
                    current_trade = None
                    continue

            # ---- Evaluate strategy signal ----
            signal = strategy.evaluate(data, i)

            if signal == OptionsSignal.OPEN and current_trade is None:
                vol = bsc.estimate_volatility(
                    close_prices, min(i + 1, _VOLATILITY_LOOKBACK)
                )
                trade_strike = strategy.get_target_strike(spot_price)
                trade_dte = strategy.get_target_days_to_expiry()
                time_to_expiry = Decimal(str(trade_dte)) / Decimal("365")

                premium = bsc.price(
                    strategy.get_option_type(),
                    spot_price,
                    trade_strike,
                    time_to_expiry,
                    self._risk_free_rate,
                    vol,
                )

                if premium <= Decimal("0"):
                    continue

                # Determine number of contracts based on available capital
                if strategy.is_short():
                    # For short options, require margin (use underlying price as collateral)
                    margin_per_contract = spot_price * Decimal(str(_CONTRACT_MULTIPLIER))
                    contracts = int(
                        (cash - self._commission) / margin_per_contract
                    )
                else:
                    # For long options, cost is the premium
                    cost_per_contract = premium * Decimal(str(_CONTRACT_MULTIPLIER))
                    contracts = int(
                        (cash - self._commission) / cost_per_contract
                    )

                # Cap at 10 contracts for risk management
                contracts = min(contracts, 10)

                if contracts > 0:
                    direction = (
                        OptionsDirection.SELL if strategy.is_short()
                        else OptionsDirection.BUY
                    )

                    current_trade = OptionsTrade(
                        option_type=strategy.get_option_type(),
                        direction=direction,
                        strike=trade_strike,
                        expiration_date=bar.date,
                        entry_date=bar.date,
                        entry_premium=premium,
                        contracts=contracts,
                        entry_volume=bar.volume,
                    )
                    trade_entry_bar = i

                    if strategy.is_short():
                        cash = cash + current_trade.total_entry_cost()
                    else:
                        cash = cash - current_trade.total_entry_cost()
                    cash = cash - self._commission

            elif (
                signal == OptionsSignal.CLOSE
                and current_trade is not None
                and current_trade.status == OptionsTradeStatus.OPEN
            ):
                bars_held = i - trade_entry_bar
                remaining_dte = max(trade_dte - bars_held, 0)
                time_to_expiry = Decimal(str(remaining_dte)) / Decimal("365")
                vol = bsc.estimate_volatility(
                    close_prices, min(i + 1, _VOLATILITY_LOOKBACK)
                )

                exit_premium = bsc.price(
                    strategy.get_option_type(),
                    spot_price,
                    trade_strike,
                    time_to_expiry,
                    self._risk_free_rate,
                    vol,
                )

                notional = exit_premium * Decimal(
                    str(current_trade.contracts * _CONTRACT_MULTIPLIER)
                )

                if strategy.is_short():
                    cash = cash - notional
                else:
                    cash = cash + notional
                cash = cash - self._commission

                current_trade.close(bar.date, exit_premium)
                result.add_trade(current_trade)
                current_trade = None

        # Force close any remaining open position at the last bar
        if (
            current_trade is not None
            and current_trade.status == OptionsTradeStatus.OPEN
        ):
            last_bar = data[-1]
            intrinsic = bsc.intrinsic_value(
                strategy.get_option_type(), last_bar.close, trade_strike
            )

            notional = intrinsic * Decimal(
                str(current_trade.contracts * _CONTRACT_MULTIPLIER)
            )

            if strategy.is_short():
                cash = cash - notional
            else:
                cash = cash + notional
            cash = cash - self._commission

            current_trade.close(last_bar.date, intrinsic)
            result.add_trade(current_trade)

        result.final_capital = cash
        result.equity_curve = equity_curve
        return result
