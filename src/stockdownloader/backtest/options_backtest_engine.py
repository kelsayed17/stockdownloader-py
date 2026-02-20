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

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.backtest.backtest_result import OptionsBacktestResult
from stockdownloader.model import OptionsTrade, OptionsDirection, OptionsTradeStatus
from stockdownloader.model.price_data import PriceData
from stockdownloader.strategy.options.options_strategies import OptionsStrategy, OptionsSignal
from stockdownloader.util import black_scholes_calculator as bsc

_CONTRACT_MULTIPLIER = 100
_DEFAULT_RISK_FREE_RATE = Decimal("0.05")
_VOLATILITY_LOOKBACK = 20
_MIN_VOLATILITY_BARS = 5
_DEFAULT_MAX_CONTRACTS = 10


@dataclass(slots=True)
class _PositionState:
    """Mutable state for the currently open options position."""

    trade: OptionsTrade | None = None
    entry_bar: int = -1
    strike: Decimal = Decimal("0")
    dte: int = 0


class OptionsBacktestEngine:
    """Runs an options strategy against historical price data and returns an
    :class:`OptionsBacktestResult`."""

    def __init__(
        self,
        initial_capital: Decimal,
        commission: Decimal,
        risk_free_rate: Decimal | None = None,
        max_contracts: int = _DEFAULT_MAX_CONTRACTS,
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
        self._max_contracts = max_contracts

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

        result = OptionsBacktestResult(strategy.name, self._initial_capital)
        cash: Decimal = self._initial_capital
        pos = _PositionState()
        equity_curve: list[Decimal] = []

        result.start_date = data[0].date
        result.end_date = data[-1].date

        close_prices: list[Decimal] = [bar.close for bar in data]

        for i, bar in enumerate(data):
            # ---- Mark-to-market equity ----
            equity = self._mark_to_market(
                cash, pos, i, bar.close, strategy, close_prices,
            )
            equity_curve.append(equity)

            # ---- Check expiration ----
            if self._check_expiration(pos, i):
                cash = self._settle_expiration(cash, pos, bar, strategy)
                result.add_trade(pos.trade)  # type: ignore[arg-type]
                pos.trade = None
                continue

            # ---- Evaluate strategy signal ----
            signal = strategy.evaluate(data, i)

            if signal == OptionsSignal.OPEN and pos.trade is None:
                cash = self._handle_entry(
                    cash, pos, i, bar, strategy, close_prices, data,
                )

            elif (
                signal == OptionsSignal.CLOSE
                and pos.trade is not None
                and pos.trade.status == OptionsTradeStatus.OPEN
            ):
                cash = self._handle_exit(
                    cash, pos, i, bar, strategy, close_prices,
                )
                result.add_trade(pos.trade)
                pos.trade = None

        # Force close any remaining open position at the last bar
        if pos.trade is not None and pos.trade.status == OptionsTradeStatus.OPEN:
            cash = self._force_close(cash, pos, data[-1], strategy)
            result.add_trade(pos.trade)

        result.final_capital = cash
        result.equity_curve = equity_curve
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_vol(
        self, close_prices: list[Decimal], bar_index: int,
    ) -> Decimal:
        """Estimate historical volatility using available price data."""
        lookback = max(min(bar_index + 1, _VOLATILITY_LOOKBACK), _MIN_VOLATILITY_BARS)
        return bsc.estimate_volatility(close_prices, lookback)

    def _mark_to_market(
        self,
        cash: Decimal,
        pos: _PositionState,
        bar_index: int,
        spot: Decimal,
        strategy: OptionsStrategy,
        close_prices: list[Decimal],
    ) -> Decimal:
        """Compute current equity including open position value."""
        if pos.trade is None or pos.trade.status != OptionsTradeStatus.OPEN:
            return cash

        remaining_dte = max(pos.dte - (bar_index - pos.entry_bar), 0)
        time_to_expiry = Decimal(str(remaining_dte)) / Decimal("365")
        vol = self._estimate_vol(close_prices, bar_index)

        current_premium = bsc.price(
            strategy.option_type, spot, pos.strike,
            time_to_expiry, self._risk_free_rate, vol,
        )
        position_value = current_premium * Decimal(
            str(pos.trade.contracts * _CONTRACT_MULTIPLIER)
        )

        if strategy.is_short():
            return cash - position_value + pos.trade.total_entry_cost()
        return cash + position_value

    def _check_expiration(
        self,
        pos: _PositionState,
        bar_index: int,
    ) -> bool:
        """Return True if the current position has expired."""
        if pos.trade is None or pos.trade.status != OptionsTradeStatus.OPEN:
            return False
        return (bar_index - pos.entry_bar) >= pos.dte

    def _settle_expiration(
        self,
        cash: Decimal,
        pos: _PositionState,
        bar: PriceData,
        strategy: OptionsStrategy,
    ) -> Decimal:
        """Settle an expired position and return updated cash."""
        assert pos.trade is not None
        intrinsic = bsc.intrinsic_value(
            strategy.option_type, bar.close, pos.strike,
        )
        notional = intrinsic * Decimal(
            str(pos.trade.contracts * _CONTRACT_MULTIPLIER)
        )

        pos.trade.expire(bar.date, intrinsic)
        if strategy.is_short():
            return cash - notional - self._commission
        return cash + notional - self._commission

    def _handle_entry(
        self,
        cash: Decimal,
        pos: _PositionState,
        bar_index: int,
        bar: PriceData,
        strategy: OptionsStrategy,
        close_prices: list[Decimal],
        data: list[PriceData],
    ) -> Decimal:
        """Open a new options position.  Returns updated cash."""
        spot = bar.close
        vol = self._estimate_vol(close_prices, bar_index)
        pos.strike = strategy.get_target_strike(spot)
        pos.dte = strategy.target_days_to_expiry
        time_to_expiry = Decimal(str(pos.dte)) / Decimal("365")

        premium = bsc.price(
            strategy.option_type, spot, pos.strike,
            time_to_expiry, self._risk_free_rate, vol,
        )
        if premium <= Decimal("0"):
            return cash

        contracts = self._size_position(cash, spot, premium, strategy)
        if contracts <= 0:
            return cash

        direction = (
            OptionsDirection.SELL if strategy.is_short()
            else OptionsDirection.BUY
        )
        exp_index = min(bar_index + pos.dte, len(data) - 1)

        pos.trade = OptionsTrade(
            option_type=strategy.option_type,
            direction=direction,
            strike=pos.strike,
            expiration_date=data[exp_index].date,
            entry_date=bar.date,
            entry_premium=premium,
            contracts=contracts,
            entry_volume=bar.volume,
        )
        pos.entry_bar = bar_index

        if strategy.is_short():
            cash = cash + pos.trade.total_entry_cost()
        else:
            cash = cash - pos.trade.total_entry_cost()
        return cash - self._commission

    def _size_position(
        self,
        cash: Decimal,
        spot: Decimal,
        premium: Decimal,
        strategy: OptionsStrategy,
    ) -> int:
        """Determine number of contracts to trade."""
        if strategy.is_short():
            margin_per_contract = spot * Decimal(str(_CONTRACT_MULTIPLIER))
            contracts = int((cash - self._commission) / margin_per_contract)
        else:
            cost_per_contract = premium * Decimal(str(_CONTRACT_MULTIPLIER))
            contracts = int((cash - self._commission) / cost_per_contract)
        return min(contracts, self._max_contracts)

    def _handle_exit(
        self,
        cash: Decimal,
        pos: _PositionState,
        bar_index: int,
        bar: PriceData,
        strategy: OptionsStrategy,
        close_prices: list[Decimal],
    ) -> Decimal:
        """Close an open position on signal.  Returns updated cash."""
        assert pos.trade is not None
        spot = bar.close
        remaining_dte = max(pos.dte - (bar_index - pos.entry_bar), 0)
        time_to_expiry = Decimal(str(remaining_dte)) / Decimal("365")
        vol = self._estimate_vol(close_prices, bar_index)

        exit_premium = bsc.price(
            strategy.option_type, spot, pos.strike,
            time_to_expiry, self._risk_free_rate, vol,
        )
        notional = exit_premium * Decimal(
            str(pos.trade.contracts * _CONTRACT_MULTIPLIER)
        )

        if strategy.is_short():
            cash = cash - notional
        else:
            cash = cash + notional
        cash = cash - self._commission

        pos.trade.close(bar.date, exit_premium)
        return cash

    def _force_close(
        self,
        cash: Decimal,
        pos: _PositionState,
        last_bar: PriceData,
        strategy: OptionsStrategy,
    ) -> Decimal:
        """Force-close an open position at end of data."""
        assert pos.trade is not None
        intrinsic = bsc.intrinsic_value(
            strategy.option_type, last_bar.close, pos.strike,
        )
        notional = intrinsic * Decimal(
            str(pos.trade.contracts * _CONTRACT_MULTIPLIER)
        )

        if strategy.is_short():
            cash = cash - notional
        else:
            cash = cash + notional
        cash = cash - self._commission

        pos.trade.close(last_bar.date, intrinsic)
        return cash
