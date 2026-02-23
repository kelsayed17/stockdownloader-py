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
A configurable *slippage_pct* (default 2 bps = 0.02%) is applied to every
fill, always working *against* the trader:

* **Buy fills** (long entry, short exit) at ``close × (1 + slippage_pct)``.
* **Sell fills** (short entry, long exit) at ``close × (1 − slippage_pct)``.

Set ``slippage_pct=0`` to disable.

Risk scaling (opt-in)
---------------------
Two composable risk-scaling mechanisms reduce position size when conditions
are unfavourable.  Both default to *off* and compose multiplicatively::

    effective_risk = base_risk × vol_ratio × dd_factor

**Volatility scaling** (``vol_scale=True``):  Tracks a rolling window of
daily ATR values.  When the current session's ATR exceeds the median of
the lookback window, ``risk_per_trade`` is scaled down proportionally
(``median / current_atr``, capped at 1.0).

**Drawdown throttle** (``dd_throttle=True``):  Monitors peak equity.
Graduated tiers reduce size as drawdown deepens (default: -5% → 0.75×,
-10% → 0.50×, -15% → halt).
"""
from __future__ import annotations

import logging
import statistics
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.model import Trade, Direction, TradeStatus
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.model.trade import IntradayAction, IntradaySignal
from stockdownloader.strategy.trading_strategy import IntradayTradingStrategy
from stockdownloader.util.math import ZERO, ONE

logger = logging.getLogger(__name__)

# Drawdown tier defaults
_DD_TIER_1 = Decimal("0.05")   # -5% → scale to 0.75
_DD_TIER_2 = Decimal("0.10")   # -10% → scale to 0.50
_DD_TIER_3 = Decimal("0.15")   # -15% → halt (0.00)
_DD_SCALE_1 = Decimal("0.75")
_DD_SCALE_2 = Decimal("0.50")


class IntradayBacktestEngine:
    """Runs an intraday strategy against historical 5-minute bar data and
    returns a :class:`BacktestResult` with trade log and equity curve."""

    def __init__(
        self,
        initial_capital: Decimal,
        risk_per_trade: Decimal = Decimal("0.01"),
        commission: Decimal = Decimal("0"),
        slippage_pct: Decimal = Decimal("0.0002"),
        *,
        vol_scale: bool = False,
        vol_lookback: int = 60,
        dd_throttle: bool = False,
        dd_tier1: Decimal = _DD_TIER_1,
        dd_tier2: Decimal = _DD_TIER_2,
        dd_tier3: Decimal = _DD_TIER_3,
    ) -> None:
        if initial_capital is None:
            raise ValueError("initial_capital must not be None")
        self._initial_capital = initial_capital
        self._risk_per_trade = risk_per_trade
        self._commission = commission
        self._slippage_pct = slippage_pct

        # -- Volatility-scaled sizing --
        self._vol_scale = vol_scale
        self._vol_lookback = vol_lookback

        # -- Drawdown throttle --
        self._dd_throttle = dd_throttle
        self._dd_tier1 = dd_tier1
        self._dd_tier2 = dd_tier2
        self._dd_tier3 = dd_tier3

    # ------------------------------------------------------------------
    # Risk-scaling helpers
    # ------------------------------------------------------------------

    def _vol_scaled_risk(
        self, daily_atr: Decimal, atr_history: list[Decimal],
    ) -> Decimal:
        """Return risk_per_trade scaled by volatility ratio.

        When current ATR exceeds the rolling median, reduce risk
        proportionally.  When ATR is at or below median, return
        the base risk (ratio capped at 1.0).
        """
        if not self._vol_scale or not atr_history or daily_atr <= ZERO:
            return self._risk_per_trade

        # Compute median of the lookback window
        window = atr_history[-self._vol_lookback:]
        median_atr = Decimal(str(statistics.median(float(a) for a in window)))

        if median_atr <= ZERO:
            return self._risk_per_trade

        vol_ratio = min(median_atr / daily_atr, ONE)
        return self._risk_per_trade * vol_ratio

    def _dd_scale_factor(self, current_equity: Decimal, peak_equity: Decimal) -> Decimal:
        """Return a drawdown scaling factor (1.0 / 0.75 / 0.50 / 0.0).

        Graduated tiers reduce position size as the drawdown deepens
        from peak equity.
        """
        if not self._dd_throttle or peak_equity <= ZERO:
            return ONE

        dd_pct = (peak_equity - current_equity) / peak_equity

        if dd_pct >= self._dd_tier3:
            return ZERO
        if dd_pct >= self._dd_tier2:
            return _DD_SCALE_2
        if dd_pct >= self._dd_tier1:
            return _DD_SCALE_1
        return ONE

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

        # -- Risk-scaling state --
        peak_equity: Decimal = self._initial_capital
        # Daily bar aggregation for vol-scaling
        atr_history: list[Decimal] = []
        current_day_atr: Decimal = ZERO
        prev_session_date: str = ""
        session_high: Decimal = ZERO
        session_low: Decimal = Decimal("999999")
        prev_close: Decimal = ZERO

        for i, bar in enumerate(data):
            bar_date = bar.date[:10]

            # -- Track daily bars for volatility scaling --
            if self._vol_scale:
                if bar_date != prev_session_date:
                    # New session: finalize previous session's true range
                    if prev_session_date and session_high > ZERO:
                        if prev_close > ZERO:
                            tr = max(
                                session_high - session_low,
                                abs(session_high - prev_close),
                                abs(session_low - prev_close),
                            )
                        else:
                            tr = session_high - session_low
                        atr_history.append(tr)
                        # Simple rolling ATR (average of last 14 TRs)
                        window = atr_history[-14:]
                        current_day_atr = sum(window) / Decimal(str(len(window)))
                    prev_close = session_low  # approx prev close
                    if prev_session_date:
                        # Use last bar's close from previous session as prev_close
                        # (the bar before this one is the last of the prev session)
                        prev_close = data[i - 1].close if i > 0 else ZERO
                    prev_session_date = bar_date
                    session_high = bar.high
                    session_low = bar.low
                else:
                    if bar.high > session_high:
                        session_high = bar.high
                    if bar.low < session_low:
                        session_low = bar.low

            # -- Update peak equity when flat --
            if self._dd_throttle and current_trade is None:
                flat_equity = cash - margin_hold
                if flat_equity > peak_equity:
                    peak_equity = flat_equity

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

                # -- Apply risk scaling --
                effective_risk = self._vol_scaled_risk(current_day_atr, atr_history)
                dd_factor = self._dd_scale_factor(cash - margin_hold, peak_equity)
                effective_risk = effective_risk * dd_factor

                shares = self._compute_shares(
                    buying_power, fill, signal.risk_per_share,
                    risk_per_trade=effective_risk,
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
        risk_per_trade: Decimal | None = None,
    ) -> int:
        """Compute position size based on risk.

        *buying_power* is the cash available for new positions (cash minus
        any existing margin holds).

        *risk_per_trade* overrides the instance default when provided (used
        by volatility scaling and drawdown throttle).
        """
        if risk_per_share <= ZERO or price <= ZERO or buying_power <= ZERO:
            return 0

        effective_risk = risk_per_trade if risk_per_trade is not None else self._risk_per_trade
        risk_amount = buying_power * effective_risk
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
