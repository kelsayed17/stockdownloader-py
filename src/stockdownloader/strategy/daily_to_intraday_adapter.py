"""Adapter that wraps any daily TradingStrategy for use with the intraday engine.

Daily strategies return simple BUY/SELL/HOLD signals and have no concept of
stop-loss, take-profit, or risk-per-share.  This adapter bridges the gap by:

1. Delegating signal generation to the wrapped daily strategy.
2. Computing ATR-based stop-loss and take-profit from recent price data.
3. Tracking position state to correctly map SELL → EXIT vs ENTER_SHORT.
4. Managing session boundaries (new trading day resets).

Usage::

    from stockdownloader.strategy.daily_to_intraday_adapter import DailyToIntradayAdapter
    from stockdownloader.strategy.daily.rsi_strategy import RSIStrategy

    daily = RSIStrategy(period=14, oversold=30, overbought=70)
    adapter = DailyToIntradayAdapter(daily)
    signal = adapter.evaluate(data, current_index)
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import (
    IntradayAction,
    IntradaySignal,
    HOLD,
)
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.strategy.trading_strategy import Signal, TradingStrategy
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.big_decimal_math import ZERO

class DailyToIntradayAdapter(IntradayTradingStrategy):
    """Wraps a daily :class:`TradingStrategy` for use with :class:`IntradayBacktestEngine`.

    Parameters
    ----------
    strategy:
        The daily strategy to adapt.
    sl_atr_mult:
        ATR multiplier for stop-loss distance (default: 1.5).
    rr:
        Risk/reward ratio for take-profit (default: 1.5).
    sl_cap:
        Maximum stop-loss distance in dollars (default: $2.00).
    allow_shorts:
        Whether SELL signals with no position open should enter short
        (default: False — only long entries).
    atr_period:
        ATR look-back period in bars (default: 14).
    max_consecutive_losses:
        Circuit-breaker: stop entering new positions after this many
        consecutive losses (default: 5).  Set to 0 to disable.
    """

    def __init__(
        self,
        strategy: TradingStrategy,
        sl_atr_mult: Decimal = Decimal("1.5"),
        rr: Decimal = Decimal("1.5"),
        sl_cap: Decimal = Decimal("2.00"),
        allow_shorts: bool = False,
        atr_period: int = 14,
        max_consecutive_losses: int = 5,
    ) -> None:
        self._strategy = strategy
        self._sl_atr_mult = sl_atr_mult
        self._rr = rr
        self._sl_cap = sl_cap
        self._allow_shorts = allow_shorts
        self._atr_period = atr_period
        self._max_consecutive_losses = max_consecutive_losses
        self._hub = IndicatorHub()

        # Position tracking
        self._in_position = False
        self._position_is_long = False
        self._current_session: str = ""

        # Circuit-breaker state
        self._consecutive_losses: int = 0
        self._circuit_tripped: bool = False
        self._entry_price: Decimal | None = None
        self._position_was_long: bool = False
        self._last_bar_price: Decimal | None = None

    @property
    def name(self) -> str:
        return f"{self._strategy.name} (adapted)"

    @property
    def warmup_period(self) -> int:
        return max(self._strategy.warmup_period, self._atr_period + 1)

    def on_session_start(self, trading_date: str) -> None:
        """Update session tracking for a new trading day.

        Unlike intraday-only strategies (e.g., DMI+VWAP), adapted daily
        strategies naturally hold positions across sessions, so we do NOT
        reset ``_in_position`` here.  Position state is managed exclusively
        by the engine callbacks :meth:`on_position_opened` and
        :meth:`on_position_closed`.

        Circuit-breaker resets on a new session — a new day gets a fresh
        start to avoid permanently locking out the strategy.
        """
        self._current_session = trading_date
        self._consecutive_losses = 0
        self._circuit_tripped = False

    # ------------------------------------------------------------------
    # Engine callbacks — keep internal state in sync with actual trades.
    # ------------------------------------------------------------------

    def on_position_opened(self, is_long: bool) -> None:
        self._in_position = True
        self._position_is_long = is_long
        self._position_was_long = is_long
        self._entry_price = self._last_bar_price

    def on_position_closed(self) -> None:
        self._in_position = False
        self._position_is_long = False

        # Circuit-breaker: check if the last trade was a loss by comparing
        # entry vs exit price.  We track entry price in on_position_opened.
        if self._entry_price is not None and self._last_bar_price is not None:
            if self._position_was_long:
                is_win = self._last_bar_price > self._entry_price
            else:
                is_win = self._last_bar_price < self._entry_price

            if is_win:
                self._consecutive_losses = 0
                self._circuit_tripped = False
            else:
                self._consecutive_losses += 1
                if (
                    self._max_consecutive_losses > 0
                    and self._consecutive_losses >= self._max_consecutive_losses
                ):
                    self._circuit_tripped = True

        self._entry_price = None
        self._position_was_long = False

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        """Evaluate the wrapped strategy and return an intraday signal."""
        # Detect session boundary
        bar = data[current_index]
        trading_date = bar.trading_date
        if trading_date != self._current_session:
            self.on_session_start(trading_date)

        # Track bar price for circuit-breaker PnL detection
        self._last_bar_price = bar.close

        # Warmup check
        if current_index < self.warmup_period:
            return HOLD

        # Get daily strategy signal
        # IntradayPriceData extends PriceData, so data is type-compatible
        daily_signal = self._strategy.evaluate(data, current_index)

        if daily_signal == Signal.HOLD:
            return HOLD

        # Circuit-breaker: skip new entries after consecutive losses
        if self._circuit_tripped and not self._in_position:
            return HOLD

        # Compute ATR-based stop/target
        atr_val = self._hub.atr(data, current_index, self._atr_period)
        if atr_val <= ZERO:
            return HOLD  # can't compute risk without ATR

        sl_distance = min(
            (atr_val * self._sl_atr_mult).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            self._sl_cap,
        )
        tp_distance = (sl_distance * self._rr).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        entry_price = bar.close

        if daily_signal == Signal.BUY:
            if self._in_position and not self._position_is_long:
                # Close existing short
                return IntradaySignal(
                    action=IntradayAction.EXIT,
                    reason=f"{self._strategy.name} BUY → exit short",
                )

            if not self._in_position:
                # Request long entry — actual flag update happens in
                # on_position_opened() callback from the engine.
                return IntradaySignal(
                    action=IntradayAction.ENTER_LONG,
                    mode=self._strategy.name,
                    stop_loss=entry_price - sl_distance,
                    take_profit=entry_price + tp_distance,
                    risk_per_share=sl_distance,
                    reason=f"{self._strategy.name} BUY signal",
                )

            # Already in long position
            return HOLD

        if daily_signal == Signal.SELL:
            if self._in_position and self._position_is_long:
                # Close existing long
                return IntradaySignal(
                    action=IntradayAction.EXIT,
                    reason=f"{self._strategy.name} SELL → exit long",
                )

            if not self._in_position and self._allow_shorts:
                # Request short entry — actual flag update happens in
                # on_position_opened() callback from the engine.
                return IntradaySignal(
                    action=IntradayAction.ENTER_SHORT,
                    mode=self._strategy.name,
                    stop_loss=entry_price + sl_distance,
                    take_profit=entry_price - tp_distance,
                    risk_per_share=sl_distance,
                    reason=f"{self._strategy.name} SELL signal (short)",
                )

            # Not in position and shorts disabled, or already in short
            return HOLD

        return HOLD
