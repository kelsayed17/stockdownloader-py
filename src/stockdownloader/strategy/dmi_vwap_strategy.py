"""DMI + Anchored VWAP intraday strategy.

Combines the Directional Movement Index (DMI/ADX) with an anchored VWAP
that resets at the start of each trading session (09:30 ET).

Entry rules
-----------
* **Long** (buy calls): price > VWAP **AND** +DI > -DI **AND** ADX > threshold
* **Short** (buy puts): price < VWAP **AND** -DI > +DI **AND** ADX > threshold
* No trade otherwise.

Exit rules
----------
* Signal reversal (opposite conditions met).
* End-of-day (last bar of session, configurable ``eod_exit_bar``).

Position sizing
---------------
Risk per share is set to ``sl_atr_mult * ATR``.  The backtest engine's
``risk_per_trade`` fraction (default 1 %) determines how many shares/contracts
to allocate per trade.  Stop-loss is ATR-based; take-profit uses a
reward/risk ratio (``rr``).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.intraday_signal import (
    HOLD,
    IntradayAction,
    IntradaySignal,
)
from stockdownloader.strategy.intraday_trading_strategy import (
    IntradayTradingStrategy,
)
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.pinescript_models import (
    Condition, Indicator, Input, StrategyDefinition,
)

_QUANT = Decimal("0.01")

@dataclass(slots=True)
class DmiVwapConfig:
    """Tunable parameters for the DMI+VWAP strategy."""

    # DMI / ADX
    dmi_period: int = 14
    adx_threshold: Decimal = Decimal("25")

    # Risk management
    atr_period: int = 14
    sl_atr_mult: Decimal = Decimal("1.5")
    rr: Decimal = Decimal("2.0")

    # Session management
    bars_per_day: int = 78  # 5-min bars 09:30-16:00
    eod_exit_bar: int = 76  # exit ~2 bars before close (15:50)
    min_entry_bar: int = 3  # skip first 15 min for indicator warmup

    # Trade management
    min_hold_bars: int = 12  # hold at least 1 hr before neutral exit
    cooldown_bars: int = 6  # wait 30 min after exit before re-entry
    max_trades_per_day: int = 3  # cap daily trades to avoid overtrading

    # Filters
    require_adx_rising: bool = False
    min_di_spread: Decimal = Decimal("0")  # minimum +DI - -DI gap

class DmiVwapStrategy(IntradayTradingStrategy):
    """Intraday strategy using DMI crossovers filtered by anchored VWAP.

    The anchored VWAP resets each session (09:30 ET).  The strategy enters
    long when price is above VWAP with bullish DMI, and short when price
    is below VWAP with bearish DMI.  All positions are closed at EOD.
    """

    def __init__(self, config: DmiVwapConfig | None = None) -> None:
        self._cfg = config or DmiVwapConfig()
        self._hub = IndicatorHub()

        # Per-session state
        self._session_date: str = ""
        self._session_bar: int = 0
        self._in_position: bool = False
        self._position_is_long: bool = False
        self._entry_bar: int = 0  # session bar when position was opened
        self._last_exit_bar: int = -999  # session bar when last exit occurred
        self._day_trades: int = 0  # trade count for current session

    # ------------------------------------------------------------------
    # ABC overrides
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "DMI+VWAP"

    @property
    def warmup_period(self) -> int:
        # Need enough bars for ADX (2 * period) + ATR (period) + session VWAP
        return max(self._cfg.dmi_period * 2 + 10, self._cfg.atr_period + 10)

    def on_session_start(self, trading_date: str) -> None:
        """Reset per-session state."""
        self._session_date = trading_date
        self._session_bar = 0
        # Force-exit any overnight position (engine handles final close,
        # but this resets our tracking).
        self._in_position = False
        self._position_is_long = False
        self._entry_bar = 0
        self._last_exit_bar = -999
        self._day_trades = 0

    # ------------------------------------------------------------------
    # Engine callbacks
    # ------------------------------------------------------------------

    def on_position_opened(self, is_long: bool) -> None:
        """Called by the engine after a position is successfully opened."""
        self._in_position = True
        self._position_is_long = is_long
        self._day_trades += 1

    def on_position_closed(self) -> None:
        """Called by the engine after a position is closed."""
        self._in_position = False
        self._position_is_long = False

    # ------------------------------------------------------------------
    # PineScript generation
    # ------------------------------------------------------------------

    def to_pinescript(self) -> StrategyDefinition:
        """Convert DMI+VWAP entry/exit logic to PineScript.

        Note: Session management (cooldown, max trades/day, min hold bars,
        confluence scoring) cannot be expressed in Pine indicator conditions.
        The Pine version captures core entry/exit logic.
        """
        return StrategyDefinition(
            name="DMI + VWAP",
            short_name="DMI-VWAP",
            description=(
                "Intraday DMI + session-anchored VWAP.\n"
                "Long: price > VWAP, +DI > -DI, ADX >= threshold.\n"
                "Short: price < VWAP, -DI > +DI, ADX >= threshold."
            ),
            inputs=[
                Input.int_("dmiPeriod", self._cfg.dmi_period, "DMI Period"),
                Input.float_("adxThreshold", float(self._cfg.adx_threshold),
                             "ADX Threshold", step=0.5),
            ],
            indicators=[
                Indicator.session_vwap("vwapValue"),
                Indicator.dmi("dmiPeriod", "dmiPeriod"),
            ],
            long_entry=Condition(
                "close > vwapValue and plusDI > minusDI "
                "and adxValue >= adxThreshold",
                "Price above VWAP + bullish DMI + strong trend",
            ),
            short_entry=Condition(
                "close < vwapValue and minusDI > plusDI "
                "and adxValue >= adxThreshold",
                "Price below VWAP + bearish DMI + strong trend",
            ),
            use_session_filter=True,
            long_label="Buy Call",
            short_label="Buy Put",
        )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        bar = data[current_index]

        # ---- Session boundary detection ----
        if current_index == 0 or bar.trading_date != data[current_index - 1].trading_date:
            # If we still have an open position from the previous session,
            # emit an EXIT before resetting state so the engine can close
            # the trade at this bar's price (first bar of new session).
            if self._in_position:
                self._last_exit_bar = self._session_bar
                self.on_session_start(bar.trading_date)
                self._session_bar += 1
                return IntradaySignal(
                    action=IntradayAction.EXIT,
                    mode="DMI_VWAP_EOD",
                    reason="Carry-over position closed at session open",
                )
            self.on_session_start(bar.trading_date)

        self._session_bar += 1
        bar_of_day = self._session_bar

        # ---- Warmup guard ----
        if current_index < self.warmup_period:
            return HOLD

        # ---- EOD exit ----
        if bar_of_day >= self._cfg.eod_exit_bar and self._in_position:
            return self._exit("DMI_VWAP_EOD", "End-of-day exit", bar_of_day)

        # ---- Compute indicators & classify conditions ----
        adx_result = self._hub.adx(data, current_index, self._cfg.dmi_period)
        vwap = self._hub.session_vwap(data, current_index)
        atr = self._hub.atr(data, current_index, self._cfg.atr_period)
        price = bar.close

        is_bullish, is_bearish = self._classify_conditions(
            data, current_index, price, vwap, adx_result,
        )

        # ---- If in position: check for exits ----
        if self._in_position:
            return self._evaluate_exits(
                bar_of_day, is_bullish, is_bearish, adx_result, vwap,
            )

        # ---- Entry filters ----
        if not self._can_enter(bar_of_day):
            return HOLD

        # ---- Entry signals ----
        return self._evaluate_entries(
            price, atr, vwap, adx_result, is_bullish, is_bearish, bar_of_day,
        )

    # ------------------------------------------------------------------
    # Evaluate helpers
    # ------------------------------------------------------------------

    def _classify_conditions(
        self,
        data: list[IntradayPriceData],
        current_index: int,
        price: Decimal,
        vwap: Decimal,
        adx_result,
    ) -> tuple[bool, bool]:
        """Classify whether current conditions are bullish or bearish."""
        cfg = self._cfg
        adx_val = adx_result.adx
        plus_di = adx_result.plus_di
        minus_di = adx_result.minus_di

        is_bullish = (
            price > vwap
            and plus_di > minus_di
            and adx_val >= cfg.adx_threshold
            and (plus_di - minus_di) >= cfg.min_di_spread
        )
        is_bearish = (
            price < vwap
            and minus_di > plus_di
            and adx_val >= cfg.adx_threshold
            and (minus_di - plus_di) >= cfg.min_di_spread
        )

        if cfg.require_adx_rising and current_index > 0:
            prev_adx = self._hub.adx(data, current_index - 1, cfg.dmi_period)
            if adx_val < prev_adx.adx:
                is_bullish = False
                is_bearish = False

        return is_bullish, is_bearish

    def _evaluate_exits(
        self,
        bar_of_day: int,
        is_bullish: bool,
        is_bearish: bool,
        adx_result,
        vwap: Decimal,
    ) -> IntradaySignal:
        """Check exit conditions for an open position."""
        adx_val = adx_result.adx
        plus_di = adx_result.plus_di
        minus_di = adx_result.minus_di
        bars_held = bar_of_day - self._entry_bar

        # Reversal exit (always allowed, even during min hold)
        if self._position_is_long and is_bearish:
            return self._exit(
                "DMI_VWAP_REVERSAL",
                f"Signal reversal: -DI({float(minus_di):.1f}) > +DI({float(plus_di):.1f}), price < VWAP",
                bar_of_day,
            )
        if not self._position_is_long and is_bullish:
            return self._exit(
                "DMI_VWAP_REVERSAL",
                f"Signal reversal: +DI({float(plus_di):.1f}) > -DI({float(minus_di):.1f}), price > VWAP",
                bar_of_day,
            )

        # Neutral exit (only after min hold period)
        if bars_held >= self._cfg.min_hold_bars:
            if self._position_is_long and not is_bullish:
                if adx_val < self._cfg.adx_threshold or minus_di >= plus_di:
                    return self._exit(
                        "DMI_VWAP_NEUTRAL",
                        f"Conditions lost: ADX={float(adx_val):.1f}",
                        bar_of_day,
                    )
            if not self._position_is_long and not is_bearish:
                if adx_val < self._cfg.adx_threshold or plus_di >= minus_di:
                    return self._exit(
                        "DMI_VWAP_NEUTRAL",
                        f"Conditions lost: ADX={float(adx_val):.1f}",
                        bar_of_day,
                    )

        return HOLD

    def _exit(self, mode: str, reason: str, bar_of_day: int) -> IntradaySignal:
        """Return an EXIT signal.

        Position state is cleared by the engine via :meth:`on_position_closed`.
        We only record ``_last_exit_bar`` here for cooldown tracking, since
        the engine always processes the EXIT signal.
        """
        self._last_exit_bar = bar_of_day
        return IntradaySignal(
            action=IntradayAction.EXIT,
            mode=mode,
            reason=reason,
        )

    def _can_enter(self, bar_of_day: int) -> bool:
        """Check whether entry filters allow a new trade."""
        if bar_of_day < self._cfg.min_entry_bar:
            return False
        # Don't enter at or after EOD exit bar — there won't be time to
        # manage the position and the EOD exit won't fire until next session.
        if bar_of_day >= self._cfg.eod_exit_bar:
            return False
        if (bar_of_day - self._last_exit_bar) < self._cfg.cooldown_bars:
            return False
        if self._day_trades >= self._cfg.max_trades_per_day:
            return False
        return True

    def _evaluate_entries(
        self,
        price: Decimal,
        atr: Decimal,
        vwap: Decimal,
        adx_result,
        is_bullish: bool,
        is_bearish: bool,
        bar_of_day: int,
    ) -> IntradaySignal:
        """Generate entry signal if conditions are met."""
        if is_bullish:
            return self._enter(
                go_long=True, price=price, atr=atr, vwap=vwap,
                adx_result=adx_result, bar_of_day=bar_of_day,
            )
        if is_bearish:
            return self._enter(
                go_long=False, price=price, atr=atr, vwap=vwap,
                adx_result=adx_result, bar_of_day=bar_of_day,
            )
        return HOLD

    def _enter(
        self,
        go_long: bool,
        price: Decimal,
        atr: Decimal,
        vwap: Decimal,
        adx_result,
        bar_of_day: int,
    ) -> IntradaySignal:
        """Build and return an entry signal.

        Position state (``_in_position``, ``_position_is_long``) is set
        by the engine via :meth:`on_position_opened` — **not** here — so
        that a rejected entry (shares = 0) doesn't leave a phantom position.

        ``_entry_bar`` is stored optimistically because it's only used for
        ``min_hold_bars`` gating, which is safe even if the entry is rejected.
        """
        sl_distance = (atr * self._cfg.sl_atr_mult).quantize(_QUANT, rounding=ROUND_HALF_UP)
        tp_distance = (sl_distance * self._cfg.rr).quantize(_QUANT, rounding=ROUND_HALF_UP)

        self._entry_bar = bar_of_day

        plus_di = adx_result.plus_di
        minus_di = adx_result.minus_di
        adx_val = adx_result.adx

        if go_long:
            dom_di, sub_di = plus_di, minus_di
            action = IntradayAction.ENTER_LONG
            mode = "DMI_VWAP_LONG"
            sl = price - sl_distance
            tp = price + tp_distance
            label = "Long"
            di_label = f"+DI({float(plus_di):.1f}) > -DI({float(minus_di):.1f})"
            rel = ">"
        else:
            dom_di, sub_di = minus_di, plus_di
            action = IntradayAction.ENTER_SHORT
            mode = "DMI_VWAP_SHORT"
            sl = price + sl_distance
            tp = price - tp_distance
            label = "Short"
            di_label = f"-DI({float(minus_di):.1f}) > +DI({float(plus_di):.1f})"
            rel = "<"

        return IntradaySignal(
            action=action,
            mode=mode,
            stop_loss=sl,
            take_profit=tp,
            risk_per_share=sl_distance,
            confluence_score=self._compute_confluence(adx_val, dom_di, sub_di, price, vwap),
            max_score=10,
            reason=(
                f"{label}: price({float(price):.2f}) {rel} VWAP({float(vwap):.2f}), "
                f"{di_label}, ADX={float(adx_val):.1f}"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confluence(
        adx: Decimal,
        dom_di: Decimal,
        sub_di: Decimal,
        price: Decimal,
        vwap: Decimal,
    ) -> int:
        """Score 0-10 based on strength of alignment."""
        score = 0
        # ADX strength tiers
        if adx >= Decimal("40"):
            score += 3
        elif adx >= Decimal("30"):
            score += 2
        else:
            score += 1

        # DI spread
        spread = dom_di - sub_di
        if spread >= Decimal("15"):
            score += 3
        elif spread >= Decimal("10"):
            score += 2
        elif spread >= Decimal("5"):
            score += 1

        # Price distance from VWAP (further = more conviction)
        if vwap > ZERO:
            dist_pct = abs(price - vwap) / vwap * Decimal("100")
            if dist_pct >= Decimal("0.3"):
                score += 2
            elif dist_pct >= Decimal("0.15"):
                score += 1

        # DI dominance (ratio)
        if sub_di > ZERO and dom_di / sub_di >= Decimal("2"):
            score += 2
        elif sub_di > ZERO and dom_di / sub_di >= Decimal("1.5"):
            score += 1

        return min(score, 10)
