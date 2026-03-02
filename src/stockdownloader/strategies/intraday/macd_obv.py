"""SPY MACD+OBV (Tournament Winner) -- intraday strategy.

Pine Script equivalent: output/pinescript/daily/spy/spy_macd_obv.pine
Tournament rank: Grand Winner | OOS Score: +74.03 | Degradation: 0.96

Entry:
  Long  -- MACD(12,26,9) bullish crossover + EMA(OBV,5) rising
  Short -- MACD(12,26,9) bearish crossunder + EMA(OBV,5) falling

Exit:
  Reverse MACD crossover, or SL/TP hit, or EOD.

Risk:
  SL = min(ATR(14) * 1.5, $2.00)
  TP = SL * 1.5 (R:R)
  Breakeven at 0.5R -- SL moves to entry +/- $0.05
  Max 4 trades/day, 3-bar spacing, circuit breaker after 3 losses,
  halt on -3% daily drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.math import ZERO
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy, InfraExitConfig
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    clamp_sl_dist,
    directional_sl_tp,
    make_entry_signal,
)

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.intraday.session import BarContext


# -- OBV EMA helper -----------------------------------------------------------


@dataclass
class _ObvEmaState:
    """Tracks EMA of OBV for rising/falling detection."""

    period: int = 5
    _ema: Decimal = ZERO
    _prev_ema: Decimal = ZERO
    _initialized: bool = False
    _count: int = 0

    def update(self, obv: Decimal) -> None:
        """Update with a new OBV value."""
        if not self._initialized:
            self._count += 1
            if self._count == 1:
                self._ema = obv
                self._prev_ema = obv
            else:
                alpha = Decimal(2) / Decimal(self.period + 1)
                self._prev_ema = self._ema
                self._ema = alpha * obv + (Decimal(1) - alpha) * self._ema
            if self._count >= self.period:
                self._initialized = True
        else:
            alpha = Decimal(2) / Decimal(self.period + 1)
            self._prev_ema = self._ema
            self._ema = alpha * obv + (Decimal(1) - alpha) * self._ema

    def is_rising(self) -> bool:
        return self._ema > self._prev_ema

    def is_falling(self) -> bool:
        return self._ema < self._prev_ema

    def reset(self) -> None:
        self._ema = ZERO
        self._prev_ema = ZERO
        self._initialized = False
        self._count = 0


# -- Config --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MACDOBVConfig(InfraExitConfig):
    """Configuration for SPY MACD+OBV strategy."""

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    obv_smooth: int = 5
    atr_len: int = 14
    sl_mult: Decimal = Decimal("1.5")
    rr_ratio: Decimal = Decimal("1.5")
    sl_cap: Decimal = Decimal("2.0")
    be_trigger: Decimal = Decimal("0.5")
    max_day: int = 4
    spacing: int = 3
    circuit: int = 3
    day_loss: Decimal = Decimal("3.0")
    allow_longs: bool = True
    allow_shorts: bool = True


# -- Strategy ------------------------------------------------------------------


class MACDOBVStrategy(BaseIntradayStrategy):
    """SPY MACD+OBV intraday strategy (Tournament Winner)."""

    def __init__(self, **overrides: object) -> None:
        self._c = MACDOBVConfig(**overrides) if overrides else MACDOBVConfig()
        self._infra = IntradayInfra(self._c, IntradayExitManager())
        super().__init__()
        self._obv_state = _ObvEmaState(period=self._c.obv_smooth)
        self._data: list[IntradayPriceData] = []
        self._idx: int = 0
        # NOTE: _prev_macd_*, _obv_state intentionally NOT reset across sessions.
        # Pine Script's ta.crossover/ta.crossunder operate on a continuous time
        # series — no day-boundary reset — so we replicate that behaviour here.
        self._prev_macd_line: Decimal = ZERO
        self._prev_macd_sig: Decimal = ZERO

    @property
    def name(self) -> str:
        return "SPY MACD+OBV"

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        self._data = data
        self._idx = current_index
        hub = self._infra.hub
        obv_val = hub.obv(data, current_index)
        self._obv_state.update(obv_val)
        return self._infra.run_bar(
            data, current_index, self._evaluate_entry, self._ENTRY_FLAGS,
        )

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state

        if s.day_trades >= c.max_day:
            return None
        if ctx.bar_of_day - s.last_entry_bar < c.spacing and s.last_entry_bar > 0:
            return None
        if not ctx.is_good_time:
            return None

        hub = self._infra.hub
        data, i = self._data, self._idx
        macd_line = hub.macd_line(data, i, c.macd_fast, c.macd_slow)
        macd_sig = hub.macd_signal(data, i, c.macd_fast, c.macd_slow, c.macd_signal)

        crossover = self._prev_macd_line <= self._prev_macd_sig and macd_line > macd_sig
        crossunder = self._prev_macd_line >= self._prev_macd_sig and macd_line < macd_sig

        self._prev_macd_line = macd_line
        self._prev_macd_sig = macd_sig

        obv_rising = self._obv_state.is_rising()
        obv_falling = self._obv_state.is_falling()

        go_long = crossover and obv_rising and c.allow_longs
        go_short = crossunder and obv_falling and c.allow_shorts

        if not go_long and not go_short:
            return None

        atr_val = ctx.atr_val
        raw_sl = atr_val * c.sl_mult
        sl_dist = clamp_sl_dist(raw_sl, c.sl_cap)
        if sl_dist is None:
            return None
        tp_dist = sl_dist * c.rr_ratio

        is_long = go_long
        sl_price, tp_price = directional_sl_tp(is_long, ctx.bar.close, sl_dist, tp_dist)

        return make_entry_signal(
            go_long=is_long,
            mode="MACD-OBV",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason="MACD cross + OBV confirm",
        )
