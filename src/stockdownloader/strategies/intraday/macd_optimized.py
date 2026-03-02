"""SPY MACD Optimized 8/35/5 (Tournament #3) -- intraday strategy.

Pine Script equivalent: output/pinescript/daily/spy/spy_macd_optimized.pine
Tournament rank: #3

Entry:
  Long  -- MACD(8,35,5) bullish crossover
  Short -- MACD(8,35,5) bearish crossunder

Exit:
  Reverse MACD crossover, or SL/TP hit, or EOD.

Risk:
  SL = min(ATR(14) * 1.5, $2.00)
  TP = SL * 1.5 (R:R)
  Breakeven at 0.5R -- SL moves to entry +/- $0.05
  Max 4 trades/day, 3-bar spacing, circuit breaker after 3 losses,
  halt on -3% daily drawdown.

Note: Pure MACD crossover -- NO OBV filter (unlike MACD+OBV).
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


# -- Config --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MACDOptimizedConfig(InfraExitConfig):
    """Configuration for SPY MACD Optimized 8/35/5 strategy."""

    macd_fast: int = 8
    macd_slow: int = 35
    macd_signal: int = 5
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


class MACDOptimizedStrategy(BaseIntradayStrategy):
    """SPY MACD Optimized 8/35/5 intraday strategy (Tournament #3)."""

    def __init__(self, **overrides: object) -> None:
        self._c = MACDOptimizedConfig(**overrides) if overrides else MACDOptimizedConfig()
        self._infra = IntradayInfra(self._c, IntradayExitManager())
        super().__init__()
        self._data: list[IntradayPriceData] = []
        self._idx: int = 0
        # NOTE: _prev_macd_* intentionally NOT reset across sessions.
        # Pine Script's ta.crossover operates on a continuous time series.
        self._prev_macd_line: Decimal = ZERO
        self._prev_macd_sig: Decimal = ZERO

    @property
    def name(self) -> str:
        return "SPY MACD 8/35/5"

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        self._data = data
        self._idx = current_index
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

        go_long = crossover and c.allow_longs
        go_short = crossunder and c.allow_shorts

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
            mode="MACD-Opt",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason="MACD 8/35/5 crossover",
        )
