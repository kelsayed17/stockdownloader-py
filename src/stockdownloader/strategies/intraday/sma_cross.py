"""SPY SMA Cross 20/21 (Tournament #2) -- intraday strategy.

Pine Script equivalent: output/pinescript/spy/spy_sma_cross.pine
Tournament rank: #2

Entry:
  Long  -- SMA(20) crosses above SMA(21)
  Short -- disabled (long only)

Exit:
  Reverse SMA crossover, or SL/TP hit, or EOD.

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


# -- Config --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SMACross2021Config(InfraExitConfig):
    """Configuration for SPY SMA Cross 20/21 strategy."""

    sma_short: int = 20
    sma_long: int = 21
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
    allow_shorts: bool = False


# -- Strategy ------------------------------------------------------------------


class SMACross2021Strategy(BaseIntradayStrategy):
    """SPY SMA Cross 20/21 intraday strategy (Tournament #2)."""

    def __init__(self, **overrides: object) -> None:
        self._c = SMACross2021Config(**overrides) if overrides else SMACross2021Config()
        self._infra = IntradayInfra(self._c, IntradayExitManager())
        super().__init__()
        self._data: list[IntradayPriceData] = []
        self._idx: int = 0
        # NOTE: _prev_sma_* intentionally NOT reset across sessions.
        # Pine Script's ta.crossover operates on a continuous time series.
        self._prev_sma_fast: Decimal = ZERO
        self._prev_sma_slow: Decimal = ZERO

    @property
    def name(self) -> str:
        return "SPY SMA 20/21"

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
        sma_fast = hub.sma(data, i, c.sma_short)
        sma_slow = hub.sma(data, i, c.sma_long)

        crossover = self._prev_sma_fast <= self._prev_sma_slow and sma_fast > sma_slow

        self._prev_sma_fast = sma_fast
        self._prev_sma_slow = sma_slow

        go_long = crossover and c.allow_longs

        if not go_long:
            return None

        atr_val = ctx.atr_val
        raw_sl = atr_val * c.sl_mult
        sl_dist = clamp_sl_dist(raw_sl, c.sl_cap)
        if sl_dist is None:
            return None
        tp_dist = sl_dist * c.rr_ratio

        sl_price, tp_price = directional_sl_tp(True, ctx.bar.close, sl_dist, tp_dist)

        return make_entry_signal(
            go_long=True,
            mode="SMA-Cross",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason="SMA 20/21 bullish crossover",
        )
