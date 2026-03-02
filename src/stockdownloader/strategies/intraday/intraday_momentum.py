"""Gao (2018) Intraday Momentum — tournament strategy.

Pine Script equivalent: ~/Desktop/Pine Scripts/intraday_momentum_test.pine
Academic paper: "Market Intraday Momentum" (Journal of Financial Economics, 2018)

Hypothesis: First half-hour return (prev close → 10:00 AM) predicts
last half-hour return (3:30 PM → 4:00 PM).  Same direction = momentum.

Signal:
  r1 = (price at bar ``r1_end_bar``) / (prev day close) - 1
  If r1 > threshold → enter at ``entry_bar`` (direction depends on mode)
  Exit at EOD (bar 78).

Modes:
  Momentum — trade same direction as r1 (Gao finding)
  Reversal — fade r1

No SL/TP — pure time-based entry/exit.  Max 1 trade/day.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.math import HUNDRED, ZERO
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy, InfraExitConfig
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    make_entry_signal,
)

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.intraday.session import BarContext


# -- Config --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntradayMomentumConfig(InfraExitConfig):
    """Configuration for Gao (2018) Intraday Momentum strategy.

    Parameters
    ----------
    mode:
        ``"momentum"`` trades in the same direction as r1 (the Gao
        finding).  ``"reversal"`` fades r1.
    r1_threshold:
        Minimum absolute r1 (%) to trigger a trade.  ``0.0`` = always
        trade if r1 is non-zero.
    r1_end_bar:
        Bar of day to capture the r1 endpoint.  ``7`` = 10:00 AM on a
        5-minute chart (first 30 minutes after open).
    entry_bar:
        Bar of day to enter the trade.  ``72`` = 3:30 PM on a 5-minute
        chart (last 30 minutes of session).
    """

    # Strategy-specific
    mode: str = "momentum"
    r1_threshold: Decimal = Decimal("0.0")
    r1_end_bar: int = 7
    entry_bar: int = 72

    # Override base defaults — this strategy is bidirectional,
    # once-per-day, with no SL/TP or trail.
    allow_longs: bool = True
    allow_shorts: bool = True
    max_day: int = 1
    spacing: int = 0
    circuit: int = 99           # effectively disabled
    day_loss: Decimal = Decimal("99.0")     # effectively disabled
    be_trigger: Decimal = Decimal("99.0")   # no breakeven
    trail_vwap: bool = False                # no VWAP trail


# -- Strategy ------------------------------------------------------------------


class IntradayMomentumStrategy(BaseIntradayStrategy):
    """Gao (2018) Intraday Momentum strategy.

    Measures the first half-hour return (r1) and enters at 3:30 PM in
    the predicted direction.  Exits at EOD.  No SL/TP — the entire edge
    comes from the time-of-day momentum pattern.

    NOTE: ``_r1_value`` and ``_r1_captured`` reset each session via
    :meth:`on_session_start`.  They do NOT carry across day boundaries
    because r1 is inherently a single-day measurement.
    """

    _ENTRY_FLAGS = {"fire_once": True}

    def __init__(self, **overrides: object) -> None:
        self._c = IntradayMomentumConfig(**overrides) if overrides else IntradayMomentumConfig()
        self._infra = IntradayInfra(self._c, IntradayExitManager())
        super().__init__()
        self._r1_value: Decimal | None = None
        self._r1_captured: bool = False

    @property
    def name(self) -> str:
        return "SPY Intraday Momentum"

    def on_session_start(self, trading_date: str) -> None:
        super().on_session_start(trading_date)
        self._r1_value = None
        self._r1_captured = False

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state

        # -- Capture r1 at the designated bar --
        if ctx.bar_of_day == c.r1_end_bar and not self._r1_captured:
            pd_close = s.pd_close
            if pd_close > ZERO:
                self._r1_value = ((ctx.bar.close / pd_close) - Decimal("1")) * HUNDRED
                self._r1_captured = True

        # -- Only enter at the designated entry bar --
        if ctx.bar_of_day != c.entry_bar:
            return None

        if s.fired_today:
            return None

        if not self._r1_captured or self._r1_value is None:
            return None

        # -- Threshold check --
        if abs(self._r1_value) < c.r1_threshold:
            return None

        # -- Determine direction --
        r1_positive = self._r1_value > ZERO
        if c.mode == "momentum":
            go_long = r1_positive
        else:  # reversal
            go_long = not r1_positive

        # -- Check direction permissions --
        if go_long and not c.allow_longs:
            return None
        if not go_long and not c.allow_shorts:
            return None

        s.fired_today = True

        # -- Entry signal with wide SL/TP (EOD exit handles close) --
        wide = Decimal("100.0")
        if go_long:
            sl_price = ctx.bar.close - wide
            tp_price = ctx.bar.close + wide
        else:
            sl_price = ctx.bar.close + wide
            tp_price = ctx.bar.close - wide

        reason = f"r1={self._r1_value:.2f}%"

        return make_entry_signal(
            go_long=go_long,
            mode="IntMom",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=Decimal("0.01"),
            reason=reason,
        )
