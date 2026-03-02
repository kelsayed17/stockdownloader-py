"""Gao Intraday Momentum strategy.

Based on Gao, Han, Li & Zhou (2018), "Market Intraday Momentum,"
Journal of Financial Economics 129(2): 394-414.

The first half-hour return (9:30-10:00) predicts the last half-hour
return (3:30-4:00).  Enter long in the last 30 min if r1 > 0, short
if r1 < 0.  Enhanced mode requires both first and penultimate
half-hour returns to agree (win rate: 54% -> 77%).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategies.intraday.base import (
    BaseIntradayStrategy,
    InfraExitConfig,
)
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    make_entry_signal,
    directional_sl_tp,
)

if TYPE_CHECKING:
    from stockdownloader.strategies.intraday.session import BarContext
    from stockdownloader.strategies.intraday.market_context import MarketContextProvider

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class GaoMomentumConfig(InfraExitConfig):
    """Config for Gao Intraday Momentum strategy."""

    entry_start_bar: int = 73
    first_hh_bars: int = 6
    penult_hh_start: int = 67
    penult_hh_end: int = 72
    require_dual_signal: bool = True
    min_r1_magnitude: Decimal = Decimal("0.0005")
    sl_atr_mult: Decimal = Decimal("1.5")
    allow_longs: bool = True
    allow_shorts: bool = True
    close_eod: bool = True
    max_day: int = 1
    vix_filter: bool = True  # Skip low-VIX days when context available


class GaoMomentumStrategy(BaseIntradayStrategy):
    """Intraday momentum: first half-hour predicts last half-hour."""

    _ENTRY_FLAGS = {"fire_once": True}

    def __init__(
        self,
        config: GaoMomentumConfig | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
        **overrides: object,
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if overrides:
            self._c = GaoMomentumConfig(**overrides)
        else:
            self._c = config or GaoMomentumConfig()
        self._infra = IntradayInfra(
            self._c, IntradayExitManager(), market_ctx_provider=market_ctx_provider,
        )
        super().__init__()
        # Session state for first half-hour tracking
        self._session_open: Decimal = ZERO
        self._first_hh_close: Decimal = ZERO
        self._r1: Decimal = ZERO
        self._r1_valid: bool = False
        self._penult_close: Decimal = ZERO
        self._r12: Decimal = ZERO
        self._r12_valid: bool = False
        self._last_session_date: str = ""

    @property
    def name(self) -> str:
        mode = "dual" if self._c.require_dual_signal else "single"
        return f"Gao Momentum ({mode})"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        bar = ctx.bar

        # Fire-once guard
        if ctx.state.fired_today:
            return None

        # Track session open and first half-hour close
        if bar.trading_date != self._last_session_date:
            self._last_session_date = bar.trading_date
            self._session_open = bar.open
            self._r1_valid = False
            self._r12_valid = False

        # Record first half-hour close
        if ctx.bar_of_day == c.first_hh_bars and self._session_open > ZERO:
            self._first_hh_close = bar.close
            self._r1 = (self._first_hh_close - self._session_open) / self._session_open
            self._r1_valid = True

        # Record penultimate half-hour close
        if ctx.bar_of_day == c.penult_hh_end and self._session_open > ZERO:
            self._penult_close = bar.close
            if self._session_open > ZERO:
                self._r12 = (self._penult_close - self._session_open) / self._session_open
            else:
                self._r12 = ZERO
            self._r12_valid = True

        # Only trade in the last 30 minutes
        if ctx.bar_of_day < c.entry_start_bar:
            return None

        # Need valid first half-hour return
        if not self._r1_valid:
            return None

        # Filter by magnitude
        if abs(self._r1) < c.min_r1_magnitude:
            return None

        # VIX filter (skip low-VIX when context available)
        if c.vix_filter and ctx.market_ctx is not None:
            if ctx.market_ctx.vix_regime == "low":
                return None

        # Determine direction
        go_long = self._r1 > ZERO
        go_short = self._r1 < ZERO

        # Dual signal check
        if c.require_dual_signal:
            if not self._r12_valid:
                return None
            r12_long = self._r12 > ZERO
            r12_short = self._r12 < ZERO
            if go_long and not r12_long:
                return None
            if go_short and not r12_short:
                return None

        # Direction filter
        if go_long and not c.allow_longs:
            return None
        if go_short and not c.allow_shorts:
            return None

        is_long = go_long

        # SL/TP
        sl_dist = ctx.atr_val * c.sl_atr_mult
        if sl_dist <= ZERO:
            return None
        # No TP -- ride to close (EOD exit handles it)
        tp_dist = sl_dist * Decimal("10")  # Very wide TP, EOD exit dominates

        sl_price, tp_price = directional_sl_tp(is_long, bar.close, sl_dist, tp_dist)

        # Mark as fired for this session
        ctx.state.fired_today = True

        r1_str = f"r1={'+'if self._r1>0 else ''}{self._r1:.4f}"
        r12_str = f" r12={'+'if self._r12>0 else ''}{self._r12:.4f}" if c.require_dual_signal else ""

        return make_entry_signal(
            go_long=is_long,
            mode="Gao-Mom",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason=f"{r1_str}{r12_str}",
        )
