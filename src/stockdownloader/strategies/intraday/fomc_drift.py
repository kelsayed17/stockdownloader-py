"""FOMC Drift strategy.

Based on Lucca & Moench (2015), "The Pre-FOMC Announcement Drift,"
Journal of Finance.  Updated by Ignatieva & Ohashi (2024) -- now
concentrated on press conference days only (Sharpe ~1.8).

Enter long at market open on FOMC press conference days.  The drift
is strongest during elevated VIX periods.
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
    from stockdownloader.strategies.intraday.market_context import (
        MarketContext,
        MarketContextProvider,
    )

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class FOMCDriftConfig(InfraExitConfig):
    """Config for FOMC Drift strategy."""

    entry_bar: int = 1
    sl_atr_mult: Decimal = Decimal("2.0")
    require_high_vix: bool = True
    vix_threshold: Decimal = Decimal("20")
    allow_longs: bool = True
    allow_shorts: bool = False
    close_eod: bool = True
    max_day: int = 1


class FOMCDriftStrategy(BaseIntradayStrategy):
    """FOMC press conference day drift: long at open, close at EOD."""

    _ENTRY_FLAGS = {"fire_once": True}

    def __init__(
        self,
        config: FOMCDriftConfig | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
        **overrides: object,
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if overrides:
            self._c = FOMCDriftConfig(**overrides)
        else:
            self._c = config or FOMCDriftConfig()
        self._infra = IntradayInfra(
            self._c, IntradayExitManager(), market_ctx_provider=market_ctx_provider,
        )
        super().__init__()
        self._enter_today: bool = False
        self._last_session_date: str = ""

    @property
    def name(self) -> str:
        vix_label = f" VIX>{self._c.vix_threshold}" if self._c.require_high_vix else ""
        return f"FOMC Drift{vix_label}"

    def _should_enter_today(self, ctx: MarketContext) -> bool:
        """Check if today is a tradeable FOMC day."""
        if not ctx.is_fomc_press_conf:
            return False
        if self._c.require_high_vix and ctx.vix_close < self._c.vix_threshold:
            return False
        return True

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c

        # Fire-once guard
        if ctx.state.fired_today:
            return None

        # Determine entry at session start
        if ctx.bar.trading_date != self._last_session_date:
            self._last_session_date = ctx.bar.trading_date
            self._enter_today = False
            if ctx.market_ctx is not None:
                self._enter_today = self._should_enter_today(ctx.market_ctx)

        # Only enter at configured bar
        if ctx.bar_of_day != c.entry_bar:
            return None

        if not self._enter_today:
            return None

        if not c.allow_longs:
            return None

        # SL/TP -- wide stop for event-driven trade
        sl_dist = ctx.atr_val * c.sl_atr_mult
        if sl_dist <= ZERO:
            return None
        tp_dist = sl_dist * Decimal("5.0")  # Wide TP, EOD exit dominates

        sl_price, tp_price = directional_sl_tp(True, ctx.bar.close, sl_dist, tp_dist)

        # Mark as fired for this session
        ctx.state.fired_today = True

        return make_entry_signal(
            go_long=True,
            mode="FOMC-Drift",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason="FOMC press conf day drift",
        )
