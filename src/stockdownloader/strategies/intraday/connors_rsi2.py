"""RSI(2) Connors Mean Reversion strategy.

Based on Larry Connors' research -- 75% win rate, profit factor 2.3,
backtested on SPY 1993-present.

Buy when RSI(2) < 5 and daily close > SMA(200).
Exit when close > 5-day SMA of closes.

Implemented as intraday: entry at session open based on prior day's
daily RSI(2) from MarketContext, close at EOD.
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
class ConnorsRSI2Config(InfraExitConfig):
    """Config for Connors RSI(2) mean reversion strategy."""

    rsi_period: int = 2
    rsi_threshold: Decimal = Decimal("5")
    sma_trend_period: int = 200
    exit_sma_period: int = 5
    entry_bar: int = 1  # Enter at market open
    sl_atr_mult: Decimal = Decimal("2.0")
    allow_longs: bool = True
    allow_shorts: bool = False  # Long-only per Connors
    close_eod: bool = True
    max_day: int = 1


class ConnorsRSI2Strategy(BaseIntradayStrategy):
    """Connors RSI(2) mean reversion: buy extreme oversold, ride the bounce."""

    _ENTRY_FLAGS = {"fire_once": True}

    def __init__(
        self,
        config: ConnorsRSI2Config | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
        **overrides: object,
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if overrides:
            self._c = ConnorsRSI2Config(**overrides)
        else:
            self._c = config or ConnorsRSI2Config()
        self._infra = IntradayInfra(
            self._c, IntradayExitManager(), market_ctx_provider=market_ctx_provider,
        )
        super().__init__()
        self._enter_today: bool = False
        self._last_session_date: str = ""

    @property
    def name(self) -> str:
        return f"Connors RSI({self._c.rsi_period})"

    def _should_enter_today(self, ctx: MarketContext) -> bool:
        """Check daily-level entry conditions from MarketContext."""
        return (
            ctx.daily_rsi2 < self._c.rsi_threshold
            and ctx.daily_close_above_sma200
        )

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c

        # Fire-once guard
        if ctx.state.fired_today:
            return None

        # Determine entry signal at session start
        if ctx.bar.trading_date != self._last_session_date:
            self._last_session_date = ctx.bar.trading_date
            self._enter_today = False
            if ctx.market_ctx is not None:
                self._enter_today = self._should_enter_today(ctx.market_ctx)

        # Only enter at the configured entry bar
        if ctx.bar_of_day != c.entry_bar:
            return None

        if not self._enter_today:
            return None

        if not c.allow_longs:
            return None

        # SL/TP
        sl_dist = ctx.atr_val * c.sl_atr_mult
        if sl_dist <= ZERO:
            return None
        tp_dist = sl_dist * Decimal("3.0")  # Wide TP, EOD exit dominates

        sl_price, tp_price = directional_sl_tp(True, ctx.bar.close, sl_dist, tp_dist)

        # Mark as fired for this session
        ctx.state.fired_today = True

        rsi_val = ctx.market_ctx.daily_rsi2 if ctx.market_ctx else Decimal("0")

        return make_entry_signal(
            go_long=True,
            mode="Connors-RSI2",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason=f"RSI(2)={rsi_val:.1f} < {c.rsi_threshold}",
        )
