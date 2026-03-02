"""VIX Regime Filter -- wraps any intraday strategy with VIX conditioning.

Cross-cutting finding from academic research: VIX filtering improves
risk-adjusted returns by 8-15 percentage points across all strategy
categories. This wrapper lets you add VIX regime filtering to any
existing strategy without modifying it.

Usage::

    from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategy
    from stockdownloader.strategies.intraday.vix_regime_filter import VixFilteredStrategy

    filtered = VixFilteredStrategy(
        inner=ORReversalStrategy(),
        allowed_regimes=["mid", "high"],
        market_ctx_provider=provider,
    )
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from stockdownloader.core.models.trade import IntradaySignal, IntradayAction
from stockdownloader.strategies.base import IntradayTradingStrategy

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.intraday.market_context import (
        MarketContext,
        MarketContextProvider,
    )

HOLD = IntradaySignal(action=IntradayAction.HOLD)


class VixFilteredStrategy(IntradayTradingStrategy):
    """Wraps any intraday strategy with VIX regime filtering.

    Delegates all calls to the inner strategy, but blocks entry signals
    when the VIX regime is not in the allowed list.
    """

    def __init__(
        self,
        inner: IntradayTradingStrategy,
        allowed_regimes: list[str] | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
    ) -> None:
        self._inner = inner
        self._allowed = allowed_regimes or ["mid", "high"]
        self._provider = market_ctx_provider
        self._current_ctx: MarketContext | None = None
        self._last_session_date: str = ""

    @property
    def name(self) -> str:
        regimes = "/".join(self._allowed)
        return f"VIX[{regimes}] {self._inner.name}"

    @property
    def warmup_period(self) -> int:
        return self._inner.warmup_period

    def _is_regime_allowed(self, regime: str) -> bool:
        return regime in self._allowed

    def on_session_start(self, trading_date: str) -> None:
        self._inner.on_session_start(trading_date)
        if self._provider is not None:
            self._current_ctx = self._provider.get_context(trading_date)

    def on_position_opened(self, is_long: bool) -> None:
        self._inner.on_position_opened(is_long)

    def on_position_closed(self) -> None:
        self._inner.on_position_closed()

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        # Detect session change and update context
        bar = data[current_index]
        if bar.trading_date != self._last_session_date:
            self._last_session_date = bar.trading_date
            if self._provider is not None:
                self._current_ctx = self._provider.get_context(bar.trading_date)

        sig = self._inner.evaluate(data, current_index)

        # Only filter entry signals -- let exits through
        if sig.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT):
            if self._current_ctx is None:
                return HOLD  # No context -> block entries
            if not self._is_regime_allowed(self._current_ctx.vix_regime):
                return HOLD
        return sig
