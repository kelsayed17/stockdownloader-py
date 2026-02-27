"""Noise Boundary Breakout strategy.

Based on Zarattini, Aziz & Barbon (2024), "Beat the Market: An Effective
Intraday Momentum Strategy for S&P500 ETF (SPY)," Swiss Finance Institute
Research Paper No. 24-97.

Defines a "noise area" around the open based on historical intraday volatility
at half-hourly checkpoints. Price beyond the noise band signals genuine
momentum; price within is noise.
"""
from __future__ import annotations

from collections import defaultdict
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
ONE = Decimal("1")


@dataclass(frozen=True, slots=True)
class NoiseBoundaryConfig(InfraExitConfig):
    """Config for Noise Boundary Breakout strategy."""

    lookback_days: int = 14
    vol_multiplier: Decimal = Decimal("1.0")
    checkpoint_interval: int = 6  # Every 30 min
    sl_atr_mult: Decimal = Decimal("2.0")
    allow_longs: bool = True
    allow_shorts: bool = True
    close_eod: bool = True
    max_day: int = 1


class NoiseBoundaryStrategy(BaseIntradayStrategy):
    """Noise boundary breakout: trade genuine moves, filter noise."""

    _ENTRY_FLAGS = {"fire_once": True}

    def __init__(
        self,
        config: NoiseBoundaryConfig | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
        **overrides: object,
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if overrides:
            self._c = NoiseBoundaryConfig(**overrides)
        else:
            self._c = config or NoiseBoundaryConfig()
        self._infra = IntradayInfra(
            self._c, IntradayExitManager(), market_ctx_provider=market_ctx_provider,
        )
        super().__init__()
        # Rolling history: checkpoint_bar -> list of |close/open - 1| values
        self._checkpoint_vols: dict[int, list[Decimal]] = defaultdict(list)
        self._session_open: Decimal = ZERO
        self._prev_close: Decimal = ZERO
        self._last_session_date: str = ""
        self._warmup_days: int = 0

    @property
    def name(self) -> str:
        return f"Noise Boundary (L{self._c.lookback_days})"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        bar = ctx.bar

        # Fire-once guard
        if ctx.state.fired_today:
            return None

        # Track session open
        if bar.trading_date != self._last_session_date:
            # Record previous session's close
            if self._last_session_date and ctx.prev_bar:
                self._prev_close = ctx.prev_bar.close
            self._last_session_date = bar.trading_date
            self._session_open = bar.open
            self._warmup_days += 1

        # Record checkpoint volatility
        if ctx.bar_of_day % c.checkpoint_interval == 0 and self._session_open > ZERO:
            cp = ctx.bar_of_day
            vol = abs(bar.close / self._session_open - ONE)
            self._checkpoint_vols[cp].append(vol)
            # Trim to lookback
            if len(self._checkpoint_vols[cp]) > c.lookback_days:
                self._checkpoint_vols[cp] = self._checkpoint_vols[cp][-c.lookback_days:]

        # Need enough warmup days
        if self._warmup_days <= c.lookback_days:
            return None

        # Only evaluate at checkpoints
        if ctx.bar_of_day % c.checkpoint_interval != 0:
            return None

        # Compute noise boundaries for this checkpoint
        cp = ctx.bar_of_day
        vols = self._checkpoint_vols.get(cp, [])
        if len(vols) < c.lookback_days:
            return None

        sigma = sum(vols[-c.lookback_days:]) / c.lookback_days
        prev_c = self._prev_close if self._prev_close > ZERO else self._session_open
        anchor_high = max(self._session_open, prev_c)
        anchor_low = min(self._session_open, prev_c)
        upper = anchor_high * (ONE + c.vol_multiplier * sigma)
        lower = anchor_low * (ONE - c.vol_multiplier * sigma)

        # Check breakout
        go_long = bar.close > upper and c.allow_longs
        go_short = bar.close < lower and c.allow_shorts

        if not go_long and not go_short:
            return None

        is_long = go_long

        # SL/TP
        sl_dist = ctx.atr_val * c.sl_atr_mult
        if sl_dist <= ZERO:
            return None
        tp_dist = sl_dist * Decimal("2.0")

        sl_price, tp_price = directional_sl_tp(is_long, bar.close, sl_dist, tp_dist)

        # Mark as fired for this session
        ctx.state.fired_today = True

        return make_entry_signal(
            go_long=is_long,
            mode="NoiseBnd",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason=f"{'Upper' if is_long else 'Lower'} noise break cp={cp}",
        )
