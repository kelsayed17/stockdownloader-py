"""Standalone Pattern Scalp strategy — opening-range manipulation fade.

Fires when the opening range is >= 30% of daily ATR (manipulation),
a reversal pattern (hammer / engulfing) appears in the opposite
direction, and RVOL confirms.  TP = fraction of OR range.

This strategy does **not** use VWAP for entry decisions.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import IntradaySignal
from stockdownloader.strategy.intraday.entry_helpers import (
    clamp_sl_dist,
    detect_reversal_patterns,
    directional_sl_tp,
    make_entry_signal,
    reversal_pattern_label,
)
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.infra import BarContext, IntradayInfra
from stockdownloader.strategy.intraday.trail_strategy import BreakevenTrail
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.util.big_decimal_math import HUNDRED, ZERO
from stockdownloader.util.pinescript_models import ModeDefinition
from stockdownloader.util.pinescript_strategies import _ps_mode

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.intraday.pattern_scalp_config import PatternScalpStrategyConfig


class PatternScalpStrategy(IntradayTradingStrategy):
    """Standalone Pattern Scalp strategy.

    Fades opening-range manipulation: fires when OR range >= daily ATR
    percentage, a reversal pattern appears in the opposite direction,
    and RVOL confirms.
    """

    def __init__(self, config: PatternScalpStrategyConfig | None = None) -> None:
        from stockdownloader.strategy.intraday.pattern_scalp_config import PatternScalpStrategyConfig

        c = config or PatternScalpStrategyConfig()
        self._c = c
        self._infra = IntradayInfra(c, IntradayExitManager(BreakevenTrail()))

    _ENTRY_FLAGS = {"fire_once": True}

    @staticmethod
    def pinescript_mode() -> ModeDefinition:
        """Return the PineScript mode definition for Pattern Scalp."""
        return _ps_mode()

    @property
    def name(self) -> str:
        return "Pattern Scalp"

    @property
    def warmup_period(self) -> int:
        return self._infra.warmup_period

    def on_session_start(self, trading_date: str) -> None:
        self._infra.on_session_start(trading_date)

    def on_position_opened(self, is_long: bool) -> None:
        self._infra.confirm_position_opened(is_long)

    def on_position_closed(self) -> None:
        self._infra.confirm_position_closed()

    def evaluate(self, data: list[IntradayPriceData], current_index: int) -> IntradaySignal:
        return self._infra.run_bar(data, current_index, self._evaluate_entry, self._ENTRY_FLAGS)

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        state = ctx.state

        if not c.ps_enable:
            return None
        if state.fired_today:
            return None
        if not state.is_manip:
            return None

        # -- Window check --
        if ctx.bar_of_day < 4 or ctx.bar_of_day > c.ps_window:
            return None

        # -- Pattern detection --
        bull_hammer, bear_hammer, bull_engulf, bear_engulf = detect_reversal_patterns(
            ctx.bar, ctx.prev_bar, ctx.atr_val, c.ps_engulf,
        )

        bull_reversal = bull_hammer or bull_engulf
        bear_reversal = bear_hammer or bear_engulf

        # -- Direction (fade the OR direction) --
        go_long = state.or_dir == -1 and bull_reversal and c.allow_longs
        go_short = state.or_dir == 1 and bear_reversal and c.allow_shorts

        if not go_long and not go_short:
            return None

        # -- RVOL filter --
        if ctx.rel_vol < c.ps_rvol:
            return None

        # -- SMA bias filter (optional) --
        if c.ps_sma_filter and state.daily_sma > ZERO:
            sma_dist = (ctx.bar.close - state.daily_sma) / state.daily_sma * HUNDRED
            daily_bias = 1 if sma_dist > Decimal("0.5") else (-1 if sma_dist < Decimal("-0.5") else 0)
            sma_ok = (
                (state.or_dir == -1 and daily_bias >= 1)
                or (state.or_dir == 1 and daily_bias <= -1)
            )
            if not sma_ok:
                return None

        # -- SL calculation --
        if c.ps_sl_mode == "Day Extreme":
            if go_long:
                sl_raw = ctx.bar.close - state.day_lod + ctx.atr_val * Decimal("0.1")
            else:
                sl_raw = state.day_hod - ctx.bar.close + ctx.atr_val * Decimal("0.1")
        else:
            sl_raw = ctx.atr_val * c.ps_sl_atr

        sl_dist = clamp_sl_dist(sl_raw, c.ps_sl_cap)
        if sl_dist is None:
            return None

        # -- TP calculation --
        tp_dist = state.or_range * (c.ps_tp_pct / HUNDRED)

        sl_price, tp_price = directional_sl_tp(go_long, ctx.bar.close, sl_dist, tp_dist)

        # -- Pattern label --
        pat = reversal_pattern_label(bull_hammer, bear_hammer, bull_engulf, bear_engulf)

        # -- Scoring --
        pts_vol = 3 if ctx.rel_vol >= Decimal("2") else (1 if ctx.rel_vol >= Decimal("1") else 0)
        pts_sr = c.w_sr if state.or_done else 0
        score = pts_vol + pts_sr
        max_score = 3 + c.w_sr + 1

        state.fired_today = True
        return make_entry_signal(
            go_long=go_long,
            mode="PS",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=f"PS {pat}",
        )
