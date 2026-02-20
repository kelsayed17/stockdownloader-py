"""Standalone VWAP Reversal strategy — sideways market band fade.

Enters when ADX < threshold (not trending), VWAP is flat, price reaches
a sigma-band extreme, and a confirmation candle prints.  TP targets VWAP.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import IntradaySignal
from stockdownloader.strategy.intraday.entry_helpers import (
    clamp_sl_dist,
    make_entry_signal,
)
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.bar_context import BarContext
from stockdownloader.strategy.intraday.infra import IntradayInfra
from stockdownloader.strategy.intraday.trail_strategy import BreakevenTrail
from stockdownloader.strategy.intraday.base_strategy import BaseIntradayStrategy
from stockdownloader.util.intraday_indicators import candle_strength
from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.pinescript_models import ModeDefinition
from stockdownloader.util.pinescript_modes import rev_mode

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.intraday.reversal_config import ReversalStrategyConfig


class ReversalStrategy(BaseIntradayStrategy):
    """Standalone VWAP Reversal strategy.

    Mean-reversion entry: ADX low (sideways market), VWAP flat,
    price reaches band extreme, confirmation candle prints.
    """

    def __init__(self, config: ReversalStrategyConfig | None = None) -> None:
        from stockdownloader.strategy.intraday.reversal_config import ReversalStrategyConfig

        c = config or ReversalStrategyConfig()
        self._c = c
        self._infra = IntradayInfra(c, IntradayExitManager(BreakevenTrail()))

    _ENTRY_FLAGS: dict[str, bool] = {}

    @staticmethod
    def pinescript_mode() -> ModeDefinition:
        """Return the PineScript mode definition for VWAP Reversal."""
        return rev_mode()

    @property
    def name(self) -> str:
        return "VWAP Reversal"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state

        # -- Spacing + day trade limit --
        ready = s.day_trades < c.max_day
        spaced = (ctx.bar_of_day - s.last_entry_bar) >= c.spacing or s.last_entry_bar <= 0
        if not ready or not spaced:
            return None

        if not c.rev_enable:
            return None
        min_bar = c.rev_can_trade_bar if c.rev_can_trade_bar > 0 else c.can_trade_bar
        if ctx.bar_of_day < min_bar:
            return None

        # -- Guard: sideways market --
        is_trending = ctx.adx_val >= c.adx_thresh
        is_sideways = (
            not is_trending
            and ctx.atr_val > ZERO
            and abs(ctx.vwap_delta) <= ctx.atr_val * c.rev_vwap_flat_tol
        )
        if not is_sideways:
            return None

        # -- Band touch filter --
        if c.rev_min_touches > 0 and s.rev_band_touches < c.rev_min_touches:
            return None

        # -- Band selection --
        vwap = ctx.vwap_bands.vwap
        upper_band, lower_band = ctx.vwap_bands.band_pair(c.rev_band)

        band_prox = ctx.atr_val * Decimal("0.3")

        near_upper = upper_band > ZERO and ctx.bar.high >= upper_band - band_prox
        near_lower = lower_band > ZERO and ctx.bar.low <= lower_band + band_prox

        # -- Candle confirmation --
        cs = candle_strength(ctx.bar, ctx.atr_val)
        bull_candle = cs.bull_candle(c.rev_body)
        bear_candle = cs.bear_candle(c.rev_body)

        # -- Not-hugging filter --
        not_hugging = s.bars_above_vwap < c.rev_hug_limit and s.bars_below_vwap < c.rev_hug_limit

        # -- Direction --
        go_long = near_lower and bull_candle and not_hugging and c.allow_longs
        go_short = (
            near_upper and bear_candle and not_hugging
            and c.allow_shorts and c.rev_shorts
        )

        if not go_long and not go_short:
            return None

        # -- S/R hard filter --
        if c.rev_require_sr and not ctx.sr_any:
            return None

        # -- SL / TP --
        sl_dist = clamp_sl_dist(ctx.atr_val * c.rev_sl_atr, c.rev_sl_cap)
        if sl_dist is None:
            return None

        if c.rev_tp_mode == "rr":
            tp_dist = sl_dist * c.rev_rr
            sl_price = ctx.bar.close - sl_dist if go_long else ctx.bar.close + sl_dist
            tp_price = ctx.bar.close + tp_dist if go_long else ctx.bar.close - tp_dist
        else:
            sl_price = ctx.bar.close - sl_dist if go_long else ctx.bar.close + sl_dist
            tp_price = vwap

        # -- R:R filter --
        reward = abs(tp_price - ctx.bar.close)
        rr = reward / sl_dist if sl_dist > ZERO else ZERO
        if rr < c.rev_min_rr:
            return None

        # -- Scoring (lighter than PB) --
        # Reversal is mean-reversion: moderate vol is ideal (not too high, not too low)
        pts_vol = c.w_vol if Decimal("0.7") <= ctx.tod_rvol <= Decimal("1.5") else 0
        pts_sr = c.w_sr if ctx.sr_score_count > 0 else 0
        pts_sr2 = 1 if ctx.sr_score_count >= 2 else 0
        pts_time = c.w_time if ctx.is_good_time else 0

        if go_long:
            pts_rsi = c.w_rsi if ctx.rsi_val <= Decimal("30") else 0
        else:
            pts_rsi = c.w_rsi if ctx.rsi_val >= Decimal("70") else 0

        score = pts_vol + pts_sr + pts_sr2 + pts_rsi + pts_time
        max_score = c.w_vol + c.w_sr + 1 + c.w_rsi + c.w_time

        if score < c.min_score:
            return None

        return make_entry_signal(
            go_long=go_long,
            mode="REV",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason="Band Reversal",
        )
