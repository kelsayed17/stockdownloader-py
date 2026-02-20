"""Standalone AVWAP Pullback strategy — trades pullbacks to anchored VWAP.

The anchored VWAP (AVWAP) persists across trading sessions, anchored to
deterministic events (FOMC meetings).  Price tends to revert to the AVWAP
level, creating high-probability pullback setups when price approaches it
with trend confirmation and candle strength.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import IntradaySignal
from stockdownloader.strategy.intraday.entry_helpers import (
    clamp_sl_dist,
    directional_sl_tp,
    make_entry_signal,
)
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.bar_context import BarContext
from stockdownloader.strategy.intraday.infra import IntradayInfra
from stockdownloader.strategy.intraday.trail_strategy import VwapRatchetTrail
from stockdownloader.strategy.intraday.base_strategy import BaseIntradayStrategy
from stockdownloader.util.intraday_indicators import candle_strength
from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.intraday.avwap_pullback_config import AVWAPPullbackConfig


class AVWAPPullbackStrategy(BaseIntradayStrategy):
    """Standalone AVWAP Pullback strategy.

    Enters when price pulls back to the anchored VWAP level with
    candle confirmation, trend alignment, and confluence scoring.
    The AVWAP persists across sessions, anchored to FOMC events.
    """

    def __init__(self, config: AVWAPPullbackConfig | None = None) -> None:
        from stockdownloader.strategy.intraday.avwap_pullback_config import (
            AVWAPPullbackConfig,
        )

        c = config or AVWAPPullbackConfig()
        self._c = c
        self._infra = IntradayInfra(c, IntradayExitManager(VwapRatchetTrail()))

    _ENTRY_FLAGS: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "AVWAP Pullback"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state
        avwap = ctx.avwap_bands

        # -- Guard: AVWAP must be valid --
        if avwap is None or not avwap.valid:
            return None

        # -- Guard: anchor age in range --
        if avwap.days_since_anchor < c.avwap_min_days:
            return None  # Too fresh — not enough accumulated significance
        if avwap.days_since_anchor > c.avwap_max_days:
            return None  # Too stale

        # -- Spacing + day trade limit --
        ready = s.day_trades < c.max_day
        spaced = (ctx.bar_of_day - s.last_entry_bar) >= c.spacing or s.last_entry_bar <= 0
        if not ready or not spaced:
            return None

        # -- Time filter --
        if ctx.bar_of_day < c.can_trade_bar:
            return None
        if not ctx.is_good_time:
            return None

        # -- AVWAP proximity zone --
        avwap_level = avwap.avwap
        dist = abs(ctx.bar.close - avwap_level)
        zone = ctx.atr_val * c.avwap_zone_atr
        if zone <= ZERO or dist > zone:
            return None

        # -- Candle confirmation --
        cs = candle_strength(ctx.bar, ctx.atr_val)
        bull_candle = cs.bull_candle(c.avwap_body_min)
        bear_candle = cs.bear_candle(c.avwap_body_min)

        # -- Direction logic --
        # AVWAP as support (price at or above): go long on bull candle
        # AVWAP as resistance (price below): go short on bear candle
        price_above_avwap = ctx.bar.close >= avwap_level
        go_long = price_above_avwap and bull_candle and c.avwap_longs and c.allow_longs
        go_short = (
            not price_above_avwap and bear_candle
            and c.avwap_shorts and c.allow_shorts
        )

        if not go_long and not go_short:
            return None

        # -- Session VWAP agreement filter --
        if c.avwap_session_vwap_agree:
            session_vwap = ctx.vwap_bands.vwap
            if go_long and ctx.bar.close < session_vwap:
                go_long = False
            if go_short and ctx.bar.close > session_vwap:
                go_short = False
            if not go_long and not go_short:
                return None

        # -- Trend filter (EMA) --
        if c.avwap_trend_bars > 0:
            if go_long and s.bull_bars < c.avwap_trend_bars:
                go_long = False
            if go_short and s.bear_bars < c.avwap_trend_bars:
                go_short = False
            if not go_long and not go_short:
                return None

        # -- HTF alignment --
        if c.avwap_htf_align:
            if go_long and ctx.htf_trend == -1:
                go_long = False
            if go_short and ctx.htf_trend == 1:
                go_short = False
            if not go_long and not go_short:
                return None

        # -- ADX trend filter: need a trending market --
        if ctx.adx_val < Decimal("22"):
            return None

        # -- Confluence scoring --
        pts_vol = c.w_vol if ctx.tod_rvol >= Decimal("1.2") else 0
        pts_sr = c.w_sr if ctx.sr_score_count > 0 else 0
        pts_time = c.w_time if ctx.is_good_time else 0

        if go_long:
            pts_rsi = c.w_rsi if ctx.rsi_val <= Decimal("40") else 0
        else:
            pts_rsi = c.w_rsi if ctx.rsi_val >= Decimal("60") else 0

        score = pts_vol + pts_sr + pts_rsi + pts_time
        max_score = c.w_vol + c.w_sr + c.w_rsi + c.w_time

        if score < c.min_score:
            return None

        # -- SL / TP --
        sl_dist = clamp_sl_dist(ctx.atr_val * c.avwap_sl_atr, c.avwap_sl_cap)
        if sl_dist is None:
            return None

        if c.avwap_tp_mode == "session_vwap":
            tp_price = ctx.vwap_bands.vwap
            sl_price = ctx.bar.close - sl_dist if go_long else ctx.bar.close + sl_dist
            # R:R sanity check
            reward = abs(tp_price - ctx.bar.close)
            rr_actual = reward / sl_dist if sl_dist > ZERO else ZERO
            if rr_actual < Decimal("0.3"):
                return None
        elif c.avwap_tp_mode == "avwap_band":
            tp_price = avwap.upper_1 if go_long else avwap.lower_1
            sl_price = ctx.bar.close - sl_dist if go_long else ctx.bar.close + sl_dist
            reward = abs(tp_price - ctx.bar.close)
            rr_actual = reward / sl_dist if sl_dist > ZERO else ZERO
            if rr_actual < Decimal("0.3"):
                return None
        else:  # "rr"
            tp_dist = sl_dist * c.avwap_rr
            sl_price, tp_price = directional_sl_tp(
                go_long, ctx.bar.close, sl_dist, tp_dist,
            )

        return make_entry_signal(
            go_long=go_long,
            mode="AVWAP",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=f"AVWAP PB ({avwap.days_since_anchor}d)",
        )
