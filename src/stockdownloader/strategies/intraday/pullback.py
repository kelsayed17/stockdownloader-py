"""Standalone VWAP Pullback strategy — trending market VWAP bounce.

Requires ADX >= threshold, EMA trend confirmed for N bars, VWAP slope
aligned, price in VWAP zone, bullish/bearish candle confirmation,
confluence score above minimum.  Filters: HTF alignment, AR ratio,
VA acceleration, CVD, LRS, day-of-week, VWAP-cross cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.base import InfraExitConfig
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    clamp_sl_dist,
    directional_sl_tp,
    make_entry_signal,
)
from stockdownloader.strategies.intraday.session import BarContext
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trail import VwapRatchetTrail
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy
from stockdownloader.indicators.intraday import candle_strength
from stockdownloader.core.math import ZERO
from stockdownloader.pinescript.models import ModeDefinition
from stockdownloader.pinescript.modes import pb_mode

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData


@dataclass(frozen=True, slots=True)
class PullbackStrategyConfig(InfraExitConfig):
    """All configurable parameters for the VWAP Pullback strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── PB entry ────────────────────────────────────────────────────────
    pb_zone: Decimal = Decimal("0.5")          # Pullback zone (× band width)
    pb_body: Decimal = Decimal("0.20")         # Min candle body (× ATR) — stronger
    rr: Decimal = Decimal("1.8")               # Risk : Reward — higher
    sl_atr: Decimal = Decimal("1.3")           # Stop loss (× ATR) — tighter
    sl_cap: Decimal = Decimal("2.00")          # Stop loss cap ($) — tighter

    # ── Trend ───────────────────────────────────────────────────────────
    trend_bars: int = 7                        # Stronger trend requirement
    # adx_thresh: inherited from base (22)
    htf_align: bool = True
    ar_filter: bool = True
    ar_thresh: Decimal = Decimal("0.9")
    ar_cap: Decimal = Decimal("1.30")
    va_filter: bool = True
    va_min: Decimal = Decimal("-0.1")
    cvd_long_filter: bool = True
    lrs_short_filter: bool = True
    lrs_thresh: Decimal = Decimal("0.08")
    max_vxc: int = 4                           # Fewer VWAP crosses allowed

    # ── Confluence (w_vol, w_sr, w_time, w_rsi, min_score inherited) ─────
    w_pq: int = 1
    w_box: int = 0
    min_score_long: int = 4                    # Same as min_score for more opportunities

    # ── Risk / session (max_day, spacing, allow_longs/shorts inherited) ─
    no_friday_short: bool = True
    no_monday_long: bool = False

    # ── Exit overrides ─────────────────────────────────────────────────
    trail_buf: Decimal = Decimal("0.10")       # Tighter trail (base: 0.15)

    # ── PB enhancements ─────────────────────────────────────────────────
    pb_vwap_bias: bool = True                  # Require VWAP bias alignment
    pb_vwap_bias_pct: Decimal = Decimal("0.6")
    pb_require_sr: bool = False
    pb_tp_mode: str = "rr"


class PullbackStrategy(BaseIntradayStrategy):
    """Standalone VWAP Pullback strategy.

    Trend-following entry: ADX confirms trending market, EMA aligned,
    price pulls back to VWAP zone, bullish/bearish candle confirms.
    """

    def __init__(self, config: PullbackStrategyConfig | None = None) -> None:
        c = config or PullbackStrategyConfig()
        self._c = c
        self._infra = IntradayInfra(c, IntradayExitManager(VwapRatchetTrail()))

    _ENTRY_FLAGS: dict[str, bool] = {}

    @staticmethod
    def pinescript_mode() -> ModeDefinition:
        """Return the PineScript mode definition for VWAP Pullback."""
        return pb_mode()

    @property
    def name(self) -> str:
        return "VWAP Pullback"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state

        # -- Spacing + day trade limit --
        ready = s.day_trades < c.max_day
        spaced = (ctx.bar_of_day - s.last_entry_bar) >= c.spacing or s.last_entry_bar <= 0
        if not ready or not spaced:
            return None

        # -- Guard: trending market --
        is_trending = ctx.adx_val >= c.adx_thresh
        if not is_trending:
            return None
        if ctx.bar_of_day < c.can_trade_bar:
            return None

        # -- Trend direction (EMA cross + VWAP slope) --
        trend_dir = self._trend_dir(
            s.bull_bars, s.bear_bars, c.trend_bars,
            ctx.vwap_delta, ctx.atr_val,
        )
        if trend_dir == 0:
            return None

        # -- VWAP zone --
        vwap = ctx.vwap_bands.vwap
        band_width = ctx.vwap_bands.std_dev
        zone_w = band_width * c.pb_zone
        dist_vwap = abs(ctx.bar.close - vwap)
        if zone_w <= ZERO or dist_vwap > zone_w:
            return None

        # -- Candle confirmation --
        cs = candle_strength(ctx.bar, ctx.atr_val)
        vwap_tol = ctx.atr_val * Decimal("0.25")

        bull_candle = cs.bull_candle(c.pb_body)
        bear_candle = cs.bear_candle(c.pb_body)

        bull_at_vwap = ctx.bar.close >= (vwap - vwap_tol) and (bull_candle or cs.bull_wick)
        bear_at_vwap = ctx.bar.close <= (vwap + vwap_tol) and (bear_candle or cs.bear_wick)
        strong_bull = cs.is_bull and cs.body >= ctx.atr_val * Decimal("0.3") and ctx.bar.close > vwap
        strong_bear = cs.is_bear and cs.body >= ctx.atr_val * Decimal("0.3") and ctx.bar.close < vwap

        # -- Determine direction --
        go_long = (
            trend_dir >= 1
            and (bull_at_vwap or strong_bull)
            and c.allow_longs
        )
        go_short = (
            trend_dir <= -1
            and (bear_at_vwap or strong_bear)
            and c.allow_shorts
        )

        if not go_long and not go_short:
            return None

        # -- VWAP crosses cap --
        if s.vwap_crosses > c.max_vxc:
            return None

        # -- VWAP session bias filter --
        if c.pb_vwap_bias and ctx.bar_of_day > 0:
            total = s.cum_bars_above_vwap + s.cum_bars_below_vwap
            if total > 0:
                ratio_above = Decimal(str(s.cum_bars_above_vwap)) / Decimal(str(total))
                ratio_below = Decimal(str(s.cum_bars_below_vwap)) / Decimal(str(total))
                if go_long and ratio_above < c.pb_vwap_bias_pct:
                    go_long = False
                if go_short and ratio_below < c.pb_vwap_bias_pct:
                    go_short = False
            if not go_long and not go_short:
                return None

        # -- Filters --
        # HTF alignment
        if c.htf_align:
            if go_long and ctx.htf_trend == -1:
                go_long = False
            if go_short and ctx.htf_trend == 1:
                go_short = False

        # AR filter (ATR ratio)
        if c.ar_filter and ctx.atr_val > ZERO:
            ar = ctx.atr_fast / ctx.atr_val
            if not (c.ar_thresh <= ar <= c.ar_cap):
                return None

        # VA filter (VWAP acceleration)
        if c.va_filter and ctx.vwap_accel < c.va_min:
            return None

        # CVD long filter
        if go_long and c.cvd_long_filter and ctx.cvd_norm <= ZERO:
            go_long = False

        # Day-of-week filters
        if go_short and c.no_friday_short and ctx.dow == 4:
            go_short = False
        if go_long and c.no_monday_long and ctx.dow == 0:
            go_long = False

        # LRS short filter
        if go_short and c.lrs_short_filter and ctx.lrs_atr > c.lrs_thresh:
            go_short = False

        if not go_long and not go_short:
            return None

        # -- S/R hard filter --
        if c.pb_require_sr and not ctx.sr_any:
            return None

        # -- Confluence scoring --
        pts_vol = c.w_vol if ctx.tod_rvol >= Decimal("1.0") else 0
        pts_sr = c.w_sr if ctx.sr_score_count > 0 else 0
        pts_sr2 = 1 if ctx.sr_score_count >= 2 else 0
        pts_time = c.w_time if ctx.is_good_time else 0
        pts_pq = c.w_pq if ctx.clean_pb else 0

        if go_long:
            pts_rsi = c.w_rsi if Decimal("45") <= ctx.rsi_val <= Decimal("70") else 0
            pts_box = c.w_box if ctx.box_pos <= Decimal("0.33") else 0
            score = pts_vol + pts_sr + pts_sr2 + pts_rsi + pts_time + pts_pq + pts_box
            if score < c.min_score_long:
                return None
        else:
            pts_rsi = c.w_rsi if Decimal("30") <= ctx.rsi_val <= Decimal("55") else 0
            pts_box = c.w_box if ctx.box_pos >= Decimal("0.67") else 0
            score = pts_vol + pts_sr + pts_sr2 + pts_rsi + pts_time + pts_pq + pts_box
            if score < c.min_score:
                return None

        max_score = c.w_vol + c.w_sr + 1 + c.w_rsi + c.w_time + c.w_pq + c.w_box

        # -- SL / TP --
        sl_dist = clamp_sl_dist(ctx.atr_val * c.sl_atr, c.sl_cap)
        if sl_dist is None:
            return None

        if c.pb_tp_mode == "vwap":
            tp_price = vwap
            sl_price = ctx.bar.close - sl_dist if go_long else ctx.bar.close + sl_dist
            reward = abs(tp_price - ctx.bar.close)
            rr_actual = reward / sl_dist if sl_dist > ZERO else ZERO
            if rr_actual < Decimal("0.3"):
                return None
        else:
            tp_dist = sl_dist * c.rr
            sl_price, tp_price = directional_sl_tp(go_long, ctx.bar.close, sl_dist, tp_dist)

        label = ("Strong" if abs(trend_dir) >= 2 else "Soft") + " Trend PB"

        return make_entry_signal(
            go_long=go_long,
            mode="PB",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=label,
        )

    @staticmethod
    def _trend_dir(
        bull_bars: int, bear_bars: int, min_bars: int,
        vwap_delta: Decimal, atr_val: Decimal,
    ) -> int:
        """Compute trend direction from EMA bars + VWAP slope.

        Returns +/-2 (strong), +/-1 (soft), 0 (no trend).
        """
        noise = atr_val * Decimal("0.10")
        if bull_bars >= min_bars:
            if vwap_delta > noise:
                return 2
            if vwap_delta >= -noise:
                return 1
        if bear_bars >= min_bars:
            if vwap_delta < -noise:
                return -2
            if vwap_delta <= noise:
                return -1
        return 0
