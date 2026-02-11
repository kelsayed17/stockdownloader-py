"""PB (Pullback) entry mode — trending market VWAP bounce.

Requires ADX ≥ threshold, EMA trend confirmed for N bars, VWAP slope
aligned, price in VWAP zone, bullish/bearish candle confirmation,
confluence score above minimum.  Filters: HTF alignment, AR ratio,
VA acceleration, CVD, LRS, day-of-week, VWAP-cross cap.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import IntradayAction, IntradaySignal
from stockdownloader.model.trade import Direction

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.vwap_strategy.session_state import SessionState
    from stockdownloader.strategy.vwap_strategy.vwap_config import VwapStrategyConfig
    from stockdownloader.util.intraday_indicators import ExtendedSessionVWAP

_ZERO = Decimal("0")


class PullbackMode:
    """Evaluate pullback entry conditions."""

    def __init__(self, config: VwapStrategyConfig) -> None:
        self._c = config

    def evaluate(
        self,
        bar: IntradayPriceData,
        prev_bar: IntradayPriceData | None,
        state: SessionState,
        vwap_bands: ExtendedSessionVWAP,
        adx_val: Decimal,
        rsi_val: Decimal,
        atr_val: Decimal,
        atr_fast: Decimal,
        ema_fast: Decimal,
        ema_slow: Decimal,
        htf_trend: int,
        cvd_norm: Decimal,
        lrs_atr: Decimal,
        vwap_delta: Decimal,
        vwap_accel: Decimal,
        tod_rvol: Decimal,
        rel_vol: Decimal,
        bar_of_day: int,
        dow: int,
        sr_any: bool,
        sr_score_count: int,
        box_pos: Decimal,
        clean_pb: bool,
        is_good_time: bool,
    ) -> IntradaySignal | None:
        c = self._c

        # ── Guard: trending market ──────────────────────────────────────
        is_trending = adx_val >= c.adx_thresh
        if not is_trending:
            return None
        if bar_of_day < c.can_trade_bar:
            return None

        # ── Trend direction (EMA cross + VWAP slope) ────────────────────
        trend_dir = self._trend_dir(
            state.bull_bars, state.bear_bars, c.trend_bars,
            vwap_delta, atr_val,
        )
        if trend_dir == 0:
            return None

        # ── VWAP zone ───────────────────────────────────────────────────
        vwap = vwap_bands.vwap
        band_width = vwap_bands.upper_2 - vwap  # 2σ width / 2 ≈ 1σ
        # PineScript uses upper1 - vwap (which is 1 std_dev = bandWidth)
        band_width = vwap_bands.std_dev
        zone_w = band_width * c.pb_zone
        dist_vwap = abs(bar.close - vwap)
        if zone_w <= _ZERO or dist_vwap > zone_w:
            return None

        # ── Candle confirmation ─────────────────────────────────────────
        body = abs(bar.close - bar.open)
        bar_range = bar.high - bar.low
        body_atr = body / atr_val if atr_val > _ZERO else _ZERO
        vwap_tol = atr_val * Decimal("0.25")

        bull_candle = bar.close > bar.open and body_atr >= c.pb_body
        bear_candle = bar.close < bar.open and body_atr >= c.pb_body
        bull_wick = bar_range > _ZERO and bar.close > bar.low + bar_range * Decimal("0.6")
        bear_wick = bar_range > _ZERO and bar.close < bar.high - bar_range * Decimal("0.6")

        bull_at_vwap = bar.close >= (vwap - vwap_tol) and (bull_candle or bull_wick)
        bear_at_vwap = bar.close <= (vwap + vwap_tol) and (bear_candle or bear_wick)
        strong_bull = bar.close > bar.open and body >= atr_val * Decimal("0.3") and bar.close > vwap
        strong_bear = bar.close < bar.open and body >= atr_val * Decimal("0.3") and bar.close < vwap

        # ── Determine direction ─────────────────────────────────────────
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

        # ── VWAP crosses cap ────────────────────────────────────────────
        if state.vwap_crosses > c.max_vxc:
            return None

        # ── Filters ─────────────────────────────────────────────────────
        # HTF alignment
        if c.htf_align:
            if go_long and htf_trend == -1:
                go_long = False
            if go_short and htf_trend == 1:
                go_short = False

        # AR filter (ATR ratio)
        if c.ar_filter and atr_val > _ZERO:
            ar = atr_fast / atr_val
            if not (c.ar_thresh <= ar <= c.ar_cap):
                return None

        # VA filter (VWAP acceleration)
        if c.va_filter and vwap_accel < c.va_min:
            return None

        # CVD long filter
        if go_long and c.cvd_long_filter and cvd_norm <= _ZERO:
            go_long = False

        # Day-of-week filters (dow: 0=Mon ... 6=Sun, PineScript: 2=Mon ... 7=Sat)
        # Python datetime.weekday(): 0=Mon, 4=Fri
        if go_short and c.no_friday_short and dow == 4:
            go_short = False
        if go_long and c.no_monday_long and dow == 0:
            go_long = False

        # LRS short filter
        if go_short and c.lrs_short_filter and lrs_atr > c.lrs_thresh:
            go_short = False

        if not go_long and not go_short:
            return None

        # ── Confluence scoring ──────────────────────────────────────────
        pts_vol = c.w_vol if tod_rvol < Decimal("1") else (1 if tod_rvol < Decimal("2") else 0)
        pts_sr = c.w_sr if sr_score_count > 0 else 0
        pts_sr2 = 1 if sr_score_count >= 2 else 0
        pts_time = c.w_time if is_good_time else 0
        pts_pq = c.w_pq if clean_pb else 0

        if go_long:
            pts_rsi = c.w_rsi if Decimal("45") <= rsi_val <= Decimal("70") else 0
            pts_box = c.w_box if box_pos <= Decimal("0.33") else 0
            score = pts_vol + pts_sr + pts_sr2 + pts_rsi + pts_time + pts_pq + pts_box
            if score < c.min_score_long:
                return None
        else:
            pts_rsi = c.w_rsi if Decimal("30") <= rsi_val <= Decimal("55") else 0
            pts_box = c.w_box if box_pos >= Decimal("0.67") else 0
            score = pts_vol + pts_sr + pts_sr2 + pts_rsi + pts_time + pts_pq + pts_box
            if score < c.min_score:
                return None

        max_score = c.w_vol + c.w_sr + 1 + c.w_rsi + c.w_time + c.w_pq + c.w_box

        # ── SL / TP ────────────────────────────────────────────────────
        sl_dist = min(atr_val * c.sl_atr, c.sl_cap)
        if sl_dist <= _ZERO:
            return None

        direction = Direction.LONG if go_long else Direction.SHORT
        if go_long:
            sl_price = bar.close - sl_dist
            tp_price = bar.close + sl_dist * c.rr
        else:
            sl_price = bar.close + sl_dist
            tp_price = bar.close - sl_dist * c.rr

        label = ("Strong" if abs(trend_dir) >= 2 else "Soft") + " Trend PB"

        return IntradaySignal(
            action=IntradayAction.ENTER_LONG if go_long else IntradayAction.ENTER_SHORT,
            mode="PB",
            stop_loss=sl_price,
            take_profit=tp_price,
            confluence_score=score,
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

        Returns ±2 (strong), ±1 (soft), 0 (no trend).
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
