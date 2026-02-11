"""REV (Reversal) entry mode — sideways market band fade.

Enters when ADX < threshold (not trending), VWAP is flat, price reaches
a σ-band extreme, and a confirmation candle prints.  TP targets VWAP.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import IntradayAction, IntradaySignal

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.vwap_strategy.session_state import SessionState
    from stockdownloader.strategy.vwap_strategy.vwap_config import VwapStrategyConfig
    from stockdownloader.util.intraday_indicators import ExtendedSessionVWAP

_ZERO = Decimal("0")


class ReversalMode:
    """Evaluate reversal (band-fade) entry conditions."""

    def __init__(self, config: VwapStrategyConfig) -> None:
        self._c = config

    def evaluate(
        self,
        bar: IntradayPriceData,
        state: SessionState,
        vwap_bands: ExtendedSessionVWAP,
        adx_val: Decimal,
        rsi_val: Decimal,
        atr_val: Decimal,
        vwap_delta: Decimal,
        tod_rvol: Decimal,
        bar_of_day: int,
        sr_score_count: int,
        is_good_time: bool,
    ) -> IntradaySignal | None:
        c = self._c

        if not c.rev_enable:
            return None
        if bar_of_day < c.can_trade_bar:
            return None

        # ── Guard: sideways market ──────────────────────────────────────
        is_trending = adx_val >= c.adx_thresh
        is_sideways = (
            not is_trending
            and atr_val > _ZERO
            and abs(vwap_delta) <= atr_val * Decimal("0.05")
        )
        if not is_sideways:
            return None

        # ── Band selection ──────────────────────────────────────────────
        bw = vwap_bands.std_dev
        vwap = vwap_bands.vwap

        if c.rev_band == "3σ":
            upper_band = vwap_bands.upper_3
            lower_band = vwap_bands.lower_3
        elif c.rev_band == "1.5σ":
            upper_band = vwap_bands.upper_15
            lower_band = vwap_bands.lower_15
        else:  # default "2σ"
            upper_band = vwap_bands.upper_2
            lower_band = vwap_bands.lower_2

        band_prox = atr_val * Decimal("0.3")

        near_upper = upper_band > _ZERO and bar.high >= upper_band - band_prox
        near_lower = lower_band > _ZERO and bar.low <= lower_band + band_prox

        # ── Candle confirmation ─────────────────────────────────────────
        body = abs(bar.close - bar.open)
        body_atr = body / atr_val if atr_val > _ZERO else _ZERO

        bull_candle = bar.close > bar.open and body_atr >= c.rev_body
        bear_candle = bar.close < bar.open and body_atr >= c.rev_body

        # ── Not-hugging filter ──────────────────────────────────────────
        not_hugging = state.bars_above_vwap < 20 and state.bars_below_vwap < 20

        # ── Direction ───────────────────────────────────────────────────
        go_long = near_lower and bull_candle and not_hugging and c.allow_longs
        go_short = (
            near_upper and bear_candle and not_hugging
            and c.allow_shorts and c.rev_shorts
        )

        if not go_long and not go_short:
            return None

        # ── SL / TP ────────────────────────────────────────────────────
        sl_dist = min(atr_val * c.rev_sl_atr, c.rev_sl_cap)
        if sl_dist <= _ZERO:
            return None

        if go_long:
            sl_price = bar.close - sl_dist
            tp_price = vwap
        else:
            sl_price = bar.close + sl_dist
            tp_price = vwap

        # ── R:R filter ──────────────────────────────────────────────────
        reward = abs(tp_price - bar.close)
        rr = reward / sl_dist if sl_dist > _ZERO else _ZERO
        if rr < c.rev_min_rr:
            return None

        # ── Scoring (lighter than PB) ───────────────────────────────────
        pts_vol = c.w_vol if tod_rvol < Decimal("1") else (1 if tod_rvol < Decimal("2") else 0)
        pts_sr = c.w_sr if sr_score_count > 0 else 0
        pts_sr2 = 1 if sr_score_count >= 2 else 0
        pts_time = c.w_time if is_good_time else 0

        if go_long:
            pts_rsi = c.w_rsi if rsi_val <= Decimal("35") else 0
        else:
            pts_rsi = c.w_rsi if rsi_val >= Decimal("65") else 0

        score = pts_vol + pts_sr + pts_sr2 + pts_rsi + pts_time
        max_score = c.w_vol + c.w_sr + 1 + c.w_rsi + c.w_time

        if score < c.min_score:
            return None

        return IntradaySignal(
            action=IntradayAction.ENTER_LONG if go_long else IntradayAction.ENTER_SHORT,
            mode="REV",
            stop_loss=sl_price,
            take_profit=tp_price,
            confluence_score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason="Band Reversal",
        )
