"""ORB (Opening Range Breakout) entry mode — momentum continuation.

Enters when a 5-minute candle closes beyond the OR extreme with strong
volume confirmation (RVOL ≥ 2.0) and VWAP alignment.  No fixed TP —
uses ATR chandelier trailing stop only.
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


class ORBreakoutMode:
    """Evaluate OR-breakout entry conditions."""

    def __init__(self, config: VwapStrategyConfig) -> None:
        self._c = config

    def evaluate(
        self,
        bar: IntradayPriceData,
        state: SessionState,
        vwap_bands: ExtendedSessionVWAP,
        atr_val: Decimal,
        rel_vol: Decimal,
        bar_of_day: int,
    ) -> IntradaySignal | None:
        c = self._c

        if not c.orb_enable:
            return None
        if state.orb_fired_today:
            return None
        if not state.or_done:
            return None
        # PS and ORB are contradictory (manipulation vs continuation)
        if state.ps_fired_today:
            return None

        # ── Window check ────────────────────────────────────────────────
        if bar_of_day <= c.or_bars or bar_of_day > c.orb_window:
            return None

        # ── Close beyond OR extreme ─────────────────────────────────────
        close_long = bar.close > state.or_high and bar.close > bar.open
        close_short = bar.close < state.or_low and bar.close < bar.open

        if not close_long and not close_short:
            return None

        # ── Body filter ─────────────────────────────────────────────────
        body = abs(bar.close - bar.open)
        body_atr = body / atr_val if atr_val > _ZERO else _ZERO
        if body_atr < c.orb_body_min:
            return None

        # ── RVOL filter ─────────────────────────────────────────────────
        if rel_vol < c.orb_rvol:
            return None

        # ── VWAP alignment ──────────────────────────────────────────────
        vwap = vwap_bands.vwap
        if c.orb_vwap_align:
            if close_long and bar.close <= vwap:
                return None
            if close_short and bar.close >= vwap:
                return None

        # ── Direction ───────────────────────────────────────────────────
        go_long = close_long and c.allow_longs
        go_short = close_short and c.allow_shorts

        if not go_long and not go_short:
            return None

        # ── SL calculation ──────────────────────────────────────────────
        if go_long:
            if c.orb_sl_mode == "OR Opposite":
                sl_dist = min(bar.close - state.or_low, atr_val * c.orb_sl_atr)
            else:
                sl_dist = atr_val * c.orb_sl_atr
            sl_dist = min(sl_dist, c.orb_sl_cap)
            sl_price = bar.close - sl_dist
        else:
            if c.orb_sl_mode == "OR Opposite":
                sl_dist = min(state.or_high - bar.close, atr_val * c.orb_sl_atr)
            else:
                sl_dist = atr_val * c.orb_sl_atr
            sl_dist = min(sl_dist, c.orb_sl_cap)
            sl_price = bar.close + sl_dist

        if sl_dist <= _ZERO:
            return None

        # ── Scoring ─────────────────────────────────────────────────────
        pts_vol = 3 if rel_vol >= Decimal("2") else (2 if rel_vol >= Decimal("1.5") else 0)
        score = pts_vol
        max_score = 3 + c.w_sr + 1

        level = "ORH" if go_long else "ORL"

        return IntradaySignal(
            action=IntradayAction.ENTER_LONG if go_long else IntradayAction.ENTER_SHORT,
            mode="ORB",
            stop_loss=sl_price,
            take_profit=_ZERO,  # No fixed TP — chandelier trail only
            confluence_score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=f"ORB Break {level}",
        )
