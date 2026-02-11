"""ORR (Opening Range Reversal) entry mode — fade OR level retests.

After the opening range completes, fades price rejection at OR H/L
extremes using hammer / engulfing patterns.  R:R filtered.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import IntradayAction, IntradaySignal
from stockdownloader.util.intraday_indicators import (
    is_bull_engulfing,
    is_bear_engulfing,
    is_hammer,
    is_inv_hammer,
)

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.vwap_strategy.session_state import SessionState
    from stockdownloader.strategy.vwap_strategy.vwap_config import VwapStrategyConfig
    from stockdownloader.util.intraday_indicators import ExtendedSessionVWAP

_ZERO = Decimal("0")
_TWO = Decimal("2")


class ORReversalMode:
    """Evaluate OR-reversal entry conditions."""

    def __init__(self, config: VwapStrategyConfig) -> None:
        self._c = config

    def evaluate(
        self,
        bar: IntradayPriceData,
        prev_bar: IntradayPriceData | None,
        state: SessionState,
        vwap_bands: ExtendedSessionVWAP,
        atr_val: Decimal,
        rel_vol: Decimal,
        bar_of_day: int,
    ) -> IntradaySignal | None:
        c = self._c

        if not c.orr_enable:
            return None
        if state.orr_fired_today:
            return None
        if not state.or_done:
            return None

        # ── Window check ────────────────────────────────────────────────
        if bar_of_day <= c.or_bars or bar_of_day > c.orr_window:
            return None

        # ── Proximity to OR extremes ────────────────────────────────────
        prox_dist = atr_val * c.orr_prox
        near_or_high = bar.high >= state.or_high - prox_dist
        near_or_low = bar.low <= state.or_low + prox_dist

        # ── Pattern detection (hammer / engulfing only) ─────────────────
        bull_signal = is_hammer(bar, atr_val) or (
            prev_bar is not None and is_bull_engulfing(bar, prev_bar, c.ps_engulf)
        )
        bear_signal = is_inv_hammer(bar, atr_val) or (
            prev_bar is not None and is_bear_engulfing(bar, prev_bar, c.ps_engulf)
        )

        # ── Direction ───────────────────────────────────────────────────
        go_long = near_or_low and bull_signal and c.allow_longs
        go_short = near_or_high and bear_signal and c.allow_shorts

        if not go_long and not go_short:
            return None

        # ── RVOL filter ─────────────────────────────────────────────────
        if rel_vol < c.orr_rvol:
            return None

        # ── SL / TP pre-calc and R:R filter ─────────────────────────────
        vwap = vwap_bands.vwap
        or_mid = (state.or_high + state.or_low) / _TWO

        if go_long:
            sl_raw = bar.close - (state.or_low - atr_val * c.orr_sl_atr)
            sl_dist = min(sl_raw, c.orr_sl_cap)
            if sl_dist <= _ZERO:
                return None
            sl_price = bar.close - sl_dist
            tp_price = vwap if c.orr_tp_mode == "VWAP" else or_mid
            reward = tp_price - bar.close
        else:
            sl_raw = (state.or_high + atr_val * c.orr_sl_atr) - bar.close
            sl_dist = min(sl_raw, c.orr_sl_cap)
            if sl_dist <= _ZERO:
                return None
            sl_price = bar.close + sl_dist
            tp_price = vwap if c.orr_tp_mode == "VWAP" else or_mid
            reward = bar.close - tp_price

        rr = reward / sl_dist if sl_dist > _ZERO else _ZERO
        if rr < c.orr_min_rr or rr > c.orr_max_rr:
            return None

        # ── Scoring ─────────────────────────────────────────────────────
        pts_vol = 3 if rel_vol >= Decimal("2") else (1 if rel_vol >= Decimal("1") else 0)
        score = pts_vol
        max_score = 3 + c.w_sr + 1

        # ── Pattern label ───────────────────────────────────────────────
        if is_hammer(bar, atr_val):
            pat = "Hammer"
        elif prev_bar and is_bull_engulfing(bar, prev_bar, c.ps_engulf):
            pat = "Engulfing"
        elif is_inv_hammer(bar, atr_val):
            pat = "Inv Hammer"
        else:
            pat = "Bear Engulf"

        level = "ORL" if go_long else "ORH"

        return IntradaySignal(
            action=IntradayAction.ENTER_LONG if go_long else IntradayAction.ENTER_SHORT,
            mode="ORR",
            stop_loss=sl_price,
            take_profit=tp_price,
            confluence_score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=f"ORR {pat} at {level}",
        )
