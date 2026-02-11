"""PS (Pattern Scalp) entry mode — opening-range manipulation fade.

Fires when the opening range is ≥ 30 % of daily ATR (manipulation),
a reversal pattern (hammer / engulfing) appears in the opposite
direction, and RVOL confirms.  TP = fraction of OR range.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import IntradayAction, IntradaySignal
from stockdownloader.util.intraday_indicators import (
    is_bear_engulfing,
    is_bull_engulfing,
    is_hammer,
    is_inv_hammer,
)

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.strategy.vwap_strategy.session_state import SessionState
    from stockdownloader.strategy.vwap_strategy.vwap_config import VwapStrategyConfig

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")


class PatternScalpMode:
    """Evaluate pattern-scalp entry conditions."""

    def __init__(self, config: VwapStrategyConfig) -> None:
        self._c = config

    def evaluate(
        self,
        bar: IntradayPriceData,
        prev_bar: IntradayPriceData | None,
        state: SessionState,
        atr_val: Decimal,
        rel_vol: Decimal,
        bar_of_day: int,
    ) -> IntradaySignal | None:
        c = self._c

        if not c.ps_enable:
            return None
        if state.ps_fired_today:
            return None
        if not state.is_manip:
            return None

        # ── Window check ────────────────────────────────────────────────
        if bar_of_day < 4 or bar_of_day > c.ps_window:
            return None

        # ── Pattern detection ───────────────────────────────────────────
        bull_hammer = is_hammer(bar, atr_val)
        bear_hammer = is_inv_hammer(bar, atr_val)
        bull_engulf = prev_bar is not None and is_bull_engulfing(bar, prev_bar, c.ps_engulf)
        bear_engulf = prev_bar is not None and is_bear_engulfing(bar, prev_bar, c.ps_engulf)

        bull_reversal = bull_hammer or bull_engulf
        bear_reversal = bear_hammer or bear_engulf

        # ── Direction (fade the OR direction) ───────────────────────────
        go_long = (
            state.or_dir == -1
            and bull_reversal
            and c.allow_longs
        )
        go_short = (
            state.or_dir == 1
            and bear_reversal
            and c.allow_shorts
        )

        if not go_long and not go_short:
            return None

        # ── RVOL filter ─────────────────────────────────────────────────
        if rel_vol < c.ps_rvol:
            return None

        # ── SMA bias filter (optional) ──────────────────────────────────
        if c.ps_sma_filter and state.daily_sma > _ZERO:
            sma_dist = (bar.close - state.daily_sma) / state.daily_sma * _HUNDRED
            daily_bias = 1 if sma_dist > Decimal("0.5") else (-1 if sma_dist < Decimal("-0.5") else 0)
            sma_ok = (
                (state.or_dir == -1 and daily_bias >= 1)
                or (state.or_dir == 1 and daily_bias <= -1)
            )
            if not sma_ok:
                return None

        # ── SL calculation ──────────────────────────────────────────────
        if c.ps_sl_mode == "Day Extreme":
            if go_long:
                sl_raw = bar.close - state.day_lod + atr_val * Decimal("0.1")
            else:
                sl_raw = state.day_hod - bar.close + atr_val * Decimal("0.1")
        else:
            sl_raw = atr_val * c.ps_sl_atr

        sl_dist = min(sl_raw, c.ps_sl_cap)
        if sl_dist <= _ZERO:
            return None

        # ── TP calculation ──────────────────────────────────────────────
        tp_dist = state.or_range * (c.ps_tp_pct / _HUNDRED)

        if go_long:
            sl_price = bar.close - sl_dist
            tp_price = bar.close + tp_dist
        else:
            sl_price = bar.close + sl_dist
            tp_price = bar.close - tp_dist

        # ── Pattern label ───────────────────────────────────────────────
        if bull_hammer or bear_hammer:
            pat = "Hammer" if bull_hammer else "Inv Hammer"
        else:
            pat = "Engulfing" if bull_engulf else "Bear Engulf"

        # ── Scoring ─────────────────────────────────────────────────────
        pts_vol = 3 if rel_vol >= Decimal("2") else (1 if rel_vol >= Decimal("1") else 0)
        pts_sr = c.w_sr if state.or_done else 0
        score = pts_vol + pts_sr
        max_score = 3 + c.w_sr + 1

        return IntradaySignal(
            action=IntradayAction.ENTER_LONG if go_long else IntradayAction.ENTER_SHORT,
            mode="PS",
            stop_loss=sl_price,
            take_profit=tp_price,
            confluence_score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=f"PS {pat}",
        )
