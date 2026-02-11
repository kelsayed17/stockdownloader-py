"""VWAP v11.2 intraday strategy — main orchestrator.

Runs five entry modes (PS → ORB → ORR → PB → REV) in priority order on
each 5-minute bar, manages per-session state, risk controls, and
delegates exit evaluation to :class:`VwapExitManager`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import HOLD, IntradayAction, IntradaySignal
from stockdownloader.model.trade import Direction
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.strategy.vwap_strategy.or_breakout_mode import ORBreakoutMode
from stockdownloader.strategy.vwap_strategy.or_reversal_mode import ORReversalMode
from stockdownloader.strategy.vwap_strategy.pattern_scalp_mode import PatternScalpMode
from stockdownloader.strategy.vwap_strategy.pullback_mode import PullbackMode
from stockdownloader.strategy.vwap_strategy.reversal_mode import ReversalMode
from stockdownloader.strategy.vwap_strategy.session_state import SessionState
from stockdownloader.strategy.vwap_strategy.vwap_config import VwapStrategyConfig
from stockdownloader.strategy.vwap_strategy.vwap_exit_manager import VwapExitManager
from stockdownloader.util.intraday_indicators import (
    ExtendedSessionVWAP,
    aggregate_to_daily,
    compute_sr_score,
    cvd_normalized,
    daily_atr_prior,
    extended_session_vwap_bands,
    lrs_normalized,
    rel_vol as _rel_vol,
    tod_rvol as _tod_rvol,
    vwap_acceleration,
    vwap_slope,
    htf_ema_trend,
)
from stockdownloader.util.moving_average_calculator import ema as _ema
from stockdownloader.util.technical_indicators import (
    adx as _adx,
    atr as _atr,
    rsi as _rsi,
)

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.price_data import PriceData

_ZERO = Decimal("0")
_INF = Decimal("999999")
_HUNDRED = Decimal("100")


class VwapStrategy(IntradayTradingStrategy):
    """VWAP v11.2 intraday strategy for 5-minute bars on SPY.

    Instantiate with a :class:`VwapStrategyConfig` (defaults match the
    PineScript v11.2 defaults), then call :meth:`evaluate` on each bar
    in chronological order.
    """

    def __init__(self, config: VwapStrategyConfig | None = None) -> None:
        self._c = config or VwapStrategyConfig()
        self._state = SessionState()
        self._exit = VwapExitManager()

        # Entry modes in priority order
        self._ps = PatternScalpMode(self._c)
        self._orb = ORBreakoutMode(self._c)
        self._orr = ORReversalMode(self._c)
        self._pb = PullbackMode(self._c)
        self._rev = ReversalMode(self._c)

        # Daily-bar cache (built incrementally)
        self._daily_bars: list[PriceData] = []
        self._last_agg_date: str = ""

    # ── ABC implementation ──────────────────────────────────────────────

    def get_name(self) -> str:
        return "VWAP v11.2"

    def get_warmup_period(self) -> int:
        # Need multiple sessions for daily ATR(14), RVOL, etc.
        return self._c.bars_per_day * 15

    def on_session_start(self, trading_date: str) -> None:
        self._state.reset(trading_date)

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        bar = data[current_index]
        s = self._state
        c = self._c

        # ── Session boundary detection ──────────────────────────────────
        if current_index == 0 or bar.trading_date != data[current_index - 1].trading_date:
            self._on_new_day(data, current_index)

        s.bar_count += 1
        bar_of_day = s.bar_count

        # ── Update Opening Range ────────────────────────────────────────
        self._update_or(bar, bar_of_day)

        # ── Update day extremes ─────────────────────────────────────────
        if bar.high > s.day_hod:
            s.day_hod = bar.high
        if bar.low < s.day_lod:
            s.day_lod = bar.low

        # ── Compute indicators ──────────────────────────────────────────
        vwap_bands = extended_session_vwap_bands(data, current_index)
        atr_val = _atr(data, current_index, c.adx_len)
        atr_fast = _atr(data, current_index, 5)
        adx_result = _adx(data, current_index, c.adx_len)
        adx_val = adx_result.adx
        rsi_val = _rsi(data, current_index, 10)
        ema_fast = _ema(data, current_index, c.ema_fast)
        ema_slow = _ema(data, current_index, c.ema_slow)
        v_delta = vwap_slope(data, current_index, c.slope_period)
        v_accel = vwap_acceleration(data, current_index, c.slope_period)
        lrs_val = lrs_normalized(data, current_index, 15)
        cvd_val = cvd_normalized(data, current_index)
        r_vol = _rel_vol(data, current_index)
        t_rvol = _tod_rvol(data, current_index, c.tod_days, c.bars_per_day)
        htf = htf_ema_trend(data, current_index)

        # ── Update trend tracking ───────────────────────────────────────
        self._update_trend(ema_fast, ema_slow, vwap_bands.vwap, bar)

        # ── Update CVD ──────────────────────────────────────────────────
        bar_range = bar.high - bar.low
        if bar_range > _ZERO:
            vd = (bar.close - bar.low) / bar_range
        else:
            vd = Decimal("0.5")
        s.cum_vd += (vd - Decimal("0.5")) * Decimal(str(bar.volume))

        # ── S/R proximity ───────────────────────────────────────────────
        sr_any, sr_score = compute_sr_score(
            bar.close,
            pd_high=s.pd_high,
            pd_low=s.pd_low,
            pd_close=s.pd_close,
            or_high=s.or_high if s.or_done else _ZERO,
            or_low=s.or_low if s.or_done else _ZERO,
            pw_high=s.pw_high,
            pw_low=s.pw_low,
            prev_vwap=s.prev_vwap_close,
            proximity_pct=c.sr_prox,
            sr_pdhlc=c.sr_pdhlc,
            sr_round=c.sr_round,
            sr_or=c.sr_or,
            sr_week_hl=c.sr_week_hl,
            sr_prev_vwap=c.sr_prev_vwap,
        )

        # ── Prior day range box position ────────────────────────────────
        pd_range = s.pd_high - s.pd_low
        if pd_range > _ZERO:
            box_pos = max(_ZERO, min(Decimal("1"), (bar.close - s.pd_low) / pd_range))
        else:
            box_pos = Decimal("0.5")

        clean_pb = s.vwap_crosses <= c.pq_max_cross
        is_good_time = (
            bar_of_day >= c.can_trade_bar
            and bar_of_day <= c.eod_bar
            and not (c.lunch_start <= bar_of_day <= c.lunch_end)
        )

        # ── Weekday (Python: 0=Mon, 4=Fri) ─────────────────────────────
        dow = bar.datetime_parsed.weekday()

        prev_bar = data[current_index - 1] if current_index > 0 else None

        # ── If in position, evaluate exits ──────────────────────────────
        if s.in_position:
            return self._exit.evaluate(
                bar, s, c, vwap_bands, atr_val, bar_of_day,
            )

        # ── Risk checks ────────────────────────────────────────────────
        if s.tripped or s.day_limited:
            return HOLD

        # Circuit breaker
        if s.consec_losses >= c.circuit:
            s.tripped = True
            return HOLD

        # Daily loss limit
        if s.session_start_equity > _ZERO:
            loss_pct = (s.session_pnl / s.session_start_equity) * _HUNDRED
            if loss_pct <= -c.day_loss:
                s.day_limited = True
                return HOLD

        # ── Spacing + day trade limit ───────────────────────────────────
        ready = s.day_trades < c.max_day
        spaced = (bar_of_day - s.last_entry_bar) >= c.spacing or s.last_entry_bar <= 0

        # ── Evaluate entry modes in priority order ──────────────────────

        # 1. Pattern Scalp
        ps_ready = not s.ps_fired_today and not s.tripped and not s.day_limited
        if ps_ready:
            sig = self._ps.evaluate(bar, prev_bar, s, atr_val, r_vol, bar_of_day)
            if sig is not None:
                return self._enter(sig, s, bar, bar_of_day, is_ps=True)

        # 2. OR Breakout
        orb_ready = not s.orb_fired_today and not s.tripped and not s.day_limited
        if orb_ready:
            sig = self._orb.evaluate(bar, s, vwap_bands, atr_val, r_vol, bar_of_day)
            if sig is not None:
                return self._enter(sig, s, bar, bar_of_day, is_orb=True)

        # 3. OR Reversal
        orr_ready = not s.orr_fired_today and not s.tripped and not s.day_limited
        if orr_ready:
            sig = self._orr.evaluate(
                bar, prev_bar, s, vwap_bands, atr_val, r_vol, bar_of_day,
            )
            if sig is not None:
                return self._enter(sig, s, bar, bar_of_day, is_orr=True)

        # 4. Pullback (needs ready + spaced)
        if ready and spaced:
            sig = self._pb.evaluate(
                bar, prev_bar, s, vwap_bands, adx_val, rsi_val,
                atr_val, atr_fast, ema_fast, ema_slow, htf, cvd_val, lrs_val,
                v_delta, v_accel, t_rvol, r_vol, bar_of_day, dow,
                sr_any, sr_score, box_pos, clean_pb, is_good_time,
            )
            if sig is not None:
                return self._enter(sig, s, bar, bar_of_day, is_pb=True)

        # 5. Reversal (needs ready + spaced)
        if ready and spaced:
            sig = self._rev.evaluate(
                bar, s, vwap_bands, adx_val, rsi_val, atr_val,
                v_delta, t_rvol, bar_of_day, sr_score, is_good_time,
            )
            if sig is not None:
                return self._enter(sig, s, bar, bar_of_day)

        return HOLD

    # ── Private helpers ─────────────────────────────────────────────────

    def _on_new_day(
        self, data: list[IntradayPriceData], current_index: int,
    ) -> None:
        """Handle session boundary: aggregate prior day, reset state."""
        bar = data[current_index]
        new_date = bar.trading_date

        # Capture prev-day VWAP close
        prev_vwap = _ZERO
        if current_index > 0:
            prev_vwap_bands = extended_session_vwap_bands(data, current_index - 1)
            prev_vwap = prev_vwap_bands.vwap

        # Aggregate intraday → daily (only if we haven't already)
        if self._last_agg_date != new_date and current_index > 0:
            self._daily_bars = aggregate_to_daily(data[:current_index])
            self._last_agg_date = new_date

        self.on_session_start(new_date)
        s = self._state

        # Previous day data from daily bars
        if self._daily_bars:
            last_d = self._daily_bars[-1]
            s.pd_high = last_d.high
            s.pd_low = last_d.low
            s.pd_close = last_d.close

        # Daily ATR
        s.daily_atr = daily_atr_prior(self._daily_bars, new_date)
        s.prev_vwap_close = prev_vwap

        # Previous week H/L (last 5 daily bars before current week)
        if len(self._daily_bars) >= 5:
            s.pw_high = max(b.high for b in self._daily_bars[-5:])
            s.pw_low = min(b.low for b in self._daily_bars[-5:])

    def _update_or(self, bar: IntradayPriceData, bar_of_day: int) -> None:
        """Update opening-range state."""
        s = self._state
        c = self._c

        if bar_of_day == 1:
            s.or_high = bar.high
            s.or_low = bar.low
            s.or_open = bar.open
        elif bar_of_day <= c.or_bars:
            if bar.high > s.or_high:
                s.or_high = bar.high
            if bar.low < s.or_low:
                s.or_low = bar.low
            if bar_of_day == c.or_bars:
                s.or_close = bar.close
                s.or_done = True
                s.or_range = s.or_high - s.or_low

                # Determine OR direction
                if s.or_close > s.or_open:
                    s.or_dir = 1
                elif s.or_close < s.or_open:
                    s.or_dir = -1
                else:
                    s.or_dir = 0

                # Check manipulation
                if s.daily_atr > _ZERO:
                    threshold = s.daily_atr * (c.ps_atr_pct / _HUNDRED)
                    s.is_manip = s.or_range >= threshold

    def _update_trend(
        self,
        ema_fast: Decimal,
        ema_slow: Decimal,
        vwap: Decimal,
        bar: IntradayPriceData,
    ) -> None:
        """Update EMA trend bars, VWAP crosses, and bars-above/below-VWAP."""
        s = self._state

        # EMA trend bars
        if ema_fast > ema_slow:
            s.bull_bars += 1
            s.bear_bars = 0
        elif ema_fast < ema_slow:
            s.bear_bars += 1
            s.bull_bars = 0
        else:
            s.bull_bars = 0
            s.bear_bars = 0

        # Trend age (bars since EMA cross)
        # Simplified: just increment; orchestrator resets on new day
        s.trend_age += 1

        # VWAP crosses
        if vwap > _ZERO:
            above = bar.close > vwap
            if s.prev_close_vs_vwap is not None:
                prev_above = s.prev_close_vs_vwap > 0
                if above != prev_above:
                    s.vwap_crosses += 1
            s.prev_close_vs_vwap = 1 if above else -1

            # Bars above/below VWAP
            if above:
                s.bars_above_vwap += 1
                s.bars_below_vwap = 0
            else:
                s.bars_below_vwap += 1
                s.bars_above_vwap = 0

    @staticmethod
    def _enter(
        signal: IntradaySignal,
        state: SessionState,
        bar: IntradayPriceData,
        bar_of_day: int,
        *,
        is_ps: bool = False,
        is_orb: bool = False,
        is_orr: bool = False,
        is_pb: bool = False,
    ) -> IntradaySignal:
        """Record entry in session state and return the signal."""
        is_long = signal.action == IntradayAction.ENTER_LONG

        state.in_position = True
        state.position_direction = Direction.LONG if is_long else Direction.SHORT
        state.entry_price = bar.close
        state.stop_loss = signal.stop_loss
        state.take_profit = signal.take_profit
        state.pending_tp = signal.take_profit
        state.entry_mode = signal.mode
        state.entry_bar = bar_of_day
        state.orig_sl = signal.stop_loss
        state.risk_amount = signal.risk_per_share
        state.be_triggered = False
        state.trailing_vwap = False
        state.trailing_atr = False
        state.trail_level = _ZERO
        state.orb_extreme = bar.high if is_long else bar.low
        state.is_pb_trade = is_pb or is_orr  # PB and ORR get VWAP trail
        state.is_orb_trade = is_orb

        # Update fire-once flags
        if is_ps:
            state.ps_fired_today = True
        elif is_orb:
            state.orb_fired_today = True
        elif is_orr:
            state.orr_fired_today = True
        else:
            state.day_trades += 1
            state.last_entry_bar = bar_of_day

        return signal
