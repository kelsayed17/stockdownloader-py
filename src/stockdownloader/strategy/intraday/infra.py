"""Shared intraday infrastructure — composed, not inherited.

Provides session detection, opening-range tracking, day extremes,
daily bar aggregation, indicator computation, trend tracking, risk
controls, and exit evaluation.  Strategies compose this helper
rather than inheriting from a template-method base class.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Callable

from stockdownloader.model.intraday_signal import HOLD, IntradayAction, IntradaySignal
from stockdownloader.model.trade import Direction
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.session_state import SessionState
from stockdownloader.util.big_decimal_math import HUNDRED, ZERO
from stockdownloader.util.indicator_hub import IndicatorHub
from stockdownloader.util.intraday_indicators import (
    ExtendedSessionVWAP,
    compute_sr_score,
    daily_atr_prior,
)

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.price_data import PriceData
    from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(slots=True)
class BarContext:
    """All computed values for the current bar, passed to entry logic."""

    bar: IntradayPriceData
    prev_bar: IntradayPriceData | None
    state: SessionState
    bar_of_day: int
    dow: int
    # Indicators
    atr_val: Decimal
    atr_fast: Decimal
    adx_val: Decimal
    rsi_val: Decimal
    ema_fast: Decimal
    ema_slow: Decimal
    htf_trend: int
    cvd_norm: Decimal
    lrs_atr: Decimal
    rel_vol: Decimal
    tod_rvol: Decimal
    # VWAP
    vwap_bands: ExtendedSessionVWAP
    vwap_delta: Decimal
    vwap_accel: Decimal
    # Derived
    sr_any: bool
    sr_score_count: int
    box_pos: Decimal
    clean_pb: bool
    is_good_time: bool


class IntradayInfra:
    """Shared intraday infrastructure — composed, not inherited.

    Handles session detection, opening-range tracking, day extremes,
    daily bar aggregation, indicator computation, trend tracking, risk
    controls, position entry bookkeeping, and exit evaluation.
    """

    def __init__(
        self,
        config: InfraExitConfig,
        exit_manager: IntradayExitManager,
    ) -> None:
        self.state = SessionState()
        self.hub = IndicatorHub()
        self.exit_mgr = exit_manager
        self._c = config
        self._daily_bars: list[PriceData] = []
        self._last_agg_date: str = ""
        self._session_start_index: int = 0

    @property
    def warmup_period(self) -> int:
        return self._c.bars_per_day * 15

    def on_session_start(self, trading_date: str) -> None:
        self.state.reset(trading_date)

    def on_new_bar(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> BarContext | None:
        """Process a new bar: session detection, OR, indicators.

        Returns ``None`` when the strategy is in a position (caller
        should delegate to :meth:`evaluate_exit`).  Otherwise returns
        a :class:`BarContext` with all computed values for entry logic.
        """
        bar = data[current_index]
        s = self.state
        c = self._c

        # -- Session boundary detection --
        if current_index == 0 or bar.trading_date != data[current_index - 1].trading_date:
            self._on_new_day(data, current_index)

        s.bar_count += 1
        bar_of_day = s.bar_count

        # -- Update Opening Range --
        self._update_or(bar, bar_of_day)

        # -- Update day extremes --
        if bar.high > s.day_hod:
            s.day_hod = bar.high
        if bar.low < s.day_lod:
            s.day_lod = bar.low

        # -- Compute indicators --
        hub = self.hub
        vwap_bands = hub.extended_session_vwap_bands(data, current_index)
        atr_val = hub.atr(data, current_index, c.adx_len)
        atr_fast = hub.atr(data, current_index, 5)
        adx_result = hub.adx(data, current_index, c.adx_len)
        adx_val = adx_result.adx
        rsi_val = hub.rsi(data, current_index, 10)
        ema_fast = hub.ema(data, current_index, c.ema_fast)
        ema_slow = hub.ema(data, current_index, c.ema_slow)
        v_delta = hub.vwap_slope(data, current_index, c.slope_period)
        v_accel = hub.vwap_acceleration(data, current_index, c.slope_period)
        lrs_val = hub.lrs_normalized(data, current_index, 15)
        cvd_val = hub.cvd_normalized(data, current_index)
        r_vol = hub.rel_vol(data, current_index)
        t_rvol = hub.tod_rvol(data, current_index, c.tod_days, c.bars_per_day)
        htf = hub.htf_ema_trend(data, current_index)

        # -- Update trend tracking --
        self._update_trend(ema_fast, ema_slow, vwap_bands.vwap, bar)

        # -- Update CVD --
        bar_range = bar.high - bar.low
        if bar_range > ZERO:
            vd = (bar.close - bar.low) / bar_range
        else:
            vd = Decimal("0.5")
        s.cum_vd += (vd - Decimal("0.5")) * Decimal(str(bar.volume))

        # -- S/R proximity --
        sr_any, sr_score = compute_sr_score(
            bar.close,
            pd_high=s.pd_high,
            pd_low=s.pd_low,
            pd_close=s.pd_close,
            or_high=s.or_high if s.or_done else ZERO,
            or_low=s.or_low if s.or_done else ZERO,
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

        # -- Band touch tracking (for REV min_touches filter) --
        rev_band_label = getattr(c, "rev_band", "2σ")
        rev_upper, rev_lower = vwap_bands.band_pair(rev_band_label)
        _band_prox = atr_val * Decimal("0.3")
        if (rev_upper > ZERO and bar.high >= rev_upper - _band_prox) or (
            rev_lower > ZERO and bar.low <= rev_lower + _band_prox
        ):
            s.rev_band_touches += 1

        # -- Prior day range box position --
        pd_range = s.pd_high - s.pd_low
        if pd_range > ZERO:
            box_pos = max(ZERO, min(Decimal("1"), (bar.close - s.pd_low) / pd_range))
        else:
            box_pos = Decimal("0.5")

        clean_pb = s.vwap_crosses <= c.pq_max_cross
        is_good_time = (
            bar_of_day >= c.can_trade_bar
            and bar_of_day <= c.eod_bar
            and not (c.lunch_start <= bar_of_day <= c.lunch_end)
        )

        dow = bar.datetime_parsed.weekday()
        prev_bar = data[current_index - 1] if current_index > 0 else None

        # -- If in position, signal caller to evaluate exits --
        if s.in_position:
            return None

        ctx = BarContext(
            bar=bar,
            prev_bar=prev_bar,
            state=s,
            bar_of_day=bar_of_day,
            dow=dow,
            atr_val=atr_val,
            atr_fast=atr_fast,
            adx_val=adx_val,
            rsi_val=rsi_val,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            htf_trend=htf,
            cvd_norm=cvd_val,
            lrs_atr=lrs_val,
            rel_vol=r_vol,
            tod_rvol=t_rvol,
            vwap_bands=vwap_bands,
            vwap_delta=v_delta,
            vwap_accel=v_accel,
            sr_any=sr_any,
            sr_score_count=sr_score,
            box_pos=box_pos,
            clean_pb=clean_pb,
            is_good_time=is_good_time,
        )
        return ctx

    def evaluate_exit(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        """Evaluate exit conditions for the current bar."""
        bar = data[current_index]
        s = self.state
        c = self._c
        vwap_bands = self.hub.extended_session_vwap_bands(data, current_index)
        atr_val = self.hub.atr(data, current_index, c.adx_len)
        return self.exit_mgr.evaluate(bar, s, c, vwap_bands, atr_val, s.bar_count)

    def check_risk(self) -> bool:
        """Return True if blocked by circuit breaker or day loss limit."""
        s = self.state
        c = self._c

        if s.tripped or s.day_limited:
            return True

        if s.consec_losses >= c.circuit:
            s.tripped = True
            return True

        if s.session_start_equity > ZERO:
            loss_pct = (s.session_pnl / s.session_start_equity) * HUNDRED
            if loss_pct <= -c.day_loss:
                s.day_limited = True
                return True

        return False

    def run_bar(
        self,
        data: list[IntradayPriceData],
        current_index: int,
        evaluate_entry: Callable[[BarContext], IntradaySignal | None],
        entry_flags: dict[str, bool],
    ) -> IntradaySignal:
        """Standard bar loop shared by all standalone strategies.

        Handles session detection, indicators, risk check, entry evaluation,
        and exit delegation in one place.
        """
        ctx = self.on_new_bar(data, current_index)
        if ctx is None:
            return self.evaluate_exit(data, current_index)
        if self.check_risk():
            return HOLD
        sig = evaluate_entry(ctx)
        if sig is not None:
            return self.record_entry(sig, ctx.bar, ctx.bar_of_day, entry_flags)
        return HOLD

    def confirm_position_opened(self, is_long: bool) -> None:
        """Called by the engine after a position is successfully opened.

        Sets ``in_position`` authoritatively.  When the engine rejects
        the entry (shares = 0), this method is never called, so the
        pending entry recorded by :meth:`record_entry` is harmless — the
        next bar's :meth:`on_new_bar` will see ``in_position = False``
        and allow a fresh entry.
        """
        s = self.state
        s.in_position = True
        s.position_direction = Direction.LONG if is_long else Direction.SHORT

    def confirm_position_closed(self) -> None:
        """Called by the engine after a position is closed.

        :class:`IntradayExitManager` already clears position state for
        strategy-initiated exits, but the engine may also force-close
        at end-of-data, so we need an explicit reset path.
        """
        s = self.state
        s.in_position = False
        s.position_direction = None
        s.be_triggered = False
        s.trailing_vwap = False
        s.trailing_atr = False
        s.trail_level = ZERO
        s.orb_extreme = ZERO

    def record_entry(
        self,
        signal: IntradaySignal,
        bar: IntradayPriceData,
        bar_of_day: int,
        entry_flags: dict[str, bool],
    ) -> IntradaySignal:
        """Record entry in session state and return the signal.

        Sets ``in_position = True`` optimistically so that
        :meth:`on_new_bar` routes the next bar to exit evaluation.
        The engine confirms via :meth:`confirm_position_opened` and
        :meth:`confirm_position_closed`; these callbacks also handle
        engine-level force-closes that the exit manager doesn't see.
        """
        s = self.state
        is_long = signal.action == IntradayAction.ENTER_LONG

        s.in_position = True
        s.position_direction = Direction.LONG if is_long else Direction.SHORT
        s.entry_price = bar.close
        s.stop_loss = signal.stop_loss
        s.take_profit = signal.take_profit
        s.pending_tp = signal.take_profit
        s.entry_mode = signal.mode
        s.entry_bar = bar_of_day
        s.orig_sl = signal.stop_loss
        s.risk_amount = signal.risk_per_share
        s.be_triggered = False
        s.trailing_vwap = False
        s.trailing_atr = False
        s.trail_level = ZERO
        s.orb_extreme = bar.high if is_long else bar.low

        # Fire-once entries don't count against day trade limit
        if not entry_flags.get("fire_once"):
            s.day_trades += 1
            s.last_entry_bar = bar_of_day

        return signal

    # -- Private helpers --

    def _on_new_day(
        self, data: list[IntradayPriceData], current_index: int,
    ) -> None:
        bar = data[current_index]
        new_date = bar.trading_date

        prev_vwap = ZERO
        if current_index > 0:
            prev_vwap_bands = self.hub.extended_session_vwap_bands(data, current_index - 1)
            prev_vwap = prev_vwap_bands.vwap

        if self._last_agg_date != new_date and current_index > 0:
            prev_start = self._session_start_index
            prev_slice = data[prev_start:current_index]
            if prev_slice:
                from stockdownloader.model.price_data import PriceData as PD

                p_date = prev_slice[0].date[:10]
                p_open = prev_slice[0].open
                p_high = max(b.high for b in prev_slice)
                p_low = min(b.low for b in prev_slice)
                p_close = prev_slice[-1].close
                p_vol = sum(b.volume for b in prev_slice)
                self._daily_bars.append(PD(
                    date=p_date, open=p_open, high=p_high, low=p_low,
                    close=p_close, adj_close=p_close, volume=p_vol,
                ))
            self._session_start_index = current_index
            self._last_agg_date = new_date

        self.on_session_start(new_date)
        s = self.state

        if self._daily_bars:
            last_d = self._daily_bars[-1]
            s.pd_high = last_d.high
            s.pd_low = last_d.low
            s.pd_close = last_d.close

        s.daily_atr = daily_atr_prior(self._daily_bars, new_date)
        s.prev_vwap_close = prev_vwap

        # Gap detection: compare today's open to previous close
        if s.pd_close > ZERO:
            gap = bar.open - s.pd_close
            threshold = (
                s.daily_atr * Decimal("0.1") if s.daily_atr > ZERO
                else Decimal("0.50")
            )
            if gap > threshold:
                s.gap_dir = 1
            elif gap < -threshold:
                s.gap_dir = -1
            else:
                s.gap_dir = 0

        if len(self._daily_bars) >= 5:
            s.pw_high = max(b.high for b in self._daily_bars[-5:])
            s.pw_low = min(b.low for b in self._daily_bars[-5:])

    def _update_or(self, bar: IntradayPriceData, bar_of_day: int) -> None:
        s = self.state
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

                if s.or_close > s.or_open:
                    s.or_dir = 1
                elif s.or_close < s.or_open:
                    s.or_dir = -1
                else:
                    s.or_dir = 0

                if s.daily_atr > ZERO:
                    threshold = s.daily_atr * (c.ps_atr_pct / HUNDRED)
                    s.is_manip = s.or_range >= threshold

    def _update_trend(
        self,
        ema_fast: Decimal,
        ema_slow: Decimal,
        vwap: Decimal,
        bar: IntradayPriceData,
    ) -> None:
        s = self.state

        if ema_fast > ema_slow:
            s.bull_bars += 1
            s.bear_bars = 0
        elif ema_fast < ema_slow:
            s.bear_bars += 1
            s.bull_bars = 0
        else:
            s.bull_bars = 0
            s.bear_bars = 0

        s.trend_age += 1

        if vwap > ZERO:
            above = bar.close > vwap
            if s.prev_close_vs_vwap is not None:
                prev_above = s.prev_close_vs_vwap > 0
                if above != prev_above:
                    s.vwap_crosses += 1
            s.prev_close_vs_vwap = 1 if above else -1

            if above:
                s.bars_above_vwap += 1
                s.bars_below_vwap = 0
                s.cum_bars_above_vwap += 1
            else:
                s.bars_below_vwap += 1
                s.bars_above_vwap = 0
                s.cum_bars_below_vwap += 1
