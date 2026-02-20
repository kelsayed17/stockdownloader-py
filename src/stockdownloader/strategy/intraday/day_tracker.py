"""Day-boundary tracking, opening-range detection, and trend state.

:class:`DayTracker` handles per-day state transitions that
:class:`IntradayInfra` delegates to:

- Daily bar aggregation from intraday bars
- Previous-day context (high, low, close, ATR, VWAP)
- NR7 compression and gap detection
- Opening range (OR) tracking and manipulation detection
- EMA / VWAP trend tracking (bull/bear bars, VWAP crosses)
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.util.big_decimal_math import HUNDRED, ZERO
from stockdownloader.util.intraday_indicators import daily_atr_prior

if TYPE_CHECKING:
    from stockdownloader.model.price_data import IntradayPriceData
    from stockdownloader.model.price_data import PriceData
    from stockdownloader.strategy.intraday.base_config import InfraExitConfig
    from stockdownloader.strategy.intraday.session_state import SessionState
    from stockdownloader.util.indicator_hub import IndicatorHub


class DayTracker:
    """Tracks day-boundary transitions and opening-range state.

    Composed by :class:`IntradayInfra` — not used directly by strategies.
    """

    def __init__(self) -> None:
        self._daily_bars: list[PriceData] = []
        self._last_agg_date: str = ""
        self._session_start_index: int = 0

    @property
    def daily_bars(self) -> list[PriceData]:
        """Aggregated daily bars (read-only access for infra)."""
        return self._daily_bars

    # -- Day boundary detection ------------------------------------------

    def on_new_day(
        self,
        data: list[IntradayPriceData],
        current_index: int,
        hub: IndicatorHub,
        state: SessionState,
    ) -> None:
        """Handle the transition to a new trading day.

        Aggregates the previous day's intraday bars into a single daily
        OHLCV bar, then sets session state for the new day: previous-day
        high/low/close, daily ATR, prior VWAP close, NR7, gap direction,
        and 5-week high/low.
        """
        bar = data[current_index]
        new_date = bar.trading_date

        prev_vwap = ZERO
        if current_index > 0:
            prev_vwap_bands = hub.extended_session_vwap_bands(data, current_index - 1)
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

        state.reset(new_date)

        if self._daily_bars:
            last_d = self._daily_bars[-1]
            state.pd_high = last_d.high
            state.pd_low = last_d.low
            state.pd_close = last_d.close

        state.daily_atr = daily_atr_prior(self._daily_bars, new_date)
        state.prev_vwap_close = prev_vwap

        # NR7 compression detection: previous day's range is the narrowest
        # of the last 7 daily bars (signals potential breakout day).
        if len(self._daily_bars) >= 7:
            prev = self._daily_bars[-1]
            prev_range = prev.high - prev.low
            state.is_nr7 = all(
                prev_range <= (d.high - d.low)
                for d in self._daily_bars[-7:-1]
            )
        else:
            state.is_nr7 = False

        # Gap detection: compare today's open to previous close
        if state.pd_close > ZERO:
            gap = bar.open - state.pd_close
            threshold = (
                state.daily_atr * Decimal("0.1") if state.daily_atr > ZERO
                else Decimal("0.50")
            )
            if gap > threshold:
                state.gap_dir = 1
            elif gap < -threshold:
                state.gap_dir = -1
            else:
                state.gap_dir = 0

        if len(self._daily_bars) >= 5:
            state.pw_high = max(b.high for b in self._daily_bars[-5:])
            state.pw_low = min(b.low for b in self._daily_bars[-5:])

    # -- Opening range tracking ------------------------------------------

    def update_or(
        self,
        bar: IntradayPriceData,
        bar_of_day: int,
        state: SessionState,
        config: InfraExitConfig,
    ) -> None:
        """Update opening range high/low/close/direction/manipulation."""
        if bar_of_day == 1:
            state.or_high = bar.high
            state.or_low = bar.low
            state.or_open = bar.open
        elif bar_of_day <= config.or_bars:
            if bar.high > state.or_high:
                state.or_high = bar.high
            if bar.low < state.or_low:
                state.or_low = bar.low
            if bar_of_day == config.or_bars:
                state.or_close = bar.close
                state.or_done = True
                state.or_range = state.or_high - state.or_low

                if state.or_close > state.or_open:
                    state.or_dir = 1
                elif state.or_close < state.or_open:
                    state.or_dir = -1
                else:
                    state.or_dir = 0

                if state.daily_atr > ZERO:
                    threshold = state.daily_atr * (config.ps_atr_pct / HUNDRED)
                    state.is_manip = state.or_range >= threshold

    # -- Trend tracking --------------------------------------------------

    @staticmethod
    def update_trend(
        ema_fast: Decimal,
        ema_slow: Decimal,
        vwap: Decimal,
        bar: IntradayPriceData,
        state: SessionState,
    ) -> None:
        """Update EMA/VWAP trend and side tracking."""
        if ema_fast > ema_slow:
            state.bull_bars += 1
            state.bear_bars = 0
        elif ema_fast < ema_slow:
            state.bear_bars += 1
            state.bull_bars = 0
        else:
            state.bull_bars = 0
            state.bear_bars = 0

        state.trend_age += 1

        if vwap > ZERO:
            above = bar.close > vwap
            if state.prev_close_vs_vwap is not None:
                prev_above = state.prev_close_vs_vwap > 0
                if above != prev_above:
                    state.vwap_crosses += 1
            state.prev_close_vs_vwap = 1 if above else -1

            if above:
                state.bars_above_vwap += 1
                state.bars_below_vwap = 0
                state.cum_bars_above_vwap += 1
            else:
                state.bars_below_vwap += 1
                state.bars_above_vwap = 0
                state.cum_bars_below_vwap += 1
