"""Per-session mutable state and bar context for intraday strategies.

:class:`SessionState` tracks all per-session state that resets at each
new trading day: opening range, day extremes, previous-day data, trend
tracking, risk management, and position tracking.

:class:`BarContext` is a pure data transfer object with no business logic.
It aggregates all indicator values, session state, and derived flags
that entry functions need to decide whether to trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.models.trade import Direction
from stockdownloader.core.math import ZERO

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.util.indicators.volume import (
        AnchoredVWAPBands,
        ExtendedSessionVWAP,
    )
    from stockdownloader.util.indicators.smc import StructureState

_INF = Decimal("999999")

class SessionState:
    """Tracks all per-session state that resets daily."""

    __slots__ = (
        "trading_date",
        "bar_count",
        # Opening Range
        "or_high",
        "or_low",
        "or_open",
        "or_close",
        "or_done",
        "or_range",
        "or_dir",
        "is_manip",
        # Day context
        "day_hod",
        "day_lod",
        "gap_dir",
        # Previous day data
        "pd_high",
        "pd_low",
        "pd_close",
        "pw_high",
        "pw_low",
        "daily_atr",
        "daily_sma",
        "prev_vwap_close",
        # Trend tracking
        "bull_bars",
        "bear_bars",
        "trend_age",
        "vwap_crosses",
        "cum_vd",
        "bars_above_vwap",
        "bars_below_vwap",
        "prev_close_vs_vwap",
        # Risk management
        "day_trades",
        "last_entry_bar",
        "consec_losses",
        "tripped",
        "day_limited",
        "session_start_equity",
        "session_pnl",
        # Position tracking
        "in_position",
        "position_direction",
        "entry_price",
        "stop_loss",
        "take_profit",
        "entry_mode",
        "entry_bar",
        "orig_sl",
        "risk_amount",
        "be_triggered",
        "trailing_vwap",
        "trailing_atr",
        "trail_level",
        "orb_extreme",
        "pending_tp",
        # Fire-once flag — reset each session so strategies that fire at
        # most once per day work correctly even when the engine never
        # calls strategy.on_session_start().
        "fired_today",
        # ORB retest pending state
        "orb_breakout_pending",
        "orb_breakout_level",
        "orb_breakout_long",
        "orb_breakout_bar",
        "orb_breakout_sl",
        # ORR breakout tracking (for require_break mode)
        "orr_break_above",
        "orr_break_below",
        # Cumulative VWAP side tracking (for PB bias filter)
        "cum_bars_above_vwap",
        "cum_bars_below_vwap",
        # REV band touch tracking
        "rev_band_touches",
        # NR7 compression detection
        "is_nr7",
    )

    def __init__(self) -> None:
        self.reset("")

    def reset(self, trading_date: str) -> None:
        """Reset all state for a new trading session."""
        self.trading_date = trading_date
        self.bar_count = 0

        # Opening Range
        self.or_high = ZERO
        self.or_low = _INF
        self.or_open = ZERO
        self.or_close = ZERO
        self.or_done = False
        self.or_range = ZERO
        self.or_dir: int = 0        # +1 bull OR, -1 bear OR
        self.is_manip = False

        # Day context
        self.day_hod = ZERO
        self.day_lod = _INF
        self.gap_dir: int = 0               # +1 gap up, -1 gap down, 0 flat

        # Previous day data (set by orchestrator before first bar)
        self.pd_high = ZERO
        self.pd_low = ZERO
        self.pd_close = ZERO
        self.pw_high = ZERO
        self.pw_low = ZERO
        self.daily_atr = ZERO
        self.daily_sma = ZERO
        self.prev_vwap_close = ZERO

        # Trend tracking
        self.bull_bars = 0
        self.bear_bars = 0
        self.trend_age = 0
        self.vwap_crosses = 0
        self.cum_vd = ZERO
        self.bars_above_vwap = 0
        self.bars_below_vwap = 0
        self.prev_close_vs_vwap: int | None = None  # +1/-1/None

        # Risk management
        self.day_trades = 0
        self.last_entry_bar = -100
        self.consec_losses = 0
        self.tripped = False
        self.day_limited = False
        self.session_start_equity = ZERO
        self.session_pnl = ZERO

        # Position tracking
        self.in_position = False
        self.position_direction: Direction | None = None
        self.entry_price = ZERO
        self.stop_loss = ZERO
        self.take_profit = ZERO
        self.entry_mode = ""
        self.entry_bar = 0
        self.orig_sl = ZERO
        self.risk_amount = ZERO
        self.be_triggered = False
        self.trailing_vwap = False
        self.trailing_atr = False
        self.trail_level = ZERO
        self.orb_extreme = ZERO
        self.pending_tp = ZERO

        # Fire-once
        self.fired_today = False

        # ORB retest pending state
        self.orb_breakout_pending = False
        self.orb_breakout_level = ZERO
        self.orb_breakout_long = True
        self.orb_breakout_bar = 0
        self.orb_breakout_sl = ZERO

        # ORR breakout tracking (for require_break mode)
        self.orr_break_above = False
        self.orr_break_below = False

        # Cumulative VWAP side tracking (for PB bias filter)
        self.cum_bars_above_vwap = 0
        self.cum_bars_below_vwap = 0

        # REV band touch tracking
        self.rev_band_touches = 0

        # NR7 compression detection
        self.is_nr7 = False


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
    # AVWAP (anchored, persists across sessions -- None when not used)
    avwap_bands: AnchoredVWAPBands | None
    # Market structure (SMC -- None when not used)
    structure: StructureState | None
    # Derived
    sr_any: bool
    sr_score_count: int
    box_pos: Decimal
    clean_pb: bool
    is_good_time: bool
