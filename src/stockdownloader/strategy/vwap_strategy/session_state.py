"""Per-session mutable state for the VWAP v11.2 strategy.

Mirrors PineScript ``var`` variables that reset at each new trading day.
"""

from __future__ import annotations

from decimal import Decimal

from stockdownloader.model.trade import Direction

_ZERO = Decimal("0")
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
        # Day extremes
        "day_hod",
        "day_lod",
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
        # Mode fire-once flags
        "ps_fired_today",
        "orr_fired_today",
        "orb_fired_today",
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
        "is_pb_trade",
        "is_orb_trade",
        "pending_tp",
    )

    def __init__(self) -> None:
        self.reset("")

    def reset(self, trading_date: str) -> None:
        """Reset all state for a new trading session."""
        self.trading_date = trading_date
        self.bar_count = 0

        # Opening Range
        self.or_high = _ZERO
        self.or_low = _INF
        self.or_open = _ZERO
        self.or_close = _ZERO
        self.or_done = False
        self.or_range = _ZERO
        self.or_dir: int = 0        # +1 bull OR, -1 bear OR
        self.is_manip = False

        # Day extremes
        self.day_hod = _ZERO
        self.day_lod = _INF

        # Previous day data (set by orchestrator before first bar)
        self.pd_high = _ZERO
        self.pd_low = _ZERO
        self.pd_close = _ZERO
        self.pw_high = _ZERO
        self.pw_low = _ZERO
        self.daily_atr = _ZERO
        self.daily_sma = _ZERO
        self.prev_vwap_close = _ZERO

        # Trend tracking
        self.bull_bars = 0
        self.bear_bars = 0
        self.trend_age = 0
        self.vwap_crosses = 0
        self.cum_vd = _ZERO
        self.bars_above_vwap = 0
        self.bars_below_vwap = 0
        self.prev_close_vs_vwap: int | None = None  # +1/-1/None

        # Mode fire-once flags
        self.ps_fired_today = False
        self.orr_fired_today = False
        self.orb_fired_today = False

        # Risk management
        self.day_trades = 0
        self.last_entry_bar = -100
        self.consec_losses = 0
        self.tripped = False
        self.day_limited = False
        self.session_start_equity = _ZERO
        self.session_pnl = _ZERO

        # Position tracking
        self.in_position = False
        self.position_direction: Direction | None = None
        self.entry_price = _ZERO
        self.stop_loss = _ZERO
        self.take_profit = _ZERO
        self.entry_mode = ""
        self.entry_bar = 0
        self.orig_sl = _ZERO
        self.risk_amount = _ZERO
        self.be_triggered = False
        self.trailing_vwap = False
        self.trailing_atr = False
        self.trail_level = _ZERO
        self.orb_extreme = _ZERO
        self.is_pb_trade = False
        self.is_orb_trade = False
        self.pending_tp = _ZERO
