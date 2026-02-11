"""Configuration for the VWAP v11.2 intraday strategy.

Every tunable parameter from the PineScript inputs is represented here
as a frozen dataclass field with the v11.2 default value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class VwapStrategyConfig:
    """All configurable parameters for VWAP v11.2."""

    # ── Pullback (PB) ────────────────────────────────────────────────────
    pb_zone: Decimal = Decimal("0.5")          # Pullback zone (× band width)
    pb_body: Decimal = Decimal("0.15")         # Min candle body (× ATR)
    rr: Decimal = Decimal("1.4")               # Risk : Reward
    sl_atr: Decimal = Decimal("1.3")           # Stop loss (× ATR)
    sl_cap: Decimal = Decimal("1.50")          # Stop loss cap ($)

    # ── Reversal (REV) ───────────────────────────────────────────────────
    rev_enable: bool = True
    rev_band: str = "2σ"                       # "1.5σ", "2σ", "3σ"
    rev_body: Decimal = Decimal("0.20")        # Min reversal body (× ATR)
    rev_sl_atr: Decimal = Decimal("1.0")       # Reversal SL (× ATR)
    rev_sl_cap: Decimal = Decimal("1.50")      # Reversal SL cap ($)
    rev_shorts: bool = False                   # Allow reversal shorts
    rev_min_rr: Decimal = Decimal("0.3")       # Min reversal R:R

    # ── Pattern Scalp (PS) ───────────────────────────────────────────────
    ps_enable: bool = True
    ps_atr_pct: Decimal = Decimal("30.0")      # Min OR range (% of daily ATR)
    ps_window: int = 12                        # PS window (last bar)
    ps_sl_mode: str = "Day Extreme"            # "Day Extreme" or "ATR-Based"
    ps_sl_atr: Decimal = Decimal("1.5")        # PS SL (× ATR)
    ps_sl_cap: Decimal = Decimal("2.50")       # PS SL cap ($)
    ps_tp_pct: Decimal = Decimal("75.0")       # PS TP (% of OR range)
    ps_rvol: Decimal = Decimal("1.0")          # PS min RVOL
    ps_sma_filter: bool = False                # Require SMA bias
    ps_sma_len: int = 50                       # Daily SMA length
    ps_engulf: Decimal = Decimal("0.35")       # Engulf retrace min

    # ── OR Reversal (ORR) ────────────────────────────────────────────────
    orr_enable: bool = False                   # Disabled by default in v11.2
    orr_window: int = 25                       # ORR window (last bar)
    orr_prox: Decimal = Decimal("0.2")         # ORR proximity (× ATR)
    orr_sl_atr: Decimal = Decimal("0.5")       # ORR SL buffer (× ATR)
    orr_sl_cap: Decimal = Decimal("2.00")      # ORR SL cap ($)
    orr_tp_mode: str = "VWAP"                  # "VWAP" or "OR Mid"
    orr_min_rr: Decimal = Decimal("0.5")       # ORR min R:R
    orr_max_rr: Decimal = Decimal("3.0")       # ORR max R:R
    orr_rvol: Decimal = Decimal("1.0")         # ORR min RVOL

    # ── OR Breakout (ORB) ────────────────────────────────────────────────
    orb_enable: bool = True
    orb_window: int = 20                       # ORB window (last bar)
    orb_rvol: Decimal = Decimal("2.0")         # ORB min RVOL
    orb_sl_mode: str = "OR Opposite"           # "OR Opposite" or "ATR-Based"
    orb_sl_atr: Decimal = Decimal("1.5")       # ORB SL (× ATR)
    orb_sl_cap: Decimal = Decimal("2.50")      # ORB SL cap ($)
    orb_vwap_align: bool = True                # Require VWAP alignment
    orb_body_min: Decimal = Decimal("0.2")     # ORB min body (× ATR)
    orb_trail_atr: Decimal = Decimal("1.5")    # ORB trail (× ATR)

    # ── Trend ────────────────────────────────────────────────────────────
    ema_fast: int = 9
    ema_slow: int = 21
    slope_period: int = 5                      # VWAP slope bars
    trend_bars: int = 3                        # Min trend bars
    adx_len: int = 14
    adx_thresh: Decimal = Decimal("21")
    htf_align: bool = True                     # Require 15m trend alignment
    ar_filter: bool = True                     # ATR ratio filter
    ar_thresh: Decimal = Decimal("0.9")        # AR min threshold
    ar_cap: Decimal = Decimal("1.15")          # AR max threshold
    va_filter: bool = True                     # VWAP acceleration filter
    va_min: Decimal = Decimal("-0.1")          # VA min threshold
    cvd_long_filter: bool = True               # CVD long filter
    lrs_short_filter: bool = True              # LRS short alignment
    lrs_thresh: Decimal = Decimal("0.08")      # LRS block threshold

    # ── Exit Management ──────────────────────────────────────────────────
    be_trigger: Decimal = Decimal("0.5")       # BE trigger (× risk)
    trail_vwap: bool = True                    # VWAP trailing stop (PB)
    trail_buf: Decimal = Decimal("0.15")       # Trail buffer (× ATR)
    trail_keep_tp: bool = True                 # Keep fixed TP with trail

    # ── Confluence ───────────────────────────────────────────────────────
    min_score: int = 3                         # Min score (PB short)
    min_score_long: int = 5                    # PB long min score
    w_vol: int = 3                             # Volume weight (inverted)
    tod_days: int = 10                         # TOD RVOL days
    w_sr: int = 2                              # S/R weight
    w_rsi: int = 1                             # RSI weight
    w_time: int = 0                            # Time weight
    w_pq: int = 1                              # PB quality weight
    w_box: int = 0                             # Box position weight
    pq_max_cross: int = 3                      # PQ max crosses
    max_vxc: int = 6                           # Max VWAP crosses

    # ── Risk ─────────────────────────────────────────────────────────────
    risk_pct: Decimal = Decimal("1.0")         # Risk/trade (%)
    fixed_cap: bool = True                     # Fixed-capital sizing
    initial_capital: Decimal = Decimal("100000")
    max_day: int = 2                           # Max trades/day
    spacing: int = 3                           # Min bars between
    circuit: int = 3                           # Circuit breaker
    day_loss: Decimal = Decimal("3.0")         # Daily loss limit (%)
    close_eod: bool = True
    allow_longs: bool = True
    allow_shorts: bool = True
    no_friday_short: bool = True               # Block Friday PB shorts
    no_monday_long: bool = True                # Block Monday PB longs

    # ── S/R Levels ───────────────────────────────────────────────────────
    sr_prox: Decimal = Decimal("0.35")         # S/R proximity (%)
    sr_pdhlc: bool = True                      # Prev day H/L/C
    sr_round: bool = True                      # Round $5 levels
    sr_or: bool = True                         # Opening range
    or_bars: int = 3                           # OR bars (first N bars)
    sr_week_hl: bool = False                   # Prev week H/L
    sr_prev_vwap: bool = False                 # Prev session VWAP close

    # ── Session ──────────────────────────────────────────────────────────
    bars_per_day: int = 78                     # 5m bars per RTH session
    can_trade_bar: int = 11                    # First bar PB can trade
    lunch_start: int = 25                      # Lunch lull start bar
    lunch_end: int = 54                        # Lunch lull end bar
    eod_bar: int = 78                          # Last bar of session
