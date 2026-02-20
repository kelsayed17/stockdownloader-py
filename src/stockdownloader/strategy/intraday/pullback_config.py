"""Configuration for the standalone VWAP Pullback strategy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(frozen=True, slots=True)
class PullbackStrategyConfig(InfraExitConfig):
    """All configurable parameters for the VWAP Pullback strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── PB entry ────────────────────────────────────────────────────────
    pb_zone: Decimal = Decimal("0.5")          # Pullback zone (× band width)
    pb_body: Decimal = Decimal("0.20")         # Min candle body (× ATR) — stronger
    rr: Decimal = Decimal("1.8")               # Risk : Reward — higher
    sl_atr: Decimal = Decimal("1.3")           # Stop loss (× ATR) — tighter
    sl_cap: Decimal = Decimal("2.00")          # Stop loss cap ($) — tighter

    # ── Trend ───────────────────────────────────────────────────────────
    trend_bars: int = 7                        # Stronger trend requirement
    # adx_thresh: inherited from base (22)
    htf_align: bool = True
    ar_filter: bool = True
    ar_thresh: Decimal = Decimal("0.9")
    ar_cap: Decimal = Decimal("1.30")
    va_filter: bool = True
    va_min: Decimal = Decimal("-0.1")
    cvd_long_filter: bool = True
    lrs_short_filter: bool = True
    lrs_thresh: Decimal = Decimal("0.08")
    max_vxc: int = 4                           # Fewer VWAP crosses allowed

    # ── Confluence (w_vol, w_sr, w_time, w_rsi, min_score inherited) ─────
    w_pq: int = 1
    w_box: int = 0
    min_score_long: int = 4                    # Same as min_score for more opportunities

    # ── Risk / session (max_day, spacing, allow_longs/shorts inherited) ─
    no_friday_short: bool = True
    no_monday_long: bool = False

    # ── Exit overrides ─────────────────────────────────────────────────
    trail_buf: Decimal = Decimal("0.10")       # Tighter trail (base: 0.15)

    # ── PB enhancements ─────────────────────────────────────────────────
    pb_vwap_bias: bool = True                  # Require VWAP bias alignment
    pb_vwap_bias_pct: Decimal = Decimal("0.6")
    pb_require_sr: bool = False
    pb_tp_mode: str = "rr"
