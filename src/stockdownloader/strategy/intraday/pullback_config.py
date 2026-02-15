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
    pb_body: Decimal = Decimal("0.15")         # Min candle body (× ATR)
    rr: Decimal = Decimal("1.4")               # Risk : Reward
    sl_atr: Decimal = Decimal("1.3")           # Stop loss (× ATR)
    sl_cap: Decimal = Decimal("1.50")          # Stop loss cap ($)

    # ── Trend ───────────────────────────────────────────────────────────
    trend_bars: int = 3
    adx_thresh: Decimal = Decimal("21")
    htf_align: bool = True
    ar_filter: bool = True
    ar_thresh: Decimal = Decimal("0.9")
    ar_cap: Decimal = Decimal("1.15")
    va_filter: bool = True
    va_min: Decimal = Decimal("-0.1")
    cvd_long_filter: bool = True
    lrs_short_filter: bool = True
    lrs_thresh: Decimal = Decimal("0.08")
    max_vxc: int = 6

    # ── Confluence ──────────────────────────────────────────────────────
    w_vol: int = 3
    w_sr: int = 2
    w_time: int = 0
    w_pq: int = 1
    w_rsi: int = 1
    w_box: int = 0
    min_score: int = 3
    min_score_long: int = 4

    # ── Risk / session ──────────────────────────────────────────────────
    max_day: int = 2
    spacing: int = 3
    allow_longs: bool = True
    allow_shorts: bool = True
    no_friday_short: bool = True
    no_monday_long: bool = True

    # ── PB enhancements ─────────────────────────────────────────────────
    pb_vwap_bias: bool = False
    pb_vwap_bias_pct: Decimal = Decimal("0.7")
    pb_require_sr: bool = False
    pb_tp_mode: str = "rr"
