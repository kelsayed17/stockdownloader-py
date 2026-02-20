"""Configuration for the AVWAP Pullback strategy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(frozen=True, slots=True)
class AVWAPPullbackConfig(InfraExitConfig):
    """All configurable parameters for the AVWAP Pullback strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── AVWAP infrastructure ─────────────────────────────────────────
    use_avwap: bool = True
    avwap_anchor_type: str = "fomc"

    # ── Entry conditions ─────────────────────────────────────────────
    avwap_zone_atr: Decimal = Decimal("0.8")       # Proximity zone (× ATR) — tighter
    avwap_min_days: int = 3                        # Min days since anchor
    avwap_max_days: int = 30                       # Max days (staleness) — fresher anchors only
    avwap_body_min: Decimal = Decimal("0.20")      # Min candle body (× ATR) — stronger confirmation
    avwap_session_vwap_agree: bool = True          # Session VWAP must agree

    # ── Direction (allow_longs, allow_shorts inherited) ──────────────
    avwap_longs: bool = True
    avwap_shorts: bool = False                     # SPY long-only bias
    avwap_trend_bars: int = 7                      # Min EMA trend bars — stronger trend
    avwap_htf_align: bool = True                   # HTF EMA agreement

    # ── SL / TP ──────────────────────────────────────────────────────
    avwap_sl_atr: Decimal = Decimal("1.3")
    avwap_sl_cap: Decimal = Decimal("2.00")        # Tighter SL cap
    avwap_rr: Decimal = Decimal("2.0")             # Higher R:R
    avwap_tp_mode: str = "rr"                      # "rr" | "session_vwap" | "avwap_band"

    # ── Overrides (max_day, w_vol, w_sr, w_rsi, w_time, min_score inherited) ─
    spacing: int = 10                              # Wider spacing (base: 5)

    # ── S/R scoring includes AVWAP ───────────────────────────────────
    sr_avwap: bool = True
