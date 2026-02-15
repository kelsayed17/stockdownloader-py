"""Configuration for the standalone OR Breakout strategy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(frozen=True, slots=True)
class ORBreakoutStrategyConfig(InfraExitConfig):
    """All configurable parameters for the OR Breakout strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── ORB entry ───────────────────────────────────────────────────────
    orb_enable: bool = True
    orb_window: int = 60
    orb_rvol: Decimal = Decimal("0.8")
    orb_sl_mode: str = "OR Opposite"
    orb_sl_atr: Decimal = Decimal("1.5")
    orb_sl_cap: Decimal = Decimal("2.50")
    orb_vwap_align: bool = False
    orb_body_min: Decimal = Decimal("0.10")
    orb_entry_mode: str = "aggressive"
    orb_retest_bars: int = 5
    orb_tp_mode: str = "trail_only"
    orb_gap_filter: bool = False
    orb_adx_filter: bool = False

    # ── Shared with entry logic ─────────────────────────────────────────
    allow_longs: bool = True
    allow_shorts: bool = True
    w_sr: int = 2
    adx_thresh: Decimal = Decimal("21")
