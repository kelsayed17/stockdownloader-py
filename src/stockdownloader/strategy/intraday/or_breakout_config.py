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
    orb_window: int = 30                            # 30-min OR (research-backed)
    orb_rvol: Decimal = Decimal("2.0")             # Higher volume requirement
    orb_sl_mode: str = "OR Midpoint"               # Tighter SL using OR midpoint
    orb_sl_atr: Decimal = Decimal("1.5")
    orb_sl_cap: Decimal = Decimal("2.00")          # Tighter cap
    orb_vwap_align: bool = True
    orb_body_min: Decimal = Decimal("0.25")        # Stronger breakout candle
    orb_entry_mode: str = "aggressive"
    orb_retest_bars: int = 5
    orb_tp_mode: str = "2x_or_range"               # Larger TP target
    orb_gap_filter: bool = True
    orb_adx_filter: bool = True
    orb_nr7_filter: bool = False
    orb_htf_align: bool = True                     # Require HTF alignment

    # ── Overrides (base provides allow_longs, allow_shorts, w_sr) ────────
    adx_thresh: Decimal = Decimal("25")            # Higher for trend-following (base: 22)
