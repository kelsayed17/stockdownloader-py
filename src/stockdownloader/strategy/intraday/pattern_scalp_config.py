"""Configuration for the standalone Pattern Scalp strategy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(frozen=True, slots=True)
class PatternScalpStrategyConfig(InfraExitConfig):
    """All configurable parameters for the Pattern Scalp strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── PS entry ────────────────────────────────────────────────────────
    ps_enable: bool = True
    ps_atr_pct: Decimal = Decimal("20.0")      # 20% of daily ATR → more manipulation days
    ps_window: int = 35                        # Wider window for more setups
    ps_engulf: Decimal = Decimal("0.25")       # 25% body-to-range (research minimum)
    ps_rvol: Decimal = Decimal("1.5")          # Require above-average volume
    ps_sma_filter: bool = True                 # Require SMA alignment
    ps_sl_mode: str = "ATR-Based"
    ps_sl_atr: Decimal = Decimal("1.3")        # Tighter SL
    ps_sl_cap: Decimal = Decimal("1.50")       # Tighter cap
    ps_tp_pct: Decimal = Decimal("75.0")
    ps_min_rr: Decimal = Decimal("1.5")        # Higher R:R to protect against losses
    ps_htf_align: bool = True

    # allow_longs, allow_shorts, w_sr inherited from base
