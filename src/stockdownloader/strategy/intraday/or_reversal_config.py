"""Configuration for the standalone OR Reversal strategy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(frozen=True, slots=True)
class ORReversalStrategyConfig(InfraExitConfig):
    """All configurable parameters for the OR Reversal strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── ORR entry ───────────────────────────────────────────────────────
    orr_enable: bool = True
    orr_window: int = 78
    orr_prox: Decimal = Decimal("0.6")
    orr_sl_atr: Decimal = Decimal("0.5")
    orr_sl_cap: Decimal = Decimal("2.00")
    orr_tp_mode: str = "OR Mid"
    orr_min_rr: Decimal = Decimal("0.5")
    orr_max_rr: Decimal = Decimal("3.0")
    orr_rvol: Decimal = Decimal("1.0")
    orr_vwap_disagree: bool = False
    orr_gap_filter: bool = True
    orr_adx_filter: bool = True
    orr_adx_max: Decimal = Decimal("30")   # Block ORR when ADX >= this (strongly trending)
    orr_require_break: bool = True

    # ── Overrides (base provides allow_longs, w_sr, be_trigger) ──────────
    ps_engulf: Decimal = Decimal("0.30")       # 30% body-to-range for quality patterns
    allow_shorts: bool = True                  # ORR trades both sides (base: False)
    adx_thresh: Decimal = Decimal("21")        # Lower for mean-reversion (base: 22)
