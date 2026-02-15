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
    orr_window: int = 60
    orr_prox: Decimal = Decimal("0.8")
    orr_sl_atr: Decimal = Decimal("0.5")
    orr_sl_cap: Decimal = Decimal("2.00")
    orr_tp_mode: str = "VWAP"
    orr_min_rr: Decimal = Decimal("0.3")
    orr_max_rr: Decimal = Decimal("3.0")
    orr_rvol: Decimal = Decimal("0.8")
    orr_vwap_disagree: bool = False
    orr_gap_filter: bool = False
    orr_adx_filter: bool = False
    orr_require_break: bool = False

    # ── Shared with entry logic ─────────────────────────────────────────
    ps_engulf: Decimal = Decimal("0.25")       # Pattern detection threshold
    allow_longs: bool = True
    allow_shorts: bool = True
    w_sr: int = 2
    adx_thresh: Decimal = Decimal("21")
