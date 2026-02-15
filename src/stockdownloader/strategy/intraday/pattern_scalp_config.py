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
    ps_window: int = 50
    ps_engulf: Decimal = Decimal("0.15")
    ps_rvol: Decimal = Decimal("0.8")
    ps_sma_filter: bool = False
    ps_sl_mode: str = "Day Extreme"
    ps_sl_atr: Decimal = Decimal("1.5")
    ps_sl_cap: Decimal = Decimal("2.50")
    ps_tp_pct: Decimal = Decimal("75.0")

    # ── Shared with entry logic ─────────────────────────────────────────
    allow_longs: bool = True
    allow_shorts: bool = True
    w_sr: int = 2
