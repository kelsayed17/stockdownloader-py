"""Configuration for the standalone VWAP Reversal strategy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(frozen=True, slots=True)
class ReversalStrategyConfig(InfraExitConfig):
    """All configurable parameters for the VWAP Reversal strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.
    """

    # ── REV entry ───────────────────────────────────────────────────────
    rev_enable: bool = True
    rev_band: str = "2σ"
    rev_body: Decimal = Decimal("0.20")
    rev_sl_atr: Decimal = Decimal("1.0")
    rev_sl_cap: Decimal = Decimal("1.50")
    rev_shorts: bool = False
    rev_min_rr: Decimal = Decimal("0.3")

    # ── REV enhancements ────────────────────────────────────────────────
    rev_can_trade_bar: int = 0
    rev_min_touches: int = 0
    rev_hug_limit: int = 20
    rev_tp_mode: str = "vwap"
    rev_rr: Decimal = Decimal("1.0")
    rev_require_sr: bool = False

    # ── Shared with entry logic ─────────────────────────────────────────
    adx_thresh: Decimal = Decimal("21")
    max_day: int = 2
    spacing: int = 3
    allow_longs: bool = True
    allow_shorts: bool = True

    # ── Confluence ──────────────────────────────────────────────────────
    w_vol: int = 3
    w_sr: int = 2
    w_time: int = 0
    w_rsi: int = 1
    min_score: int = 3
