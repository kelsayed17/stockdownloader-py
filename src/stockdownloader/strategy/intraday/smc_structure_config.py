"""Configuration for the SMC Structure strategy.

All fields are frozen (immutable after construction). Override
defaults via ``dataclasses.replace()`` or keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(frozen=True, slots=True)
class SMCStructureConfig(InfraExitConfig):
    """SMC Structure strategy configuration.

    Inherits all infrastructure / exit / trail fields from
    :class:`InfraExitConfig` and adds SMC-specific parameters.
    """

    # ── Infrastructure flag ──────────────────────────────────────────
    use_smc: bool = True

    # ── Structure detection ──────────────────────────────────────────
    smc_swing_lookback: int = 3  # Faster swing confirmation for intraday
    smc_min_impulse_atr: Decimal = Decimal("2.5")  # Moderate impulse threshold
    smc_zone_bars: int = 2  # Candles at impulse origin = zone

    # ── Entry conditions ─────────────────────────────────────────────
    smc_require_bos: bool = True  # Require BoS before entry
    smc_zone_atr: Decimal = Decimal("0.8")  # Proximity to zone (× ATR)
    smc_body_min: Decimal = Decimal("0.10")  # Lower body min for more setups
    smc_longs: bool = True
    smc_shorts: bool = False  # SPY long-only bias
    smc_sweep_entry: bool = True  # Enable sweep entries for more trades
    smc_sweep_tolerance: Decimal = Decimal("0.3")  # Sweep wick limit (× ATR)

    # ── Filters ──────────────────────────────────────────────────────
    smc_htf_align: bool = True  # HTF trend must agree
    smc_vwap_agree: bool = False  # Disabled — HTF alignment sufficient
    smc_min_age: int = 10  # Min bars since BoS for pullback
    smc_max_age: int = 60  # Max bars since BoS (staleness)

    # ── SL / TP ──────────────────────────────────────────────────────
    smc_sl_mode: str = "atr"  # ATR-based SL for consistency
    smc_sl_atr: Decimal = Decimal("2.0")  # Wider SL to avoid whipsaws
    smc_sl_cap: Decimal = Decimal("1.50")  # Tighter hard cap for risk control
    smc_rr: Decimal = Decimal("1.0")  # 1:1 R:R — high WR compensates
    smc_tp_mode: str = "rr"  # "rr" | "swing" | "vwap"

    # ── Overrides (w_vol, w_sr, w_rsi, w_time inherited) ────────────
    max_day: int = 2  # Allow 2 trades per day (base: 1)
    spacing: int = 10  # Wider spacing (base: 5)
    min_score: int = 3  # Lower min score for more setups (base: 4)
