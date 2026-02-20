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
    rev_band: str = "1σ"                           # Narrower band — more entries
    rev_body: Decimal = Decimal("0.15")            # Lower body for more setups
    rev_sl_atr: Decimal = Decimal("1.0")           # Tighter SL
    rev_sl_cap: Decimal = Decimal("1.50")
    rev_shorts: bool = False                       # SPY long-only bias
    rev_min_rr: Decimal = Decimal("1.0")           # R:R floor

    # ── REV enhancements ────────────────────────────────────────────────
    rev_can_trade_bar: int = 8                     # Skip first 40 min volatility
    rev_min_touches: int = 1                       # Require at least 1 prior band touch
    rev_hug_limit: int = 30                        # Wider hugging window
    rev_tp_mode: str = "vwap"                      # Target VWAP for higher WR
    rev_rr: Decimal = Decimal("1.0")
    rev_require_sr: bool = True                    # Require S/R confluence
    rev_vwap_flat_tol: Decimal = Decimal("0.10")   # Tighter VWAP flatness

    # ── Overrides (base provides allow_longs, allow_shorts, adx_thresh) ──
    spacing: int = 10                              # Wider spacing (base: 5)

    # ── Confluence (w_vol, w_sr, w_time inherited) ────────────────────
    w_rsi: int = 2                                 # Higher RSI weight (base: 1)
    min_score: int = 5                             # Maximum selectivity (base: 4)
