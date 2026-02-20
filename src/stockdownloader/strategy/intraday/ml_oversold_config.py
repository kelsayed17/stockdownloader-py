"""Configuration for the ML Oversold Reversion strategy.

The strategy is ML-driven: a trained gradient boosting model decides
when to enter.  This config controls confidence gating, SL/TP risk
management, and session limits — **not** indicator thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(frozen=True, slots=True)
class MLOversoldConfig(InfraExitConfig):
    """Config for :class:`MLOversoldStrategy`.

    The ML model decides *when* to trade; these fields control
    *how much* to risk and basic session limits.
    """

    # ── ML model ──────────────────────────────────────────────────────
    model_path: str = "output/models/spy/gradient_boosting_latest.joblib"

    # Probability threshold — model must be at least this confident.
    ml_threshold: Decimal = Decimal("0.60")

    # ── SL / TP (standard risk management) ────────────────────────────
    ml_sl_atr: Decimal = Decimal("1.5")
    ml_sl_cap: Decimal = Decimal("2.50")
    ml_tp_mode: str = "vwap"          # "vwap" or "rr"
    ml_rr: Decimal = Decimal("1.5")   # R:R multiplier (rr mode)
    ml_min_rr: Decimal = Decimal("0.5")

    # ── Session limits ────────────────────────────────────────────────
    # Long-only: SPY mean-reversion bias.
    allow_shorts: bool = False
    max_day: int = 2
    spacing: int = 8
