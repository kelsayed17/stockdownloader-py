"""Configuration for the pattern discovery strategy."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stockdownloader.strategy.intraday.base_config import InfraExitConfig


@dataclass(frozen=True, slots=True)
class PatternDiscoveryConfig(InfraExitConfig):
    """Configuration for the pattern discovery strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.  Pattern-specific fields control SL/TP
    sizing from historical MAE/MFE.
    """

    # ── SL/TP from pattern statistics ─────────────────────────────────
    sl_mae_multiplier: Decimal = Decimal("1.2")
    """SL = avg_mae × multiplier (buffer above historical worst case)."""

    tp_mfe_multiplier: Decimal = Decimal("0.8")
    """TP = avg_mfe × multiplier (conservative — take profit early)."""

    sl_cap: Decimal = Decimal("2.50")
    """Absolute stop-loss cap in dollars."""

    min_rr: Decimal = Decimal("1.0")
    """Minimum risk:reward ratio to enter."""

    # ── Entry filters ─────────────────────────────────────────────────
    max_day: int = 2
    """Maximum trades per day."""

    spacing: int = 3
    """Minimum bars between entries."""

    use_confirmation: bool = True
    """Check indicator confirmation conditions before entry."""

    require_htf_alignment: bool = False
    """When ``True``, reject long patterns when HTF trend is down and
    short patterns when HTF trend is up."""
