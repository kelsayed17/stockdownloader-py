"""Shared infrastructure + exit + trail configuration base class.

Every strategy-specific config inherits from :class:`InfraExitConfig`,
which holds the fields consumed by :class:`IntradayInfra`,
:class:`IntradayExitManager`, and trail strategy implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class InfraExitConfig:
    """Fields consumed by IntradayInfra, IntradayExitManager, and trail strategies.

    Every strategy config inherits from this base class.
    """

    # ── InfraConfig ─────────────────────────────────────────────────────
    adx_len: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    slope_period: int = 5
    tod_days: int = 10
    or_bars: int = 3
    ps_atr_pct: Decimal = Decimal("15.0")      # Infra: manipulation detection
    sr_prox: Decimal = Decimal("0.35")
    sr_pdhlc: bool = True
    sr_round: bool = True
    sr_or: bool = True
    sr_week_hl: bool = False
    sr_prev_vwap: bool = False
    bars_per_day: int = 78
    can_trade_bar: int = 11
    eod_bar: int = 78
    lunch_start: int = 25
    lunch_end: int = 54
    pq_max_cross: int = 3
    circuit: int = 3
    day_loss: Decimal = Decimal("3.0")

    # ── ExitConfig (bars_per_day above) ─────────────────────────────────
    be_trigger: Decimal = Decimal("0.5")
    trail_vwap: bool = True
    close_eod: bool = True
    orb_reentry_exit: bool = False
    orb_time_exit: int = 0
    orr_rebreak_exit: bool = False

    # ── TrailConfig ─────────────────────────────────────────────────────
    orb_trail_atr: Decimal = Decimal("1.5")
    trail_buf: Decimal = Decimal("0.15")
    trail_keep_tp: bool = True
