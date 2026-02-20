"""Shared infrastructure + exit + trail + entry/risk configuration base class.

Every strategy-specific config inherits from :class:`InfraExitConfig`,
which holds:

- **Infrastructure fields** consumed by :class:`IntradayInfra`
- **Exit fields** consumed by :class:`IntradayExitManager`
- **Trail fields** consumed by trail strategy implementations
- **Entry/risk fields** shared across most strategy configs (direction
  controls, ADX threshold, session limits, confluence weights)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class InfraExitConfig:
    """Base configuration for all intraday strategies.

    Every strategy config inherits from this class. Subclasses override
    specific fields to customize behaviour — fields only need to be
    redeclared when their default differs from the base value.
    """

    # ── InfraConfig ─────────────────────────────────────────────────────
    adx_len: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    slope_period: int = 5
    tod_days: int = 10
    or_bars: int = 3
    ps_atr_pct: Decimal = Decimal("30.0")      # Infra: manipulation detection
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
    be_trigger: Decimal = Decimal("0.7")
    trail_vwap: bool = True
    close_eod: bool = True
    orb_reentry_exit: bool = False
    orb_time_exit: int = 0
    orr_rebreak_exit: bool = False

    # ── TrailConfig ─────────────────────────────────────────────────────
    orb_trail_atr: Decimal = Decimal("0.8")
    trail_buf: Decimal = Decimal("0.15")
    trail_keep_tp: bool = True

    # ── Shared entry / risk ─────────────────────────────────────────────
    # Direction controls — overridden per strategy as needed.
    allow_longs: bool = True
    allow_shorts: bool = False                   # SPY long-only bias default

    # ADX trending threshold — strategies override for their context
    # (higher for trend-following, lower for mean-reversion).
    adx_thresh: Decimal = Decimal("22")

    # Session risk limits
    max_day: int = 1                             # Max trades per day
    spacing: int = 5                             # Min bars between entries

    # ── Shared confluence weights ───────────────────────────────────────
    # Weights for the confluence scoring system used by strategies that
    # score entry quality (PB, REV, AVWAP, SMC).  Strategies that don't
    # use confluence scoring simply ignore these fields.
    w_sr: int = 2                                # Support/resistance
    w_vol: int = 2                               # Volume confirmation
    w_time: int = 1                              # Time-of-day
    w_rsi: int = 1                               # RSI position
    min_score: int = 4                           # Minimum confluence score
