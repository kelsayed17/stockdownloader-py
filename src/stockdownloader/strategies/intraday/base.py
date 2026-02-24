"""Base class and shared configuration for composition-based intraday strategies.

Provides:

- :class:`InfraExitConfig` — frozen dataclass holding infrastructure, exit,
  trail, and entry/risk configuration fields shared by all intraday strategies
- :class:`BaseIntradayStrategy` — template base with boilerplate delegation

All standalone intraday strategies share the same structural pattern:

1. Store a config and an :class:`IntradayInfra` instance.
2. Delegate ``warmup_period``, ``on_session_start``,
   ``on_position_opened``, ``on_position_closed``, and ``evaluate``
   to the infra instance.
3. Provide a unique ``_evaluate_entry`` method containing the
   strategy-specific entry logic.

:class:`BaseIntradayStrategy` captures this pattern, reducing each
concrete strategy to:

- ``__init__`` — set ``self._c`` and ``self._infra`` (3-5 lines)
- ``name`` property — return display name (1 line)
- ``_evaluate_entry(ctx)`` — unique entry logic (40-200 lines)
- Optionally: ``pinescript_mode()``, helper methods
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.base import IntradayTradingStrategy

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.intraday.session import BarContext
    from stockdownloader.strategies.intraday.infra import IntradayInfra


# ── InfraExitConfig ─────────────────────────────────────────────────────


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


# ── BaseIntradayStrategy ────────────────────────────────────────────────


class BaseIntradayStrategy(IntradayTradingStrategy):
    """Template base for composition-based intraday strategies.

    Subclasses **must** implement:

    - :meth:`__init__` — create ``self._c`` (config) and ``self._infra``
      (:class:`IntradayInfra`) then call ``super().__init__()``.
    - ``name`` property — return the strategy display name.
    - :meth:`_evaluate_entry` — strategy-specific entry logic receiving
      a :class:`BarContext` and returning an optional signal.

    Subclasses **may** override:

    - ``_ENTRY_FLAGS`` — set ``{"fire_once": True}`` for single-entry
      strategies.
    - ``pinescript_mode()`` — PineScript mode definition.
    - Additional private helper methods for entry logic.

    All boilerplate delegation methods (``warmup_period``,
    ``on_session_start``, ``on_position_opened``, ``on_position_closed``,
    ``evaluate``) are provided here and should **not** be overridden.
    """

    # Subclasses that restrict to one entry per session set this.
    _ENTRY_FLAGS: dict[str, bool] = {}

    # Subclass __init__ must set these before calling super().__init__()
    _c: object  # strategy config (typed in subclass)
    _infra: IntradayInfra

    # ------------------------------------------------------------------
    # Boilerplate delegates — identical across all 7 strategies
    # ------------------------------------------------------------------

    @property
    def warmup_period(self) -> int:
        return self._infra.warmup_period

    def on_session_start(self, trading_date: str) -> None:
        self._infra.on_session_start(trading_date)

    def on_position_opened(self, is_long: bool) -> None:
        self._infra.confirm_position_opened(is_long)

    def on_position_closed(self) -> None:
        self._infra.confirm_position_closed()

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        return self._infra.run_bar(
            data, current_index, self._evaluate_entry, self._ENTRY_FLAGS
        )

    # ------------------------------------------------------------------
    # Subclass hook — unique per strategy
    # ------------------------------------------------------------------

    @abstractmethod
    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        """Evaluate entry conditions for the current bar.

        Called by :meth:`evaluate` (via ``IntradayInfra.run_bar``) only
        when all session-level guards (warmup, risk limits, spacing,
        position checks) have already passed.

        Parameters
        ----------
        ctx:
            Pre-computed bar context with indicators, VWAP bands,
            session state, and optional AVWAP / SMC structure data.

        Returns
        -------
        IntradaySignal or None
            An entry signal if conditions are met, otherwise ``None``
            to signal HOLD.
        """
