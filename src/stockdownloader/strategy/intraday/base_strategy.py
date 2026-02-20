"""Base class for composition-based intraday strategies.

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
from typing import TYPE_CHECKING

from stockdownloader.model.trade import IntradaySignal
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy

if TYPE_CHECKING:
    from stockdownloader.model.price_data import IntradayPriceData
    from stockdownloader.strategy.intraday.session_state import BarContext
    from stockdownloader.strategy.intraday.infra import IntradayInfra


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
