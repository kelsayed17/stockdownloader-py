"""GME Long Options Strategy implementation.

A directional strategy that buys options outright when conditions
indicate a strong move is likely.  Entry requires:

1. The current market *regime* to be one of the configured
   ``target_regimes`` (default: ``cycle_hot``, ``gamma_ramp``).
2. The absolute ``composite_score`` to exceed ``min_composite``.

When the score is positive the strategy buys calls; when negative it
buys puts.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Sequence

import pandas as pd

from stockdownloader.gme.options.backtester import GMEOptionsStrategy, Trade


class LongOptionsStrategy(GMEOptionsStrategy):
    """Long options strategy gated by regime and composite score.

    Parameters
    ----------
    min_composite:
        Minimum absolute composite score required for entry.
    target_regimes:
        Market regimes that permit entry.
    """

    def __init__(
        self,
        min_composite: float = 1.5,
        target_regimes: Sequence[str] = ("cycle_hot", "gamma_ramp"),
    ) -> None:
        self._min_composite = min_composite
        self._target_regimes = set(target_regimes)

    # ------------------------------------------------------------------
    # GMEOptionsStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable strategy name."""
        return "long_options"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame | None,
    ) -> list[Trade]:
        """Buy calls or puts when regime and score criteria are met.

        Only enters when ``regime`` is in ``target_regimes`` AND
        ``abs(composite_score) >= min_composite``.  Buys calls if
        score > 0, puts if score < 0.
        """
        if chain is None or chain.empty:
            return []

        regime = state.get("regime", "neutral")
        if regime not in self._target_regimes:
            return []

        composite_score = state.get("composite_score", 0.0)
        if abs(composite_score) < self._min_composite:
            return []

        all_strikes = sorted(chain["strike"].unique())
        if not all_strikes:
            return []

        atm = all_strikes[len(all_strikes) // 2]

        if composite_score > 0:
            return self._buy_call(chain, all_strikes, atm)
        return self._buy_put(chain, all_strikes, atm)

    def on_expiry(
        self,
        trade_date: date,
        positions: list,
    ) -> list[Trade]:
        """Handle expiring positions (no-op for long options)."""
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _buy_call(
        self,
        chain: pd.DataFrame,
        all_strikes: list[float],
        atm: float,
    ) -> list[Trade]:
        """Buy an ATM or near-ATM call."""
        trade = self._make_trade(chain, atm, "call", "buy")
        return [trade] if trade is not None else []

    def _buy_put(
        self,
        chain: pd.DataFrame,
        all_strikes: list[float],
        atm: float,
    ) -> list[Trade]:
        """Buy an ATM or near-ATM put."""
        trade = self._make_trade(chain, atm, "put", "buy")
        return [trade] if trade is not None else []

    @staticmethod
    def _make_trade(
        chain: pd.DataFrame,
        strike: float,
        option_type: str,
        direction: str,
    ) -> Trade | None:
        """Build a ``Trade`` from a chain row matching *strike* and *option_type*."""
        mask = (chain["strike"] == strike) & (chain["option_type"] == option_type)
        rows = chain[mask]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return Trade(
            option_ticker=row["option_ticker"],
            direction=direction,
            option_type=option_type,
            strike=strike,
            expiration=row["expiration"],
            premium=row["close"],
            contracts=1,
        )
