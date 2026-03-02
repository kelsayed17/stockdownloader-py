"""GME Strangle Strategy implementation.

A short strangle sells an OTM put and an OTM call simultaneously,
collecting premium on both sides.  This strategy benefits from
time decay when the underlying stays within the range defined by the
two short strikes.

Entry is gated on IV percentile -- the strategy only enters when
implied volatility is elevated (above a configurable percentile
threshold), since higher IV means richer premiums for the seller.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import GMEOptionsStrategy, Trade


class StrangleStrategy(GMEOptionsStrategy):
    """Short strangle strategy with IV-based entry filter.

    Parameters
    ----------
    iv_min_percentile:
        Minimum IV percentile required to enter a position.  The
        strategy skips trading when IV is below this threshold.
    """

    def __init__(self, iv_min_percentile: float = 0.50) -> None:
        self._iv_min_percentile = iv_min_percentile

    # ------------------------------------------------------------------
    # GMEOptionsStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable strategy name."""
        return "strangle"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame | None,
    ) -> list[Trade]:
        """Sell an OTM put and OTM call when IV is elevated.

        Only enters when ``iv_percentile`` exceeds
        ``iv_min_percentile``.  Returns two trades (sell put + sell
        call) or an empty list.
        """
        if chain is None or chain.empty:
            return []

        iv_percentile = state.get("iv_percentile", 0.0)
        if iv_percentile <= self._iv_min_percentile:
            return []

        all_strikes = sorted(chain["strike"].unique())
        if len(all_strikes) < 3:
            return []

        atm = all_strikes[len(all_strikes) // 2]

        # Select first OTM strikes on each side
        put_strikes = [s for s in all_strikes if s < atm]
        call_strikes = [s for s in all_strikes if s > atm]
        if not put_strikes or not call_strikes:
            return []

        short_put_strike = put_strikes[-1]   # closest OTM put
        short_call_strike = call_strikes[0]  # closest OTM call

        trades: list[Trade] = []

        # Sell OTM put
        put_trade = self._make_trade(chain, short_put_strike, "put", "sell")
        if put_trade is not None:
            trades.append(put_trade)

        # Sell OTM call
        call_trade = self._make_trade(chain, short_call_strike, "call", "sell")
        if call_trade is not None:
            trades.append(call_trade)

        return trades

    def on_expiry(
        self,
        trade_date: date,
        positions: list,
    ) -> list[Trade]:
        """Handle expiring positions (no-op for strangle)."""
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
