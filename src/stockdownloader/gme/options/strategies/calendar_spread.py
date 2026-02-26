"""GME Calendar Spread Strategy implementation.

A calendar spread exploits term-structure mis-pricing by selling a
near-term option and buying a further-dated option at the same strike.

Entry is gated on *IV term-structure slope*.  When the slope is negative
(backwardation -- near-term IV exceeds far-term IV), the strategy sells
the near-term ATM call and buys the far-term ATM call.  In contango
(slope >= 0) no trades are generated, since the edge is insufficient.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import GMEOptionsStrategy, Trade


class CalendarSpreadStrategy(GMEOptionsStrategy):
    """Calendar spread strategy triggered by IV backwardation.

    Sells the near-term ATM call and buys the far-term ATM call when
    ``iv_term_slope < 0``.
    """

    # ------------------------------------------------------------------
    # GMEOptionsStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable strategy name."""
        return "calendar_spread"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame | None,
    ) -> list[Trade]:
        """Open a calendar spread when IV term slope is negative.

        Sells near-term ATM call + buys far-term ATM call.  Skips if
        ``iv_term_slope >= 0`` (contango).
        """
        if chain is None or chain.empty:
            return []

        iv_term_slope = state.get("iv_term_slope", 0.0)
        if iv_term_slope >= 0:
            return []

        # Determine ATM strike
        all_strikes = sorted(chain["strike"].unique())
        if not all_strikes:
            return []

        atm = all_strikes[len(all_strikes) // 2]

        # Filter to ATM calls
        atm_calls = chain[
            (chain["option_type"] == "call") & (chain["strike"] == atm)
        ].copy()
        if atm_calls.empty:
            return []

        # Sort by expiration to identify near-term and far-term
        atm_calls = atm_calls.sort_values("expiration")

        if len(atm_calls) < 2:
            # Only one expiration -- cannot form a calendar spread
            return []

        near_row = atm_calls.iloc[0]
        far_row = atm_calls.iloc[-1]

        return [
            Trade(
                option_ticker=near_row["option_ticker"],
                direction="sell",
                option_type="call",
                strike=atm,
                expiration=near_row["expiration"],
                premium=near_row["close"],
                contracts=1,
            ),
            Trade(
                option_ticker=far_row["option_ticker"],
                direction="buy",
                option_type="call",
                strike=atm,
                expiration=far_row["expiration"],
                premium=far_row["close"],
                contracts=1,
            ),
        ]

    def on_expiry(
        self,
        trade_date: date,
        positions: list,
    ) -> list[Trade]:
        """Handle expiring positions (no-op for calendar spread)."""
        return []
