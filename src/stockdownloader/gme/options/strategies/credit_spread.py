"""GME Credit Spread Strategy implementation.

A credit spread collects premium by selling a closer-to-the-money option
and buying a further-out-of-the-money option of the same type to cap risk.

Direction is determined by the composite score:

* **Bullish** (``composite_score > 0``) -- bull put spread (sell put +
  buy lower-strike put).
* **Bearish** (``composite_score < 0``) -- bear call spread (sell call +
  buy higher-strike call).

A composite score of exactly zero produces no trades.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import GMEOptionsStrategy, Trade


class CreditSpreadStrategy(GMEOptionsStrategy):
    """Credit spread strategy driven by composite score direction.

    Always produces exactly two trades (sell + buy) when the composite
    score is non-zero.
    """

    # ------------------------------------------------------------------
    # GMEOptionsStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable strategy name."""
        return "credit_spread"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame | None,
    ) -> list[Trade]:
        """Open a credit spread based on composite score direction.

        Bullish (score > 0): bull put spread -- sell OTM put, buy further-OTM put.
        Bearish (score < 0): bear call spread -- sell OTM call, buy further-OTM call.
        """
        if chain is None or chain.empty:
            return []

        composite_score = state.get("composite_score", 0.0)
        if composite_score == 0:
            return []

        all_strikes = sorted(chain["strike"].unique())
        if len(all_strikes) < 3:
            return []

        atm = all_strikes[len(all_strikes) // 2]

        if composite_score > 0:
            return self._bull_put_spread(chain, all_strikes, atm)
        return self._bear_call_spread(chain, all_strikes, atm)

    def on_expiry(
        self,
        trade_date: date,
        positions: list,
    ) -> list[Trade]:
        """Handle expiring positions (no-op for credit spread)."""
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bull_put_spread(
        self,
        chain: pd.DataFrame,
        all_strikes: list[float],
        atm: float,
    ) -> list[Trade]:
        """Sell OTM put + buy further-OTM put (lower strike)."""
        put_strikes = [s for s in all_strikes if s < atm]
        if len(put_strikes) < 2:
            return []

        short_put_strike = put_strikes[-1]   # closest to ATM
        long_put_strike = put_strikes[-2]    # further OTM

        trades = [
            self._make_trade(chain, short_put_strike, "put", "sell"),
            self._make_trade(chain, long_put_strike, "put", "buy"),
        ]
        return [t for t in trades if t is not None]

    def _bear_call_spread(
        self,
        chain: pd.DataFrame,
        all_strikes: list[float],
        atm: float,
    ) -> list[Trade]:
        """Sell OTM call + buy further-OTM call (higher strike)."""
        call_strikes = [s for s in all_strikes if s > atm]
        if len(call_strikes) < 2:
            return []

        short_call_strike = call_strikes[0]   # closest to ATM
        long_call_strike = call_strikes[1]    # further OTM

        trades = [
            self._make_trade(chain, short_call_strike, "call", "sell"),
            self._make_trade(chain, long_call_strike, "call", "buy"),
        ]
        return [t for t in trades if t is not None]

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
