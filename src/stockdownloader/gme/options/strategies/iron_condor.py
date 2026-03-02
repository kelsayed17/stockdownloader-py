"""GME Iron Condor Strategy implementation.

An iron condor sells a put spread and a call spread simultaneously,
profiting when the underlying stays within a range.  The four legs are:

1. Sell OTM put  (inner put)
2. Buy further-OTM put  (outer put / wing)
3. Sell OTM call  (inner call)
4. Buy further-OTM call  (outer call / wing)

When GEX concentration exceeds a configurable threshold the wings are
widened by a multiplier, reducing gamma risk in dealer-hedging-heavy
environments.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import GMEOptionsStrategy, Trade


class IronCondorStrategy(GMEOptionsStrategy):
    """Iron condor strategy with GEX-aware wing widening.

    Parameters
    ----------
    wing_width:
        Base distance (in strike units) from the short strike to the
        long (protective) wing strike.
    gex_widen_threshold:
        GEX concentration level above which wings are widened.
    gex_widen_mult:
        Multiplier applied to ``wing_width`` when GEX concentration
        exceeds the threshold.
    """

    def __init__(
        self,
        wing_width: float = 2.5,
        gex_widen_threshold: float = 0.6,
        gex_widen_mult: float = 1.5,
    ) -> None:
        self._wing_width = wing_width
        self._gex_widen_threshold = gex_widen_threshold
        self._gex_widen_mult = gex_widen_mult

    # ------------------------------------------------------------------
    # GMEOptionsStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable strategy name."""
        return "iron_condor"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame | None,
    ) -> list[Trade]:
        """Open a 4-leg iron condor position.

        Widens wings when ``gex_concentration`` exceeds the configured
        threshold.
        """
        if chain is None or chain.empty:
            return []

        gex_concentration = state.get("gex_concentration", 0.0)
        effective_width = self._wing_width
        if gex_concentration > self._gex_widen_threshold:
            effective_width = self._wing_width * self._gex_widen_mult

        all_strikes = sorted(chain["strike"].unique())
        if len(all_strikes) < 4:
            return []

        atm = all_strikes[len(all_strikes) // 2]

        # Find short strikes (first OTM on each side)
        put_strikes = [s for s in all_strikes if s < atm]
        call_strikes = [s for s in all_strikes if s > atm]
        if not put_strikes or not call_strikes:
            return []

        short_put_strike = put_strikes[-1]   # closest OTM put
        short_call_strike = call_strikes[0]  # closest OTM call

        # Find long (wing) strikes by applying effective_width
        long_put_strike = self._find_nearest_strike(
            all_strikes, short_put_strike - effective_width
        )
        long_call_strike = self._find_nearest_strike(
            all_strikes, short_call_strike + effective_width
        )

        # Ensure wings are further out than short strikes
        if long_put_strike >= short_put_strike:
            long_put_strike = all_strikes[0]
        if long_call_strike <= short_call_strike:
            long_call_strike = all_strikes[-1]

        trades = [
            self._make_trade(chain, short_put_strike, "put", "sell"),
            self._make_trade(chain, long_put_strike, "put", "buy"),
            self._make_trade(chain, short_call_strike, "call", "sell"),
            self._make_trade(chain, long_call_strike, "call", "buy"),
        ]

        # Filter out any None results from missing chain rows
        return [t for t in trades if t is not None]

    def on_expiry(
        self,
        trade_date: date,
        positions: list,
    ) -> list[Trade]:
        """Handle expiring positions (no-op for iron condor)."""
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_nearest_strike(
        strikes: list[float], target: float
    ) -> float:
        """Return the strike from *strikes* nearest to *target*."""
        return min(strikes, key=lambda s: abs(s - target))

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
