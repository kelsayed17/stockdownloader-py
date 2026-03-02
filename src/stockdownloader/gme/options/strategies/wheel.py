"""GME Wheel Strategy implementation.

The wheel strategy cycles between selling cash-secured puts (CSPs) and
covered calls (CCs).  When *not* holding shares it sells an OTM put.  If
the put is assigned (shares acquired), it switches to selling OTM calls
until the shares are called away.

Safety filters based on T+35 FTD countdown and IV percentile prevent
entry during high-risk periods.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import GMEOptionsStrategy, Trade


class GMEWheelStrategy(GMEOptionsStrategy):
    """Wheel strategy: sell CSPs then CCs on assignment.

    Parameters
    ----------
    t35_min_days:
        Minimum days remaining on the T+35 FTD countdown to allow entry.
        Skips trading when the countdown is below this threshold to avoid
        FTD-driven volatility.
    iv_max_percentile:
        Maximum IV percentile allowed for entry.  Skips trading when
        implied volatility is above this threshold.
    target_delta:
        Target delta for strike selection (approximated via distance from
        ATM).
    """

    def __init__(
        self,
        t35_min_days: int = 5,
        iv_max_percentile: float = 0.80,
        target_delta: float = 0.30,
    ) -> None:
        self._t35_min_days = t35_min_days
        self._iv_max_percentile = iv_max_percentile
        self._target_delta = target_delta
        self._holding_shares = False

    # ------------------------------------------------------------------
    # GMEOptionsStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable strategy name."""
        return "wheel"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame | None,
    ) -> list[Trade]:
        """Evaluate the wheel strategy for a single trading day.

        Safety filters:
        - Skip if ``ftd_t35_countdown`` < ``t35_min_days``.
        - Skip if ``iv_percentile`` > ``iv_max_percentile``.

        When not holding shares, sells an OTM put (cash-secured put).
        When holding shares, sells an OTM call (covered call).
        """
        if chain is None or chain.empty:
            return []

        # Safety filters
        ftd_countdown = state.get("ftd_t35_countdown", 0)
        if ftd_countdown < self._t35_min_days:
            return []

        iv_percentile = state.get("iv_percentile", 0.0)
        if iv_percentile > self._iv_max_percentile:
            return []

        if not self._holding_shares:
            return self._sell_csp(chain)
        return self._sell_cc(chain)

    def on_expiry(
        self,
        trade_date: date,
        positions: list,
    ) -> list[Trade]:
        """Handle expiring positions (no-op for wheel)."""
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sell_csp(self, chain: pd.DataFrame) -> list[Trade]:
        """Select and sell an OTM put (cash-secured put)."""
        puts = chain[chain["option_type"] == "put"].copy()
        if puts.empty:
            return []

        # Approximate ATM as the median strike
        all_strikes = sorted(chain["strike"].unique())
        atm = all_strikes[len(all_strikes) // 2]

        # OTM puts have strike < ATM
        otm_puts = puts[puts["strike"] < atm].sort_values("strike", ascending=False)
        if otm_puts.empty:
            return []

        row = otm_puts.iloc[0]
        return [
            Trade(
                option_ticker=row["option_ticker"],
                direction="sell",
                option_type="put",
                strike=row["strike"],
                expiration=row["expiration"],
                premium=row["close"],
                contracts=1,
            )
        ]

    def _sell_cc(self, chain: pd.DataFrame) -> list[Trade]:
        """Select and sell an OTM call (covered call)."""
        calls = chain[chain["option_type"] == "call"].copy()
        if calls.empty:
            return []

        all_strikes = sorted(chain["strike"].unique())
        atm = all_strikes[len(all_strikes) // 2]

        # OTM calls have strike > ATM
        otm_calls = calls[calls["strike"] > atm].sort_values("strike", ascending=True)
        if otm_calls.empty:
            return []

        row = otm_calls.iloc[0]
        return [
            Trade(
                option_ticker=row["option_ticker"],
                direction="sell",
                option_type="call",
                strike=row["strike"],
                expiration=row["expiration"],
                premium=row["close"],
                contracts=1,
            )
        ]
