"""Weekly wheel backtest engine.

Simulates the wheel strategy (cash-secured puts -> covered calls)
on SPY with optional ML-based sell/skip filter.

The wheel lifecycle:
    CASH -> PUT_PHASE (sell CSP)
    PUT_PHASE -> HOLDING (assigned when put is ITM)
    HOLDING -> CALL_PHASE (sell CC)
    CALL_PHASE -> PUT_PHASE (called away when call is ITM)
    PUT/CALL OTM expiry -> stay in same phase, collect premium
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum


class WheelState(Enum):
    """Current phase of the wheel strategy."""
    CASH = "CASH"
    PUT_PHASE = "PUT_PHASE"
    HOLDING = "HOLDING"
    CALL_PHASE = "CALL_PHASE"


@dataclass(frozen=True, slots=True)
class WeekRecord:
    """Data for one trading week in the wheel backtest."""
    week_num: int
    expiration_date: str
    entry_date: str
    spy_price_at_entry: float
    spy_price_at_expiry: float
    put_strike: float
    call_strike: float
    put_premium: float
    call_premium: float
    ml_prob: float


class WheelBacktestEngine:
    """Runs the weekly wheel backtest.

    Parameters
    ----------
    initial_capital:
        Starting cash in dollars.
    contracts:
        Number of option contracts per trade (1 contract = 100 shares).
    skip_put_thresh:
        Skip selling puts when ML prob < this (crash danger).
    skip_call_thresh:
        Skip selling calls when ML prob > this (rally expected).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        contracts: int = 1,
        skip_put_thresh: float = 0.35,
        skip_call_thresh: float = 0.65,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._contracts = contracts
        self._skip_put_thresh = skip_put_thresh
        self._skip_call_thresh = skip_call_thresh

        # State
        self._state = WheelState.CASH
        self._shares: int = 0
        self._share_cost_basis: float = 0.0

        # Tracking
        self._total_premium: float = 0.0
        self._n_assignments: int = 0
        self._n_calls_exercised: int = 0
        self._n_puts_sold: int = 0
        self._n_calls_sold: int = 0
        self._n_puts_skipped: int = 0
        self._n_calls_skipped: int = 0
        self._equity_curve: list[float] = []
        self._weeks_processed: int = 0

    @property
    def state(self) -> WheelState:
        return self._state

    @property
    def shares_held(self) -> int:
        return self._shares

    @property
    def total_premium_collected(self) -> float:
        return self._total_premium

    @property
    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)

    def process_week(
        self,
        week: WeekRecord,
        *,
        use_ml_filter: bool = False,
    ) -> None:
        """Process one week of the wheel strategy."""
        multiplier = self._contracts * 100

        if self._state in (WheelState.CASH, WheelState.PUT_PHASE):
            # ── PUT PHASE ──
            if use_ml_filter and week.ml_prob < self._skip_put_thresh:
                self._n_puts_skipped += 1
            else:
                premium = week.put_premium * multiplier
                self._cash += premium
                self._total_premium += premium
                self._n_puts_sold += 1
                self._state = WheelState.PUT_PHASE

                if week.spy_price_at_expiry < week.put_strike:
                    # ITM: assigned
                    cost = week.put_strike * multiplier
                    self._cash -= cost
                    self._shares = multiplier
                    self._share_cost_basis = week.put_strike
                    self._n_assignments += 1
                    self._state = WheelState.CALL_PHASE

        elif self._state in (WheelState.HOLDING, WheelState.CALL_PHASE):
            # ── CALL PHASE ──
            if use_ml_filter and week.ml_prob > self._skip_call_thresh:
                self._n_calls_skipped += 1
            else:
                premium = week.call_premium * multiplier
                self._cash += premium
                self._total_premium += premium
                self._n_calls_sold += 1
                self._state = WheelState.CALL_PHASE

                if week.spy_price_at_expiry > week.call_strike:
                    # ITM: called away
                    proceeds = week.call_strike * multiplier
                    self._cash += proceeds
                    self._shares = 0
                    self._share_cost_basis = 0.0
                    self._n_calls_exercised += 1
                    self._state = WheelState.PUT_PHASE

        # Update equity curve
        equity = self._cash + self._shares * week.spy_price_at_expiry
        self._equity_curve.append(equity)
        self._weeks_processed += 1

    def compute_metrics(self) -> dict[str, float]:
        """Compute summary metrics for the backtest."""
        final_equity = self._equity_curve[-1] if self._equity_curve else self._initial_capital
        total_return = final_equity - self._initial_capital
        total_return_pct = (total_return / self._initial_capital) * 100

        # Sharpe (weekly returns, annualized)
        sharpe = 0.0
        if len(self._equity_curve) >= 2:
            returns = []
            prev = self._initial_capital
            for eq in self._equity_curve:
                returns.append((eq - prev) / prev if prev > 0 else 0.0)
                prev = eq
            if returns:
                mean_r = statistics.mean(returns)
                std_r = statistics.pstdev(returns)
                if std_r > 0:
                    sharpe = (mean_r / std_r) * math.sqrt(52)

        # Max drawdown
        max_dd_pct = 0.0
        max_dd_dollar = 0.0
        peak = self._initial_capital
        for eq in self._equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = dd / peak * 100 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_dollar = dd

        # Annualized return
        years = self._weeks_processed / 52.0 if self._weeks_processed > 0 else 1.0
        annualized = 0.0
        if years > 0 and final_equity > 0 and self._initial_capital > 0:
            annualized = ((final_equity / self._initial_capital) ** (1.0 / years) - 1.0) * 100

        return {
            "initial_capital": self._initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "total_return_dollar": total_return,
            "total_premium_collected": self._total_premium,
            "n_assignments": float(self._n_assignments),
            "n_calls_exercised": float(self._n_calls_exercised),
            "n_puts_sold": float(self._n_puts_sold),
            "n_calls_sold": float(self._n_calls_sold),
            "n_puts_skipped": float(self._n_puts_skipped),
            "n_calls_skipped": float(self._n_calls_skipped),
            "weeks": float(self._weeks_processed),
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_dollar": max_dd_dollar,
            "annualized_return_pct": annualized,
        }
