"""Weekly wheel backtest engine.

Simulates the wheel strategy (cash-secured puts -> covered calls)
on SPY with optional ML-based sell/skip filter.

The wheel lifecycle:
    CASH -> PUT_PHASE (sell CSP)
    PUT_PHASE -> HOLDING (assigned when put is ITM)
    HOLDING -> CALL_PHASE (sell CC)
    CALL_PHASE -> PUT_PHASE (called away when call is ITM)
    PUT/CALL OTM expiry -> stay in same phase, collect premium

Buy-write mode:
    Buy shares immediately, sell covered calls for income.
    If called away, immediately re-buy at market price.
    Captures buy-and-hold returns PLUS premium income.

Combined mode:
    Buy shares immediately (like buy-write), sell CCs on all held
    shares, AND sell CSPs on idle cash. Generates double premium
    income. CC count is dynamic (shares // 100). CSP count is
    capped at the contracts parameter and limited by available cash.
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
    # Collar fields (default 0 = no hedge data)
    hedge_put_strike: float = 0.0
    hedge_put_premium: float = 0.0
    # Vol scaling field
    iv_percentile: float = 0.50


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
    buy_write:
        If True, buy shares immediately and sell covered calls only.
        Captures buy-and-hold returns plus premium income.
    combined:
        If True, buy shares + sell CCs on shares + sell CSPs on idle cash.
    commission_per_contract:
        Dollar commission charged per contract on each option trade.
        Default is 0.0 (no commission).
    collar:
        If True, buy a protective put on all held shares each week.
        Deducts hedge cost from cash and pays out when SPY drops below
        the hedge put strike.
    vol_scaling:
        If True, scale effective contract count by IV percentile.
        Higher IV -> more contracts traded, capped at max_contracts.
    max_contracts:
        Maximum effective contracts when vol_scaling is enabled.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        contracts: int = 1,
        skip_put_thresh: float = 0.35,
        skip_call_thresh: float = 0.65,
        buy_write: bool = False,
        combined: bool = False,
        commission_per_contract: float = 0.0,
        collar: bool = False,
        vol_scaling: bool = False,
        max_contracts: int = 3,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._contracts = contracts
        self._skip_put_thresh = skip_put_thresh
        self._skip_call_thresh = skip_call_thresh
        self._buy_write = buy_write
        self._combined = combined
        self._commission_per_contract = commission_per_contract
        self._collar = collar
        self._vol_scaling = vol_scaling
        self._max_contracts = max_contracts

        # State
        self._state = WheelState.CASH
        self._shares: int = 0
        self._share_cost_basis: float = 0.0

        # Tracking
        self._total_premium: float = 0.0
        self._total_commissions: float = 0.0
        self._total_hedge_cost: float = 0.0
        self._total_hedge_payout: float = 0.0
        self._n_assignments: int = 0
        self._n_calls_exercised: int = 0
        self._n_rebuys: int = 0
        self._n_puts_sold: int = 0
        self._n_calls_sold: int = 0
        self._n_puts_skipped: int = 0
        self._n_calls_skipped: int = 0
        self._equity_curve: list[float] = []
        self._weeks_processed: int = 0
        self._eff_contracts_history: list[int] = []

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

    def _compute_effective_contracts(self, iv_percentile: float) -> int:
        """Compute effective contract count from IV percentile."""
        if not self._vol_scaling:
            return self._contracts
        if iv_percentile <= 0.25:
            mult = 0.5
        elif iv_percentile <= 0.50:
            mult = 1.0
        elif iv_percentile <= 0.75:
            mult = 1.5
        else:
            mult = 2.0
        return max(1, min(self._max_contracts, int(self._contracts * mult)))

    def process_week(
        self,
        week: WeekRecord,
        *,
        use_ml_filter: bool = False,
    ) -> None:
        """Process one week of the wheel strategy."""
        multiplier = self._contracts * 100

        eff_contracts = self._compute_effective_contracts(week.iv_percentile)
        if self._vol_scaling:
            self._eff_contracts_history.append(eff_contracts)

        if self._combined:
            self._process_week_combined(week, use_ml_filter, eff_contracts)
        elif self._buy_write:
            self._process_week_buy_write(week, multiplier, use_ml_filter)
        else:
            self._process_week_wheel(week, multiplier, use_ml_filter)

        # Collar: buy protective put on all held shares
        if self._collar and self._shares > 0 and week.hedge_put_premium > 0:
            n_hedge = self._shares // 100
            hedge_cost = week.hedge_put_premium * n_hedge * 100
            commission = self._commission_per_contract * n_hedge
            self._cash -= hedge_cost + commission
            self._total_hedge_cost += hedge_cost
            self._total_commissions += commission

            if week.spy_price_at_expiry < week.hedge_put_strike:
                payout = (week.hedge_put_strike - week.spy_price_at_expiry) * n_hedge * 100
                self._cash += payout
                self._total_hedge_payout += payout

        # Update equity curve
        equity = self._cash + self._shares * week.spy_price_at_expiry
        self._equity_curve.append(equity)
        self._weeks_processed += 1

    def _process_week_buy_write(
        self,
        week: WeekRecord,
        multiplier: int,
        use_ml_filter: bool,
    ) -> None:
        """Buy-write mode: hold shares + sell covered calls for income."""
        # Buy shares on first week if not already holding
        if self._shares == 0:
            cost = week.spy_price_at_entry * multiplier
            self._cash -= cost
            self._shares = multiplier
            self._share_cost_basis = week.spy_price_at_entry
            self._state = WheelState.CALL_PHASE

        # Sell covered call (unless ML filter says skip for rally)
        if use_ml_filter and week.ml_prob > self._skip_call_thresh:
            self._n_calls_skipped += 1
        else:
            gross = week.call_premium * multiplier
            commission = self._commission_per_contract * self._contracts
            self._cash += gross - commission
            self._total_premium += gross - commission
            self._total_commissions += commission
            self._n_calls_sold += 1

            if week.spy_price_at_expiry > week.call_strike:
                # ITM: called away, then immediately re-buy
                proceeds = week.call_strike * multiplier
                self._cash += proceeds
                self._n_calls_exercised += 1

                # Immediately re-buy at expiry price
                rebuy_cost = week.spy_price_at_expiry * multiplier
                self._cash -= rebuy_cost
                self._share_cost_basis = week.spy_price_at_expiry
                self._n_rebuys += 1
                # Shares stay at multiplier (still holding)

        self._state = WheelState.CALL_PHASE

    def _process_week_combined(
        self,
        week: WeekRecord,
        use_ml_filter: bool,
        eff_contracts: int | None = None,
    ) -> None:
        """Combined mode: buy-write + CSPs on idle cash."""
        if eff_contracts is None:
            eff_contracts = self._contracts
        multiplier = self._contracts * 100  # initial buy uses base contracts

        # Step 1: Buy initial shares if not holding (base contracts)
        if self._shares == 0:
            cost = week.spy_price_at_entry * multiplier
            self._cash -= cost
            self._shares = multiplier
            self._share_cost_basis = week.spy_price_at_entry
            self._state = WheelState.CALL_PHASE

        # Step 2: Sell CCs - cap at eff_contracts when vol_scaling
        n_cc = self._shares // 100
        if self._vol_scaling:
            n_cc = min(n_cc, eff_contracts)
        if n_cc > 0:
            if not (use_ml_filter and week.ml_prob > self._skip_call_thresh):
                gross = week.call_premium * n_cc * 100
                commission = self._commission_per_contract * n_cc
                self._cash += gross - commission
                self._total_premium += gross - commission
                self._total_commissions += commission
                self._n_calls_sold += n_cc

                if week.spy_price_at_expiry > week.call_strike:
                    # Called away on n_cc contracts, immediately re-buy
                    self._cash += week.call_strike * n_cc * 100
                    self._cash -= week.spy_price_at_expiry * n_cc * 100
                    self._n_calls_exercised += n_cc
                    self._n_rebuys += n_cc
                    self._share_cost_basis = week.spy_price_at_expiry
            else:
                self._n_calls_skipped += n_cc

        # Step 3: Sell CSPs (capped at eff_contracts instead of self._contracts)
        if week.put_strike > 0:
            max_csp = min(eff_contracts, int(self._cash // (week.put_strike * 100)))
        else:
            max_csp = 0
        if max_csp > 0:
            if not (use_ml_filter and week.ml_prob < self._skip_put_thresh):
                gross = week.put_premium * max_csp * 100
                commission = self._commission_per_contract * max_csp
                self._cash += gross - commission
                self._total_premium += gross - commission
                self._total_commissions += commission
                self._n_puts_sold += max_csp

                if week.spy_price_at_expiry < week.put_strike:
                    # Assigned: acquire more shares
                    cost = week.put_strike * max_csp * 100
                    self._cash -= cost
                    self._shares += max_csp * 100
                    self._n_assignments += max_csp
            else:
                self._n_puts_skipped += max_csp

        self._state = WheelState.CALL_PHASE

    def _process_week_wheel(
        self,
        week: WeekRecord,
        multiplier: int,
        use_ml_filter: bool,
    ) -> None:
        """Standard wheel mode: CSP -> assignment -> CC -> called away."""
        if self._state in (WheelState.CASH, WheelState.PUT_PHASE):
            # ── PUT PHASE ──
            if use_ml_filter and week.ml_prob < self._skip_put_thresh:
                self._n_puts_skipped += 1
            else:
                gross = week.put_premium * multiplier
                commission = self._commission_per_contract * self._contracts
                self._cash += gross - commission
                self._total_premium += gross - commission
                self._total_commissions += commission
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
                gross = week.call_premium * multiplier
                commission = self._commission_per_contract * self._contracts
                self._cash += gross - commission
                self._total_premium += gross - commission
                self._total_commissions += commission
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
            "total_commissions": self._total_commissions,
            "total_hedge_cost": self._total_hedge_cost,
            "total_hedge_payout": self._total_hedge_payout,
            "n_assignments": float(self._n_assignments),
            "n_calls_exercised": float(self._n_calls_exercised),
            "n_rebuys": float(self._n_rebuys),
            "n_puts_sold": float(self._n_puts_sold),
            "n_calls_sold": float(self._n_calls_sold),
            "n_puts_skipped": float(self._n_puts_skipped),
            "n_calls_skipped": float(self._n_calls_skipped),
            "weeks": float(self._weeks_processed),
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_dollar": max_dd_dollar,
            "annualized_return_pct": annualized,
            "avg_contracts_traded": (
                statistics.mean(self._eff_contracts_history)
                if self._eff_contracts_history
                else float(self._contracts)
            ),
            "max_contracts_traded": (
                float(max(self._eff_contracts_history))
                if self._eff_contracts_history
                else float(self._contracts)
            ),
        }
