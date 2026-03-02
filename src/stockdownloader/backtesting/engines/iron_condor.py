"""Iron condor overlay backtest engine.

Sells weekly iron condors (put spread + call spread) for defined-risk
premium income. Profits in range-bound markets, complements the
directional combined strategy.

Iron condor structure:
    Buy OTM put (10-delta)  = long_put_strike
    Sell put (30-delta)     = short_put_strike
    Sell call (30-delta)    = short_call_strike
    Buy OTM call (10-delta) = long_call_strike

Max profit: net credit (SPY stays between short strikes)
Max loss: wider spread width * 100 - net credit * 100
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ICWeekRecord:
    """Data for one iron condor trading week."""
    week_num: int
    expiration_date: str
    spy_price_at_entry: float
    spy_price_at_expiry: float
    short_put_strike: float
    long_put_strike: float
    short_call_strike: float
    long_call_strike: float
    net_credit_per_contract: float
    ml_prob: float


class IronCondorEngine:
    """Runs weekly iron condor backtest.

    Parameters
    ----------
    capital:
        Starting cash allocated to IC strategy.
    commission_per_contract:
        Dollar commission per contract leg.
    skip_put_thresh:
        Skip IC when ML prob < this (strong bearish = directional).
    skip_call_thresh:
        Skip IC when ML prob > this (strong bullish = directional).
    """

    def __init__(
        self,
        capital: float,
        commission_per_contract: float = 0.0,
        skip_put_thresh: float = 0.35,
        skip_call_thresh: float = 0.65,
    ) -> None:
        self._initial_capital = capital
        self._cash = capital
        self._commission = commission_per_contract
        self._skip_put_thresh = skip_put_thresh
        self._skip_call_thresh = skip_call_thresh

        # Tracking
        self._total_credit: float = 0.0
        self._total_loss: float = 0.0
        self._total_commissions: float = 0.0
        self._n_ics_sold: int = 0
        self._n_ics_skipped: int = 0
        self._n_max_loss_events: int = 0
        self._equity_curve: list[float] = []
        self._weeks_processed: int = 0

    @property
    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)

    def process_week(
        self,
        week: ICWeekRecord,
        *,
        use_ml_filter: bool = False,
    ) -> None:
        """Process one week of iron condor strategy."""
        # ML filter: skip when strong directional signal
        if use_ml_filter and (
            week.ml_prob < self._skip_put_thresh
            or week.ml_prob > self._skip_call_thresh
        ):
            self._n_ics_skipped += 1
            self._equity_curve.append(self._cash)
            self._weeks_processed += 1
            return

        # Position sizing
        put_spread_width = week.short_put_strike - week.long_put_strike
        call_spread_width = week.long_call_strike - week.short_call_strike
        wider = max(put_spread_width, call_spread_width)
        max_loss_per = wider * 100 - week.net_credit_per_contract * 100

        if max_loss_per <= 0:
            max_loss_per = 1.0

        commission_per_ic = self._commission * 4
        denom = max_loss_per + commission_per_ic
        n_contracts = int(self._cash // denom) if denom > 0 else 0

        if n_contracts < 1:
            self._equity_curve.append(self._cash)
            self._weeks_processed += 1
            return

        # Collect credit
        credit = week.net_credit_per_contract * n_contracts * 100
        commission = commission_per_ic * n_contracts
        self._cash += credit - commission
        self._total_credit += credit
        self._total_commissions += commission
        self._n_ics_sold += n_contracts

        # Settlement
        spy = week.spy_price_at_expiry

        # Put spread loss
        if spy <= week.long_put_strike:
            put_loss = put_spread_width * n_contracts * 100
            self._n_max_loss_events += 1
        elif spy < week.short_put_strike:
            put_loss = (week.short_put_strike - spy) * n_contracts * 100
        else:
            put_loss = 0.0

        # Call spread loss
        if spy >= week.long_call_strike:
            call_loss = call_spread_width * n_contracts * 100
            self._n_max_loss_events += 1
        elif spy > week.short_call_strike:
            call_loss = (spy - week.short_call_strike) * n_contracts * 100
        else:
            call_loss = 0.0

        total_loss = put_loss + call_loss
        self._cash -= total_loss
        self._total_loss += total_loss

        self._equity_curve.append(self._cash)
        self._weeks_processed += 1

    def compute_metrics(self) -> dict[str, float]:
        """Compute summary metrics for the IC backtest."""
        final_equity = self._equity_curve[-1] if self._equity_curve else self._initial_capital
        total_return = final_equity - self._initial_capital
        total_return_pct = (total_return / self._initial_capital) * 100 if self._initial_capital > 0 else 0.0

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
        peak = self._initial_capital
        for eq in self._equity_curve:
            if eq > peak:
                peak = eq
            dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        # Annualized
        years = self._weeks_processed / 52.0 if self._weeks_processed > 0 else 1.0
        annualized = 0.0
        if years > 0 and final_equity > 0 and self._initial_capital > 0:
            annualized = ((final_equity / self._initial_capital) ** (1.0 / years) - 1.0) * 100

        return {
            "initial_capital": self._initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "total_return_dollar": total_return,
            "total_credit": self._total_credit,
            "total_loss": self._total_loss,
            "total_commissions": self._total_commissions,
            "n_ics_sold": float(self._n_ics_sold),
            "n_ics_skipped": float(self._n_ics_skipped),
            "n_max_loss_events": float(self._n_max_loss_events),
            "weeks": float(self._weeks_processed),
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd_pct,
            "annualized_return_pct": annualized,
        }
