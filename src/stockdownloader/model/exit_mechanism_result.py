"""Result models for exit mechanism tournament evaluation."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.model.trade import Direction
from stockdownloader.util.big_decimal_math import HUNDRED, ZERO

@dataclass(frozen=True, slots=True)
class ExitMechanismTradeResult:
    """Result of running a single trade through one exit mechanism."""

    mechanism_name: str
    trade_id: int
    direction: Direction
    signal_type: str
    entry_price: Decimal
    exit_price: Decimal
    exit_datetime: str
    pnl: Decimal
    r_multiple: Decimal
    holding_bars: int
    peak_favorable: Decimal
    capture_pct: Decimal
    exit_reason: str

class ExitMechanismSummary:
    """Aggregate statistics for one exit mechanism across multiple trades."""

    def __init__(self, mechanism_name: str) -> None:
        if mechanism_name is None:
            raise ValueError("mechanism_name must not be None")
        self._mechanism_name = mechanism_name
        self._results: list[ExitMechanismTradeResult] = []

    def add_result(self, result: ExitMechanismTradeResult) -> None:
        """Record a trade result for this mechanism."""
        if result is None:
            raise ValueError("result must not be None")
        self._results.append(result)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mechanism_name(self) -> str:
        return self._mechanism_name

    @property
    def results(self) -> list[ExitMechanismTradeResult]:
        return list(self._results)

    @property
    def total_trades(self) -> int:
        return len(self._results)

    @property
    def winning_trades(self) -> int:
        return sum(1 for r in self._results if r.pnl > ZERO)

    @property
    def losing_trades(self) -> int:
        return self.total_trades - self.winning_trades

    @property
    def win_rate(self) -> Decimal:
        if self.total_trades == 0:
            return ZERO
        return (
            Decimal(str(self.winning_trades))
            / Decimal(str(self.total_trades))
            * HUNDRED
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @property
    def total_pnl(self) -> Decimal:
        return sum((r.pnl for r in self._results), ZERO)

    @property
    def avg_pnl(self) -> Decimal:
        if self.total_trades == 0:
            return ZERO
        return (self.total_pnl / Decimal(str(self.total_trades))).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

    @property
    def median_pnl(self) -> Decimal:
        if not self._results:
            return ZERO
        pnls = [r.pnl for r in self._results]
        result = statistics.median(pnls)
        return result.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    @property
    def profit_factor(self) -> Decimal:
        gross_profit = sum((r.pnl for r in self._results if r.pnl > ZERO), ZERO)
        gross_loss = sum((abs(r.pnl) for r in self._results if r.pnl <= ZERO), ZERO)
        if gross_loss == ZERO:
            return Decimal("999.99") if gross_profit > ZERO else ZERO
        return (gross_profit / gross_loss).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @property
    def avg_r_multiple(self) -> Decimal:
        if self.total_trades == 0:
            return ZERO
        total_r = sum((r.r_multiple for r in self._results), ZERO)
        return (total_r / Decimal(str(self.total_trades))).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

    @property
    def avg_capture_pct(self) -> Decimal:
        if self.total_trades == 0:
            return ZERO
        total = sum((r.capture_pct for r in self._results), ZERO)
        return (total / Decimal(str(self.total_trades))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @property
    def median_capture_pct(self) -> Decimal:
        if not self._results:
            return ZERO
        caps = [r.capture_pct for r in self._results]
        result = statistics.median(caps)
        return result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @property
    def avg_holding_bars(self) -> Decimal:
        if self.total_trades == 0:
            return ZERO
        total = sum(r.holding_bars for r in self._results)
        return (Decimal(str(total)) / Decimal(str(self.total_trades))).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

    @property
    def max_loss(self) -> Decimal:
        if not self._results:
            return ZERO
        return min(r.pnl for r in self._results)

    @property
    def max_win(self) -> Decimal:
        if not self._results:
            return ZERO
        return max(r.pnl for r in self._results)

    @property
    def pnl_std(self) -> Decimal:
        if len(self._results) < 2:
            return ZERO
        pnls = [float(r.pnl) for r in self._results]
        std = statistics.pstdev(pnls)
        return Decimal(str(std)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

    @property
    def sharpe_approx(self) -> Decimal:
        """Approximate Sharpe: mean(P&L) / std(P&L)."""
        std = self.pnl_std
        if std == ZERO:
            return ZERO
        return (self.avg_pnl / std).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    def exit_reason_counts(self) -> dict[str, int]:
        """Count occurrences of each exit reason."""
        return dict(Counter(r.exit_reason for r in self._results))
