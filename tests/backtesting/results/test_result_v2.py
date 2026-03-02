"""Tests for new institutional-grade metrics in BacktestResult.

Covers sortino_ratio, calmar_ratio, max_consecutive_losses/wins,
and avg_trade_duration_bars.
"""

from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.backtesting.results.result import BacktestResult
from stockdownloader.core.models.trade import Trade, Direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    initial: str = "100000",
    final: str = "100000",
    equity: list[str] | None = None,
    wins: int = 0,
    losses: int = 0,
) -> BacktestResult:
    """Create a BacktestResult with optional equity curve and trades."""
    r = BacktestResult("test", Decimal(initial))
    r.final_capital = Decimal(final)

    if equity:
        r.equity_curve = [Decimal(e) for e in equity]

    # Add winning trades
    for i in range(wins):
        t = Trade(Direction.LONG, f"w{i}", Decimal("100"), 10)
        t.close(f"w{i}x", Decimal("110"))
        r.add_trade(t)

    # Add losing trades
    for i in range(losses):
        t = Trade(Direction.LONG, f"l{i}", Decimal("100"), 10)
        t.close(f"l{i}x", Decimal("90"))
        r.add_trade(t)

    return r


def _make_result_with_trade_sequence(
    sequence: list[bool],
    equity: list[str] | None = None,
) -> BacktestResult:
    """Create result with a specific win/loss sequence.

    sequence: list of bools — True=win, False=loss.
    """
    r = BacktestResult("test", Decimal("100000"))
    r.final_capital = Decimal("100000")

    if equity:
        r.equity_curve = [Decimal(e) for e in equity]

    for i, is_win in enumerate(sequence):
        t = Trade(Direction.LONG, f"t{i}", Decimal("100"), 10)
        exit_price = Decimal("110") if is_win else Decimal("90")
        t.close(f"t{i}x", exit_price)
        r.add_trade(t)

    return r


# =========================================================================
# Sortino Ratio
# =========================================================================


def test_sortino_ratio_no_equity_curve():
    r = _make_result()
    assert r.sortino_ratio() == Decimal("0")


def test_sortino_ratio_single_bar():
    r = _make_result(equity=["100000"])
    assert r.sortino_ratio() == Decimal("0")


def test_sortino_ratio_flat_equity():
    """Flat equity → no returns → Sortino = 0."""
    r = _make_result(equity=["100000", "100000", "100000"])
    assert r.sortino_ratio() == Decimal("0")


def test_sortino_ratio_all_positive():
    """All positive returns → no downside → high Sortino."""
    r = _make_result(
        equity=["100000", "101000", "102010", "103030", "104060"]
    )
    sortino = r.sortino_ratio()
    # Should be very high (capped at 99.99 when no downside)
    assert sortino == Decimal("99.99")


def test_sortino_ratio_mixed_returns():
    """Mix of positive and negative returns gives finite Sortino."""
    # Construct equity with ups and downs
    r = _make_result(
        equity=["100000", "101000", "100500", "101500", "101000",
                "102000", "101800", "102500"]
    )
    sortino = r.sortino_ratio()
    # Should be positive (net positive returns with some downside)
    assert sortino > Decimal("0")


def test_sortino_ratio_all_negative():
    """All negative returns → negative Sortino."""
    r = _make_result(
        equity=["100000", "99000", "98010", "97029"]
    )
    sortino = r.sortino_ratio()
    assert sortino < Decimal("0")


def test_sortino_higher_than_sharpe_for_upside_vol():
    """Sortino should be higher than Sharpe when upside vol dominates."""
    # Big gains, small losses → Sharpe penalises all vol, Sortino doesn't
    r = _make_result(
        equity=["100000", "103000", "102500", "106000", "105800",
                "110000", "109500", "115000"]
    )
    sharpe = float(r.sharpe_ratio())
    sortino = float(r.sortino_ratio())
    # With positive skew, Sortino should exceed Sharpe
    assert sortino > sharpe


# =========================================================================
# Calmar Ratio
# =========================================================================


def test_calmar_ratio_no_drawdown():
    """Zero drawdown → Calmar = 0 (undefined)."""
    r = _make_result(
        initial="100000", final="110000",
        equity=["100000", "102000", "105000", "110000"],
    )
    # Monotonically increasing → max_drawdown = 0
    assert r.calmar_ratio() == Decimal("0")


def test_calmar_ratio_positive():
    """Positive return with drawdown → positive Calmar."""
    r = _make_result(
        initial="100000", final="110000",
        equity=["100000", "105000", "102000", "110000"],
    )
    calmar = r.calmar_ratio()
    assert calmar > Decimal("0")


def test_calmar_ratio_negative_return():
    """Negative return with drawdown → negative Calmar."""
    r = _make_result(
        initial="100000", final="95000",
        equity=["100000", "98000", "96000", "95000"],
    )
    calmar = r.calmar_ratio()
    assert calmar < Decimal("0")


def test_calmar_ratio_annualized():
    """Calmar uses annualised return, not total return.

    With bars_per_year=4 and 4-bar equity curve (= 1 year of data),
    the annualised return equals total return, so
    calmar = total_return / max_drawdown.
    """
    r = _make_result(
        initial="100000", final="110000",
        equity=["100000", "105000", "100000", "110000"],
    )
    # total_return = 10%, max_drawdown ≈ 4.76% (from 105K to 100K)
    # With bars_per_year=4, years=1 so annualised return = total return
    calmar = r.calmar_ratio(bars_per_year=4)
    expected = r.total_return / r.max_drawdown
    assert calmar == expected.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# =========================================================================
# Max Consecutive Losses
# =========================================================================


def test_max_consecutive_losses_no_trades():
    r = _make_result()
    assert r.max_consecutive_losses == 0


def test_max_consecutive_losses_all_wins():
    r = _make_result(wins=5)
    assert r.max_consecutive_losses == 0


def test_max_consecutive_losses_all_losses():
    r = _make_result(losses=4)
    assert r.max_consecutive_losses == 4


def test_max_consecutive_losses_mixed():
    # W W L L L W L L W W W
    r = _make_result_with_trade_sequence(
        [True, True, False, False, False, True, False, False, True, True, True]
    )
    assert r.max_consecutive_losses == 3


def test_max_consecutive_losses_single_loss():
    r = _make_result_with_trade_sequence([True, False, True, True])
    assert r.max_consecutive_losses == 1


# =========================================================================
# Max Consecutive Wins
# =========================================================================


def test_max_consecutive_wins_no_trades():
    r = _make_result()
    assert r.max_consecutive_wins == 0


def test_max_consecutive_wins_all_losses():
    r = _make_result(losses=3)
    assert r.max_consecutive_wins == 0


def test_max_consecutive_wins_mixed():
    # W W W L W W L L W W W W
    r = _make_result_with_trade_sequence(
        [True, True, True, False, True, True, False, False, True, True, True, True]
    )
    assert r.max_consecutive_wins == 4


# =========================================================================
# Average Trade Duration
# =========================================================================


def test_avg_trade_duration_no_trades():
    r = _make_result()
    assert r.avg_trade_duration_bars == 0.0


def test_avg_trade_duration_with_trades():
    r = _make_result(wins=3, losses=2)
    # All trades have entry_date and exit_date set
    duration = r.avg_trade_duration_bars
    assert duration >= 0.0
