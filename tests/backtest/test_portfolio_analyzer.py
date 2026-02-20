"""Tests for portfolio_analyzer — correlation, selection, equity simulation."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from stockdownloader.backtest.portfolio_analyzer import (
    _pearson_correlation,
    correlation_matrix,
    daily_returns,
    portfolio_equity_curve,
    portfolio_metrics,
    select_portfolio,
)
from stockdownloader.backtest.tournament_engine import ComboKey, ComboResult


class TestDailyReturns:
    def test_empty_curve(self):
        assert daily_returns([]) == []

    def test_single_value(self):
        assert daily_returns([Decimal("100")]) == []

    def test_two_values(self):
        returns = daily_returns([Decimal("100"), Decimal("110")])
        assert len(returns) == 1
        assert returns[0] == pytest.approx(10.0)

    def test_multiple_values(self):
        curve = [Decimal("100"), Decimal("110"), Decimal("105")]
        returns = daily_returns(curve)
        assert len(returns) == 2
        assert returns[0] == pytest.approx(10.0)
        assert returns[1] == pytest.approx(-4.5454, rel=1e-3)

    def test_flat_curve(self):
        curve = [Decimal("100")] * 5
        returns = daily_returns(curve)
        assert all(r == 0.0 for r in returns)


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert _pearson_correlation(x, y) == pytest.approx(1.0)

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert _pearson_correlation(x, y) == pytest.approx(-1.0)

    def test_uncorrelated(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 1.0, 1.0, 1.0, 1.0]  # zero variance
        assert _pearson_correlation(x, y) == 0.0

    def test_empty_series(self):
        assert _pearson_correlation([], []) == 0.0

    def test_single_element(self):
        assert _pearson_correlation([1.0], [2.0]) == 0.0

    def test_different_lengths(self):
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0]
        # Should use min length
        result = _pearson_correlation(x, y)
        assert isinstance(result, float)


class TestCorrelationMatrix:
    def _make_combo(
        self, name: str, tf: str, curve: list[Decimal],
    ) -> ComboResult:
        mock_result = MagicMock()
        mock_result.equity_curve = curve
        mock_result.total_pnl = curve[-1] - curve[0] if curve else Decimal("0")
        mock_result.total_trades = 20
        mock_result.win_rate = Decimal("60.0")
        mock_result.total_return = Decimal("5.0")

        return ComboResult(
            key=ComboKey(name, tf),
            display_name=name,
            baseline=mock_result,
        )

    def test_empty(self):
        assert correlation_matrix([]) == {}

    def test_single_combo(self):
        combo = self._make_combo("a", "5m", [Decimal("100"), Decimal("110")])
        # Single combo -> no pairs
        assert correlation_matrix([combo]) == {}

    def test_two_combos(self):
        a = self._make_combo("a", "5m", [Decimal("100"), Decimal("110"), Decimal("120")])
        b = self._make_combo("b", "5m", [Decimal("100"), Decimal("105"), Decimal("110")])
        corr = correlation_matrix([a, b])
        # Should have one pair
        assert len(corr) == 1
        # Both go up linearly -> high correlation
        key = list(corr.keys())[0]
        assert corr[key] == pytest.approx(1.0, abs=0.01)


class TestSelectPortfolio:
    def _make_combo(
        self, name: str, score: float, pnl: float = 5000,
        trades: int = 50, win_rate: float = 60.0,
    ) -> ComboResult:
        mock_result = MagicMock()
        mock_result.total_pnl = Decimal(str(pnl))
        mock_result.total_trades = trades
        mock_result.win_rate = Decimal(str(win_rate))
        mock_result.equity_curve = [Decimal("100000"), Decimal(str(100000 + pnl))]

        combo = ComboResult(
            key=ComboKey(name, "5m"),
            display_name=name,
            baseline=mock_result,
        )
        combo.tournament_score = score
        return combo

    def test_empty(self):
        assert select_portfolio([], {}) == []

    def test_single_qualifying(self):
        combos = [self._make_combo("a", 100)]
        result = select_portfolio(combos, {})
        assert len(result) == 1

    def test_max_strategies_limit(self):
        combos = [self._make_combo(f"s{i}", 100 - i) for i in range(10)]
        result = select_portfolio(combos, {}, max_strategies=3)
        assert len(result) <= 3

    def test_correlation_filter(self):
        a = self._make_combo("a", 100)
        b = self._make_combo("b", 90)
        c = self._make_combo("c", 80)

        # a and b are highly correlated, c is independent
        corr = {
            ("a @ 5m", "b @ 5m"): 0.9,
            ("a @ 5m", "c @ 5m"): 0.1,
            ("b @ 5m", "c @ 5m"): 0.1,
        }
        result = select_portfolio([a, b, c], corr, max_correlation=0.4)
        # Should pick a (best) and c (low corr), skip b (high corr with a)
        names = [r.key.strategy_name for r in result]
        assert "a" in names
        assert "c" in names
        assert "b" not in names

    def test_unprofitable_excluded(self):
        combos = [self._make_combo("loss", 50, pnl=-1000)]
        result = select_portfolio(combos, {})
        assert len(result) == 0

    def test_low_trades_excluded(self):
        combos = [self._make_combo("few", 100, trades=3)]
        result = select_portfolio(combos, {}, min_trades=10)
        # Falls back to relaxed constraints (trades >= 3)
        assert len(result) == 1


class TestPortfolioEquityCurve:
    def test_empty(self):
        assert portfolio_equity_curve([]) == []

    def test_single_strategy(self):
        mock_result = MagicMock()
        mock_result.equity_curve = [Decimal("100000"), Decimal("110000")]
        mock_result.initial_capital = Decimal("100000")

        combo = ComboResult(key=ComboKey("a", "5m"), baseline=mock_result)
        curve = portfolio_equity_curve([combo], initial_capital=100000.0)
        assert len(curve) == 2
        assert curve[0] == pytest.approx(100000.0)
        assert curve[1] == pytest.approx(110000.0)

    def test_two_strategies_equal_weight(self):
        mock_a = MagicMock()
        mock_a.equity_curve = [Decimal("100000"), Decimal("110000")]
        mock_a.initial_capital = Decimal("100000")

        mock_b = MagicMock()
        mock_b.equity_curve = [Decimal("100000"), Decimal("90000")]
        mock_b.initial_capital = Decimal("100000")

        combo_a = ComboResult(key=ComboKey("a", "5m"), baseline=mock_a)
        combo_b = ComboResult(key=ComboKey("b", "5m"), baseline=mock_b)

        curve = portfolio_equity_curve([combo_a, combo_b], initial_capital=100000.0)
        # Each gets 50k. A: 50k->55k, B: 50k->45k. Total: 100k->100k
        assert curve[0] == pytest.approx(100000.0)
        assert curve[1] == pytest.approx(100000.0)


class TestPortfolioMetrics:
    def test_empty_curve(self):
        metrics = portfolio_metrics([])
        assert metrics["total_return"] == 0.0
        assert metrics["total_pnl"] == 0.0

    def test_profitable(self):
        curve = [100000.0, 105000.0, 110000.0]
        metrics = portfolio_metrics(curve, initial_capital=100000.0)
        assert metrics["total_return"] == 10.0
        assert metrics["total_pnl"] == 10000.0
        assert metrics["max_drawdown"] == 0.0  # never drew down

    def test_drawdown(self):
        curve = [100000.0, 110000.0, 95000.0, 105000.0]
        metrics = portfolio_metrics(curve, initial_capital=100000.0)
        # Peak 110k, trough 95k -> DD = 15/110 = 13.64%
        assert metrics["max_drawdown"] == pytest.approx(13.64, abs=0.1)
