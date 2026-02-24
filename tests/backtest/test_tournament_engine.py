"""Tests for tournament_engine — data models, scoring, bracket logic."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from stockdownloader.backtest.tournament_engine import (
    ComboKey,
    ComboResult,
    MatchResult,
    MonteCarloPercentiles,
    MonteCarloResult,
    RegimeAnalysis,
    RegimeTradeStats,
    TournamentResult,
    apply_cross_timeframe_bonus,
    run_elimination_bracket,
)
from stockdownloader.strategies.regime.detector import MarketRegime


class TestComboKey:
    def test_label(self):
        key = ComboKey("smc-structure", "5m")
        assert key.label == "smc-structure @ 5m"

    def test_frozen(self):
        key = ComboKey("smc-structure", "5m")
        with pytest.raises(AttributeError):
            key.strategy_name = "other"  # type: ignore[misc]

    def test_equality(self):
        a = ComboKey("rsi", "15m")
        b = ComboKey("rsi", "15m")
        assert a == b

    def test_different_tf(self):
        a = ComboKey("rsi", "5m")
        b = ComboKey("rsi", "15m")
        assert a != b

    def test_data_source_default(self):
        key = ComboKey("smc-structure", "5m")
        assert key.data_source == "resampled"


class TestComboResult:
    def test_no_result_score(self):
        combo = ComboResult(key=ComboKey("x", "5m"))
        score = combo.compute_tournament_score(100)
        assert score == -999.0

    def test_optimized_fields_default_none(self):
        combo = ComboResult(key=ComboKey("x", "5m"))
        assert combo.optimized_kwargs is None
        assert combo.optimized is None

    def test_regime_and_mc_fields_default_none(self):
        combo = ComboResult(key=ComboKey("x", "5m"))
        assert combo.regime_analysis is None
        assert combo.monte_carlo is None

    def test_score_with_regime_bonus(self):
        """Regime bonus should additively affect tournament_score."""
        mock_result = MagicMock()
        mock_result.total_return = Decimal("5.0")
        mock_result.total_pnl = Decimal("5000")
        mock_result.total_trades = 50
        mock_result.win_rate = Decimal("60.0")
        mock_result.profit_factor = Decimal("1.5")
        mock_result.max_drawdown = Decimal("3.0")
        mock_result.max_consecutive_losses = 3
        mock_result.sortino_ratio.return_value = Decimal("1.5")
        mock_result.calmar_ratio.return_value = Decimal("2.0")

        # Without regime
        combo_no_regime = ComboResult(key=ComboKey("x", "5m"), baseline=mock_result)
        score_no_regime = combo_no_regime.compute_tournament_score(100)

        # With positive regime bonus
        regime_analysis = RegimeAnalysis(
            per_regime={},
            regime_coverage=4,
            worst_regime_pnl=10.0,
            regime_consistency=5.0,
            regime_bonus=5.0,
        )
        combo_regime = ComboResult(
            key=ComboKey("x", "5m"),
            baseline=mock_result,
            regime_analysis=regime_analysis,
        )
        score_regime = combo_regime.compute_tournament_score(100)

        assert score_regime == pytest.approx(score_no_regime + 5.0)

    def test_score_with_mc_penalty(self):
        """MC penalty should reduce tournament_score."""
        mock_result = MagicMock()
        mock_result.total_return = Decimal("5.0")
        mock_result.total_pnl = Decimal("5000")
        mock_result.total_trades = 50
        mock_result.win_rate = Decimal("60.0")
        mock_result.profit_factor = Decimal("1.5")
        mock_result.max_drawdown = Decimal("3.0")
        mock_result.max_consecutive_losses = 3
        mock_result.sortino_ratio.return_value = Decimal("1.5")
        mock_result.calmar_ratio.return_value = Decimal("2.0")

        # Without MC
        combo_no_mc = ComboResult(key=ComboKey("x", "5m"), baseline=mock_result)
        score_no_mc = combo_no_mc.compute_tournament_score(100)

        # With MC penalty
        pctl = MonteCarloPercentiles(1.0, 2.0, 3.0, 4.0, 5.0)
        mc_result = MonteCarloResult(
            n_simulations=1000,
            n_trades=50,
            max_drawdown=pctl,
            final_equity=pctl,
            total_return=pctl,
            bootstrap_return=MonteCarloPercentiles(-2.0, 0.5, 1.0, 2.0, 3.0),
            bootstrap_sharpe=pctl,
            is_robust=False,
            mc_penalty=4.0,
        )
        combo_mc = ComboResult(
            key=ComboKey("x", "5m"),
            baseline=mock_result,
            monte_carlo=mc_result,
        )
        score_mc = combo_mc.compute_tournament_score(100)

        assert score_mc == pytest.approx(score_no_mc - 4.0)

    def test_best_result_returns_baseline(self):
        combo = ComboResult(key=ComboKey("x", "5m"))
        assert combo.best_result is None

        mock_result = MagicMock()
        combo.baseline = mock_result
        assert combo.best_result is mock_result

    def test_best_result_prefers_optimized(self):
        baseline = MagicMock()
        optimized = MagicMock()
        combo = ComboResult(
            key=ComboKey("x", "5m"),
            baseline=baseline,
            optimized=optimized,
        )
        assert combo.best_result is optimized

    def test_best_result_falls_back_to_baseline(self):
        baseline = MagicMock()
        combo = ComboResult(
            key=ComboKey("x", "5m"),
            baseline=baseline,
            optimized=None,
        )
        assert combo.best_result is baseline

    def test_with_mock_result(self):
        """Score computation with a mocked BacktestResult."""
        mock_result = MagicMock()
        mock_result.total_return = Decimal("5.0")
        mock_result.total_pnl = Decimal("5000")
        mock_result.total_trades = 50
        mock_result.win_rate = Decimal("60.0")
        mock_result.profit_factor = Decimal("1.5")
        mock_result.max_drawdown = Decimal("3.0")
        mock_result.max_consecutive_losses = 3
        mock_result.sortino_ratio.return_value = Decimal("1.5")
        mock_result.calmar_ratio.return_value = Decimal("2.0")

        combo = ComboResult(key=ComboKey("x", "5m"), baseline=mock_result)
        score = combo.compute_tournament_score(100)
        assert score > 0  # profitable strategy should score positive

    def test_wf_degradation_penalizes_overfit(self):
        """Walk-forward overfit should reduce score."""
        mock_result = MagicMock()
        mock_result.total_return = Decimal("5.0")
        mock_result.total_pnl = Decimal("5000")
        mock_result.total_trades = 50
        mock_result.win_rate = Decimal("60.0")
        mock_result.profit_factor = Decimal("1.5")
        mock_result.max_drawdown = Decimal("3.0")
        mock_result.max_consecutive_losses = 2
        mock_result.sortino_ratio.return_value = Decimal("1.5")
        mock_result.calmar_ratio.return_value = Decimal("2.0")

        # Without WF
        combo_no_wf = ComboResult(key=ComboKey("x", "5m"), baseline=mock_result)
        score_no_wf = combo_no_wf.compute_tournament_score(100)

        # With OVERFIT WF
        mock_wf = MagicMock()
        mock_wf.degradation_ratio = 0.3  # OVERFIT

        combo_overfit = ComboResult(
            key=ComboKey("x", "5m"),
            baseline=mock_result,
            wf_result=mock_wf,
        )
        score_overfit = combo_overfit.compute_tournament_score(100)

        assert score_overfit < score_no_wf  # overfit penalized


class TestMatchResult:
    def test_frozen(self):
        m = MatchResult(
            winner=ComboKey("a", "5m"),
            loser=ComboKey("b", "15m"),
            winner_score=100.0,
            loser_score=50.0,
            round_num=0,
        )
        assert m.winner.strategy_name == "a"
        with pytest.raises(AttributeError):
            m.winner_score = 200.0  # type: ignore[misc]


class TestTournamentResult:
    def test_defaults(self):
        t = TournamentResult()
        assert t.combos == []
        assert t.matches == []
        assert t.bracket_champion is None
        assert t.portfolio == []
        assert t.correlation_matrix == {}


class TestEliminationBracket:
    def _make_combo(self, name: str, score: float) -> ComboResult:
        combo = ComboResult(key=ComboKey(name, "5m"))
        combo.tournament_score = score
        return combo

    def test_empty_list(self):
        matches, champion = run_elimination_bracket([], 4)
        assert matches == []
        assert champion is None

    def test_single_entry(self):
        combos = [self._make_combo("a", 100)]
        matches, champion = run_elimination_bracket(combos, 4)
        assert matches == []
        assert champion == ComboKey("a", "5m")

    def test_two_entries(self):
        combos = [
            self._make_combo("a", 100),
            self._make_combo("b", 50),
        ]
        matches, champion = run_elimination_bracket(combos, 2)
        assert len(matches) == 1
        assert champion == ComboKey("a", "5m")
        assert matches[0].winner == ComboKey("a", "5m")
        assert matches[0].loser == ComboKey("b", "5m")

    def test_four_entries_bracket(self):
        combos = [
            self._make_combo("seed1", 100),
            self._make_combo("seed2", 80),
            self._make_combo("seed3", 60),
            self._make_combo("seed4", 40),
        ]
        matches, champion = run_elimination_bracket(combos, 4)
        # 4 entries -> 2 rounds -> 3 matches total
        assert len(matches) == 3
        assert champion == ComboKey("seed1", "5m")

        # Round 0: seed1 vs seed4, seed2 vs seed3
        r0 = [m for m in matches if m.round_num == 0]
        assert len(r0) == 2

    def test_bracket_rounds_up_to_power_of_2(self):
        combos = [self._make_combo(f"s{i}", 100 - i) for i in range(5)]
        matches, champion = run_elimination_bracket(combos, 5)
        # 5 entries -> rounds down to 4 (2^2)
        assert len(matches) == 3  # 4-entry bracket: 2 + 1 = 3

    def test_eight_entries(self):
        combos = [self._make_combo(f"s{i}", 100 - i * 5) for i in range(8)]
        matches, champion = run_elimination_bracket(combos, 8)
        # 8 entries -> 3 rounds: 4 + 2 + 1 = 7 matches
        assert len(matches) == 7
        assert champion == ComboKey("s0", "5m")  # highest score

    def test_round_names(self):
        combos = [self._make_combo(f"s{i}", 100 - i * 5) for i in range(8)]
        matches, _ = run_elimination_bracket(combos, 8)

        round_names = {m.round_name for m in matches}
        assert "FINAL" in round_names
        assert "SEMIFINAL" in round_names
        assert "QUARTERFINAL" in round_names


class TestCrossTimeframeBonus:
    def _make_combo(
        self, name: str, tf: str, score: float, pnl: float = 1000.0,
    ) -> ComboResult:
        mock_result = MagicMock()
        mock_result.total_pnl = Decimal(str(pnl))
        combo = ComboResult(key=ComboKey(name, tf), baseline=mock_result)
        combo.tournament_score = score
        return combo

    def test_no_bonus_below_threshold(self):
        """Strategy profitable in only 2 TFs gets no bonus."""
        combos = [
            self._make_combo("rsi", "5m", 100.0),
            self._make_combo("rsi", "15m", 80.0),
            self._make_combo("rsi", "1h", -50.0, pnl=-500.0),
        ]
        original_scores = [c.tournament_score for c in combos]
        apply_cross_timeframe_bonus(combos, min_profitable_tfs=3)
        # No change — only 2 profitable TFs
        assert [c.tournament_score for c in combos] == original_scores

    def test_bonus_applied_when_enough_tfs(self):
        """Strategy profitable in 3+ TFs gets a bonus."""
        combos = [
            self._make_combo("rsi", "5m", 100.0),
            self._make_combo("rsi", "15m", 80.0),
            self._make_combo("rsi", "30m", 90.0),
        ]
        apply_cross_timeframe_bonus(combos, min_profitable_tfs=3)
        # All combos should have gotten a bonus
        assert combos[0].tournament_score > 100.0
        assert combos[1].tournament_score > 80.0
        assert combos[2].tournament_score > 90.0

    def test_bonus_is_consistent_across_group(self):
        """All TFs of the same strategy get the same bonus."""
        combos = [
            self._make_combo("rsi", "5m", 100.0),
            self._make_combo("rsi", "15m", 80.0),
            self._make_combo("rsi", "30m", 90.0),
            self._make_combo("macd", "5m", 50.0),  # different strategy
        ]
        apply_cross_timeframe_bonus(combos, min_profitable_tfs=3)

        rsi_bonus = combos[0].tournament_score - 100.0
        assert combos[1].tournament_score == pytest.approx(80.0 + rsi_bonus)
        assert combos[2].tournament_score == pytest.approx(90.0 + rsi_bonus)
        # MACD only has 1 profitable TF — no bonus
        assert combos[3].tournament_score == 50.0

    def test_lower_variance_gives_bigger_bonus(self):
        """More consistent scores (lower variance) → bigger bonus."""
        # Consistent strategy: scores close together
        consistent = [
            self._make_combo("rsi", "5m", 100.0),
            self._make_combo("rsi", "15m", 102.0),
            self._make_combo("rsi", "30m", 101.0),
        ]
        apply_cross_timeframe_bonus(consistent, min_profitable_tfs=3)
        consistent_bonus = consistent[0].tournament_score - 100.0

        # Inconsistent strategy: scores spread out
        inconsistent = [
            self._make_combo("macd", "5m", 200.0),
            self._make_combo("macd", "15m", 50.0),
            self._make_combo("macd", "30m", 10.0),
        ]
        apply_cross_timeframe_bonus(inconsistent, min_profitable_tfs=3)
        inconsistent_bonus = inconsistent[0].tournament_score - 200.0

        assert consistent_bonus > inconsistent_bonus

    def test_empty_list(self):
        """No crash on empty list."""
        apply_cross_timeframe_bonus([], min_profitable_tfs=3)
