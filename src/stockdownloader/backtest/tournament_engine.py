"""Multi-timeframe strategy tournament engine.

Core analysis functions, elimination brackets, and portfolio analysis
for running strategy x timeframe combinatorial backtests.

Usage::

    from stockdownloader.backtest.tournament_engine import (
        ComboKey, ComboResult, run_combo_backtest, run_combo_optimize,
        run_combo_rebacktest, run_combo_walkforward,
        run_elimination_bracket, apply_cross_timeframe_bonus,
    )
"""
from __future__ import annotations

import math

from stockdownloader.backtest.tournament_analysis import (
    _compute_percentiles,
    _compute_regime_bonus,
    _equity_max_drawdown,
    apply_cross_timeframe_bonus,
    classify_timeframe_bars,
    run_monte_carlo,
    run_regime_analysis,
)
from stockdownloader.backtest.tournament_models import (
    ComboKey,
    ComboResult,
    MatchResult,
    MonteCarloPercentiles,
    MonteCarloResult,
    RegimeAnalysis,
    RegimeTradeStats,
    TournamentResult,
)
from stockdownloader.backtest.tournament_workers import (
    run_combo_backtest,
    run_combo_optimize,
    run_combo_rebacktest,
    run_combo_walkforward,
)
from stockdownloader.core.config import INITIAL_CAPITAL, RISK_PER_TRADE


# ======================================================================
# Elimination bracket
# ======================================================================


def _round_name(round_num: int, total_rounds: int) -> str:
    """Human-readable round name."""
    remaining = total_rounds - round_num
    if remaining == 1:
        return "FINAL"
    if remaining == 2:
        return "SEMIFINAL"
    if remaining == 3:
        return "QUARTERFINAL"
    return f"Round {round_num + 1}"


def run_elimination_bracket(
    combos: list[ComboResult],
    bracket_size: int = 16,
) -> tuple[list[MatchResult], ComboKey | None]:
    """Run single-elimination bracket on top combos.

    Parameters
    ----------
    combos:
        All combo results, sorted by tournament_score descending.
    bracket_size:
        Number of combos to include (must be power of 2).

    Returns
    -------
    (matches, champion_key)
    """
    # Ensure bracket_size is a power of 2
    if not combos:
        return [], None

    actual = min(bracket_size, len(combos))
    if actual < 2:
        return [], combos[0].key

    bracket_exp = max(1, int(math.log2(actual)))
    actual = 2 ** bracket_exp

    # Seed: #1 vs #N, #2 vs #N-1, etc.
    seeded = combos[:actual]
    current_round: list[ComboResult] = []
    for i in range(actual // 2):
        current_round.append(seeded[i])
        current_round.append(seeded[actual - 1 - i])

    matches: list[MatchResult] = []
    total_rounds = bracket_exp
    round_num = 0

    while len(current_round) > 1:
        next_round: list[ComboResult] = []
        rname = _round_name(round_num, total_rounds)

        for i in range(0, len(current_round), 2):
            a = current_round[i]
            b = current_round[i + 1]

            if a.tournament_score >= b.tournament_score:
                winner, loser = a, b
            else:
                winner, loser = b, a

            matches.append(MatchResult(
                winner=winner.key,
                loser=loser.key,
                winner_score=winner.tournament_score,
                loser_score=loser.tournament_score,
                round_num=round_num,
                round_name=rname,
            ))
            next_round.append(winner)

        current_round = next_round
        round_num += 1

    champion = current_round[0].key if current_round else None
    return matches, champion
