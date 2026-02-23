# Backtest Package Restructure — Design

**Date:** 2026-02-23
**Scope:** Split `tournament_engine.py` (932 lines) by responsibility, fix report formatter duplication, complete `__init__.py` exports.

## Problem

### Monster File: `tournament_engine.py` (932 lines)

This file crams 4 distinct responsibilities into a single module:

| Responsibility | Lines | Contents |
|---|---|---|
| Data models | 44–187 | 8 dataclasses: `RegimeTradeStats`, `RegimeAnalysis`, `MonteCarloPercentiles`, `MonteCarloResult`, `ComboKey`, `ComboResult`, `MatchResult`, `TournamentResult` |
| Worker functions | 194–449 | 4 process-safe functions for multiprocessing: `run_combo_backtest`, `run_combo_optimize`, `run_combo_rebacktest`, `run_combo_walkforward` |
| Analysis | 452–845 | Regime analysis, Monte Carlo simulation, cross-TF bonus: `classify_timeframe_bars`, `_compute_regime_bonus`, `run_regime_analysis`, `_compute_percentiles`, `_equity_max_drawdown`, `run_monte_carlo`, `apply_cross_timeframe_bonus` |
| Bracket logic | 848–931 | Elimination brackets: `_round_name`, `run_elimination_bracket` |

All 4 areas depend on the models (area 1) but have no other inter-dependencies. Every function except `ComboKey.label`, `ComboResult.best_result`, and `ComboResult.compute_tournament_score` is a standalone module-level function.

### Constant Re-Export Leak

`tournament_engine.py` imports `INITIAL_CAPITAL` and `RISK_PER_TRADE` from `util.config` and re-exports them implicitly. Two callers (`app/tournament/stages.py` and `app/pattern_discovery_app.py`) import these constants from `tournament_engine` instead of `util.config` directly.

### Report Formatter Duplication

| File | Function | Body |
|---|---|---|
| `report_formatter.py` (line 50) | `scale2(value)` | `value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)` |
| `exit_tournament_report_formatter.py` (line 15) | `_s2(value)` | `value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)` |

Identical logic, different names. `exit_tournament_report_formatter.py` also has `_s3()` (3 decimal places) with no counterpart.

### Incomplete `__init__.py` Exports

Currently exports 11 names. Missing:
- `tournament_engine` types (`ComboKey`, `ComboResult`, `TournamentResult`, etc.)
- `WalkForwardValidator`, `WalkForwardResult`
- `WalkForwardOptimizer`, `WFOptResult`
- `StrategyOptimizer`, `DailyStrategyOptimizer`
- `CombinatorialTester`, `CombinatorialConfig`
- `optimizer_scoring` (`score_v2`)
- `portfolio_analyzer`

## Approach: Extract + Split

Same pattern as data/ and util/ restructures — extract by responsibility, atomic moves, no backward-compat shims.

## New Layout

```
backtest/
├── tournament_models.py        ← NEW: 8 dataclasses (~150 lines)
├── tournament_workers.py       ← NEW: 4 process-safe worker functions (~260 lines)
├── tournament_analysis.py      ← NEW: regime, Monte Carlo, cross-TF bonus (~400 lines)
├── tournament_engine.py        ← SLIM: bracket logic + re-exports for callers (~90 lines)
├── report_helpers.py           ← NEW: scale2(), scale3() shared rounding (~20 lines)
├── report_formatter.py         ← MODIFY: import scale2 from report_helpers
├── exit_tournament_report_formatter.py ← MODIFY: import from report_helpers
├── __init__.py                 ← MODIFY: add missing exports
└── ... (all other files unchanged)
```

## Detailed Design

### 1. `tournament_models.py` — Data Models

All 8 dataclasses extracted verbatim:

```python
# Dataclasses (no business logic, just containers + properties)
RegimeTradeStats       # regime performance stats
RegimeAnalysis         # per-combo regime breakdown
MonteCarloPercentiles  # percentile thresholds
MonteCarloResult       # full MC result
ComboKey               # strategy×timeframe identifier (has .label property)
ComboResult            # backtest + optimization results (has .best_result, .compute_tournament_score)
MatchResult            # head-to-head bracket match
TournamentResult       # final tournament output
```

Imports: `BacktestResult`, `BaseBacktestResult` from `backtest_result`; `WalkForwardResult` from `walk_forward`; `MarketRegime` from strategy layer.

### 2. `tournament_workers.py` — Process-Safe Workers

4 standalone functions that run in multiprocessing pools:

```python
run_combo_backtest(key, data) -> ComboResult
run_combo_optimize(key, category, data) -> ComboResult
run_combo_rebacktest(key, category, optimized_kwargs, data) -> ComboResult
run_combo_walkforward(key, data, optimized_kwargs=None) -> ComboResult
```

Imports: models from `tournament_models`; engine/scoring/strategy from existing modules.

### 3. `tournament_analysis.py` — Regime + Monte Carlo + Cross-TF

7 functions (4 public, 3 private):

```python
# Regime analysis
classify_timeframe_bars(data) -> dict
_compute_regime_bonus(per_regime) -> float  # private helper
run_regime_analysis(key, trades, regime_at_bar) -> RegimeAnalysis

# Monte Carlo
_compute_percentiles(values) -> MonteCarloPercentiles  # private helper
_equity_max_drawdown(equity) -> float                  # private helper
run_monte_carlo(key, trade_pnls, initial_capital, ...) -> MonteCarloResult

# Cross-timeframe
apply_cross_timeframe_bonus(combos, min_profitable_tfs=3) -> None
```

### 4. `tournament_engine.py` — Slim Bracket + Re-Exports

Keeps only bracket logic plus convenience re-exports so callers that import many names from `tournament_engine` continue to work without changes:

```python
# Bracket logic (stays here — this IS the "engine")
_round_name(round_num, total_rounds) -> str
run_elimination_bracket(combos, bracket_size=16) -> TournamentResult

# Re-exports for backward compat (callers import from tournament_engine)
from stockdownloader.backtest.tournament_models import *    # all 8 model classes
from stockdownloader.backtest.tournament_workers import *   # 4 worker functions
from stockdownloader.backtest.tournament_analysis import *  # 4 public analysis functions
from stockdownloader.util.config import INITIAL_CAPITAL, RISK_PER_TRADE
```

This means **zero caller changes** — `app/tournament/stages.py` and all other callers keep their existing `from stockdownloader.backtest.tournament_engine import ...` imports unchanged. The re-exports are explicit star-imports from the 3 new modules.

### 5. `report_helpers.py` — Shared Rounding Utilities

```python
from decimal import Decimal, ROUND_HALF_UP

def scale2(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def scale3(value: Decimal) -> Decimal:
    """Round a Decimal to 3 decimal places."""
    return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
```

### 6. Report Formatter Updates

- `report_formatter.py`: Remove inline `scale2`, add `from stockdownloader.backtest.report_helpers import scale2`
- `exit_tournament_report_formatter.py`: Remove inline `_s2` and `_s3`, add `from stockdownloader.backtest.report_helpers import scale2 as _s2, scale3 as _s3`

The `exit_tournament_report_formatter.py` keeps its private naming convention via aliased imports — no need to rename all internal call sites.

### 7. `__init__.py` — Complete Exports

Add all missing public types and modules:

```python
# Tournament
from stockdownloader.backtest.tournament_models import (
    ComboKey, ComboResult, MatchResult, TournamentResult,
    RegimeTradeStats, RegimeAnalysis,
    MonteCarloPercentiles, MonteCarloResult,
)
from stockdownloader.backtest import tournament_engine

# Walk-forward
from stockdownloader.backtest.walk_forward import WalkForwardValidator, WalkForwardResult
from stockdownloader.backtest.walk_forward_optimizer import WalkForwardOptimizer, WFOptResult

# Optimizers
from stockdownloader.backtest.strategy_optimizer import StrategyOptimizer
from stockdownloader.backtest.daily_strategy_optimizer import DailyStrategyOptimizer
from stockdownloader.backtest.combinatorial_tester import CombinatorialTester, CombinatorialConfig

# Scoring
from stockdownloader.backtest.optimizer_scoring import score_v2
```

## External Callers — Impact Analysis

| Caller | Imports From tournament_engine | Changes Needed |
|---|---|---|
| `app/tournament/stages.py` | 14 names (models, workers, analysis, bracket) | **NONE** — re-exports in tournament_engine |
| `app/tournament/cli.py` | 2 names (TournamentResult, apply_cross_timeframe_bonus) | **NONE** |
| `app/tournament/helpers.py` | 2 names (ComboKey, ComboResult) | **NONE** |
| `app/pattern_discovery_app.py` | 5 names (INITIAL_CAPITAL, RISK_PER_TRADE, run_monte_carlo, classify_timeframe_bars, ComboKey) | **NONE** |
| `tests/backtest/test_tournament_engine.py` | 10 names | **NONE** |
| `tests/backtest/test_tournament_regime.py` | 5 names (incl. private `_compute_regime_bonus`) | **NONE** |
| `tests/backtest/test_tournament_monte_carlo.py` | 6 names (incl. private `_compute_percentiles`, `_equity_max_drawdown`) | **NONE** |
| `tests/app/test_tournament_app.py` | 2 names | **NONE** |
| `backtest/portfolio_analyzer.py` | 1 name (TYPE_CHECKING) | **NONE** |

All callers unaffected thanks to re-exports.

## Files Changed

| File | Action | Notes |
|------|--------|-------|
| `backtest/tournament_models.py` | **NEW** | 8 dataclasses (~150 lines) |
| `backtest/tournament_workers.py` | **NEW** | 4 worker functions (~260 lines) |
| `backtest/tournament_analysis.py` | **NEW** | Regime + MC + cross-TF (~400 lines) |
| `backtest/tournament_engine.py` | **MODIFY** | Slim to bracket + re-exports (~90 lines) |
| `backtest/report_helpers.py` | **NEW** | scale2(), scale3() (~20 lines) |
| `backtest/report_formatter.py` | **MODIFY** | Import scale2 from report_helpers |
| `backtest/exit_tournament_report_formatter.py` | **MODIFY** | Import from report_helpers |
| `backtest/__init__.py` | **MODIFY** | Add missing exports |
| `tests/backtest/test_report_helpers.py` | **NEW** | Tests for scale2, scale3 |

## What's NOT Changing

- `backtest_engine.py`, `backtest_result.py` — foundational, stable
- `intraday_backtest_engine.py` — self-contained
- `options_backtest_engine.py` — self-contained
- `optimizer_base.py`, `optimizer_scoring.py` — shared infrastructure
- `strategy_optimizer.py`, `daily_strategy_optimizer.py` — self-contained
- `walk_forward.py`, `walk_forward_optimizer.py` — self-contained
- `combinatorial_tester.py` — self-contained
- `portfolio_analyzer.py` — self-contained
- `exit_tournament_engine.py`, `exit_tournament_result.py` — self-contained
- All external callers — **zero changes** thanks to re-exports
