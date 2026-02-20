# Codebase Cleanup Phase 2 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Commit Phase 1 cleanup, fix stale references, consolidate hardcoded paths into constants, convert print→logging in app modules, and split oversized files into modules.

**Architecture:** Mechanical refactoring — no behavior changes. Every task ends with `pytest` green. Constants centralized in `util/constants.py`. Large files split along natural function-group boundaries.

**Tech Stack:** Python 3.11+, pytest, logging stdlib

---

## Task 1: Commit Phase 1 Cleanup

All 262 uncommitted changes from the previous session's 7-phase cleanup need to be committed.

**Step 1: Stage and commit all pending changes**

```bash
cd /Users/kelsayed/Documents/GitHub/stockdownloader-py/.claude/worktrees/vigorous-easley
git add -A
git commit -m "refactor: massive codebase cleanup and reorganization

- Delete 5 redundant scripts (duplicated by CLI apps)
- Archive 8 research scripts to scripts/archive/
- Reorganize data by symbol: data/spy/5m_bars.csv
- Reorganize output by symbol: pinescript/{general,spy,gme,composite}, models/spy, patterns/spy
- Consolidate docs into docs/reports/, docs/gme/, docs/pinescript/
- Delete stale ML pipeline experiments, beta PineScript, deprecated data
- Update all path references across 30+ files
- Add data/cache/ to .gitignore
- Add ML Oversold strategy (ml-oversold) with PineScript mode
- Reformat JSON configs to consistent 4-space indent
- Update README with new directory structure

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Step 2: Verify commit succeeded**

```bash
git log --oneline -1
git status --short  # should be clean or only untracked
```

**Step 3: Run tests to confirm baseline**

```bash
pytest tests/ --ignore=tests/data/test_sec_ownership_client.py -x -q
```
Expected: 2927 passed

---

## Task 2: Fix Stale Script Reference

**Files:**
- Modify: `src/stockdownloader/strategy/intraday/ml_oversold_strategy.py:64-65`

**Step 1: Fix the stale reference**

Change line 65 from:
```python
"Run scripts/spy_ml_train.py first.",
```
to:
```python
"Run `ml-train` CLI command first.",
```

**Step 2: Run tests**

```bash
pytest tests/strategy/test_ml_oversold_strategy.py -v
```
Expected: all pass

**Step 3: Commit**

```bash
git add src/stockdownloader/strategy/intraday/ml_oversold_strategy.py
git commit -m "fix: update stale script reference to ml-train CLI command

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Expand constants.py with Path Constants

**Files:**
- Modify: `src/stockdownloader/util/constants.py`

**Step 1: Add new path constants after line 32**

```python
#: Default models directory.
DEFAULT_MODELS_DIR: Path = DEFAULT_OUTPUT_DIR / "models"

#: Default PineScript output directory.
DEFAULT_PINESCRIPT_DIR: Path = DEFAULT_OUTPUT_DIR / "pinescript"

#: Default ML pipeline output directory.
DEFAULT_ML_PIPELINE_DIR: Path = DEFAULT_OUTPUT_DIR / "ml_pipeline"

#: Default alert history file.
DEFAULT_ALERT_HISTORY: Path = DEFAULT_OUTPUT_DIR / "alert_history.json"

#: Default pattern catalog directory.
DEFAULT_PATTERNS_DIR: Path = DEFAULT_OUTPUT_DIR / "patterns"
```

**Step 2: Run tests**

```bash
pytest tests/ --ignore=tests/data/test_sec_ownership_client.py -x -q
```
Expected: all pass (additive-only change)

**Step 3: Commit**

```bash
git add src/stockdownloader/util/constants.py
git commit -m "refactor: add path constants for models, pinescript, patterns, alerts

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Replace Hardcoded Paths in App Modules

**Files to modify** (replace hardcoded strings with constants imports):
- `src/stockdownloader/app/walk_forward_app.py` — `"data/spy/5m_bars.csv"` → `str(DEFAULT_DATA_FILE)`
- `src/stockdownloader/app/pipeline_app.py` — `"data/spy/5m_bars.csv"` → `str(DEFAULT_DATA_FILE)`
- `src/stockdownloader/app/optimize_app.py` — `"data/spy/5m_bars.csv"` → `str(DEFAULT_DATA_FILE)`
- `src/stockdownloader/app/generate_pinescript.py` — `"output/pinescript"` → `str(DEFAULT_PINESCRIPT_DIR)`
- `src/stockdownloader/app/monitor_app.py` — `"output/alert_history.json"` → `str(DEFAULT_ALERT_HISTORY)`
- `src/stockdownloader/app/ml_pipeline_app.py` — `"output/ml_pipeline"` → `str(DEFAULT_ML_PIPELINE_DIR)`
- `src/stockdownloader/app/ml_train_app.py` — `"output/models/spy"` → `str(DEFAULT_MODELS_DIR / "spy")`
- `src/stockdownloader/app/_ml_helpers.py` — `"output/ml_pipeline"` → `str(DEFAULT_ML_PIPELINE_DIR)`
- `src/stockdownloader/app/gme_ml_pipeline_app.py` — `"output/ml_pipeline"` → `str(DEFAULT_ML_PIPELINE_DIR)`
- `src/stockdownloader/app/spy_ml_pipeline_app.py` — `"output/spy_ml_pipeline"` → `str(DEFAULT_ML_PIPELINE_DIR)`

Each file needs `from stockdownloader.util.constants import <relevant constants>` added to imports.

**Step 1: Make all replacements**

For each file, add the import and replace the hardcoded string. Use `str()` wrapper since argparse defaults need strings.

**Step 2: Run full test suite**

```bash
pytest tests/ --ignore=tests/data/test_sec_ownership_client.py -x -q
```
Expected: all pass

**Step 3: Commit**

```bash
git add src/stockdownloader/app/*.py
git commit -m "refactor: replace hardcoded paths with constants in app modules

10 app files now import from util.constants instead of using
hardcoded 'data/spy/5m_bars.csv' and 'output/...' strings.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Convert print() to logging in App Modules

**Scope:** The 428 print() calls in app/ are mostly intentional CLI output (progress bars, results tables). Only convert diagnostic/debug prints — NOT user-facing output.

**Target files** (ones with diagnostic prints, not CLI output):
- `src/stockdownloader/analysis/pattern_analyzer.py` — 7 prints → logging
- `src/stockdownloader/analysis/options_gamma_analyzer.py` — 3 prints → logging
- `src/stockdownloader/app/optimize_app.py` — convert debug prints to logger.debug

**Step 1: For each file:**
1. Add `import logging` and `logger = logging.getLogger(__name__)` at top
2. Replace diagnostic `print(...)` with `logger.info(...)` or `logger.debug(...)`
3. Leave user-facing CLI output as `print()` — these are intentional

**Step 2: Run tests**

```bash
pytest tests/ --ignore=tests/data/test_sec_ownership_client.py -x -q
```

**Step 3: Commit**

```bash
git add src/stockdownloader/analysis/*.py src/stockdownloader/app/optimize_app.py
git commit -m "refactor: convert diagnostic print() to logging in analysis modules

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Split tournament_app.py (1,310 lines → modules)

**Current structure:** 16 functions all in one file. Natural groupings:
- Tournament stages: `_run_round_robin`, `_run_optimization`, `_run_walkforward`, `_run_regime_analysis`, `_run_monte_carlo_stage`, `_run_bracket`, `_run_portfolio`
- Utilities: `_box_title`, `_status_label`, `_load_data`, `_build_skip_set`
- CLI: `_parse_args`, `main`, `main_walkforward`, `main_baseline`, `main_greedy`

**Target structure:**
```
src/stockdownloader/app/tournament/
├── __init__.py          ← re-exports main, main_walkforward, etc.
├── stages.py            ← 7 stage functions
├── helpers.py           ← _box_title, _status_label, _load_data, _build_skip_set
└── cli.py               ← _parse_args, main, main_walkforward, main_baseline, main_greedy
```

**Step 1: Create tournament/ package**

Create `__init__.py`, `stages.py`, `helpers.py`, `cli.py` by moving functions from `tournament_app.py`.

**Step 2: Update `tournament_app.py` to thin re-export wrapper**

Replace contents with:
```python
"""Grand tournament — re-exports from tournament package."""
from stockdownloader.app.tournament.cli import (
    main,
    main_baseline,
    main_greedy,
    main_walkforward,
)

__all__ = ["main", "main_walkforward", "main_baseline", "main_greedy"]
```

**Step 3: Verify pyproject.toml entry points still work**

The entry points reference `stockdownloader.app.tournament_app:main` — this still works because the re-export wrapper provides `main`.

**Step 4: Run tests**

```bash
pytest tests/ --ignore=tests/data/test_sec_ownership_client.py -x -q
```

**Step 5: Commit**

```bash
git add src/stockdownloader/app/tournament/ src/stockdownloader/app/tournament_app.py
git commit -m "refactor: split tournament_app.py into tournament/ package

1,310 lines → stages.py, helpers.py, cli.py with thin re-export wrapper.
Entry points unchanged.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Split pinescript_generator.py (1,213 lines → modules)

**Current structure:** One massive class `PineScriptGenerator` + 3 standalone functions.

**Target structure:**
```
src/stockdownloader/util/pinescript/
├── __init__.py          ← re-exports PineScriptGenerator
├── generator.py         ← main PineScriptGenerator class (keep as-is but with imports)
├── converters.py        ← mode_to_strategy, strategy_to_mode, _indicators_as_lines
```

Actually, the class itself is 1080 lines — splitting it further requires understanding method groups. A simpler approach:

**Step 1: Extract standalone functions**

Move `_indicators_as_lines`, `mode_to_strategy`, `strategy_to_mode` to `converters.py`.

**Step 2: Create `pinescript/` package, move generator**

Move `PineScriptGenerator` class to `pinescript/generator.py`, update imports.

**Step 3: Make `pinescript_generator.py` a thin re-export**

```python
"""PineScript generation — re-exports from pinescript package."""
from stockdownloader.util.pinescript.generator import PineScriptGenerator
from stockdownloader.util.pinescript.converters import mode_to_strategy, strategy_to_mode

__all__ = ["PineScriptGenerator", "mode_to_strategy", "strategy_to_mode"]
```

**Step 4: Run tests**

```bash
pytest tests/ --ignore=tests/data/test_sec_ownership_client.py -x -q
```

**Step 5: Commit**

```bash
git add src/stockdownloader/util/pinescript/ src/stockdownloader/util/pinescript_generator.py
git commit -m "refactor: split pinescript_generator.py into pinescript/ package

1,213 lines → generator.py + converters.py with thin re-export wrapper.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Split gme_analyzer.py (1,187 lines → modules)

**Current structure:** 9 dataclasses + 16 analysis functions, naturally grouped by analysis type.

**Target structure:**
```
src/stockdownloader/analysis/gme/
├── __init__.py          ← re-exports all public functions and dataclasses
├── models.py            ← 9 dataclasses (FilingImpact, KeyPeriod, OptionsAnalysis, etc.)
├── event_study.py       ← run_event_study, _compute_log_returns, _autocorrelation, _ols_market_model
├── regime.py            ← detect_volatility_regimes, detect_structural_breaks, detect_key_periods
├── distribution.py      ← analyze_return_distribution, compute_volume_profile, compute_price_statistics
├── options.py           ← analyze_options_chain, _compute_max_pain, _highest_oi_strike, etc.
└── filings.py           ← correlate_filings_with_price, _find_nearest_index
```

**Step 1: Create gme/ package with sub-modules**

Move functions into their respective files by analysis group.

**Step 2: Make `gme_analyzer.py` a thin re-export**

**Step 3: Run tests**

```bash
pytest tests/ --ignore=tests/data/test_sec_ownership_client.py -x -q
```

**Step 4: Commit**

```bash
git add src/stockdownloader/analysis/gme/ src/stockdownloader/analysis/gme_analyzer.py
git commit -m "refactor: split gme_analyzer.py into gme/ analysis package

1,187 lines → models.py, event_study.py, regime.py, distribution.py,
options.py, filings.py with thin re-export wrapper.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 9: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ --ignore=tests/data/test_sec_ownership_client.py -x -q
```
Expected: 2927+ tests pass

**Step 2: Verify imports work**

```bash
python3 -c "from stockdownloader.app.tournament_app import main; print('tournament OK')"
python3 -c "from stockdownloader.util.pinescript_generator import PineScriptGenerator; print('pinescript OK')"
python3 -c "from stockdownloader.analysis.gme_analyzer import run_event_study; print('gme OK')"
python3 -c "from stockdownloader.strategy.intraday import MLOversoldStrategy; print('ml OK')"
```

**Step 3: Verify no stale references remain**

```bash
grep -r "scripts/spy_ml_train" src/  # should return nothing
grep -r "spy_5m_bars.csv" src/       # should return nothing
```

---

## Summary

| Task | What | Risk | Lines Affected |
|------|------|------|----------------|
| 1 | Commit Phase 1 | None (already tested) | 262 files |
| 2 | Fix stale ref | Trivial | 1 line |
| 3 | Add constants | Additive only | ~15 lines |
| 4 | Replace hardcoded paths | Low (string defaults) | ~30 lines across 10 files |
| 5 | Print → logging | Low | ~15 lines across 3 files |
| 6 | Split tournament_app | Medium (import paths) | 1,310 lines reorganized |
| 7 | Split pinescript_generator | Medium (import paths) | 1,213 lines reorganized |
| 8 | Split gme_analyzer | Medium (import paths) | 1,187 lines reorganized |
| 9 | Final verification | None | 0 lines |
