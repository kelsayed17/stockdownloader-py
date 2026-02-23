# App Package Restructure — Design

**Date:** 2026-02-23
**Scope:** Split `pipeline_app.py` (949 lines) into a `pipeline/` subpackage, split `tournament/stages.py` (866 lines) into core vs. advanced stages, consolidate duplicated helpers.

## Problem

### Monster File: `pipeline_app.py` (949 lines)

Mixes 6 responsibilities: data model, utility helpers, 4 pipeline stages (baseline, optimize, re-backtest, walk-forward), holistic report formatting, and CLI/main entry point. Follows the same anti-pattern that `tournament/stages.py` already solved by splitting into `cli.py` + `helpers.py` + `stages.py` — except pipeline never got that treatment.

### Monster File: `tournament/stages.py` (866 lines)

All 7 tournament stages crammed into one file. The first 3 stages (round-robin, optimization, walk-forward) are the core pipeline; the last 4 (regime analysis, Monte Carlo, bracket, portfolio) are advanced analysis stages that were added later. Natural split: core vs. advanced.

### Triple Duplication of `_status_label`

Identical 7-line function defined in 3 places:
- `pipeline_app.py` line 706
- `optimize_app.py` line 313
- `tournament/helpers.py` line 27

### Double Duplication of `_box_title`

Identical 6-line function defined in 2 places:
- `pattern_discovery_app.py` line 83
- `tournament/helpers.py` line 18

## Approach

1. **Split `pipeline_app.py`** into `pipeline/` subpackage following the existing `tournament/` pattern
2. **Split `tournament/stages.py`** into core + advanced stages
3. **Consolidate `_status_label` and `_box_title`** into `app_helpers.py` (the canonical shared helpers module)
4. **Wire all callers** to import from the new canonical locations

## New Layout

```
app/
├── app_helpers.py               ← MODIFY: add status_label(), box_title()
│
├── pipeline/                    ← NEW subpackage (replaces pipeline_app.py)
│   ├── __init__.py              ← NEW: re-export main(), SlotResult, _OPTIMIZE_MODES
│   ├── models.py                ← NEW: SlotResult dataclass (~60 lines)
│   ├── helpers.py               ← NEW: _print_fn, _load_intraday_data, _load_daily_data, _unique_days (~60 lines)
│   ├── stages.py                ← NEW: 4 stage functions + workers (~540 lines)
│   ├── report.py                ← NEW: _print_holistic_report (~100 lines)
│   └── cli.py                   ← NEW: main(), _run_pipeline(), argparse (~130 lines)
│
├── pipeline_app.py              ← SLIM: re-exports for backward compat (~15 lines)
│
├── tournament/
│   ├── __init__.py              ← UNCHANGED
│   ├── cli.py                   ← UNCHANGED
│   ├── helpers.py               ← MODIFY: remove _box_title/_status_label, import from app_helpers
│   ├── stages.py                ← SLIM: core stages only (~330 lines, was 866)
│   └── stages_advanced.py       ← NEW: regime, MC, bracket, portfolio (~540 lines)
│
├── optimize_app.py              ← MODIFY: remove _status_label, import from app_helpers
├── pattern_discovery_app.py     ← MODIFY: remove _box_title, import from app_helpers
└── ... (all other files unchanged)
```

## Detailed Design

### 1. `app_helpers.py` — Consolidate Shared Helpers

Add two public functions (no underscore — they're now package-level utilities):

```python
def status_label(degradation: float) -> str:
    """Map walk-forward degradation ratio to a human-readable label."""
    if degradation >= 0.8:
        return "ROBUST"
    if degradation >= 0.5:
        return "ACCEPTABLE"
    return "OVERFIT"

def box_title(title: str, width: int = 100) -> str:
    """Render a Unicode box around a title string."""
    lines = [
        "╔" + "═" * width + "╗",
        "║" + title.center(width) + "║",
        "╚" + "═" * width + "╝",
    ]
    return "\n".join(lines)
```

### 2. `pipeline/` Subpackage

**`pipeline/models.py`**: `SlotResult` dataclass — verbatim from `pipeline_app.py` lines 69-123.

**`pipeline/helpers.py`**: 4 utility functions from `pipeline_app.py`:
- `_print_fn()` (line 130)
- `_load_intraday_data()` (line 138)
- `_load_daily_data()` (line 150)
- `_unique_days()` (line 160)

**`pipeline/stages.py`**: All 4 stage functions + their worker functions:
- `_baseline_one_slot()` + `_run_baseline()` (Stage 1)
- `_optimize_one_strategy()` + `_run_optimize()` + `_run_optimize_wf()` + `_run_optimize_full()` (Stage 2)
- `_rebacktest_one_slot()` + `_run_rebacktest()` (Stage 3)
- `_validate_one_slot()` + `_run_walkforward()` (Stage 4)

Uses `status_label` from `app_helpers` instead of the inline `_status_label`.

**`pipeline/report.py`**: `_print_holistic_report()` — pure formatting.

**`pipeline/cli.py`**: `main()`, `_run_pipeline()`, `_build_parser()`, constants (`_CATEGORIES`, `_STAGES`, `_OPTIMIZE_MODES`, `_MP_CTX`).

**`pipeline/__init__.py`**: Re-exports for backward compatibility:
```python
from stockdownloader.app.pipeline.models import SlotResult
from stockdownloader.app.pipeline.cli import main, _OPTIMIZE_MODES
```

**`pipeline_app.py`** (slim): Backward compat shim:
```python
"""Backward-compatibility re-exports — use ``app.pipeline`` directly."""
from stockdownloader.app.pipeline import main, SlotResult, _OPTIMIZE_MODES  # noqa: F401
```

This ensures `from stockdownloader.app.pipeline_app import main` still works for `pyproject.toml` console scripts and tests.

### 3. `tournament/stages.py` → Split Core + Advanced

**`tournament/stages.py`** (slim, ~330 lines): Keep core stages:
- `_run_round_robin()` — Stage 1
- `_run_optimization()` — Stage 1.5
- `_run_walkforward()` — Stage 2

**`tournament/stages_advanced.py`** (new, ~540 lines): Advanced analysis:
- `_run_regime_analysis()` — Stage 2.5
- `_run_monte_carlo_stage()` — Stage 2.75
- `_run_bracket()` — Stage 3
- `_run_portfolio()` — Stage 4

**`tournament/cli.py`**: Update to import advanced stages from `stages_advanced`.

### 4. Caller Updates

| File | Change |
|------|--------|
| `tournament/helpers.py` | Remove `_box_title`, `_status_label`; import `box_title`, `status_label` from `app_helpers` |
| `optimize_app.py` | Remove `_status_label`; import `status_label` from `app_helpers` |
| `pattern_discovery_app.py` | Remove `_box_title`; import `box_title` from `app_helpers` |
| `tournament/cli.py` | Add import from `stages_advanced` for advanced stage functions |
| `tournament/stages.py` | Import `_box_title` as `box_title`, `_status_label` as `status_label` from `app_helpers` |

## External Callers — Impact Analysis

| Caller | Current Import | Impact |
|---|---|---|
| `pyproject.toml` console scripts | `stockdownloader.app.pipeline_app:main` | **NONE** — `pipeline_app.py` slim re-exports `main` |
| `tests/app/test_pipeline_app.py` | `pipeline_app.main`, `SlotResult`, `_OPTIMIZE_MODES` | **NONE** — re-exports |
| `tests/app/test_tournament_app.py` | `tournament.cli.main`, `tournament.helpers.*` | **NONE** — helpers re-exports from app_helpers |
| `ml/pipeline/stage_data.py` | `app.app_helpers.fetch_daily_data` | **NONE** — app_helpers unchanged |

## Files Changed

| File | Action | Notes |
|------|--------|-------|
| `app/app_helpers.py` | **MODIFY** | Add `status_label()`, `box_title()` |
| `app/pipeline/__init__.py` | **NEW** | Re-exports main, SlotResult, _OPTIMIZE_MODES |
| `app/pipeline/models.py` | **NEW** | SlotResult dataclass |
| `app/pipeline/helpers.py` | **NEW** | 4 utility functions |
| `app/pipeline/stages.py` | **NEW** | 4 stage functions + workers |
| `app/pipeline/report.py` | **NEW** | Holistic report formatter |
| `app/pipeline/cli.py` | **NEW** | main(), _run_pipeline(), argparse |
| `app/pipeline_app.py` | **MODIFY** | Slim re-export shim |
| `app/tournament/stages.py` | **MODIFY** | Slim to core stages only |
| `app/tournament/stages_advanced.py` | **NEW** | 4 advanced stage functions |
| `app/tournament/cli.py` | **MODIFY** | Import from stages_advanced |
| `app/tournament/helpers.py` | **MODIFY** | Remove duplication, import from app_helpers |
| `app/optimize_app.py` | **MODIFY** | Remove _status_label, import from app_helpers |
| `app/pattern_discovery_app.py` | **MODIFY** | Remove _box_title, import from app_helpers |
| `tests/app/test_app_helpers.py` | **MODIFY** | Add tests for status_label, box_title |

## What's NOT Changing

- `backtest_app.py` (556 lines) — 5 entry points but logically grouped
- `gme_analysis_app.py` (715 lines) — report formatting is long but single-purpose
- `_ml_helpers.py`, `ml_train_app.py`, `spy_ml_pipeline_app.py` — well-organized
- `monitor_app.py`, `generate_pinescript.py`, `symbol_analysis_app.py` — small and focused
- `tournament_apps.py` — two independent tournament types, not related to `tournament/`
- `value_screener_app.py` — self-contained
- `pinescript_catalog/` — declarative catalogs
- `app/__init__.py` — documentation-only (intentionally no re-exports)
