# GME Consolidation & Non-Src Cleanup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate all GME-specific code into `src/stockdownloader/gme/` and clean up the non-src directory structure (delete archived scripts, completed plans, merge reports/).

**Architecture:** Move 6 analysis files, 2 app files, 1 pinescript file into a new `gme/` top-level package. Promote 2 scripts from `scripts/` into the package. Rewrite 11 files' imports via the existing `scripts/rewrite_imports.py` tool. Move 3 test files into `tests/gme/`. Delete `scripts/archive/` (9 files), `docs/plans/` (34 files), merge `reports/GME/` into `docs/reports/GME/`.

**Tech Stack:** Python, `scripts/rewrite_imports.py` (JSON-driven import rewriter), `pyproject.toml` entry points

---

### Task 1: Create `gme/` package scaffold and move analysis files

**Files:**
- Create: `src/stockdownloader/gme/__init__.py`
- Create: `src/stockdownloader/gme/analysis/__init__.py`
- Move: `src/stockdownloader/analysis/gme/models.py` → `src/stockdownloader/gme/analysis/models.py`
- Move: `src/stockdownloader/analysis/gme/regime.py` → `src/stockdownloader/gme/analysis/regime.py`
- Move: `src/stockdownloader/analysis/gme/event_study.py` → `src/stockdownloader/gme/analysis/event_study.py`
- Move: `src/stockdownloader/analysis/gme/distribution.py` → `src/stockdownloader/gme/analysis/distribution.py`
- Move: `src/stockdownloader/analysis/gme/options.py` → `src/stockdownloader/gme/analysis/options.py`
- Delete: `src/stockdownloader/analysis/gme/` (entire directory, after moving files)

**Step 1: Create directories and `__init__.py` files**

```bash
cd /Users/kelsayed/Documents/GitHub/stockdownloader-py/.claude/worktrees/vigorous-easley
mkdir -p src/stockdownloader/gme/analysis
```

Create `src/stockdownloader/gme/__init__.py`:
```python
"""GME (GameStop) analysis, pipelines, and tooling."""
```

Create `src/stockdownloader/gme/analysis/__init__.py` — copy the exact content from `src/stockdownloader/analysis/gme/__init__.py`, but change every `stockdownloader.analysis.gme.` prefix to `stockdownloader.gme.analysis.`:

```python
"""GME deep analysis -- modular analysis package.

Re-exports all public dataclasses and analysis functions so callers
can use ``from stockdownloader.gme.analysis import run_event_study``.
"""

from stockdownloader.gme.analysis.distribution import (
    analyze_return_distribution,
    compute_price_statistics,
    compute_volume_profile,
)
from stockdownloader.gme.analysis.event_study import (
    correlate_filings_with_price,
    run_event_study,
)
from stockdownloader.gme.analysis.models import (
    EventStudyResult,
    FilingImpact,
    KeyPeriod,
    OptionsAnalysis,
    PriceStatistics,
    ReturnDistribution,
    StructuralBreak,
    VolatilityRegime,
    VolumeProfile,
)
from stockdownloader.gme.analysis.options import analyze_options_chain
from stockdownloader.gme.analysis.regime import (
    detect_key_periods,
    detect_structural_breaks,
    detect_volatility_regimes,
)

__all__ = [
    # Models
    "EventStudyResult",
    "FilingImpact",
    "KeyPeriod",
    "OptionsAnalysis",
    "PriceStatistics",
    "ReturnDistribution",
    "StructuralBreak",
    "VolatilityRegime",
    "VolumeProfile",
    # Event study
    "run_event_study",
    # Regime / structural
    "detect_volatility_regimes",
    "detect_structural_breaks",
    "detect_key_periods",
    # Distribution / volume / stats
    "analyze_return_distribution",
    "compute_volume_profile",
    "compute_price_statistics",
    # Options
    "analyze_options_chain",
    # Filings
    "correlate_filings_with_price",
]
```

**Step 2: Move the 5 analysis files with `git mv`**

```bash
git mv src/stockdownloader/analysis/gme/models.py src/stockdownloader/gme/analysis/models.py
git mv src/stockdownloader/analysis/gme/regime.py src/stockdownloader/gme/analysis/regime.py
git mv src/stockdownloader/analysis/gme/event_study.py src/stockdownloader/gme/analysis/event_study.py
git mv src/stockdownloader/analysis/gme/distribution.py src/stockdownloader/gme/analysis/distribution.py
git mv src/stockdownloader/analysis/gme/options.py src/stockdownloader/gme/analysis/options.py
```

**Step 3: Delete the old `analysis/gme/` directory**

```bash
rm src/stockdownloader/analysis/gme/__init__.py
rmdir src/stockdownloader/analysis/gme
git add src/stockdownloader/analysis/gme/
```

**Step 4: Rewrite internal imports in moved files**

The 5 moved files reference each other using `stockdownloader.analysis.gme.*` paths. Create a JSON mapping and run the rewrite script:

```bash
cat > /tmp/gme_imports.json << 'EOF'
{
  "stockdownloader.analysis.gme.models": "stockdownloader.gme.analysis.models",
  "stockdownloader.analysis.gme.regime": "stockdownloader.gme.analysis.regime",
  "stockdownloader.analysis.gme.event_study": "stockdownloader.gme.analysis.event_study",
  "stockdownloader.analysis.gme.distribution": "stockdownloader.gme.analysis.distribution",
  "stockdownloader.analysis.gme.options": "stockdownloader.gme.analysis.options",
  "stockdownloader.analysis.gme": "stockdownloader.gme.analysis",
  "stockdownloader.app.gme_pipeline": "stockdownloader.gme.pipeline",
  "stockdownloader.app.gme": "stockdownloader.gme.app",
  "stockdownloader.app.pinescript_catalog.gme_prediction": "stockdownloader.gme.prediction"
}
EOF

python3 scripts/rewrite_imports.py /tmp/gme_imports.json --dry-run
```

Review the dry-run output. Then apply:
```bash
python3 scripts/rewrite_imports.py /tmp/gme_imports.json
```

**Step 5: Verify imports parse correctly**

```bash
python3 -c "from stockdownloader.gme.analysis import run_event_study; print('OK')"
python3 -c "from stockdownloader.gme.analysis.models import EventStudyResult; print('OK')"
```

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: create gme/ package, move analysis/gme/ into gme/analysis/"
```

---

### Task 2: Move app files and promote scripts into `gme/`

**Files:**
- Move: `src/stockdownloader/app/gme.py` → `src/stockdownloader/gme/app.py`
- Move: `src/stockdownloader/app/gme_pipeline.py` → `src/stockdownloader/gme/pipeline.py`
- Move: `src/stockdownloader/app/pinescript_catalog/gme_prediction.py` → `src/stockdownloader/gme/prediction.py`
- Move: `scripts/gme_options_analysis.py` → `src/stockdownloader/gme/options_analysis.py`
- Move: `scripts/regsho_targeted_backfill.py` → `src/stockdownloader/gme/regsho_backfill.py`

**Step 1: Move app files with `git mv`**

```bash
cd /Users/kelsayed/Documents/GitHub/stockdownloader-py/.claude/worktrees/vigorous-easley
git mv src/stockdownloader/app/gme.py src/stockdownloader/gme/app.py
git mv src/stockdownloader/app/gme_pipeline.py src/stockdownloader/gme/pipeline.py
git mv src/stockdownloader/app/pinescript_catalog/gme_prediction.py src/stockdownloader/gme/prediction.py
```

**Step 2: Move and fix promoted scripts**

```bash
git mv scripts/gme_options_analysis.py src/stockdownloader/gme/options_analysis.py
git mv scripts/regsho_targeted_backfill.py src/stockdownloader/gme/regsho_backfill.py
```

Fix `gme/regsho_backfill.py` — remove the `sys.path.insert(0, ...)` hack (lines 34-35) since it's now inside the package:

Delete these two lines:
```python
# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

Fix `gme/regsho_backfill.py` — update `DATA_DIR` to use the project root pattern (the file used `Path(__file__).resolve().parent.parent / "data"` which assumed it was in `scripts/`). Change to:
```python
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
```
Or better, make it a parameter of `main()` — derive from project root the same way other app modules do. Check how `app/gme.py` (now `gme/app.py`) resolves data paths and follow the same pattern.

Fix `gme/options_analysis.py` — update `DATA_DIR` from `Path(__file__).resolve().parent.parent / "data"` (assumed `scripts/`) to the correct depth. Since it's now at `src/stockdownloader/gme/options_analysis.py`, the project root is 4 parents up:
```python
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
```

**Step 3: Update `catalogs.py` import**

The import rewrite script (run in Task 1) should have already rewritten `stockdownloader.app.pinescript_catalog.gme_prediction` → `stockdownloader.gme.prediction` in `catalogs.py`. Verify:

```bash
grep "gme_prediction" src/stockdownloader/app/pinescript_catalog/catalogs.py
```

Expected: `from stockdownloader.gme.prediction import gme_prediction_strategy`

If not already rewritten, manually fix line 46 of `catalogs.py`.

**Step 4: Verify imports**

```bash
python3 -c "from stockdownloader.gme.app import main; print('OK')"
python3 -c "from stockdownloader.gme.pipeline import main; print('OK')"
python3 -c "from stockdownloader.gme.prediction import gme_prediction_strategy; print('OK')"
python3 -c "from stockdownloader.gme.options_analysis import main; print('OK')"
python3 -c "from stockdownloader.gme.regsho_backfill import main; print('OK')"
```

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move app/gme*.py and promote scripts into gme/ package"
```

---

### Task 3: Move test files and update pyproject.toml entry points

**Files:**
- Create: `tests/gme/__init__.py` (empty)
- Move: `tests/analysis/test_gme_analyzer.py` → `tests/gme/test_analyzer.py`
- Move: `tests/analysis/test_gme_analyzer_statistics.py` → `tests/gme/test_statistics.py`
- Move: `tests/pinescript/test_gme_prediction.py` → `tests/gme/test_prediction.py`
- Modify: `pyproject.toml` (entry points)

**Step 1: Create test directory and move files**

```bash
cd /Users/kelsayed/Documents/GitHub/stockdownloader-py/.claude/worktrees/vigorous-easley
mkdir -p tests/gme
touch tests/gme/__init__.py
git mv tests/analysis/test_gme_analyzer.py tests/gme/test_analyzer.py
git mv tests/analysis/test_gme_analyzer_statistics.py tests/gme/test_statistics.py
git mv tests/pinescript/test_gme_prediction.py tests/gme/test_prediction.py
```

The import rewrite script (run in Task 1) should have already rewritten imports inside these test files. Verify:

```bash
grep "stockdownloader.analysis.gme" tests/gme/*.py
grep "stockdownloader.app.pinescript_catalog.gme_prediction" tests/gme/*.py
```

Both should return no matches (all rewritten to `stockdownloader.gme.*`).

**Step 2: Update pyproject.toml entry points**

In `pyproject.toml`, change:
```
gme-ml-pipeline = "stockdownloader.app.gme_pipeline:main"
gme-analysis = "stockdownloader.app.gme:main"
```
To:
```
gme-ml-pipeline = "stockdownloader.gme.pipeline:main"
gme-analysis = "stockdownloader.gme.app:main"
```

Add two new entry points:
```
gme-options-analysis = "stockdownloader.gme.options_analysis:main"
gme-regsho-backfill = "stockdownloader.gme.regsho_backfill:main"
```

**Step 3: Run full test suite**

```bash
python3 -m pytest tests/ -x -q --tb=short
```

Expected: 3,297 passed, 44 deselected (same as baseline). If any fail, fix before proceeding.

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move GME tests to tests/gme/, update pyproject.toml entry points"
```

---

### Task 4: Delete `scripts/archive/` and `docs/plans/`

**Files:**
- Delete: `scripts/archive/` (9 archived one-off scripts)
- Delete: `docs/plans/` (34 completed plan docs)

**Step 1: Delete `scripts/archive/`**

```bash
cd /Users/kelsayed/Documents/GitHub/stockdownloader-py/.claude/worktrees/vigorous-easley
git rm -r scripts/archive/
```

**Step 2: Delete `docs/plans/`**

The current plan doc (`docs/plans/2026-02-23-gme-consolidation-and-non-src-cleanup-plan.md`) and design doc are also in this directory. They will be deleted along with the 32 other completed plan docs. This is intentional — all plan docs from the codebase overhaul are artifacts of the process, not living documentation.

```bash
git rm -r docs/plans/
```

**Step 3: Verify no imports reference archived scripts**

```bash
grep -r "scripts/archive" src/ tests/ 2>/dev/null || echo "Clean"
grep -r "scripts.archive" src/ tests/ 2>/dev/null || echo "Clean"
```

**Step 4: Commit**

```bash
git add -A
git commit -m "cleanup: delete scripts/archive/ (9 files) and docs/plans/ (34 files)"
```

---

### Task 5: Merge `reports/` into `docs/reports/` and final verification

**Files:**
- Move: `reports/GME/*` → `docs/reports/GME/*`
- Delete: `reports/` root directory
- Verify: full test suite, no stale references

**Step 1: Merge `reports/GME/` into `docs/reports/GME/`**

```bash
cd /Users/kelsayed/Documents/GitHub/stockdownloader-py/.claude/worktrees/vigorous-easley
mkdir -p docs/reports/GME
git mv reports/GME/* docs/reports/GME/
rmdir reports/GME reports
git add reports/
```

Files being moved (5):
- `reports/GME/holistic_aligned.csv`
- `reports/GME/report_data.json`
- `reports/GME/comprehensive_report.txt`
- `reports/GME/comprehensive_report.md`
- `reports/GME/analysis_short_interest_theory.md`

**Step 2: Check for references to old `reports/` path**

```bash
grep -rn "reports/GME" src/ tests/ scripts/ 2>/dev/null
```

If any code references `reports/GME/`, update to `docs/reports/GME/`. Known candidate: `scripts/archive/gme_holistic_report.py` — but that was deleted in Task 4.

**Step 3: Final structure verification**

```bash
# Verify expected structure
ls scripts/                     # Should have: download_all.py, rewrite_imports.py, profile_strategy.py
ls docs/                        # Should have: design/, gme/, reports/ (no plans/)
ls docs/reports/                # Should have: *.md files + GME/
ls docs/reports/GME/            # Should have: 5 files from old reports/GME/
ls src/stockdownloader/gme/     # Should have: __init__.py, analysis/, app.py, pipeline.py, prediction.py, options_analysis.py, regsho_backfill.py
ls tests/gme/                   # Should have: __init__.py, test_analyzer.py, test_statistics.py, test_prediction.py
```

**Step 4: Run full test suite**

```bash
python3 -m pytest tests/ -x -q --tb=short
```

Expected: 3,297 passed, 44 deselected.

**Step 5: Verify no stale import references**

```bash
grep -rn "stockdownloader.analysis.gme" src/ tests/ scripts/
grep -rn "stockdownloader.app.gme_pipeline" src/ tests/ scripts/
grep -rn "stockdownloader.app.gme" src/ tests/ scripts/ | grep -v "stockdownloader.gme"
grep -rn "stockdownloader.app.pinescript_catalog.gme_prediction" src/ tests/ scripts/
```

All should return empty (no stale references).

**Step 6: Commit**

```bash
git add -A
git commit -m "cleanup: merge reports/GME/ into docs/reports/GME/, verify final structure"
```
