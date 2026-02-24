# GME Consolidation & Non-Src Cleanup — Design

## Goal

Consolidate all GME-specific code into a top-level `gme/` package and clean up the non-src directory structure (scripts, docs, reports).

## Approach

**GME Top-Level Package + Flat Non-Src** — Create `src/stockdownloader/gme/` as a dedicated namespace that owns all GME-specific code. Outside `src/`, delete archived scripts and completed plans, merge duplicate report directories.

## GME Package Structure

```
src/stockdownloader/gme/
    __init__.py
    analysis/                      # FROM: analysis/gme/
        __init__.py
        models.py
        regime.py
        event_study.py
        distribution.py
        options.py
    app.py                         # FROM: app/gme.py
    pipeline.py                    # FROM: app/gme_pipeline.py
    prediction.py                  # FROM: app/pinescript_catalog/gme_prediction.py
    options_analysis.py            # FROM: scripts/gme_options_analysis.py (promoted)
    regsho_backfill.py             # FROM: scripts/regsho_targeted_backfill.py (promoted)
```

### Source File Moves

| From (src/stockdownloader/) | To (src/stockdownloader/) |
|---|---|
| `analysis/gme/__init__.py` | `gme/analysis/__init__.py` |
| `analysis/gme/models.py` | `gme/analysis/models.py` |
| `analysis/gme/regime.py` | `gme/analysis/regime.py` |
| `analysis/gme/event_study.py` | `gme/analysis/event_study.py` |
| `analysis/gme/distribution.py` | `gme/analysis/distribution.py` |
| `analysis/gme/options.py` | `gme/analysis/options.py` |
| `app/gme.py` | `gme/app.py` |
| `app/gme_pipeline.py` | `gme/pipeline.py` |
| `app/pinescript_catalog/gme_prediction.py` | `gme/prediction.py` |

### Script Promotions (scripts/ → src/)

| From | To (src/stockdownloader/) |
|---|---|
| `scripts/gme_options_analysis.py` | `gme/options_analysis.py` |
| `scripts/regsho_targeted_backfill.py` | `gme/regsho_backfill.py` |

### Test File Moves

| From (tests/) | To (tests/) |
|---|---|
| `analysis/test_gme_analyzer.py` | `gme/test_analyzer.py` |
| `analysis/test_gme_analyzer_statistics.py` | `gme/test_statistics.py` |
| `pinescript/test_gme_prediction.py` | `gme/test_prediction.py` |

### Import Rewrite Mappings

```json
{
  "stockdownloader.analysis.gme.models": "stockdownloader.gme.analysis.models",
  "stockdownloader.analysis.gme.regime": "stockdownloader.gme.analysis.regime",
  "stockdownloader.analysis.gme.event_study": "stockdownloader.gme.analysis.event_study",
  "stockdownloader.analysis.gme.distribution": "stockdownloader.gme.analysis.distribution",
  "stockdownloader.analysis.gme.options": "stockdownloader.gme.analysis.options",
  "stockdownloader.analysis.gme import": "stockdownloader.gme.analysis import",
  "stockdownloader.app.gme_pipeline": "stockdownloader.gme.pipeline",
  "stockdownloader.app.gme": "stockdownloader.gme.app"
}
```

### Pyproject.toml Entry Points

Update existing:
- `gme-analysis` → `stockdownloader.gme.app:main`
- `gme-ml-pipeline` → `stockdownloader.gme.pipeline:main`

Add new:
- `gme-options-analysis` → `stockdownloader.gme.options_analysis:main`
- `gme-regsho-backfill` → `stockdownloader.gme.regsho_backfill:main`

## Non-Src Cleanup

### Target Structure

```
scripts/                           # General tooling only (3 files)
    download_all.py
    rewrite_imports.py
    profile_strategy.py

config/                            # Unchanged
    strategies/
    backtest/
    ml/

data/                              # Unchanged
    GME/
    GMEWS/
    SPY/
    cache/

docs/
    gme/                           # GME analysis docs (unchanged)
    design/                        # Design docs (unchanged)
    reports/                       # Merged — absorbs root reports/
        GME/                       # FROM: reports/GME/

output/                            # Unchanged
```

### Deletions

- `scripts/archive/` — entire directory (9 archived one-off scripts)
- `docs/plans/` — entire directory (24 completed plan docs from codebase overhaul)
- `reports/` root directory — after merging contents into `docs/reports/`

### Merges

- `reports/GME/*` → `docs/reports/GME/*`
