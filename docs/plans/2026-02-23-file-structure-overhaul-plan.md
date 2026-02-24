# File Structure Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the entire `stockdownloader` package from 8 loosely-organized packages into 10 clearly-named flat-domain packages with consistent file naming, rewriting all imports.

**Architecture:** Phase migration using a reusable import-rewrite script. Each round moves one package group, rewrites all imports referencing moved files across the entire codebase, and commits. Test files move alongside source files. Clean break — no backward-compat shims.

**Tech Stack:** Python, git mv, sed/Python for import rewriting

**Reference:** See `docs/plans/2026-02-23-file-structure-overhaul-design.md` for the complete file mapping.

---

### Task 1: Create the import rewrite migration script

**Files:**
- Create: `scripts/rewrite_imports.py`

Create a Python script that takes a JSON mapping file of old→new module paths and rewrites all Python imports across the codebase. The script must handle:

1. `from stockdownloader.old.path import Name` → `from stockdownloader.new.path import Name`
2. `from stockdownloader.old.path import (Name1, Name2)` (multiline imports)
3. `import stockdownloader.old.path` → `import stockdownloader.new.path`
4. String references in lazy imports: `"stockdownloader.old.path"` inside `importlib.import_module()` calls
5. TYPE_CHECKING-guarded imports (same syntax, just under `if TYPE_CHECKING:`)

**Usage:**
```bash
python3 scripts/rewrite_imports.py mappings.json [--dry-run]
```

**mappings.json format:**
```json
{
  "stockdownloader.model.financial_models": "stockdownloader.core.models.financial",
  "stockdownloader.model.price_data": "stockdownloader.core.models.price"
}
```

The script should:
- Search all `.py` files in `src/` and `tests/`
- Apply replacements (longest match first to avoid partial matches)
- Report how many files changed and how many replacements made
- Support `--dry-run` to preview changes

**Verification:**
```bash
python3 scripts/rewrite_imports.py --help
python3 scripts/rewrite_imports.py test_mappings.json --dry-run
```

**Commit:** `feat: add import rewrite migration script`

---

### Task 2: Migrate `model/` + util basics → `core/`

**Source file moves** (prefix all with `src/stockdownloader/`):

| git mv FROM | git mv TO |
|---|---|
| `model/financial_models.py` | `core/models/financial.py` |
| `model/price_data.py` | `core/models/price.py` |
| `model/options.py` | `core/models/options.py` |
| `model/trade.py` | `core/models/trade.py` |
| `model/regulatory_records.py` | `core/models/regulatory.py` |
| `model/alert_result.py` | `core/models/alert.py` |
| `model/signal_advisory.py` | `core/models/signal.py` |
| `model/unified_market_data.py` | `core/models/market_data.py` |
| `model/symbol_info.py` | `core/models/symbol.py` |
| `model/indicator_values.py` | `core/models/indicator.py` |
| `model/exit_mechanism_result.py` | `core/models/exit_result.py` |
| `util/math.py` | `core/math.py` |
| `util/config.py` | `core/config.py` |
| `util/io.py` | `core/io.py` |
| `util/timeframe.py` | `core/timeframe.py` |

**Test file moves** (prefix all with `tests/`):

| git mv FROM | git mv TO |
|---|---|
| `model/` | `core/models/` |

Note: Move the entire `tests/model/` directory to `tests/core/models/`. If tests reference `util.math`, `util.config`, `util.io`, or `util.timeframe` they need import rewrites too.

**Import rewrite mappings:**
```json
{
  "stockdownloader.model.financial_models": "stockdownloader.core.models.financial",
  "stockdownloader.model.price_data": "stockdownloader.core.models.price",
  "stockdownloader.model.options": "stockdownloader.core.models.options",
  "stockdownloader.model.trade": "stockdownloader.core.models.trade",
  "stockdownloader.model.regulatory_records": "stockdownloader.core.models.regulatory",
  "stockdownloader.model.alert_result": "stockdownloader.core.models.alert",
  "stockdownloader.model.signal_advisory": "stockdownloader.core.models.signal",
  "stockdownloader.model.unified_market_data": "stockdownloader.core.models.market_data",
  "stockdownloader.model.symbol_info": "stockdownloader.core.models.symbol",
  "stockdownloader.model.indicator_values": "stockdownloader.core.models.indicator",
  "stockdownloader.model.exit_mechanism_result": "stockdownloader.core.models.exit_result",
  "stockdownloader.model import": "stockdownloader.core.models import",
  "stockdownloader.util.math": "stockdownloader.core.math",
  "stockdownloader.util.config": "stockdownloader.core.config",
  "stockdownloader.util.io": "stockdownloader.core.io",
  "stockdownloader.util.timeframe": "stockdownloader.core.timeframe"
}
```

**Steps:**
1. Create directories: `src/stockdownloader/core/`, `src/stockdownloader/core/models/`, `tests/core/`, `tests/core/models/`
2. Create `__init__.py` files for `core/` and `core/models/`
3. `git mv` all files per the table above
4. Write the import mappings JSON and run the rewrite script
5. Create `core/__init__.py` re-exporting key symbols (mirror what `model/__init__.py` had)
6. Create `core/models/__init__.py` re-exporting all model classes
7. Delete old `model/` directory if empty
8. Run: `python3 -m pytest tests/ -x -q`
9. Commit: `refactor: migrate model/ + util basics to core/`

**Blast radius:** ~251 files for model imports + ~86 files for util basics = highest risk round. Run tests after EVERY mapping application.

---

### Task 3: Migrate `data/` → reorganize into subpackages

**Source file moves** (within `src/stockdownloader/data/`):

| git mv FROM | git mv TO |
|---|---|
| `sec_common.py` | `sec/common.py` |
| `sec_edgar_client.py` | `sec/edgar_client.py` |
| `sec_ftd_client.py` | `sec/ftd_client.py` |
| `sec_insider_client.py` | `sec/insider_client.py` |
| `sec_insider_parsers.py` | `sec/insider_parser.py` |
| `bulk_insider_parser.py` | `sec/bulk_insider_parser.py` |
| `sec_ownership_client.py` | `sec/ownership_client.py` |
| `sec_ownership_parsers.py` | `sec/ownership_parser.py` |
| `bulk_ownership_parser.py` | `sec/bulk_ownership_parser.py` |
| `sec_parser_utils.py` | `sec/parser_utils.py` |
| `finra_base_client.py` | `finra/base_client.py` |
| `finra_dark_pool_client.py` | `finra/dark_pool_client.py` |
| `finra_short_interest_client.py` | `finra/short_interest_client.py` |
| `finra_short_volume_client.py` | `finra/short_volume_client.py` |
| `yahoo_base_client.py` | `market/yahoo_base_client.py` |
| `yahoo_data_client.py` | `market/yahoo_data_client.py` |
| `yahoo_finance_client.py` | `market/yahoo_finance_client.py` |
| `yahoo_options_client.py` | `market/yahoo_options_client.py` |
| `morningstar_client.py` | `market/morningstar_client.py` |
| `polygon_data_client.py` | `market/polygon_client.py` |
| `tradier_options_client.py` | `market/tradier_client.py` |
| `occ_options_client.py` | `market/occ_client.py` |
| `borrow_rate.py` | `market/borrow_rate.py` |
| `regsho_threshold_client.py` | `regsho/threshold_client.py` |
| `regsho_sources.py` | `regsho/sources.py` |
| `nyse_regsho.py` | `regsho/nyse.py` |
| `data_parsers.py` | `parsers.py` |
| `intraday_data_accumulator.py` | `accumulator.py` |
| `full_history_fetcher.py` | `history_fetcher.py` |
| `stock_list_downloader.py` | `stock_list.py` |
| `tradingview_trade_loader.py` | `tv_trade_loader.py` |

Files that stay in place (no move): `base_client.py`, `intraday_csv.py`

**Test file moves:** Reorganize `tests/data/` to mirror with `sec/`, `finra/`, `market/`, `regsho/` subdirs. Move test files to match their source counterparts.

**Import rewrite mappings:**
All `stockdownloader.data.sec_*` → `stockdownloader.data.sec.*`, `stockdownloader.data.finra_*` → `stockdownloader.data.finra.*`, etc. Generate the full mapping from the file move table.

**Steps:**
1. Create subdirectories: `data/sec/`, `data/finra/`, `data/market/`, `data/regsho/` (source + tests)
2. Create `__init__.py` for each new subpackage
3. `git mv` all files
4. Run rewrite script with data mappings
5. Update `data/__init__.py` re-exports
6. Run: `python3 -m pytest tests/ -x -q`
7. Commit: `refactor: reorganize data/ into sec/, finra/, market/, regsho/ subpackages`

---

### Task 4: Migrate `util/indicators/` → `indicators/`, `util/pinescript/` → `pinescript/`, `util/options/` → `analysis/options/`

**Source file moves:**

| git mv FROM | git mv TO |
|---|---|
| `util/indicators/_core.py` | `indicators/core.py` |
| `util/indicators/volatility.py` | `indicators/volatility.py` |
| `util/indicators/momentum.py` | `indicators/momentum.py` |
| `util/indicators/trend.py` | `indicators/trend.py` |
| `util/indicators/volume.py` | `indicators/volume.py` |
| `util/indicators/intraday.py` | `indicators/intraday.py` |
| `util/indicators/smc.py` | `indicators/smc.py` |
| `util/indicators/htf.py` | `indicators/htf.py` |
| `util/indicators/hub.py` | `indicators/hub.py` |
| `util/indicators/hub_intraday.py` | `indicators/hub_intraday.py` |
| `util/pinescript/models.py` | `pinescript/models.py` |
| `util/pinescript/generator.py` | `pinescript/generator.py` |
| `util/pinescript/strategy_renderer.py` | `pinescript/strategy_renderer.py` |
| `util/pinescript/composite_renderer.py` | `pinescript/composite_renderer.py` |
| `util/pinescript/modes.py` | `pinescript/modes.py` |
| `util/pinescript/ml_export.py` | `pinescript/ml_export.py` |
| `util/options/black_scholes.py` | `analysis/options/pricing.py` |

**Import rewrite mappings:**
- `stockdownloader.util.indicators._core` → `stockdownloader.indicators.core`
- `stockdownloader.util.indicators.*` → `stockdownloader.indicators.*`
- `stockdownloader.util.pinescript.*` → `stockdownloader.pinescript.*`
- `stockdownloader.util.options.black_scholes` → `stockdownloader.analysis.options.pricing`

**Test file moves:** Move `tests/util/` indicator/pinescript tests to `tests/indicators/` and `tests/pinescript/`.

After this round, `util/` directory should be empty and can be deleted.

**Commit:** `refactor: promote indicators/, pinescript/ from util/, dissolve util/`

---

### Task 5: Migrate `strategy/signals/` → `signals/`, rename `strategy/` → `strategies/`

This is two logical moves in one round:

**Part A — Promote signals:**

| git mv FROM | git mv TO |
|---|---|
| `strategy/signals/signal_generator.py` | `signals/generator.py` |
| `strategy/signals/stacked_signal_engine.py` | `signals/engine.py` |
| `strategy/signals/stacked_intraday_strategy.py` | `signals/intraday_adapter.py` |
| `strategy/signals/stacked_daily_strategy.py` | `signals/daily_adapter.py` |
| `strategy/signals/multi_timeframe_aligner.py` | `signals/timeframe_aligner.py` |
| `strategy/signals/generators/trend.py` | `signals/generators/trend.py` |
| `strategy/signals/generators/momentum.py` | `signals/generators/momentum.py` |
| `strategy/signals/generators/volume.py` | `signals/generators/volume.py` |
| `strategy/signals/generators/volatility.py` | `signals/generators/volatility.py` |

**Part B — Rename strategy → strategies + rename files:**

| git mv FROM | git mv TO |
|---|---|
| `strategy/trading_strategy.py` | `strategies/base.py` |
| `strategy/base_registry.py` | `strategies/registry.py` |
| `strategy/registration_loader.py` | `strategies/loader.py` |
| `strategy/daily/simple_strategies.py` | `strategies/daily/simple.py` |
| `strategy/daily/bollinger_band_rsi_strategy.py` | `strategies/daily/bollinger_rsi.py` |
| `strategy/daily/breakout_strategy.py` | `strategies/daily/breakout.py` |
| `strategy/daily/multi_indicator_strategy.py` | `strategies/daily/multi_indicator.py` |
| `strategy/daily/momentum_confluence_strategy.py` | `strategies/daily/momentum.py` |
| `strategy/intraday/base_strategy.py` | `strategies/intraday/base.py` |
| `strategy/intraday/session_state.py` | `strategies/intraday/session.py` |
| `strategy/intraday/trade_management.py` | `strategies/intraday/trade_mgmt.py` |
| `strategy/intraday/trail_strategy.py` | `strategies/intraday/trail.py` |
| `strategy/intraday/pullback_strategy.py` | `strategies/intraday/pullback.py` |
| `strategy/intraday/reversal_strategy.py` | `strategies/intraday/reversal.py` |
| `strategy/intraday/or_breakout_strategy.py` | `strategies/intraday/or_breakout.py` |
| `strategy/intraday/or_reversal_strategy.py` | `strategies/intraday/or_reversal.py` |
| `strategy/intraday/dmi_vwap_strategy.py` | `strategies/intraday/dmi_vwap.py` |
| `strategy/intraday/smc_structure_strategy.py` | `strategies/intraday/smc_structure.py` |
| `strategy/intraday/avwap_pullback_strategy.py` | `strategies/intraday/avwap_pullback.py` |
| `strategy/intraday/ml_oversold_strategy.py` | `strategies/intraday/ml_oversold.py` |
| `strategy/intraday/pattern_scalp_strategy.py` | `strategies/intraday/pattern_scalp.py` |
| `strategy/intraday/pattern_discovery_strategy.py` | `strategies/intraday/pattern_discovery.py` |
| `strategy/intraday/daily_to_intraday_adapter.py` | `strategies/intraday/daily_adapter.py` |
| (keep path for files without rename) | |
| `strategy/intraday/infra.py` | `strategies/intraday/infra.py` |
| `strategy/intraday/day_tracker.py` | `strategies/intraday/day_tracker.py` |
| `strategy/options/options_strategies.py` | `strategies/options/strategies.py` |
| `strategy/exit_mechanisms/trailing_exit_base.py` | `strategies/exits/base.py` |
| `strategy/exit_mechanisms/vwap_exits.py` | `strategies/exits/vwap.py` |
| `strategy/exit_mechanisms/trail_exits.py` | `strategies/exits/trailing.py` |
| `strategy/exit_mechanisms/composite_exits.py` | `strategies/exits/composite.py` |
| `strategy/regime/regime_detector.py` | `strategies/regime/detector.py` |
| `strategy/regime/regime_strategy_map.py` | `strategies/regime/strategy_map.py` |
| `strategy/regime/ensemble_strategy.py` | `strategies/regime/ensemble.py` |

**Import rewrite mappings:** All `stockdownloader.strategy.signals.*` → `stockdownloader.signals.*`, all `stockdownloader.strategy.*` → `stockdownloader.strategies.*` with file renames.

**Important:** Strategy files contain string-based registry lookups (class names in config JSON files). Check for any string references like `"strategy."` or module path strings in config files and update those too.

**Commit:** `refactor: promote signals/, rename strategy/ to strategies/`

---

### Task 6: Reorganize `analysis/`, minor renames in `ml/`

**analysis/ reorganization:**

| git mv FROM | git mv TO |
|---|---|
| `analysis/formula_calculator.py` | `analysis/formula.py` |
| `analysis/options_gex.py` | `analysis/options/gex.py` |
| `analysis/options_gamma_analyzer.py` | `analysis/options/gamma_analyzer.py` |
| `analysis/value_screener.py` | `analysis/value/screener.py` |
| `analysis/value_scoring.py` | `analysis/value/scoring.py` |

Note: `analysis/options/pricing.py` was already moved in Task 4.

Files that stay: `pattern_analyzer.py`, `pattern_encoder.py`, `alert_generator.py`, `alert_store.py`, `signal_advisor.py`, `gme/`, `pattern_discovery/`.

**ml/ minor renames:**

| git mv FROM | git mv TO |
|---|---|
| `ml/alternative_data_store.py` | `ml/alt_data_store.py` |
| `ml/hmm_regime_detector.py` | `ml/hmm_detector.py` |
| `ml/pipeline/stage_hybrid_strategies.py` | `ml/pipeline/stage_hybrid.py` |

**Import rewrite mappings:** Generate from file moves.

**Commit:** `refactor: reorganize analysis/ subpackages, minor ml/ renames`

---

### Task 7: Rename `backtest/` → `backtesting/` with internal reorganization

**Source file moves:**

| git mv FROM | git mv TO |
|---|---|
| `backtest/backtest_engine.py` | `backtesting/engines/daily.py` |
| `backtest/intraday_backtest_engine.py` | `backtesting/engines/intraday.py` |
| `backtest/options_backtest_engine.py` | `backtesting/engines/options.py` |
| `backtest/backtest_result.py` | `backtesting/results/result.py` |
| `backtest/report_formatter.py` | `backtesting/results/formatter.py` |
| `backtest/report_helpers.py` | `backtesting/results/helpers.py` |
| `backtest/optimizer_base.py` | `backtesting/optimization/base.py` |
| `backtest/optimizer_scoring.py` | `backtesting/optimization/scoring.py` |
| `backtest/strategy_optimizer.py` | `backtesting/optimization/strategy.py` |
| `backtest/daily_strategy_optimizer.py` | `backtesting/optimization/daily.py` |
| `backtest/walk_forward.py` | `backtesting/optimization/walk_forward.py` |
| `backtest/walk_forward_optimizer.py` | `backtesting/optimization/wf_optimizer.py` |
| `backtest/combinatorial_tester.py` | `backtesting/combinatorial.py` |
| `backtest/portfolio_analyzer.py` | `backtesting/portfolio.py` |
| `backtest/tournament_engine.py` | `backtesting/tournament/engine.py` |
| `backtest/tournament_models.py` | `backtesting/tournament/models.py` |
| `backtest/tournament_workers.py` | `backtesting/tournament/workers.py` |
| `backtest/tournament_analysis.py` | `backtesting/tournament/analysis.py` |
| `backtest/exit_tournament_engine.py` | `backtesting/tournament/exit_engine.py` |
| `backtest/exit_tournament_result.py` | `backtesting/tournament/exit_result.py` |
| `backtest/exit_tournament_report_formatter.py` | `backtesting/tournament/exit_report.py` |

**Test file moves:** Move `tests/backtest/` → `tests/backtesting/` with matching substructure.

**Commit:** `refactor: rename backtest/ to backtesting/ with internal reorganization`

---

### Task 8: Rename `app/` CLI files

**Source file moves** (within `src/stockdownloader/app/`):

| git mv FROM | git mv TO |
|---|---|
| `symbol_analysis_app.py` | `analysis.py` |
| `gme_analysis_app.py` | `gme.py` |
| `backtest_app.py` | `backtest.py` |
| `optimize_app.py` | `optimize.py` |
| `monitor_app.py` | `monitor.py` |
| `ml_train_app.py` | `ml_train.py` |
| `spy_ml_pipeline_app.py` | `ml_pipeline.py` |
| `gme_ml_pipeline_app.py` | `gme_pipeline.py` |
| `tournament_apps.py` | `tournament.py` |
| `pattern_discovery_app.py` | `pattern_discovery.py` |
| `value_screener_app.py` | `value_screener.py` |
| `generate_pinescript.py` | `pinescript.py` |
| `app_helpers.py` | `helpers.py` |
| `_ml_helpers.py` | `ml_helpers.py` |

Sub-packages (`pipeline/`, `tournament/`, `pinescript_catalog/`) keep their names.

**Test file moves:** Rename corresponding test files in `tests/app/`.

**Commit:** `refactor: rename app/ CLI files for consistency`

---

### Task 9: Final verification and cleanup

**Steps:**
1. Delete any empty old directories (model/, util/, strategy/, backtest/)
2. Verify no import references to old paths remain:
   ```bash
   grep -r "stockdownloader\.model\." src/ tests/ --include="*.py" | grep -v "__pycache__"
   grep -r "stockdownloader\.util\." src/ tests/ --include="*.py" | grep -v "__pycache__"
   grep -r "stockdownloader\.strategy\." src/ tests/ --include="*.py" | grep -v "__pycache__"
   grep -r "stockdownloader\.backtest\." src/ tests/ --include="*.py" | grep -v "__pycache__"
   ```
   All should return empty.
3. Full regression: `python3 -m pytest tests/ -x -q`
4. Expected: 3,297 passed
5. Verify new directory structure matches design:
   ```bash
   find src/stockdownloader/ -type d | sort
   ```
6. Commit: `cleanup: remove old directories, verify final structure`

---

## Verification

```bash
# No references to old paths
grep -r "stockdownloader\.model\b" src/ tests/ --include="*.py" | grep -v __pycache__ | wc -l  # should be 0
grep -r "stockdownloader\.util\b" src/ tests/ --include="*.py" | grep -v __pycache__ | wc -l   # should be 0
grep -r "stockdownloader\.strategy\b" src/ tests/ --include="*.py" | grep -v __pycache__ | wc -l # should be 0
grep -r "stockdownloader\.backtest\b" src/ tests/ --include="*.py" | grep -v __pycache__ | wc -l # should be 0

# Full test suite
python3 -m pytest tests/ -x -q

# Directory structure
find src/stockdownloader/ -type d -not -name __pycache__ | sort
```
