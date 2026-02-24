# File Structure Overhaul — Design Document

## Goal

Restructure the entire `stockdownloader` package from 8 loosely-organized packages into 10 clearly-named, flat-domain packages with consistent file naming. Clean break — all imports rewritten, no backward-compatibility shims.

## Architecture

**Flat Domain with Subpackages** — each domain is a top-level package with shallow imports:

```
stockdownloader/
├── core/           # 15 files — models, types, math, config
├── data/           # 34 files — external data clients (reorganized by source)
├── indicators/     # 10 files — technical indicator library
├── signals/        # 9 files  — signal generators + engine
├── strategies/     # 33 files — all strategy implementations
├── analysis/       # 19 files — analysis & screening tools
├── backtesting/    # 22 files — engines, optimizers, tournament
├── ml/             # 18 files — training, prediction, pipeline
├── pinescript/     # 6 files  — Pine Script v6 generation
└── app/            # 31 files — CLI entry points
```

**Key transformations:**
- `model/` + util basics → `core/`
- `util/` dissolved: `indicators/` and `pinescript/` promoted to top-level; `math`, `config`, `io`, `timeframe` move to `core/`; `options/black_scholes.py` moves to `analysis/options/pricing.py`
- `strategy/signals/` promoted to top-level `signals/`
- `strategy/` (minus signals) → `strategies/` (pluralized)
- `backtest/` → `backtesting/`
- `data/` reorganized into subpackages: `sec/`, `finra/`, `market/`, `regsho/`

**Naming conventions:**
- Data clients: `*_client.py` suffix
- Parsers: `*_parser.py` suffix (singular)
- Strategy files: drop `_strategy` suffix (already in `strategies/` package)
- Base classes: `base.py`
- No `_` prefix on public modules (e.g., `_core.py` → `core.py`)

## Complete File Mapping

### `core/` — Models, Types, Utilities

| New Path | Old Path |
|---|---|
| `core/models/financial.py` | `model/financial_models.py` |
| `core/models/price.py` | `model/price_data.py` |
| `core/models/options.py` | `model/options.py` |
| `core/models/trade.py` | `model/trade.py` |
| `core/models/regulatory.py` | `model/regulatory_records.py` |
| `core/models/alert.py` | `model/alert_result.py` |
| `core/models/signal.py` | `model/signal_advisory.py` |
| `core/models/market_data.py` | `model/unified_market_data.py` |
| `core/models/symbol.py` | `model/symbol_info.py` |
| `core/models/indicator.py` | `model/indicator_values.py` |
| `core/models/exit_result.py` | `model/exit_mechanism_result.py` |
| `core/math.py` | `util/math.py` |
| `core/config.py` | `util/config.py` |
| `core/io.py` | `util/io.py` |
| `core/timeframe.py` | `util/timeframe.py` |

### `data/` — External Data Access

| New Path | Old Path |
|---|---|
| `data/base_client.py` | `data/base_client.py` |
| `data/parsers.py` | `data/data_parsers.py` |
| `data/intraday_csv.py` | `data/intraday_csv.py` |
| `data/accumulator.py` | `data/intraday_data_accumulator.py` |
| `data/history_fetcher.py` | `data/full_history_fetcher.py` |
| `data/stock_list.py` | `data/stock_list_downloader.py` |
| `data/tv_trade_loader.py` | `data/tradingview_trade_loader.py` |
| `data/sec/common.py` | `data/sec_common.py` |
| `data/sec/edgar_client.py` | `data/sec_edgar_client.py` |
| `data/sec/ftd_client.py` | `data/sec_ftd_client.py` |
| `data/sec/insider_client.py` | `data/sec_insider_client.py` |
| `data/sec/insider_parser.py` | `data/sec_insider_parsers.py` |
| `data/sec/bulk_insider_parser.py` | `data/bulk_insider_parser.py` |
| `data/sec/ownership_client.py` | `data/sec_ownership_client.py` |
| `data/sec/ownership_parser.py` | `data/sec_ownership_parsers.py` |
| `data/sec/bulk_ownership_parser.py` | `data/bulk_ownership_parser.py` |
| `data/sec/parser_utils.py` | `data/sec_parser_utils.py` |
| `data/finra/base_client.py` | `data/finra_base_client.py` |
| `data/finra/dark_pool_client.py` | `data/finra_dark_pool_client.py` |
| `data/finra/short_interest_client.py` | `data/finra_short_interest_client.py` |
| `data/finra/short_volume_client.py` | `data/finra_short_volume_client.py` |
| `data/market/yahoo_base_client.py` | `data/yahoo_base_client.py` |
| `data/market/yahoo_data_client.py` | `data/yahoo_data_client.py` |
| `data/market/yahoo_finance_client.py` | `data/yahoo_finance_client.py` |
| `data/market/yahoo_options_client.py` | `data/yahoo_options_client.py` |
| `data/market/morningstar_client.py` | `data/morningstar_client.py` |
| `data/market/polygon_client.py` | `data/polygon_data_client.py` |
| `data/market/tradier_client.py` | `data/tradier_options_client.py` |
| `data/market/occ_client.py` | `data/occ_options_client.py` |
| `data/market/borrow_rate.py` | `data/borrow_rate.py` |
| `data/regsho/threshold_client.py` | `data/regsho_threshold_client.py` |
| `data/regsho/sources.py` | `data/regsho_sources.py` |
| `data/regsho/nyse.py` | `data/nyse_regsho.py` |

### `indicators/` — Technical Indicators

| New Path | Old Path |
|---|---|
| `indicators/core.py` | `util/indicators/_core.py` |
| `indicators/volatility.py` | `util/indicators/volatility.py` |
| `indicators/momentum.py` | `util/indicators/momentum.py` |
| `indicators/trend.py` | `util/indicators/trend.py` |
| `indicators/volume.py` | `util/indicators/volume.py` |
| `indicators/intraday.py` | `util/indicators/intraday.py` |
| `indicators/smc.py` | `util/indicators/smc.py` |
| `indicators/htf.py` | `util/indicators/htf.py` |
| `indicators/hub.py` | `util/indicators/hub.py` |
| `indicators/hub_intraday.py` | `util/indicators/hub_intraday.py` |

### `signals/` — Signal Generation

| New Path | Old Path |
|---|---|
| `signals/generator.py` | `strategy/signals/signal_generator.py` |
| `signals/engine.py` | `strategy/signals/stacked_signal_engine.py` |
| `signals/intraday_adapter.py` | `strategy/signals/stacked_intraday_strategy.py` |
| `signals/daily_adapter.py` | `strategy/signals/stacked_daily_strategy.py` |
| `signals/timeframe_aligner.py` | `strategy/signals/multi_timeframe_aligner.py` |
| `signals/generators/trend.py` | `strategy/signals/generators/trend.py` |
| `signals/generators/momentum.py` | `strategy/signals/generators/momentum.py` |
| `signals/generators/volume.py` | `strategy/signals/generators/volume.py` |
| `signals/generators/volatility.py` | `strategy/signals/generators/volatility.py` |

### `strategies/` — Trading Strategies

| New Path | Old Path |
|---|---|
| `strategies/base.py` | `strategy/trading_strategy.py` |
| `strategies/registry.py` | `strategy/base_registry.py` |
| `strategies/loader.py` | `strategy/registration_loader.py` |
| `strategies/daily/simple.py` | `strategy/daily/simple_strategies.py` |
| `strategies/daily/bollinger_rsi.py` | `strategy/daily/bollinger_band_rsi_strategy.py` |
| `strategies/daily/breakout.py` | `strategy/daily/breakout_strategy.py` |
| `strategies/daily/multi_indicator.py` | `strategy/daily/multi_indicator_strategy.py` |
| `strategies/daily/momentum.py` | `strategy/daily/momentum_confluence_strategy.py` |
| `strategies/intraday/infra.py` | `strategy/intraday/infra.py` |
| `strategies/intraday/base.py` | `strategy/intraday/base_strategy.py` |
| `strategies/intraday/session.py` | `strategy/intraday/session_state.py` |
| `strategies/intraday/day_tracker.py` | `strategy/intraday/day_tracker.py` |
| `strategies/intraday/trade_mgmt.py` | `strategy/intraday/trade_management.py` |
| `strategies/intraday/trail.py` | `strategy/intraday/trail_strategy.py` |
| `strategies/intraday/pullback.py` | `strategy/intraday/pullback_strategy.py` |
| `strategies/intraday/reversal.py` | `strategy/intraday/reversal_strategy.py` |
| `strategies/intraday/or_breakout.py` | `strategy/intraday/or_breakout_strategy.py` |
| `strategies/intraday/or_reversal.py` | `strategy/intraday/or_reversal_strategy.py` |
| `strategies/intraday/dmi_vwap.py` | `strategy/intraday/dmi_vwap_strategy.py` |
| `strategies/intraday/smc_structure.py` | `strategy/intraday/smc_structure_strategy.py` |
| `strategies/intraday/avwap_pullback.py` | `strategy/intraday/avwap_pullback_strategy.py` |
| `strategies/intraday/ml_oversold.py` | `strategy/intraday/ml_oversold_strategy.py` |
| `strategies/intraday/pattern_scalp.py` | `strategy/intraday/pattern_scalp_strategy.py` |
| `strategies/intraday/pattern_discovery.py` | `strategy/intraday/pattern_discovery_strategy.py` |
| `strategies/intraday/daily_adapter.py` | `strategy/intraday/daily_to_intraday_adapter.py` |
| `strategies/options/strategies.py` | `strategy/options/options_strategies.py` |
| `strategies/exits/base.py` | `strategy/exit_mechanisms/trailing_exit_base.py` |
| `strategies/exits/vwap.py` | `strategy/exit_mechanisms/vwap_exits.py` |
| `strategies/exits/trailing.py` | `strategy/exit_mechanisms/trail_exits.py` |
| `strategies/exits/composite.py` | `strategy/exit_mechanisms/composite_exits.py` |
| `strategies/regime/detector.py` | `strategy/regime/regime_detector.py` |
| `strategies/regime/strategy_map.py` | `strategy/regime/regime_strategy_map.py` |
| `strategies/regime/ensemble.py` | `strategy/regime/ensemble_strategy.py` |

### `analysis/` — Analysis Tools

| New Path | Old Path |
|---|---|
| `analysis/formula.py` | `analysis/formula_calculator.py` |
| `analysis/pattern_analyzer.py` | `analysis/pattern_analyzer.py` |
| `analysis/pattern_encoder.py` | `analysis/pattern_encoder.py` |
| `analysis/alert_generator.py` | `analysis/alert_generator.py` |
| `analysis/alert_store.py` | `analysis/alert_store.py` |
| `analysis/signal_advisor.py` | `analysis/signal_advisor.py` |
| `analysis/options/gex.py` | `analysis/options_gex.py` |
| `analysis/options/gamma_analyzer.py` | `analysis/options_gamma_analyzer.py` |
| `analysis/options/pricing.py` | `util/options/black_scholes.py` |
| `analysis/value/screener.py` | `analysis/value_screener.py` |
| `analysis/value/scoring.py` | `analysis/value_scoring.py` |
| `analysis/gme/models.py` | `analysis/gme/models.py` |
| `analysis/gme/options.py` | `analysis/gme/options.py` |
| `analysis/gme/regime.py` | `analysis/gme/regime.py` |
| `analysis/gme/distribution.py` | `analysis/gme/distribution.py` |
| `analysis/gme/event_study.py` | `analysis/gme/event_study.py` |
| `analysis/pattern_discovery/models.py` | `analysis/pattern_discovery/models.py` |
| `analysis/pattern_discovery/miner.py` | `analysis/pattern_discovery/miner.py` |
| `analysis/pattern_discovery/filters.py` | `analysis/pattern_discovery/filters.py` |
| `analysis/pattern_discovery/catalog.py` | `analysis/pattern_discovery/catalog.py` |

### `backtesting/` — Backtesting Engines

| New Path | Old Path |
|---|---|
| `backtesting/engines/daily.py` | `backtest/backtest_engine.py` |
| `backtesting/engines/intraday.py` | `backtest/intraday_backtest_engine.py` |
| `backtesting/engines/options.py` | `backtest/options_backtest_engine.py` |
| `backtesting/results/result.py` | `backtest/backtest_result.py` |
| `backtesting/results/formatter.py` | `backtest/report_formatter.py` |
| `backtesting/results/helpers.py` | `backtest/report_helpers.py` |
| `backtesting/optimization/base.py` | `backtest/optimizer_base.py` |
| `backtesting/optimization/scoring.py` | `backtest/optimizer_scoring.py` |
| `backtesting/optimization/strategy.py` | `backtest/strategy_optimizer.py` |
| `backtesting/optimization/daily.py` | `backtest/daily_strategy_optimizer.py` |
| `backtesting/optimization/walk_forward.py` | `backtest/walk_forward.py` |
| `backtesting/optimization/wf_optimizer.py` | `backtest/walk_forward_optimizer.py` |
| `backtesting/combinatorial.py` | `backtest/combinatorial_tester.py` |
| `backtesting/portfolio.py` | `backtest/portfolio_analyzer.py` |
| `backtesting/tournament/engine.py` | `backtest/tournament_engine.py` |
| `backtesting/tournament/models.py` | `backtest/tournament_models.py` |
| `backtesting/tournament/workers.py` | `backtest/tournament_workers.py` |
| `backtesting/tournament/analysis.py` | `backtest/tournament_analysis.py` |
| `backtesting/tournament/exit_engine.py` | `backtest/exit_tournament_engine.py` |
| `backtesting/tournament/exit_result.py` | `backtest/exit_tournament_result.py` |
| `backtesting/tournament/exit_report.py` | `backtest/exit_tournament_report_formatter.py` |

### `ml/` — Machine Learning

| New Path | Old Path |
|---|---|
| `ml/feature_extractor.py` | `ml/feature_extractor.py` |
| `ml/dataset_builder.py` | `ml/dataset_builder.py` |
| `ml/trainer.py` | `ml/trainer.py` |
| `ml/trainer_tuning.py` | `ml/trainer_tuning.py` |
| `ml/predictor.py` | `ml/predictor.py` |
| `ml/model_store.py` | `ml/model_store.py` |
| `ml/alt_data_store.py` | `ml/alternative_data_store.py` |
| `ml/hmm_detector.py` | `ml/hmm_regime_detector.py` |
| `ml/pipeline/config.py` | `ml/pipeline/config.py` |
| `ml/pipeline/results.py` | `ml/pipeline/results.py` |
| `ml/pipeline/orchestrator.py` | `ml/pipeline/orchestrator.py` |
| `ml/pipeline/stage_data.py` | `ml/pipeline/stage_data.py` |
| `ml/pipeline/stage_training.py` | `ml/pipeline/stage_training.py` |
| `ml/pipeline/stage_convergence.py` | `ml/pipeline/stage_convergence.py` |
| `ml/pipeline/stage_hybrid.py` | `ml/pipeline/stage_hybrid_strategies.py` |
| `ml/pipeline/stage_backtest.py` | `ml/pipeline/stage_backtest.py` |
| `ml/pipeline/stage_selection.py` | `ml/pipeline/stage_selection.py` |
| `ml/pipeline/walk_forward.py` | `ml/pipeline/walk_forward.py` |

### `pinescript/` — Pine Script Generation

| New Path | Old Path |
|---|---|
| `pinescript/models.py` | `util/pinescript/models.py` |
| `pinescript/generator.py` | `util/pinescript/generator.py` |
| `pinescript/strategy_renderer.py` | `util/pinescript/strategy_renderer.py` |
| `pinescript/composite_renderer.py` | `util/pinescript/composite_renderer.py` |
| `pinescript/modes.py` | `util/pinescript/modes.py` |
| `pinescript/ml_export.py` | `util/pinescript/ml_export.py` |

### `app/` — CLI Entry Points

| New Path | Old Path |
|---|---|
| `app/analysis.py` | `app/symbol_analysis_app.py` |
| `app/gme.py` | `app/gme_analysis_app.py` |
| `app/backtest.py` | `app/backtest_app.py` |
| `app/optimize.py` | `app/optimize_app.py` |
| `app/monitor.py` | `app/monitor_app.py` |
| `app/ml_train.py` | `app/ml_train_app.py` |
| `app/ml_pipeline.py` | `app/spy_ml_pipeline_app.py` |
| `app/gme_pipeline.py` | `app/gme_ml_pipeline_app.py` |
| `app/tournament.py` | `app/tournament_apps.py` |
| `app/pattern_discovery.py` | `app/pattern_discovery_app.py` |
| `app/value_screener.py` | `app/value_screener_app.py` |
| `app/pinescript.py` | `app/generate_pinescript.py` |
| `app/helpers.py` | `app/app_helpers.py` |
| `app/ml_helpers.py` | `app/_ml_helpers.py` |
| `app/pipeline_app.py` | `app/pipeline_app.py` |
| `app/pipeline/cli.py` | `app/pipeline/cli.py` |
| `app/pipeline/stages.py` | `app/pipeline/stages.py` |
| `app/pipeline/models.py` | `app/pipeline/models.py` |
| `app/pipeline/helpers.py` | `app/pipeline/helpers.py` |
| `app/pipeline/report.py` | `app/pipeline/report.py` |
| `app/tournament/cli.py` | `app/tournament/cli.py` |
| `app/tournament/stages.py` | `app/tournament/stages.py` |
| `app/tournament/stages_advanced.py` | `app/tournament/stages_advanced.py` |
| `app/tournament/helpers.py` | `app/tournament/helpers.py` |
| `app/pinescript_catalog/catalogs.py` | `app/pinescript_catalog/catalogs.py` |
| `app/pinescript_catalog/gme_prediction.py` | `app/pinescript_catalog/gme_prediction.py` |
| `app/pinescript_catalog/spy_strategies.py` | `app/pinescript_catalog/spy_strategies.py` |

### Tests — Mirror Source Structure

Test directory mirrors source structure exactly:
```
tests/
├── core/           (was tests/model/ + tests/util/ core tests)
├── data/           (was tests/data/, with sec/, finra/, market/, regsho/ subdirs)
├── indicators/     (was tests/util/ indicator tests)
├── signals/        (was tests/strategy/signals/)
├── strategies/     (was tests/strategy/ minus signals)
├── analysis/       (keep, reorganize to match)
├── backtesting/    (was tests/backtest/)
├── ml/             (keep)
├── pinescript/     (new, from tests/util/ pinescript tests)
├── app/            (keep)
├── e2e/            (keep)
├── integration/    (keep)
└── live/           (keep)
```

## Execution Strategy

This restructure touches every file in the codebase. Execute in rounds:

1. **Round 1: core/** — Move model/ + util basics. Rewrite all `model.*` and `util.math/config/io` imports.
2. **Round 2: data/** — Reorganize into subpackages. Rewrite all `data.*` imports.
3. **Round 3: indicators/ + pinescript/** — Promote from util/. Rewrite `util.indicators.*` and `util.pinescript.*` imports.
4. **Round 4: signals/** — Promote from strategy/signals/. Rewrite `strategy.signals.*` imports.
5. **Round 5: strategies/** — Rename strategy/ → strategies/, rename files. Rewrite all `strategy.*` imports.
6. **Round 6: analysis/** — Reorganize subpackages, absorb black_scholes. Rewrite changed `analysis.*` imports.
7. **Round 7: backtesting/** — Rename backtest/ → backtesting/, reorganize. Rewrite all `backtest.*` imports.
8. **Round 8: ml/** — Minor renames. Rewrite changed `ml.*` imports.
9. **Round 9: app/** — Rename CLI files. Rewrite internal imports.
10. **Round 10: tests/** — Move all test files to mirror new structure.
11. **Round 11: cleanup** — Delete old directories, verify, final regression.

Each round: move files, rewrite imports in ALL source + test files, run full regression, commit.
