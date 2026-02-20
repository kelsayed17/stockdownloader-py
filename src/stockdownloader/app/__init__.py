"""Application entry points (CLI commands).

Each module provides a ``main()`` function registered as a console script
in ``pyproject.toml``.  Key apps:

- **backtest_app** — Unified daily / intraday / options strategy backtest
- **symbol_analysis_app** — Unified analysis (daily + options + alerts)
- **tournament_app** — Strategy tournaments (walk-forward, baseline, greedy)
- **exit_tournament_app** — Exit mechanism testing
- **signal_stack_tournament** — Signal combinatorial search
- **spy_ml_pipeline_app** — ML pipelines (generic ``main_generic`` + SPY multi-TF ``main``)
- **gme_ml_pipeline_app** — GME alt-data ML pipeline
- **ml_train_app** — Standalone ML model trainer
- **generate_pinescript** — Export strategies to TradingView Pine Script v6
- **pipeline_app** — Full data + backtest orchestration pipeline
- **optimize_app** — Greedy-sequential parameter optimization
- **walk_forward_app** — Standalone walk-forward validation
- **gme_analysis_app** — 6-stage GME analysis
- **monitor_app** — Signal advisory / monitoring
- **pattern_discovery_app** — Pattern mining
- **value_screener_app** — Deep-value screener
- **intraday_accumulate_app** — Intraday data accumulation

Shared helpers:

- **app_helpers** — Constants, data loading, argparse builders, banners
- **_ml_helpers** — ML pipeline shared plumbing (dep check, config builders)
"""

__all__: list[str] = []
