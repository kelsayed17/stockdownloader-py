"""Application entry points (CLI commands).

Each module provides a ``main()`` function registered as a console script
in ``pyproject.toml``.  Key apps:

- **backtest** — Unified daily / intraday / options strategy backtest
- **analysis** — Unified analysis (daily + options + alerts)
- **tournament** — Exit mechanism testing + signal combinatorial search
- **ml_pipeline** — ML pipelines (generic ``main_generic`` + SPY multi-TF ``main``)
- **gme_pipeline** — GME alt-data ML pipeline
- **ml_train** — Standalone ML model trainer
- **pinescript** — Export strategies to TradingView Pine Script v6
- **pipeline_app** — Full data + backtest orchestration pipeline
- **optimize** — Parameter optimization + walk-forward validation
- **gme** — 6-stage GME analysis
- **monitor** — Signal advisory / monitoring
- **pattern_discovery** — Pattern mining
- **value_screener** — Deep-value screener
- **backtest.main_accumulate** — Intraday data accumulation

Sub-packages:

- **tournament/** — Strategy tournaments (walk-forward, baseline, greedy)
- **pipeline/** — Pipeline stages and orchestration
- **pinescript_catalog/** — Pine Script strategy catalogs

Shared helpers:

- **helpers** — Constants, data loading, argparse builders, banners
- **ml_helpers** — ML pipeline shared plumbing (dep check, config builders)
"""

__all__: list[str] = []
