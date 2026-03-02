# SPY ML Ensemble Strategy with Deep Surrogate PineScript Export — Design

## Goal

Build an ML ensemble that determines the best strategy to trade SPY, using daily and intraday data along with all available indicators and signals. The final output is a production-ready PineScript v6 strategy with buy/sell label overlays drawn at every trade, green/red background shading during positions, and an ML confidence panel.

## Architecture Overview

Four phases chained sequentially:

1. **Training Phase** — Train 3 model types (GradientBoosting, RandomForest, LogisticRegression) across a hyperparameter grid using the existing 6-stage ML pipeline. Use all 63 Pine-compatible features (excluding alt-data/HMM features that map to constants in Pine). Download fresh SPY data (daily 10y + intraday 60d) from Yahoo Finance before training.

2. **Ensemble Phase** — Combine the top 3-5 models (by OOS accuracy) via soft-voting (averaged `predict_proba`). A new `EnsemblePredictor` class wraps multiple trained `TrainingResult` models.

3. **Deep Surrogate Phase** — Train a `DecisionTreeRegressor` (depth 10, min_samples_leaf 10) to approximate the ensemble's continuous probability predictions. Use the ensemble's averaged feature importances to select the top 25 features. Measure fidelity via R² (target > 0.75).

4. **PineScript Export Phase** — Export the deep surrogate to a v2 strategy-mode PineScript with ML-driven dynamic TP/SL, visual overlays, and multi-timeframe confirmation.

## Ensemble Model

### Construction

```
EnsemblePredictor
  ├── GradientBoosting (best OOS accuracy — captures non-linear patterns)
  ├── RandomForest (diversity — different bias than GB, bagging vs boosting)
  └── LogisticRegression (linear baseline — catches what trees miss)
```

For each bar, `EnsemblePredictor.predict_proba(X)` averages the individual models' P(profitable) outputs. Selection criteria: top 3-5 models by OOS accuracy from the training grid.

### Existing Infrastructure Reused

- `MLTrainer` — trains individual models with expanding-window cross-validation
- `FeatureExtractor` — extracts 63-dimensional feature vectors
- `DatasetBuilder` — builds labeled (X, y) training matrices
- `MLModelConfig` — hyperparameter configuration
- Training grid: forward_periods (5, 10, 20), profit_thresholds (0.3%, 0.5%, 1%), model types, class balance options

### New Components

- `EnsemblePredictor` — wraps N `TrainingResult` models, averages probabilities
- `EnsembleBuilder` — selects top models from grid results, constructs ensemble

## Deep Surrogate

### Why Regression Instead of Classification

The existing `DecisionTreeExporter` uses `DecisionTreeClassifier` which only learns binary labels (profitable / not profitable). For an ensemble approximation, we want the surrogate to learn the **continuous probability surface** — the nuanced confidence levels between 0 and 1 that the ensemble produces. A `DecisionTreeRegressor` directly predicts P(profitable) as a continuous value.

### Parameters (vs existing defaults)

| Parameter | Current | New |
|-----------|---------|-----|
| `max_depth` | 6 | 10 |
| `min_samples_leaf` | 20 | 10 |
| `top_n` features | 15 | 25 |
| Model class | DecisionTreeClassifier | DecisionTreeRegressor |
| Training target | Binary labels | Ensemble probabilities |

### Fidelity Measurement

After training, measure R² between ensemble predictions and surrogate predictions on held-out OOS data. Target: R² > 0.75. If fidelity is too low, reduce depth or increase min_samples_leaf to reduce overfitting.

### PineScript Translation

The existing `DecisionTreeExporter.tree_to_pine_lines()` already converts tree nodes to intermediate Pine Script variables (`float tn0 = ...`). The only change is that leaf nodes output the regression value (mean target) instead of class probability — but both are floats in [0, 1], so the Pine code structure is identical.

The deeper tree (depth 10) will produce more intermediate variables (~100-200 lines of ternary assignments). The `tree_to_pine_lines()` method already handles this by breaking into named intermediate variables to stay under TradingView's line-length limits.

## PineScript Output

### Strategy Mode (v2 pattern)

Full `strategy()` with `strategy.entry()` / `strategy.exit()`, following the existing SPY v2 strategy pattern.

### Entry Logic

- **Long entry:** `mlProb > buyThresh` (default 0.65) AND intraday confirmation via `request.security(syminfo.tickerid, "15", ta.vwap)` — price must be above 15-min VWAP
- **Short entry:** `mlProb < sellThresh` (default 0.35) AND price below 15-min VWAP
- Multi-timeframe confirmation prevents entering against intraday momentum

### ML-Driven Dynamic Exits

Instead of fixed ATR multipliers, the ML confidence score scales TP/SL dynamically:

```
confidence = math.abs(mlProb - 0.5) * 2.0    // 0.0 = uncertain, 1.0 = max confident

dynamicSL = atrVal * (baseSL - confidence * 0.5)   // High confidence → tighter stop
dynamicTP = atrVal * (baseTP + confidence * 1.0)    // High confidence → wider target
```

- High confidence: tighter stops (less room for reversal), wider targets (expecting bigger moves)
- Low confidence: wider stops, tighter targets (take profit quickly)

### Visual Overlays

1. **Buy labels** (`▲`) at entry bars, **sell labels** (`▼`) at exit bars
2. **Green background** shading during long positions, **red** during short
3. **ML confidence line** in a separate pane (0-100 scale)
4. **Entry price / TP / SL levels** plotted as horizontal lines during active positions

### Risk Management

- Circuit breaker: max 3 consecutive losses → pause trading for 5 bars
- EOD close option (configurable input)
- Position sizing: fixed 100% equity per trade (configurable)

## Validation & Tournament

### Walk-Forward Validation

1. **Ensemble training**: 10-year SPY data, expanding windows (min 3y train, 6-month OOS)
2. **Surrogate fidelity**: R² measured at each window (must stay > 0.70)
3. **PineScript backtest**: Backtest the surrogate's decisions using daily `BacktestEngine`

### Tournament Against Existing Strategies

Must outperform all 3 existing SPY strategies:

| Metric | Minimum Threshold | Must Beat Existing? |
|--------|-------------------|---------------------|
| Sharpe ratio | > 1.0 | Must be highest |
| Max drawdown | < 15% | Must be lowest |
| Win rate | > 55% | — |
| OOS total return | — | Must beat MACD+OBV (+74%) |

If the ML ensemble cannot beat existing strategies, report the comparison but do not add to catalog.

### Success Artifacts

- Trained ensemble models → `output/ml/spy_ensemble/`
- Surrogate tree PineScript → `output/pinescript/spy_ml_ensemble.pine`
- Tournament comparison report → terminal output
- If winner: strategy factory added to `spy_strategies.py` catalog

## Data Sources

### Price Data (Yahoo Finance)

- Daily: 10-year SPY history (`range_=10y, interval=1d`)
- Intraday: 60-day 5-minute bars (`range_=60d, interval=5m`)
- Fetched fresh at pipeline start using `YahooDataClient`

### Credentials (Environment Variables)

API credentials are passed via environment variables or runtime arguments — never hardcoded:

- `POLYGON_API_KEY` — for high-frequency intraday data if needed
- `FINRA_CLIENT_ID` / `FINRA_CLIENT_SECRET` — for short interest / dark pool data

### Feature Set

All 63 Pine-compatible features from `FeatureExtractor`:
- 26 normalized technical indicators
- 17 atomic signal scores
- 5 regime features
- 15 enhanced features (lags, volatility, momentum derivatives, interactions, calendar)

Alternative data features (17) and HMM features (5) are excluded from surrogate training since they map to constants in PineScript.

## New Files

| File | Purpose |
|------|---------|
| `src/stockdownloader/ml/ensemble.py` | `EnsemblePredictor`, `EnsembleBuilder` |
| `src/stockdownloader/ml/deep_surrogate.py` | Extended `DecisionTreeExporter` with regression mode, depth 10, fidelity measurement |
| `src/stockdownloader/app/spy_ml_ensemble.py` | CLI entry point — orchestrates the full 4-phase pipeline |
| `src/stockdownloader/app/pinescript_catalog/spy_ml_ensemble.py` | PineScript strategy factory with ML-driven exits, visual overlays |
| `tests/ml/test_ensemble.py` | Unit tests for ensemble predictor |
| `tests/ml/test_deep_surrogate.py` | Unit tests for regression surrogate + fidelity |
| `tests/pinescript/test_spy_ml_ensemble.py` | Tests for PineScript generation |

## Modified Files

| File | Change |
|------|--------|
| `src/stockdownloader/pinescript/ml_export.py` | Add `DeepSurrogateExporter` (regression mode, configurable depth/features) |
| `src/stockdownloader/app/pinescript_catalog/catalogs.py` | Register new strategy in `STRATEGY_CATALOG` |
| `pyproject.toml` | Add `spy-ml-ensemble` entry point |
