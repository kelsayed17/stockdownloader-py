# ML Algorithm Expansion — Design

## Goal

Add XGBoost, LightGBM, CatBoost, SVM, MLP, and KNN to the ML training infrastructure, implement diversity-aware ensemble selection with stacking, and create a `spy-ml-mega` pipeline that exhaustively searches all algorithms to build the best possible ensemble for PineScript export.

## Architecture: Dual-Track

**Track A — Grid Expansion**: Add 6 new model types to `MLTrainer._build_model()` so they integrate with the existing training pipeline (`TrainingResult`, expanding-window CV, `EnsembleBuilder`, `DeepSurrogateExporter`).

**Track B — Mega Pipeline**: Create `spy-ml-mega` CLI that runs an 18-config grid across all model types, uses diversity-aware ensemble selection, and optionally builds a stacking ensemble.

## Track A: New Model Types in MLTrainer

### Models

| `model_type` | Library | Key configuration |
|---|---|---|
| `"xgboost"` | XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `eval_metric="logloss"`, `use_label_encoder=False` |
| `"lightgbm"` | LightGBM | `n_estimators`, `max_depth`, `learning_rate`, `verbose=-1` (suppress output) |
| `"catboost"` | CatBoost | `iterations=n_estimators`, `depth=max_depth`, `learning_rate`, `verbose=0` |
| `"svm"` | sklearn SVC | `kernel="rbf"`, `probability=True`, `C=1.0`, `gamma="scale"` |
| `"mlp"` | sklearn MLPClassifier | `hidden_layer_sizes=(128, 64)`, `activation="relu"`, `max_iter=500`, `early_stopping=True` |
| `"knn"` | sklearn KNeighborsClassifier | `n_neighbors=15`, `weights="distance"`, `n_jobs=-1` |

### Integration

Each model produces a sklearn-compatible object with `predict_proba()`. All existing infrastructure works unchanged:
- `MLTrainer.train()` → `TrainingResult` (with OOS accuracy, AUC, feature importances)
- `EnsembleBuilder.select_top(n)` / `select_diverse(n)` → `EnsemblePredictor`
- `DeepSurrogateExporter` trains regression surrogate on ensemble probabilities
- PineScript export via `spy_ml_ensemble_strategy()`

### Graceful Degradation

XGBoost, LightGBM, and CatBoost imports wrapped in try/except. If not installed, `_build_model()` raises `ImportError` with: `"model_type='xgboost' requires: pip install xgboost>=2.0"`.

### Feature Importances

- XGBoost / LightGBM / CatBoost: `model.feature_importances_` (same as sklearn trees)
- SVM: `abs(model.coef_)` for linear kernel, uniform for non-linear (no native importances)
- MLP: `abs(model.coefs_[0]).mean(axis=1)` (input layer weights)
- KNN: uniform (no native importances)

### Class Balance

- XGBoost: `scale_pos_weight = n_neg / n_pos`
- LightGBM: `is_unbalance=True`
- CatBoost: `auto_class_weights="Balanced"`
- SVM: `class_weight="balanced"` (always on, inherent to margin)
- MLP / KNN: no class balance (not applicable)

## Track B: Diversity-Aware Ensemble Selection

### `EnsembleBuilder.select_diverse(n, diversity_weight=0.4)`

Instead of picking top N by accuracy:

1. **Seed** with the #1 model by OOS accuracy
2. **Greedily add** the candidate maximizing: `score = (1 - w) * accuracy + w * (1 - max_corr)` where `max_corr` is the maximum Pearson correlation between the candidate's OOS predictions and any existing member's predictions
3. **Repeat** until N models selected

This ensures the ensemble contains models that disagree — a strong XGBoost + a strong SVM + a strong KNN will outperform 3 strong boosters that all predict the same outputs.

### Stacking Ensemble

New `ensemble_method` parameter on `EnsemblePredictor`:

- `"soft_vote"` (default): Current behavior — average `predict_proba` across models
- `"stacking"`: Train a `LogisticRegression` meta-learner on the base models' OOS predictions. The meta-learner learns optimal per-model weights. Uses temporal CV for meta-learner training (no leakage).

Stacking requires OOS predictions from each base model. These are already available in `TrainingResult` (the model is evaluated on the held-out test set). The meta-learner trains on the test-set predictions and corresponding labels.

## Track B: spy-ml-mega Pipeline

### Grid (18 configs)

| Model Type | Balanced | Unbalanced |
|---|---|---|
| gradient_boosting | ✓ | ✓ |
| random_forest | ✓ | ✓ |
| extra_trees | ✓ | ✓ |
| hist_gradient_boosting | ✓ | ✓ |
| logistic_regression | — | ✓ |
| xgboost | ✓ | ✓ |
| lightgbm | ✓ | ✓ |
| catboost | ✓ | ✓ |
| svm | — | ✓ |
| mlp | — | ✓ |
| knn | — | ✓ |

Quick mode: 6 configs (xgboost, lightgbm, catboost, random_forest, logistic_regression, svm — unbalanced only).

### Pipeline Stages (8)

1. Download SPY data (reuses existing `YahooDataClient`)
2. Build feature dataset (reuses existing `DatasetBuilder`)
3. Train full model grid (18 configs)
4. **Rank & report** — print leaderboard sorted by accuracy, print prediction correlation matrix
5. **Build ensemble** using `select_diverse(n)` — diversity-aware selection
6. **Optional stacking** — compare soft-vote vs. stacking ensemble, pick the better one
7. Train deep surrogate, measure R²
8. Export PineScript + run portfolio tournament

### CLI Args

Inherits all common ML args plus:
- `--ensemble-method {soft_vote,stacking,both}` (default: `both`)
- `--diversity-weight` (default: 0.4)
- `--top-models` (default: 5, larger than current 3 since we have more diverse models)

### Output

Same PineScript strategy factory (`spy_ml_ensemble_strategy`) and same `_run_tournament()` — the mega pipeline just produces a better ensemble.

## Dependencies

Add to `pyproject.toml` `[ml]` extra:

```toml
ml = [
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
    "xgboost>=2.0",
    "lightgbm>=4.0",
    "catboost>=1.2",
]
```

## New Files

| File | Purpose |
|---|---|
| `src/stockdownloader/app/spy_ml_mega.py` | CLI entry point for 8-stage mega pipeline |
| `tests/ml/test_ensemble_diversity.py` | Tests for `select_diverse()` and stacking |
| `tests/app/test_spy_ml_mega.py` | Tests for mega pipeline arg parsing and grid construction |

## Modified Files

| File | Change |
|---|---|
| `src/stockdownloader/ml/trainer.py` | Add 6 new model types to `_build_model()` with graceful ImportError |
| `src/stockdownloader/ml/ensemble.py` | Add `select_diverse()`, stacking `ensemble_method` |
| `pyproject.toml` | Add xgboost, lightgbm, catboost to `[ml]` extra; add `spy-ml-mega` entry point |
| `tests/ml/test_trainer.py` | Parameterized tests for each new model type |
| `tests/ml/test_ensemble.py` | Tests for `select_diverse()` |
