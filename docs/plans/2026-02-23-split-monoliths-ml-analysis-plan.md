# Round 14: Split Monoliths — ml/ & analysis/ Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split 4 monoliths (trainer.py 786 lines, options_gamma_analyzer.py 748 lines, value_screener.py 662 lines, stage_backtest.py 658 lines) into focused modules, targeting all files under 500 lines. feature_extractor.py (602 lines) inspected and deemed marginal — skip.

**Architecture:** Extract the largest self-contained chunk from each file — tuning/CV from trainer, GEX/max-pain static methods from options analyzer, scoring dimension functions from value screener, walk-forward methods from backtest stage — into dedicated modules. Original files keep thin wrappers calling extracted functions for full backward compatibility.

**Tech Stack:** Python, no new dependencies

---

### Task 1: Extract tuning & CV logic from trainer.py

**Files:**
- Create: `src/stockdownloader/ml/trainer_tuning.py`
- Modify: `src/stockdownloader/ml/trainer.py`

**What to extract** — all hyperparameter tuning, CV, and helper classes:

Move these to `trainer_tuning.py` as module-level functions/classes:

1. `TimeSeriesExpandingCV` class (lines 99-154, **55 lines**) — temporal expanding cross-validation. Move as-is.

2. `ScaledModel` class (was `_ScaledModel`, lines 754-773, **19 lines**) — wraps (scaler, model) for inference. Rename to drop leading underscore since it's now a public module member.

3. `grid_combos(grid)` function (was `_grid_combos`, lines 780-787, **7 lines**) — cartesian product of param grid.

4. `try_ensemble(trainer, X_train, y_train, X_test, y_test, sample_weight)` function (was `_try_ensemble`, lines 489-532, **43 lines**) — trains primary + LR ensemble. Takes `trainer` instance as first param to access `trainer._build_model()` and `trainer.config.model_type`.

5. `log_baseline_comparison(X_train, y_train, X_test, y_test, model_accuracy)` function (was `_log_baseline_comparison`, lines 710-746, **36 lines**) — trains LR baseline, logs improvement. Uses module-level `logger`, no instance state.

6. `run_tuning(trainer, dataset, param_grid=None)` function (was `train_with_tuning`, lines 562-708, **146 lines**) — full temporal grid search. Takes `trainer: MLTrainer` as first param to access `trainer._select_features()`, `trainer._compute_sample_weights()`, `trainer.config`, `trainer.train()`.

The new module needs these imports:
- `logging`, `itertools.product`, `numpy`, `sklearn` (LogisticRegression, accuracy_score, roc_auc_score)
- `MLModelConfig`, `TrainingResult` from `trainer` (use `TYPE_CHECKING` guard to avoid circular import)

**In trainer.py**: Replace extracted bodies with imports + thin wrappers:
```python
from stockdownloader.ml.trainer_tuning import (
    TimeSeriesExpandingCV,
    ScaledModel as _ScaledModel,
    grid_combos as _grid_combos,
    try_ensemble as _try_ensemble_impl,
    log_baseline_comparison as _log_baseline_impl,
    run_tuning as _run_tuning_impl,
)

# In MLTrainer class:
def _try_ensemble(self, X_train, y_train, X_test, y_test, sample_weight):
    return _try_ensemble_impl(self, X_train, y_train, X_test, y_test, sample_weight)

def train_with_tuning(self, dataset, param_grid=None):
    return _run_tuning_impl(self, dataset, param_grid)

def _log_baseline_comparison(self, X_train, y_train, X_test, y_test, model_accuracy):
    return _log_baseline_impl(X_train, y_train, X_test, y_test, model_accuracy)
```

This preserves all existing imports: `TimeSeriesExpandingCV` re-exported, `MLTrainer.train_with_tuning()` method still exists.

**Tests:**
1. Run: `python3 -m pytest tests/ml/test_trainer.py -x -q`
2. Run: `python3 -m pytest tests/ml/pipeline/test_stage_backtest.py tests/ml/pipeline/test_stage_training.py -x -q`
3. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract tuning & CV logic from MLTrainer`

### Task 2: Extract GEX, max-pain, and signal interpreters from options_gamma_analyzer.py

**Files:**
- Create: `src/stockdownloader/analysis/options_gex.py`
- Modify: `src/stockdownloader/analysis/options_gamma_analyzer.py`

**What to extract** — all static computational methods:

Move these to `options_gex.py` as module-level functions (all were `@staticmethod`, no instance state):

GEX computation:
- `compute_gex_by_strike(chain, price)` (was `_compute_gex_by_strike`, lines 341-435, **94 lines**) — aggregates call/put gamma by strike
- `compute_expiry_gex(chain, expiry, price)` (was `_compute_expiry_gex`, lines 437-447, **10 lines**) — single-expiry net GEX
- `find_gex_flip(gex_map, price)` (was `_find_gex_flip`, lines 449-479, **30 lines**) — nearest sign-change strike

Max pain:
- `compute_max_pain(chain, price)` (was `_compute_max_pain`, lines 485-530, **45 lines**) — OI-based max pain
- `compute_max_pain_from_volume(chain, price)` (was `_compute_max_pain_from_volume`, lines 532-573, **41 lines**) — volume-based fallback

Signal interpretation:
- `interpret_gex(net_gex, price)` (was `_interpret_gex`, lines 632-645, **13 lines**)
- `interpret_max_pain(max_pain, price)` (was `_interpret_max_pain`, lines 647-657, **10 lines**)
- `interpret_pcr(pcr_vol, pcr_oi)` (was `_interpret_pcr`, lines 659-671, **12 lines**)
- `interpret_pcr_volume_only(pcr_vol)` (was `_interpret_pcr_volume_only`, lines 673-684, **11 lines**)
- `combine_signals(gex_signal, mp_signal, pcr_signal)` (was `_combine_signals`, lines 686-708, **22 lines**)

The new module needs: `logging`, `StrikeGamma` (import from `options_gamma_analyzer` or move the dataclass). Move `StrikeGamma` to `options_gex.py` since it's only used by GEX computation. Re-export it from `options_gamma_analyzer.py` if anyone imports it directly (currently no external imports of `StrikeGamma`).

Also move `_MULT = 100` constant (used by `compute_gex_by_strike`).

**In options_gamma_analyzer.py**: Replace `@staticmethod` methods with calls to imported functions:
```python
from stockdownloader.analysis.options_gex import (
    StrikeGamma,
    compute_gex_by_strike,
    compute_expiry_gex,
    find_gex_flip,
    compute_max_pain,
    compute_max_pain_from_volume,
    interpret_gex,
    interpret_max_pain,
    interpret_pcr,
    interpret_pcr_volume_only,
    combine_signals,
)
```

Update `analyze()` method to call module-level functions instead of `self._compute_*` / `self._interpret_*`. Also update `_find_unusual_activity()` to use `_MULT` from gex module or keep a local copy (it uses `_MULT` for unusual volume threshold — check if it does).

**Tests:**
1. No dedicated test file exists for options_gamma_analyzer. Run import verification:
   ```bash
   python3 -c "from stockdownloader.analysis.options_gamma_analyzer import OptionsGammaAnalyzer, OptionsFlowReport; print('OK')"
   python3 -c "from stockdownloader.analysis import OptionsGammaAnalyzer, OptionsFlowReport; print('OK')"
   ```
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract GEX & max-pain calculators from OptionsGammaAnalyzer`

### Task 3: Extract scoring dimensions from value_screener.py

**Files:**
- Create: `src/stockdownloader/analysis/value_scoring.py`
- Modify: `src/stockdownloader/analysis/value_screener.py`

**What to extract** — all dimension scorers + helpers:

Move these to `value_scoring.py` as module-level functions. All `_score_*` methods used `self._w` (weights dict) and `self._continuous_score` — pass `w` as parameter and import `continuous_score` directly.

Helpers (both were `@staticmethod`):
- `continuous_score(value, best, worst, max_points)` (was `_continuous_score`, lines 631-650, **19 lines**) — linear interpolation scorer
- `graham_number(eps, book_value)` (was `graham_number`, lines 652-663, **11 lines**) — `sqrt(22.5 * EPS * BV)`. Note: this was already a public static method.
- `compute_piotroski(quote, detailed)` (was `_compute_piotroski`, lines 562-625, **63 lines**) — 9-point F-Score

Dimension scorers (convert `self._w` → `w` parameter, `self._continuous_score(...)` → `continuous_score(...)`):
- `score_valuation(w, quote, detailed)` (was `_score_valuation`, lines 281-365, **84 lines**)
- `score_growth(w, quote, detailed)` (was `_score_growth`, lines 367-419, **52 lines**)
- `score_quality(w, detailed, quote)` (was `_score_quality`, lines 421-486, **65 lines**)
- `score_income(w, quote, detailed)` (was `_score_income`, lines 488-556, **68 lines**)

The new module needs: `math`, `Decimal`, `TYPE_CHECKING` imports for `DetailedFinancialData`, `QuoteData`.

Also move `_ZERO` and `_ONE` Decimal constants if they're only used by scoring functions. If `coarse_filter` or `score()` also use them, keep copies in both files or import from value_scoring.

**In value_screener.py**: Replace method bodies with calls to imported functions:
```python
from stockdownloader.analysis.value_scoring import (
    continuous_score,
    graham_number,
    compute_piotroski,
    score_valuation,
    score_growth,
    score_quality,
    score_income,
)

class ValueScreener:
    def _score_valuation(self, quote, detailed):
        return score_valuation(self._w, quote, detailed)

    def _score_growth(self, quote, detailed):
        return score_growth(self._w, quote, detailed)

    def _score_quality(self, detailed, quote):
        return score_quality(self._w, detailed, quote)

    def _score_income(self, quote, detailed):
        return score_income(self._w, quote, detailed)

    @staticmethod
    def _compute_piotroski(quote, detailed):
        return compute_piotroski(quote, detailed)

    @staticmethod
    def _continuous_score(value, best, worst, max_points):
        return continuous_score(value, best, worst, max_points)

    @staticmethod
    def graham_number(eps, book_value):
        return graham_number(eps, book_value)
```

**Tests:**
1. Run: `python3 -m pytest tests/analysis/test_value_screener.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract scoring dimensions from ValueScreener`

### Task 4: Extract walk-forward logic from stage_backtest.py

**Files:**
- Create: `src/stockdownloader/ml/pipeline/walk_forward.py`
- Modify: `src/stockdownloader/ml/pipeline/stage_backtest.py`

**What to extract** — walk-forward validation methods:

Move these to `walk_forward.py` as module-level functions:

- `recreate_strategy(entry)` (was `_recreate_strategy`, lines 641-658, **17 lines**, @staticmethod) — rebuilds baseline strategy from registry
- `walk_forward(config, entry, data, initial_capital, commission, slippage_pct, compute_score_fn)` (was `_walk_forward`, lines 349-441, **92 lines**) — simple temporal walk-forward. Takes `config` and `compute_score_fn` instead of `self`.
- `parallel_walk_forward(config, top_entries, data, initial_capital, commission, slippage_pct, compute_score_fn, print_fn)` (was `_parallel_walk_forward`, lines 310-347, **37 lines**) — runs walk_forward for all entries via thread pool.
- `true_walk_forward(config, entry, data, initial_capital, commission, slippage_pct, compute_score_fn, print_fn, data_result)` (was `_true_walk_forward`, lines 447-639, **192 lines**) — full ML retraining per window. Uses lazy imports for `DatasetBuilder`, `FeatureExtractor`, `MLTrainer`, `HybridStage`.

The new module needs: `logging`, `concurrent.futures.ThreadPoolExecutor`, `BacktestEngine`, `BacktestConfig`, result types.

For `walk_forward` and `true_walk_forward`: they used `self.config` and `self._compute_score()`. Pass `config` directly and `compute_score_fn` as a callable parameter (it's a `@staticmethod` so can be passed as `BacktestStage._compute_score`).

For `true_walk_forward`: uses lazy imports inside the method body for `DatasetBuilder`, `FeatureExtractor`, `MLTrainer`, `HybridStage` — keep the lazy imports in the extracted function.

**In stage_backtest.py**: Replace method bodies with calls to imported functions:
```python
from stockdownloader.ml.pipeline.walk_forward import (
    walk_forward as _walk_forward_impl,
    parallel_walk_forward as _parallel_walk_forward_impl,
    true_walk_forward as _true_walk_forward_impl,
    recreate_strategy,
)

class BacktestStage:
    def _walk_forward(self, entry, data, initial_capital, commission, slippage_pct):
        return _walk_forward_impl(
            self.config, entry, data, initial_capital, commission, slippage_pct,
            self._compute_score,
        )

    def _parallel_walk_forward(self, top_entries, data, ...):
        return _parallel_walk_forward_impl(
            self.config, top_entries, data, ...,
            self._compute_score, self._print,
        )

    def _true_walk_forward(self, entry, data, ..., data_result):
        return _true_walk_forward_impl(
            self.config, entry, data, ...,
            self._compute_score, self._print, data_result,
        )

    @staticmethod
    def _recreate_strategy(entry):
        return recreate_strategy(entry)
```

**Tests:**
1. Run: `python3 -m pytest tests/ml/pipeline/test_stage_backtest.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract walk-forward logic from BacktestStage`

### Task 5: Verify line counts and full regression

**Steps:**

1. Verify line counts:
```bash
wc -l src/stockdownloader/ml/trainer.py \
      src/stockdownloader/ml/trainer_tuning.py \
      src/stockdownloader/analysis/options_gamma_analyzer.py \
      src/stockdownloader/analysis/options_gex.py \
      src/stockdownloader/analysis/value_screener.py \
      src/stockdownloader/analysis/value_scoring.py \
      src/stockdownloader/ml/pipeline/stage_backtest.py \
      src/stockdownloader/ml/pipeline/walk_forward.py
```

**Expected results:**
| File | Before | After |
|------|--------|-------|
| `trainer.py` | 786 | ~480 |
| `trainer_tuning.py` | (new) | ~320 |
| `options_gamma_analyzer.py` | 748 | ~460 |
| `options_gex.py` | (new) | ~310 |
| `value_screener.py` | 662 | ~300 |
| `value_scoring.py` | (new) | ~380 |
| `stage_backtest.py` | 658 | ~320 |
| `walk_forward.py` | (new) | ~360 |

2. Verify imports work:
```bash
python3 -c "from stockdownloader.ml.trainer import MLTrainer, MLModelConfig, TrainingResult, TimeSeriesExpandingCV; print('OK')"
python3 -c "from stockdownloader.analysis.options_gamma_analyzer import OptionsGammaAnalyzer, OptionsFlowReport; print('OK')"
python3 -c "from stockdownloader.analysis.value_screener import ValueScreener; print('OK')"
python3 -c "from stockdownloader.ml.pipeline.stage_backtest import BacktestStage; print('OK')"
```

3. Full regression: `python3 -m pytest tests/ -x -q`

---

## Verification

```bash
python3 -m pytest tests/ -x -q
wc -l src/stockdownloader/ml/trainer.py src/stockdownloader/ml/trainer_tuning.py
wc -l src/stockdownloader/analysis/options_gamma_analyzer.py src/stockdownloader/analysis/options_gex.py
wc -l src/stockdownloader/analysis/value_screener.py src/stockdownloader/analysis/value_scoring.py
wc -l src/stockdownloader/ml/pipeline/stage_backtest.py src/stockdownloader/ml/pipeline/walk_forward.py
```
