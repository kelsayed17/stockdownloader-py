# ML Package Restructure Design (Round 7)

**Goal:** Populate the empty `ml/__init__.py` and `ml/pipeline/__init__.py` with comprehensive exports.

---

## Current State

The ml/ package is 5,068 lines across 18 files with good internal structure. The problem is visibility — both `__init__.py` files are effectively empty:

- `ml/__init__.py`: 1 export (`AlternativeDataStore`) out of ~13 major public classes
- `ml/pipeline/__init__.py`: 0 exports despite containing orchestrator, 8 config classes, 11 result classes, and 6 stage classes

## What We're Doing

**Fix 1: Expand `ml/__init__.py`** — Add exports for all major public classes:
- Training: `MLTrainer`, `MLModelConfig`, `TrainingResult`
- Features: `FeatureExtractor`, `FeatureVector`
- Dataset: `DatasetBuilder`, `LabelConfig`, `MLDataset`
- Inference: `MLPredictor`
- Storage: `ModelStore`, `ModelMetadata`
- HMM: `HMMRegimeDetector`, `RegimeSnapshot`
- Alt data: `AlternativeDataStore`, `AlternativeDataSnapshot`

**Fix 2: Expand `ml/pipeline/__init__.py`** — Add exports for:
- Orchestrator: `MLPipelineOrchestrator`
- Config: `PipelineConfig`, `DataConfig`, `TrainingGridConfig`, `ConvergenceConfig`, `HybridConfig`, `BacktestConfig`, `SelectionConfig`, `AltDataConfig`
- Results: `PipelineResult`, `DataResult`, `TrainingStageResult`, `ConvergenceResult`, `HybridStageResult`, `BacktestStageResult`
- Stages: `DataStage`, `TrainingStage`, `ConvergenceStage`, `HybridStage`, `BacktestStage`, `SelectionStage`

## What We're NOT Doing

- **Not splitting `trainer.py`** (786 lines) — It has clear internal structure (config, CV, trainer, scaled model wrapper) and changing it would require updating 17 test files.
- **Not splitting `stage_backtest.py`** (658 lines) — Single-responsibility stage with complex walk-forward logic.
- **Not renaming `_ALT_DATA_NAMES`** — It's a module-level constant used externally; renaming would be a breaking change.
