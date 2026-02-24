"""Machine learning layer for signal confidence boosting.

Exports
-------
Training & Inference:
    MLTrainer, MLModelConfig, TrainingResult
    MLPredictor

Feature Engineering:
    FeatureExtractor, FeatureVector

Dataset Construction:
    DatasetBuilder, LabelConfig, MLDataset

Model Persistence:
    ModelStore, ModelMetadata

Regime Detection:
    HMMRegimeDetector, RegimeSnapshot

Alternative Data:
    AlternativeDataStore, AlternativeDataSnapshot
"""

from stockdownloader.ml.alt_data_store import (
    AlternativeDataSnapshot,
    AlternativeDataStore,
)
from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig, MLDataset
from stockdownloader.ml.feature_extractor import FeatureExtractor, FeatureVector
from stockdownloader.ml.hmm_detector import HMMRegimeDetector, RegimeSnapshot
from stockdownloader.ml.model_store import ModelMetadata, ModelStore
from stockdownloader.ml.predictor import MLPredictor
from stockdownloader.ml.trainer import MLModelConfig, MLTrainer, TrainingResult

__all__ = [
    # Training & Inference
    "MLTrainer",
    "MLModelConfig",
    "TrainingResult",
    "MLPredictor",
    # Feature Engineering
    "FeatureExtractor",
    "FeatureVector",
    # Dataset Construction
    "DatasetBuilder",
    "LabelConfig",
    "MLDataset",
    # Model Persistence
    "ModelStore",
    "ModelMetadata",
    # Regime Detection
    "HMMRegimeDetector",
    "RegimeSnapshot",
    # Alternative Data
    "AlternativeDataStore",
    "AlternativeDataSnapshot",
]
